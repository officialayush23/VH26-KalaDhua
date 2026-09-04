"""The feature builder.

This module is the single most important file to keep in sync with the Rust
engine. ``engine/aura-core`` implements the identical maths on the hot path;
``tests/golden/feature_vectors.json`` is the shared fixture that both sides
assert against.

Design rules, in order of importance:

1. **No leakage.** Every feature emitted for the event at time ``t`` is derived
   only from events at or before ``t``. Concretely: per-key counters are
   *decayed and read* before the current access is folded in, and the running
   regeneration-cost quantiles are *read* before the current observation is
   folded in (the current regeneration latency is only known *after* the miss
   has been served, so it is future information at decision time).
2. **Streaming.** One pass over the trace in timestamp order, O(unique keys)
   memory. The Rust side has the same constraint, so nothing here may look
   ahead or hold the whole trace.
3. **Deterministic.** No hashing with a random seed, no dict-iteration order
   dependence, no floating point reductions whose order could change.

Feature vector, in contract order (contract section 5 ``feature_names``)::

    0  log_age_ms             ln(1 + ms since this key's previous access)
    1  log_inter_arrival_ms   ln(1 + EWMA(gap), alpha = 0.3)
    2  freq_1m                decayed access counter, time constant 60 s
    3  freq_5m                decayed access counter, time constant 300 s
    4  freq_1h                decayed access counter, time constant 3600 s
    5  ewma_fast              decayed access counter, half-life 5 s
    6  ewma_slow              decayed access counter, half-life 60 s
    7  trend                  ln((ewma_fast + eps) / (ewma_slow + eps))
    8  acceleration           trend - previous trend for this key
    9  log_size_bytes         ln(1 + size_bytes)
    10 log_regen_p50_ms       ln(1 + running p50 regen latency for the group)
    11 cost_variance_ratio    regen_p95_ms / max(regen_p50_ms, 1)
    12 regen_cost_usd         priced cost vector (contract section 8 pricing)
    13 ttl_remaining_frac     freshness of the resident copy, in [0, 1]
    14 cache_pressure         used_bytes / capacity_bytes before this access
    15 app_id                 stable small integer per application name
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

from .config import FeatureConfig, Pricing

FEATURE_NAMES: tuple[str, ...] = (
    "log_age_ms",
    "log_inter_arrival_ms",
    "freq_1m",
    "freq_5m",
    "freq_1h",
    "ewma_fast",
    "ewma_slow",
    "trend",
    "acceleration",
    "log_size_bytes",
    "log_regen_p50_ms",
    "cost_variance_ratio",
    "regen_cost_usd",
    "ttl_remaining_frac",
    "cache_pressure",
    "app_id",
)

N_FEATURES = len(FEATURE_NAMES)

# Feature groups, used by the ablation harness in train_gbdt.py.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "recency": ("log_age_ms", "log_inter_arrival_ms"),
    "frequency": ("freq_1m", "freq_5m", "freq_1h"),
    "trend": ("ewma_fast", "ewma_slow", "trend", "acceleration"),
    "cost": ("log_regen_p50_ms", "cost_variance_ratio", "regen_cost_usd"),
    "context": ("log_size_bytes", "ttl_remaining_frac", "cache_pressure", "app_id"),
}

# Applications AURA ships with. Anything else gets a deterministic hashed id so
# that a new application never renumbers an existing one.
KNOWN_APPLICATIONS: dict[str, int] = {
    "recommendation": 0,
    "analytics": 1,
    "content": 2,
}
_HASHED_APP_BASE = 3
_HASHED_APP_MODULO = 1021

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_U64 = 0xFFFFFFFFFFFFFFFF

_LN2 = 0.6931471805599453


def fnv1a64(text: str) -> int:
    """FNV-1a over the UTF-8 bytes of ``text``. Trivial to reimplement in Rust."""
    h = _FNV64_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV64_PRIME) & _U64
    return h


def app_id(application: str) -> int:
    """Stable small integer per application name.

    Known applications keep their reserved ids forever. Unknown ones are mapped
    into ``[3, 1023]`` by FNV-1a, which is stable across processes, machines and
    languages (unlike Python's salted ``hash``).
    """
    known = KNOWN_APPLICATIONS.get(application)
    if known is not None:
        return known
    return _HASHED_APP_BASE + int(fnv1a64(application) % _HASHED_APP_MODULO)


@dataclass(slots=True)
class AccessEvent:
    """One row of the trace, already parsed. Contract section 4 column order."""

    ts_ms: float
    key_id: int
    application: str
    object_type: str
    size_bytes: int
    ttl_ms: float
    sla_class: str
    cpu_ms: float
    gpu_ms: float
    db_ms: float
    network_bytes: float
    api_calls: float
    api_cost_usd: float
    regen_latency_ms: float
    scenario: str
    regime: str


@dataclass(slots=True)
class _KeyState:
    last_ts_ms: float
    ewma_gap_ms: float = 0.0
    seen_gap: bool = False
    freq: list[float] = field(default_factory=list)
    ewma_fast: float = 0.0
    ewma_slow: float = 0.0
    prev_trend: float = 0.0
    seen_trend: bool = False


@dataclass(slots=True)
class _CostState:
    """Streaming pinball-loss quantile estimators for regeneration latency.

    A full histogram per (application, object_type) would be exact but would not
    survive the Rust hot path, so both sides use the same SGD update:

        lr   = quantile_lr * max(estimate, 1.0)
        est += lr * q          if observation >  estimate
        est -= lr * (1 - q)    if observation <= estimate

    seeded with the first observation.
    """

    p50: float = 0.0
    p95: float = 0.0
    seen: bool = False


@dataclass(slots=True)
class _Entry:
    size_bytes: int
    fill_ts_ms: float
    ttl_ms: float


class CachePressureSim:
    """A minimal LRU replica whose only job is to produce ``cache_pressure``.

    ``cache_pressure`` is defined by the contract as ``used_bytes /
    capacity_bytes`` *at that moment*. Offline we do not have the engine's
    occupancy series, so we reconstruct a deterministic one: an LRU cache of the
    configured capacity, filled on miss or on TTL expiry. This is not the
    engine's policy, and it is not meant to be -- it is a stable, reproducible
    occupancy signal that correlates with the pressure the engine will see.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = max(int(capacity_bytes), 1)
        self.used_bytes = 0
        self._entries: OrderedDict[int, _Entry] = OrderedDict()

    @property
    def pressure(self) -> float:
        return min(self.used_bytes / self.capacity_bytes, 1.0)

    def resident(self, key_id: int, now_ms: float) -> _Entry | None:
        entry = self._entries.get(key_id)
        if entry is None:
            return None
        if entry.ttl_ms > 0.0 and now_ms - entry.fill_ts_ms >= entry.ttl_ms:
            return None
        return entry

    def touch(self, key_id: int, now_ms: float, size_bytes: int, ttl_ms: float) -> bool:
        """Serve one access. Returns True on a hit."""
        entry = self.resident(key_id, now_ms)
        if entry is not None:
            self._entries.move_to_end(key_id)
            return True
        self._fill(key_id, now_ms, size_bytes, ttl_ms)
        return False

    def _fill(self, key_id: int, now_ms: float, size_bytes: int, ttl_ms: float) -> None:
        stale = self._entries.pop(key_id, None)
        if stale is not None:
            self.used_bytes -= stale.size_bytes
        size = max(int(size_bytes), 0)
        if size > self.capacity_bytes:
            return
        while self.used_bytes + size > self.capacity_bytes and self._entries:
            _, victim = self._entries.popitem(last=False)
            self.used_bytes -= victim.size_bytes
        self._entries[key_id] = _Entry(size, now_ms, ttl_ms)
        self.used_bytes += size


class FeatureBuilder:
    """Stateful, single-pass builder. Feed it events in timestamp order."""

    def __init__(
        self,
        cfg: FeatureConfig | None = None,
        pricing: Pricing | None = None,
    ) -> None:
        self.cfg = cfg or FeatureConfig()
        self.pricing = pricing or Pricing()
        self._keys: dict[int, _KeyState] = {}
        self._costs: dict[tuple[str, str], _CostState] = {}
        self.cache = CachePressureSim(self.cfg.sim_capacity_bytes)
        self._n_windows = len(self.cfg.freq_windows_ms)
        self._last_ts_ms: float = float("-inf")

    # -- helpers ---------------------------------------------------------

    def _quantile_update(self, state: _CostState, observation: float) -> None:
        if not state.seen:
            state.p50 = observation
            state.p95 = observation
            state.seen = True
            return
        lr = self.cfg.quantile_lr
        step50 = lr * max(state.p50, 1.0)
        state.p50 += step50 * 0.5 if observation > state.p50 else -step50 * 0.5
        step95 = lr * max(state.p95, 1.0)
        state.p95 += step95 * 0.95 if observation > state.p95 else -step95 * 0.05
        state.p95 = max(state.p95, state.p50)

    # -- main ------------------------------------------------------------

    def transform(self, event: AccessEvent) -> list[float]:
        """Return the 16-element feature vector for ``event``.

        The builder mutates its internal state, so calling this twice with the
        same event is not idempotent -- that is intentional, it mirrors the
        engine, where every access advances the counters exactly once.
        """
        if event.ts_ms < self._last_ts_ms:
            raise ValueError(
                f"trace is not sorted by ts_ms ({event.ts_ms} after {self._last_ts_ms}); "
                "sort the trace before building features"
            )
        self._last_ts_ms = event.ts_ms

        # 14: read occupancy before this access mutates it.
        cache_pressure = self.cache.pressure

        # 13: freshness of the copy that is resident right now.
        entry = self.cache.resident(event.key_id, event.ts_ms)
        if entry is None or entry.ttl_ms <= 0.0:
            ttl_remaining_frac = 0.0
        else:
            elapsed = event.ts_ms - entry.fill_ts_ms
            ttl_remaining_frac = min(max(1.0 - elapsed / entry.ttl_ms, 0.0), 1.0)

        state = self._keys.get(event.key_id)
        first_access = state is None
        if state is None:
            state = _KeyState(last_ts_ms=event.ts_ms, freq=[0.0] * self._n_windows)
            self._keys[event.key_id] = state
        dt_ms = 0.0 if first_access else max(event.ts_ms - state.last_ts_ms, 0.0)

        # 0, 1: recency. The first access of a key has no measurable gap, so it
        # reports zero rather than a fabricated one.
        log_age_ms = math.log1p(dt_ms)
        if not first_access:
            if state.seen_gap:
                alpha = self.cfg.inter_arrival_alpha
                state.ewma_gap_ms = alpha * dt_ms + (1.0 - alpha) * state.ewma_gap_ms
            else:
                state.ewma_gap_ms = dt_ms
                state.seen_gap = True
        log_inter_arrival_ms = math.log1p(state.ewma_gap_ms)

        # 2-4: decayed frequency counters. Decay, read, then fold in this access.
        freqs: list[float] = []
        for i, window_ms in enumerate(self.cfg.freq_windows_ms):
            decayed = state.freq[i] * math.exp(-dt_ms / window_ms) if dt_ms > 0.0 else state.freq[i]
            freqs.append(decayed)
            state.freq[i] = decayed + 1.0

        # 5-8: trend. Same decay-read-fold order, half-life parameterised.
        dt_s = dt_ms / 1000.0
        fast = state.ewma_fast * math.exp(-_LN2 * dt_s / self.cfg.half_life_fast_s)
        slow = state.ewma_slow * math.exp(-_LN2 * dt_s / self.cfg.half_life_slow_s)
        state.ewma_fast = fast + 1.0
        state.ewma_slow = slow + 1.0
        eps = self.cfg.trend_eps
        trend = math.log((fast + eps) / (slow + eps))
        acceleration = trend - state.prev_trend if state.seen_trend else 0.0
        state.prev_trend = trend
        state.seen_trend = True

        state.last_ts_ms = event.ts_ms

        # 9: size.
        log_size_bytes = math.log1p(max(float(event.size_bytes), 0.0))

        # 10-11: regeneration cost shape, read before folding in this event.
        group = (event.application, event.object_type)
        cost_state = self._costs.get(group)
        if cost_state is None:
            cost_state = _CostState()
            self._costs[group] = cost_state
        regen_p50 = cost_state.p50
        regen_p95 = cost_state.p95
        log_regen_p50_ms = math.log1p(max(regen_p50, 0.0))
        cost_variance_ratio = regen_p95 / max(regen_p50, 1.0)
        if event.regen_latency_ms > 0.0:
            self._quantile_update(cost_state, event.regen_latency_ms)

        # 12: priced cost vector.
        regen_cost_usd = self.pricing.regen_cost_usd(
            event.cpu_ms,
            event.gpu_ms,
            event.db_ms,
            event.network_bytes,
            event.api_cost_usd,
        )

        # Advance the occupancy replica last, so nothing above saw it.
        self.cache.touch(event.key_id, event.ts_ms, event.size_bytes, event.ttl_ms)

        return [
            log_age_ms,
            log_inter_arrival_ms,
            freqs[0],
            freqs[1],
            freqs[2],
            fast,
            slow,
            trend,
            acceleration,
            log_size_bytes,
            log_regen_p50_ms,
            cost_variance_ratio,
            regen_cost_usd,
            ttl_remaining_frac,
            cache_pressure,
            float(app_id(event.application)),
        ]

    def transform_many(self, events: Iterable[AccessEvent]) -> Iterator[list[float]]:
        for event in events:
            yield self.transform(event)


def feature_dict(vector: list[float]) -> dict[str, float]:
    """Name a raw feature vector. Useful in tests and in the explain payload."""
    if len(vector) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {len(vector)}")
    return dict(zip(FEATURE_NAMES, vector, strict=True))


def vector_from_dict(named: Mapping[str, float]) -> list[float]:
    """Inverse of :func:`feature_dict`, in contract order."""
    return [float(named[name]) for name in FEATURE_NAMES]
