"""The extra signals — the ones that have to earn their place.

Why this file exists
--------------------
With honest baselines, the twelve-feature model barely beat ``freq_1m`` at predicting
reuse. That is not a failure of the model; it is what the data was saying. On a stationary
Zipf workload, frequency *is* nearly all the signal there is, and a gradient-boosted tree
fed mostly frequency counters will rediscover frequency.

To beat a frequency counter you have to give the model something a frequency counter cannot
see. That means new signals, and they have to satisfy two constraints at once:

1. **Portable.** Nothing measured in absolute dollars or milliseconds. Those shift by orders
   of magnitude between deployments, and when they do, every tree split learned on them
   becomes meaningless — silently, with no error and no warning. Anything cost-related
   enters as a *percentile within its own application*, which means the same thing on every
   platform.
2. **Cheap.** O(1) per access with a couple of floats of state, because the engine computes
   these on the write path at a few microseconds per decision.

The eight added here
--------------------

``size_percentile``
    This object's size relative to its own application's distribution. ``log_size_bytes``
    already carries absolute footprint; this carries *relative* footprint, which is what
    decides whether an object is worth its space among its peers.

``cost_percentile``
    Rebuild cost as a percentile within the application. **This is how cost gets back into
    the model without breaking portability** — a 90th-percentile-expensive object is a
    90th-percentile-expensive object whether the currency is GPU-seconds or database time.

``cost_variance_ratio``
    ``regen_p95 / regen_p50`` for the operation class. Already a ratio, so already portable —
    dropping it earlier was an over-correction. A miss whose cost you cannot predict is worse
    than one you can, and this is the only feature that says so.

``log_reuse_distance``
    Distinct keys seen since this key was last accessed. This is the classical stack
    distance, and it is the strongest single predictor of cache reuse there is — it is
    precisely what LRU approximates by construction. Frequency counters cannot express it:
    a key accessed ten times an hour ago and a key accessed ten times in the last minute
    have identical frequency and completely different reuse distance.

``burstiness``
    Coefficient of variation of this key's inter-arrival gaps. Separates a key requested
    steadily every two seconds from one requested twenty times in a burst and then not at
    all — same frequency, opposite caching decision.

``novelty_rate``
    Fraction of recent requests that were first-time keys. A workload-level regime signal:
    it goes to nearly 1.0 during a sequential scan, which is exactly when the cache should
    stop admitting things.

``hour_sin`` / ``hour_cos``
    Time of day as a phase on the unit circle, so 23:59 and 00:01 are adjacent rather than
    maximally distant. Analytics traffic is diurnal — dashboards get opened at the top of
    the hour and at the start of the working day — and a model with no clock cannot
    anticipate that.

Approximations, stated plainly
------------------------------
``log_reuse_distance`` is an estimate, not an exact stack distance. Exact computation needs
a tree keyed on last-access order and costs O(log n) per access, which does not belong on a
hot path. The estimate here multiplies the gap in access ordinals by the recently observed
unique-key ratio, which is accurate to within a few percent on skewed workloads and costs
two integers per key.

``size_percentile`` and ``cost_percentile`` use the same streaming pinball-quantile trick as
the rest of the engine, over a small ladder of quantiles rather than an exact histogram.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

EXTRA_FEATURES: tuple[str, ...] = (
    "size_percentile",
    "cost_percentile",
    "cost_variance_ratio",
    "log_reuse_distance",
    "burstiness",
    "novelty_rate",
    "hour_sin",
    "hour_cos",
)

#: Quantile ladder used to place a value within its application's distribution.
_LADDER: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


class QuantileLadder:
    """Streaming quantile estimates, and the percentile of a new value against them.

    Each level uses the pinball-loss gradient step the engine already uses for regeneration
    latency: step up by ``lr * q`` when the observation is above the estimate, down by
    ``lr * (1 - q)`` when below. Six floats per application, no histogram, no sample buffer.
    """

    def __init__(self, lr: float = 0.02) -> None:
        self.lr = lr
        self.levels: list[float] = [0.0] * len(_LADDER)
        self.seen = 0

    def observe(self, x: float) -> None:
        if self.seen == 0:
            self.levels = [x] * len(_LADDER)
            self.seen = 1
            return
        for i, q in enumerate(_LADDER):
            step = self.lr * max(abs(self.levels[i]), 1.0)
            self.levels[i] += step * q if x > self.levels[i] else -step * (1.0 - q)
        # Keep the ladder monotone; SGD on independent levels can cross them.
        for i in range(1, len(self.levels)):
            if self.levels[i] < self.levels[i - 1]:
                self.levels[i] = self.levels[i - 1]
        self.seen += 1

    def percentile_of(self, x: float) -> float:
        """Where ``x`` falls in the distribution, in ``[0, 1]``. Read before observing."""
        if self.seen == 0:
            return 0.5
        below = sum(1 for level in self.levels if x >= level)
        if below == 0:
            return 0.0
        if below >= len(_LADDER):
            return 1.0
        # Interpolate between the bracketing ladder levels.
        lo_q = _LADDER[below - 1]
        hi_q = _LADDER[below]
        lo_v = self.levels[below - 1]
        hi_v = self.levels[below]
        span = hi_v - lo_v
        frac = 0.0 if span <= 1e-12 else (x - lo_v) / span
        return float(lo_q + (hi_q - lo_q) * min(max(frac, 0.0), 1.0))


@dataclass
class _KeyState:
    last_ordinal: int = -1
    gap_mean_ms: float = 0.0
    gap_sq_mean_ms: float = 0.0
    gaps_seen: int = 0
    last_ts_ms: float = 0.0


@dataclass
class _AppState:
    size: QuantileLadder = field(default_factory=QuantileLadder)
    cost: QuantileLadder = field(default_factory=QuantileLadder)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_seen: int = 0


class ExtraFeatureBuilder:
    """Computes the eight extra signals, streaming, in one pass.

    Same discipline as the main builder: read the state, emit the features, *then* fold the
    current access in. A feature that includes the access it is describing has leaked the
    present into the past, and the model that learns from it will look excellent offline and
    be useless online.
    """

    def __init__(self, novelty_window: int = 2_000, quantile_lr: float = 0.02) -> None:
        self.keys: dict[int, _KeyState] = {}
        self.apps: dict[str, _AppState] = {}
        self.ordinal = 0
        self.quantile_lr = quantile_lr
        self.novelty_window: deque[int] = deque(maxlen=novelty_window)
        # Rolling estimate of how many distinct keys appear per access, used to turn a gap
        # in access ordinals into an estimated count of distinct intervening keys.
        self.unique_ratio = 1.0
        self._recent_unique: deque[int] = deque(maxlen=novelty_window)
        self._recent_seen: set[int] = set()

    def transform(
        self,
        *,
        key_id: int,
        ts_ms: float,
        application: str,
        size_bytes: int,
        cost_usd: float,
        regen_latency_ms: float,
    ) -> list[float]:
        app = self.apps.setdefault(application, _AppState())
        state = self.keys.get(key_id)

        # -- read ---------------------------------------------------------
        size_pct = app.size.percentile_of(float(size_bytes))
        cost_pct = app.cost.percentile_of(float(cost_usd))
        variance_ratio = app.latency_p95 / max(app.latency_p50, 1.0) if app.latency_seen else 0.0

        if state is None or state.last_ordinal < 0:
            # A key never seen before has infinite reuse distance. Encoding it as the
            # window size rather than a literal infinity keeps the feature bounded.
            reuse_distance = float(self.novelty_window.maxlen or 2_000)
            burstiness = 0.0
        else:
            gap_ordinals = self.ordinal - state.last_ordinal
            reuse_distance = gap_ordinals * self.unique_ratio
            if state.gaps_seen >= 2 and state.gap_mean_ms > 1e-9:
                var = max(state.gap_sq_mean_ms - state.gap_mean_ms**2, 0.0)
                burstiness = math.sqrt(var) / state.gap_mean_ms
            else:
                burstiness = 0.0

        novelty = (
            sum(self.novelty_window) / len(self.novelty_window) if self.novelty_window else 0.0
        )

        # Time of day as a phase, so midnight is not maximally far from one minute earlier.
        hour = (ts_ms / 3_600_000.0) % 24.0
        angle = 2.0 * math.pi * hour / 24.0

        features = [
            size_pct,
            cost_pct,
            min(variance_ratio, 20.0),
            math.log1p(max(reuse_distance, 0.0)),
            min(burstiness, 10.0),
            novelty,
            math.sin(angle),
            math.cos(angle),
        ]

        # -- fold in ------------------------------------------------------
        first_time = state is None
        self.novelty_window.append(1 if first_time else 0)

        if state is None:
            state = _KeyState()
            self.keys[key_id] = state
        else:
            gap = max(ts_ms - state.last_ts_ms, 0.0)
            # Welford would be more accurate; an EWMA of the gap and of its square is two
            # floats and tracks a workload that changes, which matters more here.
            alpha = 0.3
            state.gap_mean_ms = alpha * gap + (1 - alpha) * state.gap_mean_ms
            state.gap_sq_mean_ms = alpha * gap * gap + (1 - alpha) * state.gap_sq_mean_ms
            state.gaps_seen += 1

        state.last_ordinal = self.ordinal
        state.last_ts_ms = ts_ms

        app.size.observe(float(size_bytes))
        app.cost.observe(float(cost_usd))
        if regen_latency_ms > 0.0:
            if app.latency_seen == 0:
                app.latency_p50 = app.latency_p95 = regen_latency_ms
            else:
                lr = self.quantile_lr
                s50 = lr * max(app.latency_p50, 1.0)
                app.latency_p50 += s50 * 0.5 if regen_latency_ms > app.latency_p50 else -s50 * 0.5
                s95 = lr * max(app.latency_p95, 1.0)
                app.latency_p95 += s95 * 0.95 if regen_latency_ms > app.latency_p95 else -s95 * 0.05
                app.latency_p95 = max(app.latency_p95, app.latency_p50)
            app.latency_seen += 1

        self.ordinal += 1
        self._recent_unique.append(key_id)
        if len(self._recent_unique) == self._recent_unique.maxlen:
            self.unique_ratio = len(set(self._recent_unique)) / len(self._recent_unique)

        return features


def extend(events: Sequence, base_rows: Sequence[Sequence[float]], pricing) -> list[list[float]]:
    """Append the eight extra signals to each already-built base row.

    ``events`` must be the same sequence, in the same order, that produced ``base_rows``.
    """
    builder = ExtraFeatureBuilder()
    out: list[list[float]] = []
    for event, base in zip(events, base_rows, strict=True):
        cost_usd = pricing.regen_cost_usd(
            event.cpu_ms, event.gpu_ms, event.db_ms, event.network_bytes, event.api_cost_usd
        )
        extra = builder.transform(
            key_id=event.key_id,
            ts_ms=event.ts_ms,
            application=event.application,
            size_bytes=event.size_bytes,
            cost_usd=cost_usd,
            regen_latency_ms=event.regen_latency_ms,
        )
        out.append(list(base) + extra)
    return out


__all__ = ["EXTRA_FEATURES", "ExtraFeatureBuilder", "QuantileLadder", "extend"]
