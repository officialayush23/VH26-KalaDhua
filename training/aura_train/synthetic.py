"""A pure-Python trace generator.

The Rust simulator is the real source of traces. This generator exists for three
reasons: it makes the test suite hermetic, it gives the Colab notebook a path
that never dead-ends when the user has no traces of their own, and it produces
every regime the split configuration expects, so the whole pipeline can be
exercised without a Rust toolchain.

It is deliberately simple and dependency-free (``random`` only, no numpy) so it
can be pasted into a fresh notebook if it ever has to be.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .features import AccessEvent, fnv1a64
from .traces import TraceMeta, write_aura_trace

LOG = logging.getLogger(__name__)

# regime -> (scenario id, description)
REGIMES: dict[str, str] = {
    "steady": "steady",
    "zipf_shift_moderate": "zipf_shift_moderate",
    "analytics_stable": "analytics_stable",
    "flash_crowd": "flash_crowd",
    "scan": "scan",
    "expensive_tail": "expensive_tail",
    "cost_spike": "cost_spike",
}


@dataclass(frozen=True)
class AppProfile:
    """Per-application cost and object shape, loosely matching the demo apps."""

    name: str
    object_type: str
    size_mean: int
    size_sigma: float
    ttl_ms: float
    cpu_ms: float
    gpu_ms: float
    db_ms: float
    api_cost_usd: float
    sla_class: str


APP_PROFILES: tuple[AppProfile, ...] = (
    AppProfile("recommendation", "ranking_result", 1_800_000, 0.45, 300_000,
               320.0, 80.0, 140.0, 0.002, "high"),
    AppProfile("analytics", "aggregate", 41_000, 0.8, 600_000,
               60.0, 0.0, 240.0, 0.0, "normal"),
    AppProfile("content", "rendered_page", 180_000, 0.6, 120_000,
               45.0, 0.0, 30.0, 0.0, "normal"),
)


def _zipf_weights(n: int, alpha: float) -> list[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def _cumulative(weights: list[float]) -> list[float]:
    out: list[float] = []
    running = 0.0
    for w in weights:
        running += w
        out.append(running)
    return out


def _sample(cumulative: list[float], rng: random.Random) -> int:
    target = rng.random()
    low, high = 0, len(cumulative) - 1
    while low < high:
        mid = (low + high) // 2
        if cumulative[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def generate_events(
    regime: str,
    requests: int,
    unique_keys: int = 4000,
    duration_s: float = 3600.0,
    seed: int = 42,
) -> Iterator[AccessEvent]:
    """Yield ``requests`` events for one regime.

    Each regime bends one dimension of the workload:

    ``steady``               stationary Zipf(0.9) over the whole key space.
    ``zipf_shift_moderate``  the popularity ranking rotates slowly.
    ``analytics_stable``     few keys, long TTLs, db-heavy, very regular reuse.
    ``flash_crowd``          a 60 s window where 20 keys take 80% of traffic.
    ``scan``                 a long sweep of never-reused keys over a warm set.
    ``expensive_tail``       reuse is uncorrelated with cost; the cheap head is
                             hot, and a small set of very expensive objects is
                             reused just often enough to be worth keeping.
    ``cost_spike``           regeneration cost of one application jumps 8x
                             halfway through.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}, expected one of {sorted(REGIMES)}")
    # fnv1a64, not the builtin hash(): PYTHONHASHSEED must not change a trace.
    rng = random.Random(seed ^ (fnv1a64(regime) & 0xFFFF))
    duration_ms = duration_s * 1000.0
    step_ms = duration_ms / max(requests, 1)

    alpha = 1.15 if regime == "analytics_stable" else 0.9
    keys = 400 if regime == "analytics_stable" else unique_keys
    cumulative = _cumulative(_zipf_weights(keys, alpha))
    permutation = list(range(keys))

    flash_keys = list(range(20))
    scan_cursor = keys

    for i in range(requests):
        ts_ms = i * step_ms
        progress = i / max(requests - 1, 1)

        if regime == "zipf_shift_moderate" and i % max(requests // 40, 1) == 0 and i:
            # Rotate the popularity ranking a little; keys keep their identity
            # but change rank, which is what a real popularity shift looks like.
            shift = max(keys // 50, 1)
            permutation = permutation[shift:] + permutation[:shift]

        if regime == "flash_crowd" and 0.4 <= progress < 0.47 and rng.random() < 0.8:
            key_index = rng.choice(flash_keys)
        elif regime == "scan" and 0.3 <= progress < 0.7 and rng.random() < 0.55:
            scan_cursor += 1
            key_index = scan_cursor
        else:
            key_index = permutation[_sample(cumulative, rng)]

        if regime == "analytics_stable":
            profile = APP_PROFILES[1]
        elif regime == "expensive_tail":
            profile = APP_PROFILES[0] if key_index % 37 == 0 else APP_PROFILES[2]
        else:
            profile = APP_PROFILES[key_index % len(APP_PROFILES)]

        drawn = rng.lognormvariate(math.log(profile.size_mean), profile.size_sigma)
        size_bytes = max(int(drawn), 64)

        cost_multiplier = 1.0
        if regime == "expensive_tail" and key_index % 37 == 0:
            cost_multiplier = 6.0
        if regime == "cost_spike" and progress >= 0.5 and profile.name == "analytics":
            cost_multiplier = 8.0

        cpu_ms = profile.cpu_ms * cost_multiplier * rng.uniform(0.7, 1.4)
        gpu_ms = profile.gpu_ms * cost_multiplier * rng.uniform(0.7, 1.4)
        db_ms = profile.db_ms * cost_multiplier * rng.uniform(0.6, 1.6)
        regen_latency_ms = (cpu_ms + gpu_ms + db_ms) * rng.uniform(0.9, 1.6)

        yield AccessEvent(
            ts_ms=ts_ms,
            key_id=int(key_index),
            application=profile.name,
            object_type=profile.object_type,
            size_bytes=size_bytes,
            ttl_ms=profile.ttl_ms,
            sla_class=profile.sla_class,
            cpu_ms=cpu_ms,
            gpu_ms=gpu_ms,
            db_ms=db_ms,
            network_bytes=float(size_bytes),
            api_calls=1.0 if profile.api_cost_usd > 0 else 0.0,
            api_cost_usd=profile.api_cost_usd * cost_multiplier,
            regen_latency_ms=regen_latency_ms,
            scenario=REGIMES[regime],
            regime=regime,
        )


def generate_trace_set(
    out_dir: Path,
    requests_per_regime: int = 60_000,
    regimes: tuple[str, ...] | None = None,
    unique_keys: int = 4000,
    duration_s: float = 3600.0,
    seed: int = 42,
) -> list[Path]:
    """Write one ``<regime>.csv.gz`` per regime and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = regimes or tuple(REGIMES)
    written: list[Path] = []
    for regime in chosen:
        path = out_dir / f"{regime}.csv.gz"
        events = generate_events(
            regime,
            requests=requests_per_regime,
            unique_keys=unique_keys,
            duration_s=duration_s,
            seed=seed,
        )
        meta = TraceMeta(
            scenario=REGIMES[regime],
            seed=seed,
            generator_version=1,
            applications=tuple(p.name for p in APP_PROFILES),
        )
        rows = write_aura_trace(path, events, meta)
        LOG.info("wrote %s (%d rows)", path, rows)
        written.append(path)
    return written
