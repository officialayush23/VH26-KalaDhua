"""Golden-vector tests for the feature builder.

``tests/golden/feature_vectors.json`` is a shared fixture: this test replays it
through the Python feature builder, and the Rust test in
``engine/aura-core/tests/feature_parity.rs`` replays the same file through the
Rust one. Both assert the same expected vectors, so the two implementations
cannot drift apart without a test going red on one side or the other.

Regenerate the fixture (only when the feature definitions intentionally change,
which is a contract change) with::

    python tests/test_features_parity.py --regenerate

Run the tests with ``pytest tests`` or, without pytest installed::

    python tests/test_features_parity.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aura_train.config import FeatureConfig, Pricing, TrainingConfig  # noqa: E402
from aura_train.export import (  # noqa: E402
    build_linear_bundle,
    bundle_schema_errors,
    predict_bundle,
)
from aura_train.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    AccessEvent,
    FeatureBuilder,
    app_id,
    fnv1a64,
)
from aura_train.labels import build_labels  # noqa: E402
from aura_train.train_linear import cold_start_model  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "feature_vectors.json"
TOLERANCE = 1e-9

# The exact feature list from contract section 5. Written out literally rather
# than imported, so that an accidental edit to features.py fails this test.
CONTRACT_FEATURE_NAMES = [
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
]


def golden_events() -> list[AccessEvent]:
    """A small, hand-built trace that exercises every branch of the builder.

    It deliberately contains: a key's first access, a repeat access at a
    non-trivial gap, two accesses at the same millisecond, a TTL expiry, an
    object larger than the (tiny) simulated capacity, two applications, and one
    application name that is not in the reserved id table.
    """
    def event(**kwargs: object) -> AccessEvent:
        base = {
            "ts_ms": 0.0,
            "key_id": 1,
            "application": "recommendation",
            "object_type": "ranking_result",
            "size_bytes": 1_000_000,
            "ttl_ms": 60_000.0,
            "sla_class": "high",
            "cpu_ms": 320.0,
            "gpu_ms": 80.0,
            "db_ms": 140.0,
            "network_bytes": 1_000_000.0,
            "api_calls": 1.0,
            "api_cost_usd": 0.002,
            "regen_latency_ms": 540.0,
            "scenario": "golden",
            "regime": "steady",
        }
        base.update(kwargs)
        return AccessEvent(**base)  # type: ignore[arg-type]

    return [
        event(ts_ms=0.0, key_id=1),
        event(ts_ms=1_000.0, key_id=2, application="analytics", object_type="aggregate",
              size_bytes=41_000, ttl_ms=600_000.0, cpu_ms=60.0, gpu_ms=0.0, db_ms=240.0,
              api_calls=0.0, api_cost_usd=0.0, regen_latency_ms=300.0, sla_class="normal"),
        event(ts_ms=1_500.0, key_id=1),
        event(ts_ms=1_500.0, key_id=1),
        event(ts_ms=9_000.0, key_id=1),
        event(ts_ms=9_500.0, key_id=2, application="analytics", object_type="aggregate",
              size_bytes=41_000, ttl_ms=600_000.0, cpu_ms=60.0, gpu_ms=0.0, db_ms=240.0,
              api_calls=0.0, api_cost_usd=0.0, regen_latency_ms=900.0, sla_class="normal"),
        event(ts_ms=70_000.0, key_id=1),
        event(ts_ms=70_100.0, key_id=3, application="content", object_type="rendered_page",
              size_bytes=180_000, ttl_ms=120_000.0, cpu_ms=45.0, gpu_ms=0.0, db_ms=30.0,
              api_calls=0.0, api_cost_usd=0.0, regen_latency_ms=90.0, sla_class="normal"),
        event(ts_ms=70_200.0, key_id=4, application="billing", object_type="invoice",
              size_bytes=8_000, ttl_ms=30_000.0, cpu_ms=12.0, gpu_ms=0.0, db_ms=95.0,
              api_calls=2.0, api_cost_usd=0.0004, regen_latency_ms=140.0, sla_class="critical"),
        event(ts_ms=71_000.0, key_id=5, size_bytes=64_000_000, ttl_ms=60_000.0),
        event(ts_ms=90_000.0, key_id=3, application="content", object_type="rendered_page",
              size_bytes=180_000, ttl_ms=120_000.0, cpu_ms=45.0, gpu_ms=0.0, db_ms=30.0,
              api_calls=0.0, api_cost_usd=0.0, regen_latency_ms=110.0, sla_class="normal"),
        event(ts_ms=600_000.0, key_id=1),
    ]


def golden_feature_config() -> FeatureConfig:
    """A small capacity, so the golden trace actually exercises eviction."""
    return FeatureConfig(sim_capacity_bytes=2_000_000)


def build_golden() -> dict[str, object]:
    cfg = golden_feature_config()
    pricing = Pricing()
    builder = FeatureBuilder(cfg, pricing)
    cases = []
    for event in golden_events():
        vector = builder.transform(event)
        cases.append({"event": asdict(event), "features": vector})
    return {
        "description": (
            "Golden feature vectors shared by the Python and Rust feature builders. "
            "Replay the events in order through a fresh builder configured with "
            "`config` below and compare each vector to `features` within 1e-9."
        ),
        "feature_names": list(FEATURE_NAMES),
        "config": {
            "inter_arrival_alpha": cfg.inter_arrival_alpha,
            "freq_windows_ms": list(cfg.freq_windows_ms),
            "half_life_fast_s": cfg.half_life_fast_s,
            "half_life_slow_s": cfg.half_life_slow_s,
            "trend_eps": cfg.trend_eps,
            "quantile_lr": cfg.quantile_lr,
            "sim_capacity_bytes": cfg.sim_capacity_bytes,
            "pricing": {
                "cpu_ms_usd": pricing.cpu_ms_usd,
                "gpu_ms_usd": pricing.gpu_ms_usd,
                "db_ms_usd": pricing.db_ms_usd,
                "network_gb_usd": pricing.network_gb_usd,
            },
        },
        "app_ids": {
            "recommendation": 0,
            "analytics": 1,
            "content": 2,
            "billing": app_id("billing"),
        },
        "cases": cases,
    }


def regenerate() -> Path:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build_golden(), indent=2) + "\n")
    return GOLDEN_PATH


def load_golden() -> dict[str, object]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} is missing; regenerate it with "
            "`python tests/test_features_parity.py --regenerate`"
        )
    return json.loads(GOLDEN_PATH.read_text())


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_feature_names_match_contract() -> None:
    assert list(FEATURE_NAMES) == CONTRACT_FEATURE_NAMES
    assert N_FEATURES == 16


def test_golden_vectors() -> None:
    golden = load_golden()
    assert golden["feature_names"] == CONTRACT_FEATURE_NAMES
    builder = FeatureBuilder(golden_feature_config(), Pricing())
    for i, case in enumerate(golden["cases"]):
        event = AccessEvent(**case["event"])
        got = builder.transform(event)
        expected = case["features"]
        assert len(got) == N_FEATURES
        for name, a, b in zip(FEATURE_NAMES, got, expected, strict=True):
            assert math.isclose(a, b, rel_tol=0.0, abs_tol=TOLERANCE), (
                f"case {i} feature {name}: {a!r} != {b!r}"
            )


def test_golden_config_matches_defaults() -> None:
    """The fixture must not silently encode different constants to the library."""
    golden = load_golden()
    cfg = golden_feature_config()
    stored = golden["config"]
    assert stored["inter_arrival_alpha"] == cfg.inter_arrival_alpha
    assert stored["freq_windows_ms"] == list(cfg.freq_windows_ms)
    assert stored["half_life_fast_s"] == cfg.half_life_fast_s
    assert stored["half_life_slow_s"] == cfg.half_life_slow_s
    assert stored["quantile_lr"] == cfg.quantile_lr
    assert stored["sim_capacity_bytes"] == cfg.sim_capacity_bytes


def test_app_id_is_stable_across_processes() -> None:
    """``app_id`` must not depend on PYTHONHASHSEED."""
    assert app_id("recommendation") == 0
    assert app_id("analytics") == 1
    assert app_id("content") == 2
    assert app_id("billing") == app_id("billing")
    assert 3 <= app_id("billing") < 3 + 1021
    # FNV-1a reference vectors.
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_features_do_not_look_ahead() -> None:
    """Appending future events must not change any earlier feature vector."""
    events = golden_events()
    prefix = FeatureBuilder(golden_feature_config(), Pricing())
    prefix_vectors = [prefix.transform(e) for e in events[:6]]
    full = FeatureBuilder(golden_feature_config(), Pricing())
    full_vectors = [full.transform(e) for e in events]
    for i, (a, b) in enumerate(zip(prefix_vectors, full_vectors[:6], strict=True)):
        assert a == b, f"vector {i} changed when later events were appended"


def test_unsorted_trace_is_rejected() -> None:
    builder = FeatureBuilder(golden_feature_config(), Pricing())
    events = golden_events()
    builder.transform(events[4])
    try:
        builder.transform(events[0])
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("out-of-order event was accepted")


def test_labels_and_censoring() -> None:
    key_ids = [1, 2, 1, 3, 1]
    ts_ms = [0.0, 100.0, 5_000.0, 6_000.0, 90_000.0]
    result = build_labels(key_ids, ts_ms, [10_000, 60_000], trace_end_ms=90_000.0)
    # key 1 at t=0 is reused at 5000 -> positive at both horizons
    assert result.labels[10_000][0] == 1
    # key 2 is never reused, and 100 + 10_000 <= 90_000, so it is a true negative
    assert result.labels[10_000][1] == 0
    assert result.censored[10_000][1] == 0
    # the final access has no future left to observe at either horizon
    assert result.labels[10_000][4] == 0
    assert result.censored[10_000][4] == 1
    assert result.censored[60_000][4] == 1
    # key 1 at t=5000 is reused at 90_000: outside 10 s, outside 60 s
    assert result.labels[10_000][2] == 0
    assert result.labels[60_000][2] == 0


def test_linear_bundle_round_trip() -> None:
    """The reference scorer in export.py must reproduce the model exactly."""
    model = cold_start_model(TrainingConfig().primary_horizon_ms)
    bundle = build_linear_bundle(model)
    assert bundle_schema_errors(bundle) == []
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, N_FEATURES)) * np.array([abs(v) for v in model.scale]) + model.mean
    delta = float(np.max(np.abs(predict_bundle(bundle, x) - model.predict(x))))
    assert delta < 1e-12, delta


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a test harness has to report everything
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        print(f"wrote {regenerate()}")
        raise SystemExit(0)
    raise SystemExit(_run_all())
