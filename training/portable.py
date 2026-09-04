"""Train the portable reuse model, and keep retraining it.

One command, three jobs:

    python -m portable bootstrap    # synthetic data -> first bundle, from nothing
    python -m portable train        # build dataset, train, evaluate, export
    python -m portable watch        # do that on a loop, forever, publishing each time

What the model is
-----------------
Three binary classifiers, one per horizon, each answering exactly one question:

    given what I know about this object right now, will it be requested again
    within 10 / 60 / 600 seconds?

Nothing about cost, value, or eviction. The economics are deterministic and live in the
engine, downstream of this. That separation is what lets a price change take effect
instantly without a retrain, and what lets the model work on an application it has never
seen.

The twelve features
-------------------
Deliberately a *subset* of the sixteen the engine computes. The four that were dropped —
``regen_cost_usd``, ``log_regen_p50_ms``, ``cost_variance_ratio`` and ``app_id`` — are all
platform-bound. Absolute dollars and milliseconds shift by orders of magnitude between
deployments, so a tree split learned on one is meaningless on another, and the failure is
silent: no error, just quietly worse decisions. ``app_id`` is worse still, a categorical
that cannot generalise to an application it never saw during training.

What remains describes *behaviour*, and behaviour transfers:

    recency     log_age_ms, log_inter_arrival_ms
    frequency   freq_1m, freq_5m, freq_1h
    trend       ewma_fast, ewma_slow, trend, acceleration
    context     log_size_bytes, ttl_remaining_frac, cache_pressure

Where the data comes from
-------------------------
In priority order, and it falls through automatically:

1. **The running engine.** ``GET /v1/training/rows`` drains the decision journal, which
   already holds the exact feature vector each decision was made from plus the labels that
   arrived later. This is the best source by a wide margin — the engine computed those
   features itself, so there is no possibility of drift between training and serving — and
   it is what makes retraining continuous rather than a one-off.
2. **The user simulator's request log**, ``apps/runs/requests.jsonl``, replayed through the
   feature builder.
3. **Synthetic traces**, for bootstrapping from nothing.

Labels are built by looking *forward* in the trace, which is legitimate at training time and
impossible at serving time — the split that keeps this honest is in `_label`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aura_train.config import FeatureConfig, Pricing  # noqa: E402
from features_v2 import EXTRA_FEATURES, extend  # noqa: E402
from aura_train.features import (  # noqa: E402
    FEATURE_NAMES,
    AccessEvent,
    CachePressureSim,
    FeatureBuilder,
)

LOG = logging.getLogger("aura.portable")

# --------------------------------------------------------------------------- the contract

PORTABLE_FEATURES: tuple[str, ...] = (
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
    "ttl_remaining_frac",
    "cache_pressure",
)

DROPPED_FEATURES: tuple[str, ...] = (
    "log_regen_p50_ms",
    "cost_variance_ratio",
    "regen_cost_usd",
    "app_id",
)

#: The full model input: twelve portable behavioural features plus the eight extra signals
#: that a frequency counter cannot express. Absolute cost never appears; cost enters as a
#: percentile within its own application, which means the same thing on every platform.
MODEL_FEATURES: tuple[str, ...] = PORTABLE_FEATURES + EXTRA_FEATURES

HORIZONS_MS: tuple[int, ...] = (10_000, 60_000, 600_000)

#: Index of each portable feature within the engine's full 16-element vector.
PORTABLE_INDEX: tuple[int, ...] = tuple(FEATURE_NAMES.index(name) for name in PORTABLE_FEATURES)

assert len(PORTABLE_FEATURES) == 12
assert set(PORTABLE_FEATURES) | set(DROPPED_FEATURES) == set(FEATURE_NAMES)


def horizon_label(ms: int) -> str:
    return f"h{ms // 1000}s"


# --------------------------------------------------------------------------- dataset


@dataclass
class Dataset:
    x: np.ndarray
    y: dict[str, np.ndarray]
    t: np.ndarray
    application: np.ndarray
    source: str = "unknown"
    feature_names: list[str] = field(default_factory=lambda: list(PORTABLE_FEATURES))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self),
            "features": self.x.shape[1] if len(self) else 0,
            "source": self.source,
            "applications": sorted({str(a) for a in self.application.tolist()}),
            "positive_rate": {k: round(float(v.mean()), 4) for k, v in self.y.items()},
        }


def _label(events: Sequence[AccessEvent], horizons_ms: Sequence[int]) -> dict[str, np.ndarray]:
    """Future-reuse labels, by one backward pass over the trace.

    For the access to key ``k`` at time ``t``, the label for horizon ``h`` is 1 if ``k``
    appears again at any time in ``(t, t + h]``. Nothing but the *next* occurrence matters,
    so one reverse pass with a dictionary of "when did I last see this key, reading
    backwards" is enough.

    Rows whose horizon extends past the end of the trace are **censored**: we cannot know
    whether reuse happened, and labelling them 0 would teach the model that the end of every
    trace is a cold region. They are dropped by :func:`build_dataset`.
    """
    n = len(events)
    next_at = np.full(n, np.inf)
    seen: dict[int, float] = {}
    for i in range(n - 1, -1, -1):
        key = events[i].key_id
        if key in seen:
            next_at[i] = seen[key]
        seen[key] = events[i].ts_ms

    end = events[-1].ts_ms if n else 0.0
    out: dict[str, np.ndarray] = {}
    for h in horizons_ms:
        name = horizon_label(h)
        ts = np.array([e.ts_ms for e in events])
        out[name] = (next_at <= ts + h).astype(np.int8)
        out[f"censored_{name}"] = ((next_at == np.inf) & (ts + h > end)).astype(np.int8)
    return out


def build_dataset(
    events: Sequence[AccessEvent],
    *,
    sim_capacity_bytes: int = 536_870_912,
    source: str = "trace",
    extras: bool = True,
) -> Dataset:
    """Replay a trace chronologically, emitting one feature row per access.

    ``extras=False`` builds the twelve-feature baseline, so the value of the extra signals
    can be measured rather than assumed.
    """
    if not events:
        raise ValueError("no events to build a dataset from")

    events = sorted(events, key=lambda e: e.ts_ms)
    builder = FeatureBuilder(FeatureConfig(), Pricing())
    rows: list[list[float]] = []
    # `FeatureBuilder` keeps its own LRU replica to derive `cache_pressure` and
    # `ttl_remaining_frac`: the engine's real occupancy is not available offline, so both
    # sides agree on a deterministic reconstruction instead. Not the engine's policy, and
    # not meant to be — a reproducible occupancy signal that tracks what it will see.
    builder.cache = CachePressureSim(sim_capacity_bytes)
    for event in events:
        full = builder.transform(event)
        # `transform` returns all sixteen; keep the twelve that transfer between platforms.
        rows.append([full[i] for i in PORTABLE_INDEX])

    if extras:
        rows = extend(events, rows, Pricing())

    labels = _label(events, HORIZONS_MS)
    keep = np.ones(len(events), dtype=bool)
    for h in HORIZONS_MS:
        keep &= labels[f"censored_{horizon_label(h)}"] == 0

    x = np.asarray(rows, dtype=np.float64)[keep]
    y = {horizon_label(h): labels[horizon_label(h)][keep] for h in HORIZONS_MS}
    t = np.array([e.ts_ms for e in events])[keep]
    app = np.array([e.application for e in events])[keep]

    dropped = int((~keep).sum())
    if dropped:
        LOG.info("dropped %d censored row(s) at the end of the trace", dropped)
    return Dataset(x=x, y=y, t=t, application=app, source=source,
                   feature_names=list(MODEL_FEATURES) if extras else list(PORTABLE_FEATURES))


# --------------------------------------------------------------------------- sources


def events_from_engine(base_url: str, limit: int = 200_000) -> list[AccessEvent]:
    """Drain the engine's decision journal.

    These rows already carry the exact feature vector the engine used, so strictly speaking
    we do not need to recompute features from them at all — but going through the same
    builder keeps one code path and makes a drift bug impossible to hide.
    """
    url = f"{base_url.rstrip('/')}/v1/training/rows?limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        LOG.info("engine journal unavailable (%s); falling back", exc)
        return []

    rows = payload.get("rows", [])
    LOG.info("drained %d row(s) from the engine journal", len(rows))
    return [_event_from_journal(r) for r in rows]


def _event_from_journal(row: dict[str, Any]) -> AccessEvent:
    return AccessEvent(
        ts_ms=float(row.get("decided_ms", 0.0)),
        key_id=int(row.get("key", 0)),
        application=str(row.get("application", "unknown")),
        object_type=str(row.get("object_type", "object")),
        size_bytes=int(row.get("size_bytes", 0)),
        ttl_ms=float(row.get("ttl_ms", 0.0)),
        sla_class=str(row.get("sla_class", "normal")),
        cpu_ms=0.0,
        gpu_ms=0.0,
        db_ms=0.0,
        network_bytes=0.0,
        api_calls=0.0,
        api_cost_usd=0.0,
        regen_latency_ms=0.0,
        scenario=str(row.get("scenario", "live")),
        regime=str(row.get("regime", "live")),
    )


def events_from_request_log(path: Path, limit: int | None = None) -> list[AccessEvent]:
    """Replay the user simulator's per-request log.

    The log records what users actually asked for, including the epoch bumps a click causes,
    so the reuse structure in it is real rather than scripted. Object sizes are not in the
    log, so they are approximated per action — which is fine, because `log_size_bytes` is
    one feature of twelve and the reuse signal lives in the timing.
    """
    if not path.exists():
        LOG.info("request log %s not present; falling back", path)
        return []

    sizes = {
        "home": 1_800_000,
        "click": 1_800_000,
        "view": 240_000,
        "search": 90_000,
        "cart": 240_000,
        "purchase": 40_000,
        "dashboard": 60_000,
    }
    ttls = {"home": 300_000.0, "click": 300_000.0, "dashboard": 900_000.0}
    services = {"recommendation": "ranking_result", "analytics": "query_result", "content": "blob"}

    events: list[AccessEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = rec.get("action", "view")
            service = rec.get("service", "content")
            # The key is what the application would have used, epoch included. That epoch is
            # the whole reason a click produces an unavoidable miss.
            key = hash((service, action, rec.get("user_id"), rec.get("epoch"))) & 0xFFFFFFFF
            events.append(
                AccessEvent(
                    ts_ms=float(rec.get("t", 0.0)) * 1000.0,
                    key_id=key,
                    application=service,
                    object_type=services.get(service, "object"),
                    size_bytes=sizes.get(action, 100_000),
                    ttl_ms=ttls.get(action, 600_000.0),
                    sla_class="normal",
                    cpu_ms=0.0,
                    gpu_ms=0.0,
                    db_ms=0.0,
                    network_bytes=0.0,
                    api_calls=0.0,
                    api_cost_usd=0.0,
                    regen_latency_ms=float(rec.get("service_ms", 0.0)),
                    scenario="live",
                    regime="live",
                )
            )
    LOG.info("read %d event(s) from %s", len(events), path)
    return events


def events_synthetic(count: int = 400_000, seed: int = 42) -> list[AccessEvent]:
    """Bootstrap data covering every regime, so the first model is not blind.

    Every regime is concatenated onto one timeline rather than trained separately. The
    model has to learn a single rule that works under all of them, which is the whole
    point: a cache does not get told which regime it is in.
    """
    from aura_train.synthetic import REGIMES, generate_events

    per_regime = max(count // len(REGIMES), 5_000)
    events: list[AccessEvent] = []
    offset = 0.0
    for regime in REGIMES:
        batch = list(generate_events(regime, requests=per_regime, seed=seed))
        for e in batch:
            e.ts_ms += offset
            # Keep the keyspaces disjoint, or two regimes share keys and the reuse signal
            # from one leaks into the other.
            e.key_id += hash(regime) % 1_000_000 * 10
            events.append(e)
        if batch:
            offset = events[-1].ts_ms + 5_000.0
        LOG.info("synthetic regime %-22s %7d events", regime, len(batch))
    return events


def collect_events(args: argparse.Namespace) -> tuple[list[AccessEvent], str]:
    """Try each source in order of quality and return the first that yields enough."""
    if args.source in ("auto", "engine"):
        events = events_from_engine(args.aura_url, args.limit)
        if len(events) >= args.min_rows:
            return events, "engine_journal"
        if args.source == "engine":
            raise SystemExit(
                f"engine journal returned {len(events)} rows, need {args.min_rows}. "
                "Let the system run longer, or use --source auto."
            )

    if args.source in ("auto", "requests"):
        events = events_from_request_log(Path(args.request_log), args.limit)
        if len(events) >= args.min_rows:
            return events, "request_log"
        if args.source == "requests":
            raise SystemExit(f"request log has {len(events)} rows, need {args.min_rows}")

    LOG.info("using synthetic traces")
    return events_synthetic(args.synthetic_events, args.seed), "synthetic"


# --------------------------------------------------------------------------- splitting


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    kind: str
    held_out: str = ""


def make_split(ds: Dataset, hold_out_application: str | None) -> Split:
    """Split temporally, and optionally by application.

    Never randomly. A random split puts an access and its own future in the same set, which
    inflates every metric and produces a model that collapses the first time it meets a key
    it has not seen.

    Holding out an entire application is the stronger claim and the one worth making: it
    tests whether the model works on a workload it has never been trained on, which is
    exactly what happens when someone plugs a third application into the cache.
    """
    n = len(ds)
    order = np.argsort(ds.t, kind="stable")

    if hold_out_application:
        mask = ds.application == hold_out_application
        if mask.sum() < 50:
            LOG.warning(
                "application %r has only %d rows; falling back to a temporal split",
                hold_out_application,
                int(mask.sum()),
            )
        else:
            rest = order[~mask[order]]
            cut = int(len(rest) * 0.85)
            return Split(
                train=rest[:cut],
                val=rest[cut:],
                test=order[mask[order]],
                kind="held_out_application",
                held_out=hold_out_application,
            )

    a, b = int(n * 0.70), int(n * 0.85)
    return Split(train=order[:a], val=order[a:b], test=order[b:], kind="temporal")


# --------------------------------------------------------------------------- metrics


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC. No sklearn needed, and exact rather than approximated."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks within ties, or a constant predictor scores 1.0 instead of 0.5.
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision. The metric that matters when positives are rare."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y_sorted).sum() / y.sum())


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error, and the reliability table behind it.

    A model that says 0.7 should be right about 70% of the time. If it is not, the engine's
    economics are multiplying a number that does not mean what it claims.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table: list[dict[str, float]] = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not mask.any():
            continue
        conf, acc, share = float(p[mask].mean()), float(y[mask].mean()), float(mask.mean())
        ece += share * abs(conf - acc)
        table.append({"bin": round(float(edges[i]), 2), "predicted": round(conf, 4),
                      "actual": round(acc, 4), "share": round(share, 4)})
    return float(ece), table


# --------------------------------------------------------------------------- training


def _fit_one(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, seed: int) -> tuple[Any, str]:
    """LightGBM when available, histogram gradient boosting otherwise.

    The fallback is not a compromise for the demo: both produce the same bundle format, and
    `export.py` flattens either into the same tree encoding the Rust walker reads.
    """
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=63,
            learning_rate=0.06,
            n_estimators=400,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )
        model.fit(
            x, y,
            eval_set=[(xv, yv)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        return model, "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, n_iter_no_change=40, random_state=seed,
        )
        model.fit(x, y)
        return model, "sklearn_hgb"


def train(ds: Dataset, split: Split, seed: int = 42) -> dict[str, Any]:
    """Fit one model per horizon and evaluate each on the held-out set."""
    results: dict[str, Any] = {"split": split.kind, "held_out": split.held_out, "horizons": {}}

    for h in HORIZONS_MS:
        name = horizon_label(h)
        y = ds.y[name]
        xt, yt = ds.x[split.train], y[split.train]
        xv, yv = ds.x[split.val], y[split.val]
        xs, ys = ds.x[split.test], y[split.test]

        if yt.sum() == 0 or yt.sum() == len(yt):
            LOG.warning("horizon %s has a single class in training; skipping", name)
            continue

        model, backend = _fit_one(xt, yt, xv, yv, seed)
        p = model.predict_proba(xs)[:, 1]
        ece, table = calibration(ys, p)

        results["horizons"][name] = {
            "backend": backend,
            "model": model,
            "n_train": int(len(yt)),
            "n_test": int(len(ys)),
            "positive_rate_train": round(float(yt.mean()), 4),
            "positive_rate_test": round(float(ys.mean()), 4),
            "roc_auc": round(roc_auc(ys, p), 4),
            "pr_auc": round(pr_auc(ys, p), 4),
            "log_loss": round(log_loss(ys, p), 4),
            "calibration_error": round(ece, 4),
            "reliability": table,
        }
        LOG.info(
            "%s  auc=%.4f  pr_auc=%.4f  logloss=%.4f  ece=%.4f  positives=%.1f%%  (%s)",
            name,
            results["horizons"][name]["roc_auc"],
            results["horizons"][name]["pr_auc"],
            results["horizons"][name]["log_loss"],
            ece,
            100 * float(ys.mean()),
            backend,
        )
    return results


def replay_gain(ds: Dataset, split: Split, results: dict[str, Any]) -> dict[str, Any]:
    """Does the model make better *decisions* than the heuristics it has to beat?

    AUC is not the objective. This is a decision test: with room for only a fraction of the
    held-out objects, which ranking retains the ones that are actually reused within 60
    seconds? The model is compared against the two signals the classical policies use.

    - **LRU-like**: keep the most recently accessed. Note the correction below — a key's
      *first* access has ``log_age_ms == 0``, which naively reads as "just used" when it in
      fact means "never seen before". Ranking those first is how you accidentally build a
      baseline that looks terrible and flatter your own model by a factor of ten.
    - **LFU-like**: keep the most frequently accessed in the last minute.
    """
    entry = results["horizons"].get("h60s")
    if entry is None:
        return {}
    model = entry["model"]
    idx = split.test
    if len(idx) < 1_000:
        return {}

    x = ds.x[idx]
    truth = ds.y["h60s"][idx]
    age = x[:, ds.feature_names.index("log_age_ms")]
    freq = x[:, ds.feature_names.index("freq_1m")]

    # A first access reports age 0 and frequency 0 together. Treat it as maximally old for
    # the recency baseline, so LRU is judged on what it would actually do.
    first_access = freq <= 1e-9
    recency_score = np.where(first_access, -np.inf, -age)

    scores = {
        "model": model.predict_proba(x)[:, 1],
        "lru_like": recency_score,
        "lfu_like": freq,
    }

    def precision_at(score: np.ndarray, keep_fraction: float) -> float:
        k = max(1, int(len(score) * keep_fraction))
        finite = np.where(np.isfinite(score), score, -1e300)
        chosen = np.argpartition(-finite, k - 1)[:k]
        return float(truth[chosen].mean())

    out: dict[str, Any] = {"base_rate": round(float(truth.mean()), 4)}
    for frac in (0.05, 0.10, 0.25):
        pct = int(frac * 100)
        row = {name: round(precision_at(s, frac), 4) for name, s in scores.items()}
        best_baseline = max(row["lru_like"], row["lfu_like"])
        row["lift_vs_best_baseline"] = round(
            (row["model"] - best_baseline) / max(best_baseline, 1e-9), 4
        )
        out[f"keep_top_{pct}pct"] = row
    return out


# --------------------------------------------------------------------------- export


def export_bundles(results: dict[str, Any], out_dir: Path, ds: Dataset) -> list[Path]:
    """Write one Rust-loadable bundle per horizon."""
    from aura_train import export as export_mod

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for h in HORIZONS_MS:
        name = horizon_label(h)
        entry = results["horizons"].get(name)
        if entry is None:
            continue
        model = entry["model"]

        if entry["backend"] == "lightgbm":
            trees = export_mod.trees_from_lightgbm(model.booster_)
        else:
            trees = export_mod.trees_from_sklearn(model)

        bundle = {
            "schema_version": 1,
            "name": f"reuse_gbdt_{name}",
            "kind": "lightgbm_gbdt",
            "horizon_ms": h,
            "version": export_mod.utc_version(),
            "git_sha": export_mod.git_sha(),
            # The engine projects its own feature vector onto these names, so this list is
            # the contract. Twelve names, not sixteen.
            "feature_names": list(ds.feature_names),
            "normalization": None,
            "objective": "binary",
            "sigmoid_output": True,
            "trees": [t.to_json() for t in trees],
            "linear_weights": None,
            "metrics": {
                k: v for k, v in entry.items() if k not in ("model", "reliability")
            } | {"rows": len(ds), "source": ds.source},
        }

        path = out_dir / f"reuse_gbdt_{name}.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        written.append(path)
        LOG.info("wrote %s (%d trees, %.1f KB)", path, len(bundle["trees"]), path.stat().st_size / 1024)
    return written


def reload_engine(base_url: str, models_dir: Path) -> bool:
    """Tell a running engine to pick the new bundles up. No restart."""
    body = json.dumps({"source": "file", "path": str(models_dir)}).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{base_url.rstrip('/')}/v1/model/reload",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            LOG.info("engine reloaded: %s", resp.read().decode()[:200])
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        LOG.warning("engine reload failed (%s); the bundles are on disk regardless", exc)
        return False


# --------------------------------------------------------------------------- commands


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    events, source = collect_events(args)
    if len(events) < args.min_rows:
        raise SystemExit(f"only {len(events)} events available, need {args.min_rows}")

    ds = build_dataset(events, source=source, extras=not args.no_extras)
    LOG.info("dataset %s", json.dumps(ds.summary()))

    split = make_split(ds, args.hold_out_application)
    LOG.info(
        "split %s: train=%d val=%d test=%d%s",
        split.kind, len(split.train), len(split.val), len(split.test),
        f" (held out {split.held_out})" if split.held_out else "",
    )

    results = train(ds, split, seed=args.seed)
    gain = replay_gain(ds, split, results)
    if gain:
        LOG.info("replay gain %s", json.dumps(gain))

    written = export_bundles(results, Path(args.models_dir), ds)
    if args.reload and written:
        reload_engine(args.aura_url, Path(args.models_dir))

    report = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": ds.summary(),
        "split": {"kind": split.kind, "held_out": split.held_out,
                  "train": len(split.train), "test": len(split.test)},
        "horizons": {
            k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in results["horizons"].items()
        },
        "replay_gain": gain,
        "bundles": [str(p) for p in written],
    }
    report_path = Path(args.report_dir) / "portable_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("report written to %s", report_path)
    return report


def run_watch(args: argparse.Namespace) -> int:
    """Retrain forever.

    The interval is not a tuning knob so much as an honesty constraint: the journal settles
    decisions at 60 s and retires them at 600 s, so retraining more often than about ten
    minutes just refits the same rows.
    """
    LOG.info("retraining every %.0fs; ctrl-c to stop", args.every_s)
    round_no = 0
    while True:
        round_no += 1
        started = time.monotonic()
        try:
            report = run_once(args)
            best = report["horizons"].get("h60s", {})
            LOG.info(
                "round %d complete: %d rows, h60s auc=%s ece=%s",
                round_no, report["dataset"]["rows"],
                best.get("roc_auc"), best.get("calibration_error"),
            )
        except SystemExit as exc:
            LOG.warning("round %d skipped: %s", round_no, exc)
        except Exception as exc:  # noqa: BLE001 - a retraining loop must not die
            LOG.exception("round %d failed: %s", round_no, exc)

        sleep_for = max(5.0, args.every_s - (time.monotonic() - started))
        time.sleep(sleep_for)


def main() -> int:
    p = argparse.ArgumentParser(prog="portable", description="Train AURA's portable reuse model")
    p.add_argument("command", choices=["bootstrap", "train", "watch", "features"])
    p.add_argument("--source", default="auto", choices=["auto", "engine", "requests", "synthetic"])
    p.add_argument("--aura-url", default=os.environ.get("AURA_URL", "http://localhost:8080"))
    p.add_argument("--request-log", default="../apps/runs/requests.jsonl")
    p.add_argument("--models-dir", default="../engine/models")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--synthetic-events", type=int, default=400_000)
    p.add_argument("--limit", type=int, default=400_000)
    p.add_argument("--min-rows", type=int, default=5_000)
    p.add_argument("--hold-out-application", default=None,
                   help="train without this application and test only on it")
    p.add_argument("--every-s", type=float, default=600.0, help="watch mode interval")
    p.add_argument("--no-reload", dest="reload", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-extras", action="store_true",
                   help="train on the 12 base features only, to measure what the extras buy")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    if args.command == "features":
        print(f"{len(MODEL_FEATURES)} model features\n")
        print(f"  base ({len(PORTABLE_FEATURES)}) - behavioural, read from the engine vector:")
        for i, name in enumerate(PORTABLE_FEATURES):
            print(f"    {i:2d}  {name:<24} engine index {PORTABLE_INDEX[i]}")
        print(f"\n  extra ({len(EXTRA_FEATURES)}) - signals a frequency counter cannot express:")
        for i, name in enumerate(EXTRA_FEATURES, start=len(PORTABLE_FEATURES)):
            print(f"    {i:2d}  {name}")
        print(f"\n  removed as platform-bound: {', '.join(DROPPED_FEATURES)}")
        print("    absolute dollars and milliseconds do not transfer between deployments;")
        print("    cost re-enters as cost_percentile, which does.")
        return 0

    if args.command == "bootstrap":
        args.source = "synthetic"
    if args.command == "watch":
        return run_watch(args)

    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
