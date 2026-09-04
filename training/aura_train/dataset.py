"""Dataset assembly: traces in, feature/label shards out.

The build is streaming per trace file: features are produced in one forward
pass (no lookahead, see ``features.py``), then labels are produced in one
reverse pass over the same trace's ``(key_id, ts_ms)`` columns. Only those two
columns plus the feature block for a single trace are ever resident, so a
100 M-row corpus builds by processing one trace at a time.

Shards are Parquet when ``pyarrow`` is installed and gzipped CSV otherwise, so
the pipeline runs on a bare Python install. The reader hides the difference.

The split is regime-stratified, never random -- see ``SplitConfig``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .features import FEATURE_NAMES, AccessEvent, FeatureBuilder
from .labels import build_labels
from .traces import TraceMeta, discover_traces, read_trace

LOG = logging.getLogger(__name__)

META_COLUMNS: tuple[str, ...] = (
    "ts_ms",
    "key_id",
    "application",
    "scenario",
    "regime",
    "size_bytes",
    "split",
)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_UNUSED = "unused"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def shard_suffix() -> str:
    return ".parquet" if _parquet_available() else ".csv.gz"


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write one shard, choosing the best available columnar format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, compression="gzip")
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def shard_paths(dataset_dir: Path) -> list[Path]:
    found = sorted(dataset_dir.glob("shard_*.parquet")) + sorted(dataset_dir.glob("shard_*.csv.gz"))
    return found


def load_dataset(cfg: TrainingConfig) -> pd.DataFrame:
    """Concatenate every shard. Small corpora only; the trainer streams shards."""
    paths = shard_paths(cfg.dataset_dir)
    if not paths:
        raise FileNotFoundError(
            f"no dataset shards under {cfg.dataset_dir}; run `build-dataset` first"
        )
    frames = [read_table(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


@dataclass
class BuildReport:
    """What a dataset build produced. Printed by the CLI, asserted by tests."""

    rows: int
    shards: list[Path]
    positive_rate: dict[str, float]
    rows_by_split: dict[str, int]
    rows_by_regime: dict[str, int]
    censored_dropped: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "split": list(self.rows_by_split),
                "rows": [self.rows_by_split[k] for k in self.rows_by_split],
            }
        )


def _assign_split(cfg: TrainingConfig, regime: str, ts_ms: np.ndarray) -> np.ndarray:
    """Regime-stratified split with a time-ordered validation tail."""
    n = len(ts_ms)
    out = np.full(n, SPLIT_UNUSED, dtype=object)
    if regime in cfg.split.test_regimes:
        out[:] = SPLIT_TEST
        return out
    if regime in cfg.split.train_regimes:
        if n == 0:
            return out
        cutoff = np.quantile(ts_ms, 1.0 - cfg.split.val_tail_fraction)
        out[:] = np.where(ts_ms <= cutoff, SPLIT_TRAIN, SPLIT_VAL)
        return out
    # A regime nobody declared: keep it, but only as extra test material.
    out[:] = SPLIT_TEST
    return out


def frame_from_events(
    cfg: TrainingConfig,
    events: Iterable[AccessEvent],
    drop_censored: bool = True,
) -> pd.DataFrame:
    """Turn one trace's events into a labelled feature frame."""
    builder = FeatureBuilder(cfg.features, cfg.pricing)
    rows: list[list[float]] = []
    ts_ms: list[float] = []
    key_ids: list[int] = []
    applications: list[str] = []
    scenarios: list[str] = []
    regimes: list[str] = []
    sizes: list[int] = []

    for event in events:
        rows.append(builder.transform(event))
        ts_ms.append(event.ts_ms)
        key_ids.append(event.key_id)
        applications.append(event.application)
        scenarios.append(event.scenario)
        regimes.append(event.regime)
        sizes.append(event.size_bytes)

    if not rows:
        return pd.DataFrame(columns=list(FEATURE_NAMES) + list(META_COLUMNS))

    frame = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=list(FEATURE_NAMES))
    frame["ts_ms"] = ts_ms
    frame["key_id"] = key_ids
    frame["application"] = applications
    frame["scenario"] = scenarios
    frame["regime"] = regimes
    frame["size_bytes"] = sizes

    labelled = build_labels(key_ids, ts_ms, cfg.horizons_ms, trace_end_ms=max(ts_ms))
    for horizon in cfg.horizons_ms:
        frame[cfg.label_column(horizon)] = np.asarray(labelled.labels[horizon], dtype=np.int8)
        frame[cfg.censored_column(horizon)] = np.asarray(
            labelled.censored[horizon], dtype=np.int8
        )

    regime = regimes[0] if regimes else "unknown"
    frame["split"] = _assign_split(cfg, regime, frame["ts_ms"].to_numpy())

    if drop_censored:
        # A row censored at the primary horizon is unusable for the model we
        # actually ship, so it goes. Rows censored only at longer horizons are
        # kept; the per-horizon trainer filters again on its own flag.
        keep = frame[cfg.censored_column(cfg.primary_horizon_ms)] == 0
        frame = frame.loc[keep].reset_index(drop=True)
    return frame


def build_dataset(
    cfg: TrainingConfig,
    trace_paths: Iterable[Path] | None = None,
    progress: bool = False,
) -> BuildReport:
    """Build the full dataset from every trace under ``cfg.trace_dir``."""
    cfg.ensure_dirs()
    paths = list(trace_paths) if trace_paths is not None else discover_traces(cfg.trace_dir)
    if not paths:
        raise FileNotFoundError(
            f"no traces found under {cfg.trace_dir}. Generate some with "
            "`python -m aura_train.cli synth`, or point AURA_TRAIN_TRACE_DIR at "
            "a directory of aura-bench traces."
        )

    for stale in shard_paths(cfg.dataset_dir):
        stale.unlink()

    suffix = shard_suffix()
    shards: list[Path] = []
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    total_rows = 0
    censored_dropped = 0
    rows_by_split: dict[str, int] = {}
    rows_by_regime: dict[str, int] = {}
    positives: dict[str, list[int]] = {cfg.label_column(h): [0, 0] for h in cfg.horizons_ms}

    def flush() -> None:
        nonlocal buffer, buffered_rows
        if not buffer:
            return
        frame = pd.concat(buffer, ignore_index=True)
        path = cfg.dataset_dir / f"shard_{len(shards):05d}{suffix}"
        write_table(frame, path)
        shards.append(path)
        LOG.info("wrote %s (%d rows)", path, len(frame))
        buffer = []
        buffered_rows = 0

    for path in paths:
        meta = TraceMeta.from_path(path)
        events = read_trace(path, limit=cfg.max_rows_per_trace)
        raw_count = 0

        def counted(source: Iterator[AccessEvent]) -> Iterator[AccessEvent]:
            nonlocal raw_count
            for event in source:
                raw_count += 1
                yield event

        frame = frame_from_events(cfg, counted(events))
        if frame.empty:
            LOG.warning("%s produced no rows", path)
            continue
        censored_dropped += raw_count - len(frame)
        total_rows += len(frame)

        for split, count in frame["split"].value_counts().items():
            rows_by_split[str(split)] = rows_by_split.get(str(split), 0) + int(count)
        for regime, count in frame["regime"].value_counts().items():
            rows_by_regime[str(regime)] = rows_by_regime.get(str(regime), 0) + int(count)
        for horizon in cfg.horizons_ms:
            column = cfg.label_column(horizon)
            positives[column][0] += int(frame[column].sum())
            positives[column][1] += int(len(frame))

        if progress:
            LOG.info(
                "%s scenario=%s rows=%d kept=%d",
                path.name,
                meta.scenario,
                raw_count,
                len(frame),
            )

        buffer.append(frame)
        buffered_rows += len(frame)
        if buffered_rows >= cfg.shard_rows:
            flush()

    flush()
    if not shards:
        raise RuntimeError("dataset build produced no shards")

    positive_rate = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in positives.items()}
    return BuildReport(
        rows=total_rows,
        shards=shards,
        positive_rate=positive_rate,
        rows_by_split=rows_by_split,
        rows_by_regime=rows_by_regime,
        censored_dropped=censored_dropped,
    )


def class_balance_table(report: BuildReport) -> pd.DataFrame:
    """The class-balance table the notebook prints after a build."""
    return pd.DataFrame(
        {
            "label": list(report.positive_rate),
            "positive_rate": [report.positive_rate[k] for k in report.positive_rate],
            "rows": [report.rows] * len(report.positive_rate),
        }
    )


def split_frames(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a loaded dataset into (train, val, test) views."""
    return (
        frame.loc[frame["split"] == SPLIT_TRAIN],
        frame.loc[frame["split"] == SPLIT_VAL],
        frame.loc[frame["split"] == SPLIT_TEST],
    )


def xy(
    frame: pd.DataFrame,
    label_column: str,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and label vector, in contract order."""
    names = list(feature_names)
    x = frame[names].to_numpy(dtype=np.float64, copy=False)
    y = frame[label_column].to_numpy(dtype=np.int32, copy=False)
    return x, y
