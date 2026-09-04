"""Command line entry point.

    python -m aura_train.cli synth          # generate traces (no Rust needed)
    python -m aura_train.cli build-dataset  # traces  -> feature/label shards
    python -m aura_train.cli train          # shards  -> boosters + linear model
    python -m aura_train.cli evaluate       # boosters-> per-regime tables + figures
    python -m aura_train.cli export         # boosters-> model_bundle.json + ONNX
    python -m aura_train.cli push           # bundles -> Supabase, flip is_active
    python -m aura_train.cli pull           # Supabase-> local bundle
    python -m aura_train.cli all            # everything above except push

This module is the only place in the package that prints; everywhere else logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import TrainingConfig, load_config
from .dataset import (
    SPLIT_TEST,
    build_dataset,
    class_balance_table,
    load_dataset,
    shard_paths,
)
from .evaluate import evaluate
from .export import (
    bundle_schema_errors,
    export_all,
    read_bundle,
    sample_rows,
)
from .features import FEATURE_NAMES
from .train_gbdt import ABLATIONS, TrainedGbdt, run_ablations, train_gbdt
from .train_linear import LinearModel, train_linear

LOG = logging.getLogger("aura_train")

TRAINED_SUFFIX = ".trained.pkl"


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print(*parts: object) -> None:
    print(*parts, file=sys.stdout, flush=True)


def _trained_path(cfg: TrainingConfig, horizon_ms: int) -> Path:
    return cfg.model_dir / f"reuse_gbdt_{cfg.horizon_label(horizon_ms)}{TRAINED_SUFFIX}"


def _linear_path(cfg: TrainingConfig) -> Path:
    label = cfg.horizon_label(cfg.primary_horizon_ms)
    return cfg.model_dir / f"reuse_linear_{label}{TRAINED_SUFFIX}"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_synth(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    from .synthetic import generate_trace_set

    paths = generate_trace_set(
        cfg.trace_dir,
        requests_per_regime=args.requests,
        unique_keys=args.unique_keys,
        duration_s=args.duration_s,
        seed=cfg.seed,
    )
    _print(f"wrote {len(paths)} traces to {cfg.trace_dir}")
    for path in paths:
        _print(f"  {path}  ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


def cmd_build_dataset(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    report = build_dataset(cfg, progress=True)
    _print(f"rows: {report.rows}   shards: {len(report.shards)}   "
           f"censored rows dropped: {report.censored_dropped}")
    _print("")
    _print("rows by split")
    for split, count in sorted(report.rows_by_split.items()):
        _print(f"  {split:8s} {count:>10,d}")
    _print("")
    _print("rows by regime")
    for regime, count in sorted(report.rows_by_regime.items()):
        _print(f"  {regime:22s} {count:>10,d}")
    _print("")
    _print("class balance")
    _print(class_balance_table(report).to_string(index=False))
    return 0


def _train_all(cfg: TrainingConfig, frame: pd.DataFrame, backend: str | None) -> tuple[
    list[TrainedGbdt], LinearModel
]:
    models: list[TrainedGbdt] = []
    for horizon in cfg.horizons_ms:
        trained = train_gbdt(cfg, frame, horizon, FEATURE_NAMES, backend=backend)
        models.append(trained)
        with _trained_path(cfg, horizon).open("wb") as handle:
            pickle.dump(trained, handle)
    linear = train_linear(cfg, frame, cfg.primary_horizon_ms)
    with _linear_path(cfg).open("wb") as handle:
        pickle.dump(linear, handle)
    return models, linear


def cmd_train(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    frame = load_dataset(cfg)
    models, linear = _train_all(cfg, frame, args.backend)

    _print(f"{'model':22s} {'backend':12s} {'trees':>6s} {'val_auc':>8s} {'val_pr':>8s} "
           f"{'logloss':>8s} {'seconds':>8s}")
    for trained in models:
        _print(
            f"reuse_gbdt_{cfg.horizon_label(trained.horizon_ms):11s} "
            f"{trained.backend:12s} {trained.best_iteration:6d} "
            f"{trained.metrics.get('auc', float('nan')):8.4f} "
            f"{trained.metrics.get('pr_auc', float('nan')):8.4f} "
            f"{trained.metrics.get('logloss', float('nan')):8.4f} "
            f"{trained.train_seconds:8.1f}"
        )
    _print(
        f"{'reuse_linear':22s} {'sgd':12s} {'-':>6s} "
        f"{linear.metrics.get('auc', float('nan')):8.4f} "
        f"{linear.metrics.get('pr_auc', float('nan')):8.4f} "
        f"{linear.metrics.get('logloss', float('nan')):8.4f}"
    )

    if args.ablations:
        table = run_ablations(cfg, frame, cfg.primary_horizon_ms, ABLATIONS, backend=args.backend)
        _print("")
        _print("ablations (primary horizon)")
        _print(table.to_string(index=False))
        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.report_dir / f"ablations_{cfg.horizon_label(cfg.primary_horizon_ms)}.csv"
        table.to_csv(out, index=False)
        _print(f"wrote {out}")
    return 0


def _load_trained(cfg: TrainingConfig) -> tuple[list[TrainedGbdt], LinearModel | None]:
    models: list[TrainedGbdt] = []
    for horizon in cfg.horizons_ms:
        path = _trained_path(cfg, horizon)
        if path.exists():
            with path.open("rb") as handle:
                models.append(pickle.load(handle))
    linear = None
    if _linear_path(cfg).exists():
        with _linear_path(cfg).open("rb") as handle:
            linear = pickle.load(handle)
    if not models:
        raise FileNotFoundError(
            f"no trained models under {cfg.model_dir}; run `train` first"
        )
    return models, linear


def cmd_evaluate(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    frame = load_dataset(cfg)
    models, _ = _load_trained(cfg)
    primary = next(
        (m for m in models if m.horizon_ms == cfg.primary_horizon_ms), models[0]
    )
    ablations = None
    ablation_csv = cfg.report_dir / f"ablations_{cfg.horizon_label(primary.horizon_ms)}.csv"
    if args.ablations:
        ablations = run_ablations(cfg, frame, primary.horizon_ms, ABLATIONS, backend=args.backend)
    elif ablation_csv.exists():
        # `train --ablations` already produced this; no reason to retrain them.
        ablations = pd.read_csv(ablation_csv)
    report = evaluate(
        cfg,
        frame,
        primary.predict,
        importance=primary.gain_importance(),
        horizon_ms=primary.horizon_ms,
        ablations=ablations,
    )
    for line in report.summary_lines():
        _print(line)
    _print("")
    for label, path in report.paths.items():
        _print(f"{label:18s} {path}")
    return 0


def cmd_export(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    models, linear = _load_trained(cfg)
    frame = load_dataset(cfg)
    test = frame.loc[frame["split"] == SPLIT_TEST]
    source = test if not test.empty else frame
    matrix = source[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    parity_rows = sample_rows(matrix, 1000, cfg.seed)

    artifacts = export_all(
        cfg.model_dir, models, linear, parity_rows, write_onnx=not args.no_onnx
    )
    _print(f"{'bundle':26s} {'parity delta':>14s}  path")
    for artifact in artifacts:
        errors = bundle_schema_errors(read_bundle(artifact.bundle_path))
        if errors:
            for error in errors:
                _print(f"  SCHEMA ERROR {artifact.name}: {error}")
            return 1
        _print(f"{artifact.name:26s} {artifact.parity_delta:14.3e}  {artifact.bundle_path}")
        if artifact.onnx_path is not None:
            _print(f"{'':26s} {'':>14s}  {artifact.onnx_path}")
    return 0


def cmd_push(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    from .supabase_io import Session, register_model, set_active, upload_bundle

    session = Session()
    bundles = sorted(cfg.model_dir.glob("reuse_*.json"))
    if not bundles:
        _print(f"no bundles under {cfg.model_dir}; run `export` first")
        return 1
    for bundle_path in bundles:
        bundle = read_bundle(bundle_path)
        onnx_path = bundle_path.with_suffix(".onnx")
        storage_path, onnx_storage = upload_bundle(
            bundle_path,
            onnx_path if onnx_path.exists() else None,
            bucket=cfg.storage_bucket,
            session=session,
        )
        register_model(
            bundle_path,
            storage_path,
            onnx_storage,
            is_active=args.activate,
            session=session,
        )
        if args.activate:
            set_active(str(bundle["name"]), str(bundle["version"]), session=session)
        _print(f"pushed {bundle['name']} {bundle['version']} -> {storage_path}"
               f"{' (active)' if args.activate else ''}")
    _print("")
    _print('reload the running engine with:')
    _print('  curl -X POST http://localhost:8080/v1/model/reload '
           '-H "Content-Type: application/json" -d \'{"source":"supabase"}\'')
    return 0


def cmd_pull(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    from .supabase_io import download_active_bundle

    cfg.ensure_dirs()
    dest = cfg.model_dir / f"{args.name}.json"
    bundle = download_active_bundle(args.name, dest=dest, bucket=cfg.storage_bucket)
    errors = bundle_schema_errors(bundle)
    if errors:
        for error in errors:
            _print(f"SCHEMA ERROR: {error}")
        return 1
    _print(f"pulled {bundle['name']} {bundle['version']} -> {dest}")
    _print(json.dumps(bundle.get("metrics", {}), indent=2))
    return 0


def cmd_all(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    if not shard_paths(cfg.dataset_dir) or args.rebuild:
        from .traces import discover_traces

        if not discover_traces(cfg.trace_dir):
            _print(f"no traces under {cfg.trace_dir}; generating synthetic ones")
            cmd_synth(cfg, args)
        rc = cmd_build_dataset(cfg, args)
        if rc:
            return rc
    _print("")
    rc = cmd_train(cfg, args)
    if rc:
        return rc
    _print("")
    # `train` has already written the ablation table; evaluate picks it up from
    # disk rather than retraining five models to print the same numbers twice.
    args.ablations = False
    rc = cmd_evaluate(cfg, args)
    if rc:
        return rc
    _print("")
    return cmd_export(cfg, args)


def cmd_inspect(cfg: TrainingConfig, args: argparse.Namespace) -> int:
    from .traces import detect_format, discover_traces

    for path in discover_traces(cfg.trace_dir):
        _print(f"{path}  format={detect_format(path)}  {path.stat().st_size / 1024:.0f} KiB")
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aura_train.cli",
        description="AURA training pipeline",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--max-rows-per-trace", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--backend",
        choices=("lightgbm", "sklearn_hist"),
        help="force a GBDT backend instead of auto-detecting lightgbm",
    )
    parser.add_argument("--ablations", action="store_true", help="run the feature-group ablations")
    parser.add_argument("--no-onnx", action="store_true", help="skip the ONNX export")
    parser.add_argument("--activate", action="store_true", help="push: mark the model active")
    parser.add_argument("--rebuild", action="store_true", help="all: rebuild the dataset")
    parser.add_argument("--name", default="reuse_gbdt_h60s", help="pull: model name")
    parser.add_argument("--requests", type=int, default=60_000, help="synth: rows per regime")
    parser.add_argument("--unique-keys", type=int, default=4_000, help="synth: key space")
    parser.add_argument("--duration-s", type=float, default=3600.0, help="synth: virtual duration")

    parser.add_argument(
        "command",
        choices=(
            "synth",
            "build-dataset",
            "train",
            "evaluate",
            "export",
            "push",
            "pull",
            "all",
            "inspect",
        ),
    )
    return parser


COMMANDS = {
    "synth": cmd_synth,
    "build-dataset": cmd_build_dataset,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "push": cmd_push,
    "pull": cmd_pull,
    "all": cmd_all,
    "inspect": cmd_inspect,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    overrides: dict[str, Any] = {
        "trace_dir": args.trace_dir,
        "dataset_dir": args.dataset_dir,
        "model_dir": args.model_dir,
        "report_dir": args.report_dir,
        "max_rows_per_trace": args.max_rows_per_trace,
        "seed": args.seed,
    }
    cfg = load_config(**{k: v for k, v in overrides.items() if v is not None})
    cfg.ensure_dirs()
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
