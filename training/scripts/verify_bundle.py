#!/usr/bin/env python3
"""Verify a model bundle before it is allowed anywhere near the engine.

Three checks, in increasing order of how much they would have hurt to miss:

1. **Schema.** Every key the contract requires is present, the feature list is
   in contract order, and every tree index points somewhere valid.
2. **Numerics.** No NaN or infinity in any threshold, leaf value or
   normalisation constant. The Rust walker has no way to report a poisoned
   tree; it would just serve garbage probabilities.
3. **Parity.** Re-score 1000 rows through the reference walker and compare
   against the trainer's own prediction. Anything above 1e-6 fails.

Usage::

    python scripts/verify_bundle.py models/reuse_gbdt_h60s.json
    python scripts/verify_bundle.py models/reuse_gbdt_h60s.json --rows 5000
    python scripts/verify_bundle.py models/*.json --dataset-dir data/dataset
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aura_train.config import load_config  # noqa: E402
from aura_train.dataset import load_dataset, shard_paths  # noqa: E402
from aura_train.export import (  # noqa: E402
    bundle_schema_errors,
    predict_bundle,
    read_bundle,
)
from aura_train.features import FEATURE_NAMES, N_FEATURES  # noqa: E402

LOG = logging.getLogger("verify_bundle")
TOLERANCE = 1e-6


def numeric_errors(bundle: dict) -> list[str]:
    errors: list[str] = []
    for key in ("mean", "scale"):
        values = np.asarray(bundle["normalization"][key], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            errors.append(f"normalization.{key} contains a non-finite value")
    if np.any(np.abs(np.asarray(bundle["normalization"]["scale"])) < 1e-12):
        errors.append("normalization.scale contains a zero, which would divide by zero")
    for t, tree in enumerate(bundle["trees"]):
        for key in ("threshold", "leaf_value"):
            values = np.asarray(tree[key], dtype=np.float64)
            if values.size and not np.all(np.isfinite(values)):
                errors.append(f"tree {t}: {key} contains a non-finite value")
    weights = bundle.get("linear_weights")
    if weights:
        coef = np.asarray(weights["coef"], dtype=np.float64)
        if not np.all(np.isfinite(coef)) or not np.isfinite(float(weights["intercept"])):
            errors.append("linear_weights contain a non-finite value")
    return errors


def score_rows(dataset_dir: Path | None, n: int, seed: int) -> np.ndarray:
    """Real feature rows when a dataset exists, otherwise a plausible sample.

    Synthetic rows are a weaker test -- they cannot exercise thresholds the
    trees actually learned -- so the dataset path is strongly preferred and the
    fallback says so.
    """
    if dataset_dir is not None and shard_paths(dataset_dir):
        cfg = load_config(dataset_dir=dataset_dir)
        frame = load_dataset(cfg)
        x = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        if len(x) > n:
            rng = np.random.default_rng(seed)
            x = x[rng.choice(len(x), size=n, replace=False)]
        return x
    LOG.warning(
        "no dataset shards found; scoring on synthetic rows, which is a weaker check"
    )
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_FEATURES)) * 3.0
    x[:, FEATURE_NAMES.index("ttl_remaining_frac")] = rng.random(n)
    x[:, FEATURE_NAMES.index("cache_pressure")] = rng.random(n)
    x[:, FEATURE_NAMES.index("app_id")] = rng.integers(0, 3, size=n)
    return np.abs(x) * np.where(rng.random((n, N_FEATURES)) > 0.5, 1.0, -1.0)


def trained_reference(bundle_path: Path):  # noqa: ANN201 - the pickle is untyped by nature
    """The trainer object that produced this bundle, if it is still on disk."""
    candidate = bundle_path.with_name(bundle_path.stem + ".trained.pkl")
    if not candidate.exists():
        return None
    with candidate.open("rb") as handle:
        return pickle.load(handle)


def verify(bundle_path: Path, dataset_dir: Path | None, rows: int, seed: int) -> list[str]:
    bundle = read_bundle(bundle_path)
    problems = bundle_schema_errors(bundle)
    problems += numeric_errors(bundle)
    if problems:
        return problems

    x = score_rows(dataset_dir, rows, seed)
    probabilities = predict_bundle(bundle, x)
    if not np.all(np.isfinite(probabilities)):
        problems.append("the bundle produced a non-finite probability")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        problems.append(
            f"probabilities out of range: [{probabilities.min():.6f}, {probabilities.max():.6f}]"
        )

    reference = trained_reference(bundle_path)
    if reference is None:
        LOG.warning(
            "no %s alongside the bundle, so the parity check was skipped; "
            "run this on the machine that trained the model",
            bundle_path.stem + ".trained.pkl",
        )
    else:
        delta = float(np.max(np.abs(probabilities - np.asarray(reference.predict(x)).ravel())))
        LOG.info("parity delta over %d rows: %.3e", len(x), delta)
        if not np.isfinite(delta) or delta > TOLERANCE:
            problems.append(f"parity delta {delta:.3e} exceeds {TOLERANCE:.1e}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate an AURA model bundle")
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    failed = 0
    for bundle_path in args.bundles:
        problems = verify(bundle_path, args.dataset_dir, args.rows, args.seed)
        if problems:
            failed += 1
            print(f"FAIL {bundle_path}")
            for problem in problems:
                print(f"     {problem}")
        else:
            bundle = read_bundle(bundle_path)
            print(
                f"ok   {bundle_path}  {bundle['name']} {bundle['version']} "
                f"kind={bundle['kind']} trees={len(bundle['trees'])}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
