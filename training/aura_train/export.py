"""Model export: trained booster -> ``model_bundle.json`` (+ optional ONNX).

The bundle is the only thing the Rust engine ever sees, so this file is the
other half of the interface contract (section 5). It also contains a reference
tree walker, ``predict_bundle``, which is a line-for-line description of what
``aura-core`` has to do. If the Rust walker and this walker disagree, the bundle
encoding is ambiguous and this file is where the ambiguity has to be resolved.

Tree encoding, restated from the contract because it is easy to get wrong:

* Node ``i`` of a tree has ``split_feature[i]`` and ``threshold[i]``.
* ``left[i]`` / ``right[i]``: **positive or zero** means "internal node with
  that index"; **negative** means "leaf ``-(v) - 1``" into ``leaf_value``.
  Index 0 is unambiguous because the root is never anyone's child.
* The comparison is ``feature <= threshold -> left``.
* A missing or NaN feature goes left.
* A tree with no splits is encoded as a single node whose left and right both
  point at leaf 0.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES
from .train_gbdt import BACKEND_LIGHTGBM, BACKEND_SKLEARN, TrainedGbdt
from .train_linear import LinearModel

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
KIND_GBDT = "lightgbm_gbdt"
KIND_LINEAR = "linear_logistic"
DECISION_TYPE_NUMERICAL = 2
DECISION_TYPE_LEAF = 0


def utc_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def storage_version(version: str) -> str:
    """``2026-09-04T11:20:00Z -> 2026-09-04T11-20-00Z`` for object keys."""
    return version.replace(":", "-")


def git_sha(repo_root: Path | None = None) -> str:
    """Best-effort commit id, read from the environment or from ``.git``.

    Deliberately does not shell out: this runs inside Colab and inside CI
    containers where ``git`` may be absent or the checkout may be a tarball.
    """
    for key in ("AURA_GIT_SHA", "GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    root = repo_root or Path(__file__).resolve().parents[2]
    head = root / ".git" / "HEAD"
    try:
        if not head.exists():
            return "unknown"
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            target = root / ".git" / ref[5:]
            return target.read_text().strip() if target.exists() else "unknown"
        return ref
    except OSError:
        return "unknown"


def bundle_name(kind: str, horizon_ms: int) -> str:
    """``reuse_gbdt_h60s`` / ``reuse_linear_h60s``."""
    stem = "gbdt" if kind == KIND_GBDT else "linear"
    return f"reuse_{stem}_h{horizon_ms // 1000}s"


# --------------------------------------------------------------------------
# Tree extraction
# --------------------------------------------------------------------------


@dataclass
class FlatTree:
    split_feature: list[int]
    threshold: list[float]
    left: list[int]
    right: list[int]
    leaf_value: list[float]
    decision_type: list[int]

    def to_json(self) -> dict[str, list[float] | list[int]]:
        return {
            "split_feature": self.split_feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
            "leaf_value": self.leaf_value,
            "decision_type": self.decision_type,
        }


def _constant_tree(value: float) -> FlatTree:
    return FlatTree(
        split_feature=[0],
        threshold=[0.0],
        left=[-1],
        right=[-1],
        leaf_value=[float(value)],
        decision_type=[DECISION_TYPE_LEAF],
    )


def _flatten_lightgbm_tree(root: dict[str, Any]) -> FlatTree:
    tree = FlatTree([], [], [], [], [], [])

    def visit(node: dict[str, Any]) -> int:
        if "split_feature" not in node:
            tree.leaf_value.append(float(node.get("leaf_value", 0.0)))
            return -len(tree.leaf_value)  # -(index) - 1
        index = len(tree.split_feature)
        tree.split_feature.append(int(node["split_feature"]))
        tree.threshold.append(float(node["threshold"]))
        tree.decision_type.append(DECISION_TYPE_NUMERICAL)
        tree.left.append(0)
        tree.right.append(0)
        tree.left[index] = visit(node["left_child"])
        tree.right[index] = visit(node["right_child"])
        return index

    if "split_feature" not in root:
        return _constant_tree(float(root.get("leaf_value", 0.0)))
    visit(root)
    return tree


def trees_from_lightgbm(booster: Any, num_iteration: int | None = None) -> list[FlatTree]:
    dumped = booster.dump_model(num_iteration=num_iteration)
    trees: list[FlatTree] = []
    for info in dumped["tree_info"]:
        trees.append(_flatten_lightgbm_tree(info["tree_structure"]))
    return trees


def _flatten_sklearn_predictor(nodes: Any) -> FlatTree:
    tree = FlatTree([], [], [], [], [], [])

    def visit(node_index: int) -> int:
        node = nodes[node_index]
        if bool(node["is_leaf"]):
            tree.leaf_value.append(float(node["value"]))
            return -len(tree.leaf_value)
        index = len(tree.split_feature)
        tree.split_feature.append(int(node["feature_idx"]))
        tree.threshold.append(float(node["num_threshold"]))
        tree.decision_type.append(DECISION_TYPE_NUMERICAL)
        tree.left.append(0)
        tree.right.append(0)
        tree.left[index] = visit(int(node["left"]))
        tree.right[index] = visit(int(node["right"]))
        return index

    if bool(nodes[0]["is_leaf"]):
        return _constant_tree(float(nodes[0]["value"]))
    visit(0)
    return tree


def trees_from_sklearn(model: Any) -> list[FlatTree]:
    """Convert a fitted ``HistGradientBoostingClassifier`` to the same encoding.

    The class baseline (the log-odds prior sklearn fits before boosting) has no
    home in the contract's tree list, so it becomes a single-leaf tree. The Rust
    walker sums leaf values across all trees regardless, so this is exact.
    """
    baseline = float(np.asarray(model._baseline_prediction).ravel()[0])  # noqa: SLF001
    trees = [_constant_tree(baseline)]
    for stage in model._predictors:  # noqa: SLF001
        for predictor in stage:
            trees.append(_flatten_sklearn_predictor(predictor.nodes))
    return trees


# --------------------------------------------------------------------------
# Reference scorer -- the specification the Rust walker must satisfy
# --------------------------------------------------------------------------


def _walk(tree: dict[str, Any], row: np.ndarray) -> float:
    left = tree["left"]
    right = tree["right"]
    split_feature = tree["split_feature"]
    threshold = tree["threshold"]
    leaf_value = tree["leaf_value"]
    node = 0
    for _ in range(len(split_feature) + 1):
        if tree["decision_type"][node] == DECISION_TYPE_LEAF:
            return float(leaf_value[0])
        value = row[split_feature[node]]
        go_left = not np.isfinite(value) or value <= threshold[node]
        child = left[node] if go_left else right[node]
        if child < 0:
            return float(leaf_value[-child - 1])
        node = child
    raise ValueError("tree traversal did not terminate; the bundle is malformed")


def predict_bundle(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Score a bundle exactly the way ``aura-core`` must.

    Used by ``scripts/verify_bundle.py`` and by the export-time parity check.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    mean = np.asarray(bundle["normalization"]["mean"], dtype=np.float64)
    scale = np.asarray(bundle["normalization"]["scale"], dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    z = (x - mean) / scale

    if bundle["kind"] == KIND_LINEAR:
        weights = bundle["linear_weights"]
        coef = np.asarray(weights["coef"], dtype=np.float64)
        raw = z @ coef + float(weights["intercept"])
    else:
        raw = np.zeros(len(z), dtype=np.float64)
        for tree in bundle["trees"]:
            for i in range(len(z)):
                raw[i] += _walk(tree, z[i])

    if bundle.get("sigmoid_output", True):
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
    return raw


# --------------------------------------------------------------------------
# Bundle construction
# --------------------------------------------------------------------------


def _identity_normalization(n: int = N_FEATURES) -> dict[str, list[float]]:
    return {"mean": [0.0] * n, "scale": [1.0] * n}


def build_gbdt_bundle(
    trained: TrainedGbdt,
    metrics: dict[str, float] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Serialise a trained GBDT into the contract's bundle shape."""
    if tuple(trained.feature_names) != FEATURE_NAMES:
        raise ValueError(
            "only a model trained on the full 16-feature vector can be exported; "
            f"got {len(trained.feature_names)} features. Ablation models are for "
            "the report, not for the engine."
        )
    if trained.backend == BACKEND_LIGHTGBM:
        trees = trees_from_lightgbm(trained.model, trained.best_iteration or None)
    elif trained.backend == BACKEND_SKLEARN:
        trees = trees_from_sklearn(trained.model)
    else:
        raise ValueError(f"cannot export backend {trained.backend!r}")

    merged = dict(trained.metrics)
    merged.update(metrics or {})
    version = utc_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name or bundle_name(KIND_GBDT, trained.horizon_ms),
        "kind": KIND_GBDT,
        "horizon_ms": int(trained.horizon_ms),
        "version": version,
        "git_sha": git_sha(),
        "feature_names": list(FEATURE_NAMES),
        # Trees are scale invariant, so the engine applies an identity
        # transform. The field still has to be present and correct, because the
        # linear bundle uses the same code path with real statistics.
        "normalization": _identity_normalization(),
        "objective": "binary",
        "sigmoid_output": True,
        "trees": [t.to_json() for t in trees],
        "linear_weights": None,
        "metrics": {
            "auc": float(merged.get("auc", float("nan"))),
            "pr_auc": float(merged.get("pr_auc", float("nan"))),
            "logloss": float(merged.get("logloss", float("nan"))),
            "n_train": int(trained.n_train),
            "trees": len(trees),
            "backend": trained.backend,
        },
    }


def build_linear_bundle(
    model: LinearModel,
    metrics: dict[str, float] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Serialise the cold-start / online logistic model."""
    if tuple(model.feature_names) != FEATURE_NAMES:
        raise ValueError("the linear bundle must use the full 16-feature vector")
    merged = dict(model.metrics)
    merged.update(metrics or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name or bundle_name(KIND_LINEAR, model.horizon_ms),
        "kind": KIND_LINEAR,
        "horizon_ms": int(model.horizon_ms),
        "version": utc_version(),
        "git_sha": git_sha(),
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "mean": [float(v) for v in model.mean],
            "scale": [float(v) for v in model.scale],
        },
        "objective": "binary",
        "sigmoid_output": True,
        "trees": [],
        "linear_weights": {
            "coef": [float(v) for v in model.coef],
            "intercept": float(model.intercept),
        },
        "metrics": {
            "auc": float(merged.get("auc", float("nan"))),
            "pr_auc": float(merged.get("pr_auc", float("nan"))),
            "logloss": float(merged.get("logloss", float("nan"))),
            "n_train": int(model.n_train),
            "fitted": bool(model.fitted),
        },
    }


class BundleParityError(AssertionError):
    """The exported bundle does not reproduce the trainer's own predictions."""


def assert_parity(
    bundle: dict[str, Any],
    reference: TrainedGbdt | LinearModel,
    x: np.ndarray,
    tolerance: float = 1e-6,
) -> float:
    """Score the same rows both ways and assert they agree.

    This is the check that catches every tree-encoding mistake: an off-by-one in
    the leaf index, a flipped comparison, a forgotten baseline term. It runs at
    export time, not just in the test suite, because an unnoticed encoding bug
    would be served to the engine.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    from_bundle = predict_bundle(bundle, x)
    from_model = np.asarray(reference.predict(x), dtype=np.float64).ravel()
    delta = float(np.max(np.abs(from_bundle - from_model))) if len(x) else 0.0
    if not np.isfinite(delta) or delta > tolerance:
        raise BundleParityError(
            f"bundle/model disagreement of {delta:.3e} exceeds {tolerance:.1e} "
            f"over {len(x)} rows"
        )
    LOG.info("parity check passed over %d rows (max delta %.3e)", len(x), delta)
    return delta


def write_bundle(bundle: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, allow_nan=True) + "\n")
    LOG.info(
        "wrote %s (%d trees, %.1f KiB)",
        path,
        len(bundle["trees"]),
        path.stat().st_size / 1024,
    )
    return path


def read_bundle(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------


def export_onnx(trained: TrainedGbdt, path: Path) -> Path | None:
    """Best-effort ONNX export.

    The JSON bundle is authoritative: the Rust engine's default path is the pure
    tree walker, and ONNX only exists for people who want to serve the same
    model from a different runtime. If no converter is installed we log and move
    on rather than failing the pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_types = None
    try:
        from skl2onnx.common.data_types import FloatTensorType

        initial_types = [("features", FloatTensorType([None, N_FEATURES]))]
    except ImportError:
        pass

    try:
        if trained.backend == BACKEND_LIGHTGBM:
            from onnxmltools import convert_lightgbm

            onnx_model = convert_lightgbm(
                trained.model, initial_types=initial_types, target_opset=15
            )
        else:
            from skl2onnx import convert_sklearn

            onnx_model = convert_sklearn(
                trained.model, initial_types=initial_types, target_opset=15
            )
    except ImportError as exc:
        LOG.warning(
            "ONNX export skipped (%s). Install onnxmltools/skl2onnx to enable it; "
            "the JSON bundle is unaffected.",
            exc,
        )
        return None

    path.write_bytes(onnx_model.SerializeToString())
    LOG.info("wrote %s", path)
    return path


def verify_onnx(path: Path, x: np.ndarray, reference: np.ndarray, tolerance: float = 1e-4) -> float:
    """Run the exported graph under onnxruntime and compare probabilities."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(x, dtype=np.float32)})
    probabilities = None
    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 2 and array.shape[1] == 2:
            probabilities = array[:, 1]
            break
        if array.dtype == object:
            probabilities = np.array([float(d[1]) for d in output])
            break
    if probabilities is None:
        raise ValueError("could not locate a probability output in the ONNX graph")
    delta = float(np.max(np.abs(probabilities - np.asarray(reference).ravel())))
    if delta > tolerance:
        raise BundleParityError(f"ONNX disagreement {delta:.3e} exceeds {tolerance:.1e}")
    return delta


# --------------------------------------------------------------------------
# Whole-run export
# --------------------------------------------------------------------------


@dataclass
class ExportedArtifact:
    name: str
    bundle_path: Path
    onnx_path: Path | None
    parity_delta: float
    metrics: dict[str, float]


def export_all(
    model_dir: Path,
    gbdt_models: Iterable[TrainedGbdt],
    linear: LinearModel | None,
    parity_rows: np.ndarray,
    write_onnx: bool = True,
) -> list[ExportedArtifact]:
    """Export every trained head, verifying parity for each one."""
    out: list[ExportedArtifact] = []
    sample = np.atleast_2d(np.asarray(parity_rows, dtype=np.float64))
    for trained in gbdt_models:
        bundle = build_gbdt_bundle(trained)
        name = str(bundle["name"])
        bundle_path = write_bundle(bundle, model_dir / f"{name}.json")
        delta = assert_parity(bundle, trained, sample)
        onnx_path = export_onnx(trained, model_dir / f"{name}.onnx") if write_onnx else None
        out.append(
            ExportedArtifact(name, bundle_path, onnx_path, delta, dict(trained.metrics))
        )
    if linear is not None:
        bundle = build_linear_bundle(linear)
        name = str(bundle["name"])
        bundle_path = write_bundle(bundle, model_dir / f"{name}.json")
        delta = assert_parity(bundle, linear, sample)
        out.append(ExportedArtifact(name, bundle_path, None, delta, dict(linear.metrics)))
    return out


def sample_rows(x: np.ndarray, n: int = 1000, seed: int = 42) -> np.ndarray:
    """Deterministic subsample used for the parity check."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def bundle_schema_errors(bundle: dict[str, Any]) -> list[str]:
    """Structural validation of a bundle against contract section 5."""
    errors: list[str] = []
    required = (
        "schema_version",
        "name",
        "kind",
        "horizon_ms",
        "version",
        "git_sha",
        "feature_names",
        "normalization",
        "objective",
        "sigmoid_output",
        "trees",
        "linear_weights",
        "metrics",
    )
    for key in required:
        if key not in bundle:
            errors.append(f"missing key: {key}")
    if errors:
        return errors

    if bundle["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {bundle['schema_version']} != {SCHEMA_VERSION}")
    if bundle["kind"] not in (KIND_GBDT, KIND_LINEAR):
        errors.append(f"unknown kind {bundle['kind']!r}")
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        errors.append("feature_names does not match the contract order")
    for key in ("mean", "scale"):
        values: Sequence[float] = bundle["normalization"].get(key, [])
        if len(values) != N_FEATURES:
            errors.append(f"normalization.{key} has {len(values)} entries, expected {N_FEATURES}")
    if bundle["kind"] == KIND_LINEAR:
        weights = bundle.get("linear_weights") or {}
        if len(weights.get("coef", [])) != N_FEATURES:
            errors.append("linear_weights.coef must have one entry per feature")
        if "intercept" not in weights:
            errors.append("linear_weights.intercept is missing")
    else:
        if not bundle["trees"]:
            errors.append("a gbdt bundle must contain at least one tree")

    for t, tree in enumerate(bundle["trees"]):
        n_nodes = len(tree["split_feature"])
        for key in ("threshold", "left", "right", "decision_type"):
            if len(tree[key]) != n_nodes:
                errors.append(f"tree {t}: {key} has {len(tree[key])} entries, expected {n_nodes}")
        n_leaves = len(tree["leaf_value"])
        for i in range(n_nodes):
            if tree["decision_type"][i] == DECISION_TYPE_LEAF:
                continue
            if not 0 <= tree["split_feature"][i] < N_FEATURES:
                errors.append(f"tree {t} node {i}: feature index out of range")
            for side in ("left", "right"):
                child = tree[side][i]
                if child < 0:
                    if -child - 1 >= n_leaves:
                        errors.append(f"tree {t} node {i}: {side} leaf index out of range")
                elif child >= n_nodes:
                    errors.append(f"tree {t} node {i}: {side} node index out of range")
    return errors
