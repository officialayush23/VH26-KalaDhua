"""Gradient-boosted reuse model.

LightGBM is the model we ship. It is not always installable (locked-down CI
images, air-gapped boxes, a Colab runtime that lost its wheel cache), so this
module also drives scikit-learn's ``HistGradientBoostingClassifier``, which is
the same algorithm family -- histogram binning, leaf-wise-ish growth, shrinkage
baked into the leaf values -- and, importantly, exports to the *same*
``model_bundle.json`` tree encoding. The Rust tree walker cannot tell the two
apart.

Whichever backend runs, the artifact contract in ``export.py`` is identical.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .dataset import SPLIT_TRAIN, SPLIT_VAL, xy
from .features import FEATURE_GROUPS, FEATURE_NAMES

LOG = logging.getLogger(__name__)

BACKEND_LIGHTGBM = "lightgbm"
BACKEND_SKLEARN = "sklearn_hist"


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(preferred: str | None = None) -> str:
    """Pick a backend, warning loudly when we fall back."""
    if preferred == BACKEND_SKLEARN:
        return BACKEND_SKLEARN
    if lightgbm_available():
        return BACKEND_LIGHTGBM
    if preferred == BACKEND_LIGHTGBM:
        raise RuntimeError("lightgbm was requested but is not installed")
    LOG.warning(
        "lightgbm is not installed; falling back to sklearn HistGradientBoosting. "
        "The exported bundle is byte-compatible, but the numbers will differ "
        "slightly from a LightGBM run."
    )
    return BACKEND_SKLEARN


@dataclass
class TrainedGbdt:
    """A trained booster plus everything the exporter and evaluator need."""

    backend: str
    model: Any
    feature_names: tuple[str, ...]
    label_column: str
    horizon_ms: int
    best_iteration: int
    metrics: dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    n_train: int = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Probability of reuse within the horizon, one value per row."""
        if self.backend == BACKEND_LIGHTGBM:
            return np.asarray(
                self.model.predict(x, num_iteration=self.best_iteration or None),
                dtype=np.float64,
            )
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        """Pre-sigmoid score. Used by the export parity check."""
        if self.backend == BACKEND_LIGHTGBM:
            return np.asarray(
                self.model.predict(x, raw_score=True, num_iteration=self.best_iteration or None),
                dtype=np.float64,
            )
        return np.asarray(self.model.decision_function(x), dtype=np.float64).ravel()

    def gain_importance(self) -> dict[str, float]:
        """Total split gain per feature, normalised to sum to 1."""
        if self.backend == BACKEND_LIGHTGBM:
            raw = self.model.feature_importance(importance_type="gain")
            gains = {name: float(v) for name, v in zip(self.feature_names, raw, strict=True)}
        else:
            gains = {name: 0.0 for name in self.feature_names}
            for stage in self.model._predictors:  # noqa: SLF001 - no public accessor exists
                for predictor in stage:
                    nodes = predictor.nodes
                    internal = nodes[nodes["is_leaf"] == 0]
                    for feature_idx, gain in zip(
                        internal["feature_idx"], internal["gain"], strict=True
                    ):
                        gains[self.feature_names[int(feature_idx)]] += float(gain)
        total = sum(gains.values()) or 1.0
        return {k: v / total for k, v in sorted(gains.items(), key=lambda kv: -kv[1])}


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int32)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    out: dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    out["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    out["positive_rate"] = float(y_true.mean())
    out["n"] = float(len(y_true))
    return out


def train_gbdt(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    horizon_ms: int | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    backend: str | None = None,
) -> TrainedGbdt:
    """Train one horizon's reuse model on the ``train`` split, early-stopping on ``val``."""
    horizon = horizon_ms or cfg.primary_horizon_ms
    label_column = cfg.label_column(horizon)
    censored_column = cfg.censored_column(horizon)
    names = tuple(feature_names)

    usable = frame.loc[frame[censored_column] == 0] if censored_column in frame else frame
    train_frame = usable.loc[usable["split"] == SPLIT_TRAIN]
    val_frame = usable.loc[usable["split"] == SPLIT_VAL]
    if train_frame.empty:
        raise ValueError("training split is empty; check the regimes in SplitConfig")
    if val_frame.empty:
        LOG.warning("validation split is empty; early stopping disabled")

    x_train, y_train = xy(train_frame, label_column, names)
    x_val, y_val = (
        xy(val_frame, label_column, names) if not val_frame.empty else (None, None)
    )

    chosen = resolve_backend(backend)
    started = time.perf_counter()
    if chosen == BACKEND_LIGHTGBM:
        model, best_iteration = _fit_lightgbm(cfg, x_train, y_train, x_val, y_val, names)
    else:
        model, best_iteration = _fit_sklearn(cfg, x_train, y_train, x_val, y_val)
    elapsed = time.perf_counter() - started

    trained = TrainedGbdt(
        backend=chosen,
        model=model,
        feature_names=names,
        label_column=label_column,
        horizon_ms=horizon,
        best_iteration=best_iteration,
        train_seconds=elapsed,
        n_train=int(len(y_train)),
    )
    if x_val is not None:
        trained.metrics = _binary_metrics(y_val, trained.predict(x_val))
    LOG.info(
        "trained %s h=%dms backend=%s trees=%d in %.1fs val_auc=%.4f",
        label_column,
        horizon,
        chosen,
        best_iteration,
        elapsed,
        trained.metrics.get("auc", float("nan")),
    )
    return trained


def _fit_lightgbm(
    cfg: TrainingConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    names: tuple[str, ...],
) -> tuple[Any, int]:
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": cfg.gbdt.num_leaves,
        "learning_rate": cfg.gbdt.learning_rate,
        "min_data_in_leaf": cfg.gbdt.min_child_samples,
        "bagging_fraction": cfg.gbdt.subsample,
        "bagging_freq": cfg.gbdt.subsample_freq,
        "feature_fraction": cfg.gbdt.colsample_bytree,
        "lambda_l2": cfg.gbdt.reg_lambda,
        "max_bin": cfg.gbdt.max_bin,
        "seed": cfg.gbdt.seed,
        "deterministic": True,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=list(names))
    valid_sets = []
    callbacks = [lgb.log_evaluation(period=0)]
    if x_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
        valid_sets.append(lgb.Dataset(x_val, label=y_val, reference=train_set))
        callbacks.append(lgb.early_stopping(cfg.gbdt.early_stopping_rounds, verbose=False))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.gbdt.n_estimators,
        valid_sets=valid_sets or None,
        callbacks=callbacks,
    )
    return booster, int(booster.best_iteration or booster.num_trees())


def _fit_sklearn(
    cfg: TrainingConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> tuple[Any, int]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=cfg.gbdt.n_estimators,
        learning_rate=cfg.gbdt.learning_rate,
        max_leaf_nodes=cfg.gbdt.num_leaves,
        min_samples_leaf=cfg.gbdt.min_child_samples,
        l2_regularization=cfg.gbdt.reg_lambda,
        max_bins=min(cfg.gbdt.max_bin, 255),
        early_stopping=False,
        random_state=cfg.gbdt.seed,
    )
    model.fit(x_train, y_train)
    n_trees = len(model._predictors)  # noqa: SLF001

    if x_val is None or y_val is None or len(np.unique(y_val)) < 2:
        return model, n_trees

    # Our own early stopping, against the regime-held-out validation split
    # rather than sklearn's random internal one.
    best_iteration = n_trees
    best_loss = float("inf")
    losses: list[float] = []
    for i, decision in enumerate(model.staged_decision_function(x_val), start=1):
        prob = 1.0 / (1.0 + np.exp(-np.asarray(decision).ravel()))
        loss = float(log_loss(y_val, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]))
        losses.append(loss)
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_iteration = i
        elif i - best_iteration >= cfg.gbdt.early_stopping_rounds:
            break
    if best_iteration < n_trees:
        LOG.info("early stopping at iteration %d of %d", best_iteration, n_trees)
        model._predictors = model._predictors[:best_iteration]  # noqa: SLF001
    return model, best_iteration


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------

ABLATIONS: tuple[str, ...] = ("cost", "trend", "frequency", "recency")


def ablation_feature_sets(
    groups: Iterable[str] = ABLATIONS,
) -> dict[str, tuple[str, ...]]:
    """``{"full": all 16, "drop_cost": 13, ...}``."""
    sets: dict[str, tuple[str, ...]] = {"full": FEATURE_NAMES}
    for group in groups:
        dropped = set(FEATURE_GROUPS[group])
        sets[f"drop_{group}"] = tuple(n for n in FEATURE_NAMES if n not in dropped)
    return sets


def run_ablations(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    horizon_ms: int | None = None,
    groups: Iterable[str] = ABLATIONS,
    backend: str | None = None,
) -> pd.DataFrame:
    """Train one model per feature subset and report validation and test AUC.

    This is the honest version of "our features matter": if dropping the cost
    block does not move the test AUC, the cost block is decoration.
    """
    horizon = horizon_ms or cfg.primary_horizon_ms
    label_column = cfg.label_column(horizon)
    censored_column = cfg.censored_column(horizon)
    usable = frame.loc[frame[censored_column] == 0] if censored_column in frame else frame
    test_frame = usable.loc[usable["split"] == "test"]

    records: list[dict[str, object]] = []
    for name, names in ablation_feature_sets(groups).items():
        trained = train_gbdt(cfg, frame, horizon, names, backend=backend)
        row: dict[str, object] = {
            "variant": name,
            "n_features": len(names),
            "val_auc": trained.metrics.get("auc", float("nan")),
            "val_pr_auc": trained.metrics.get("pr_auc", float("nan")),
            "val_logloss": trained.metrics.get("logloss", float("nan")),
            "trees": trained.best_iteration,
            "train_seconds": round(trained.train_seconds, 2),
        }
        if not test_frame.empty:
            x_test, y_test = xy(test_frame, label_column, names)
            test_metrics = _binary_metrics(y_test, trained.predict(x_test))
            row["test_auc"] = test_metrics["auc"]
            row["test_pr_auc"] = test_metrics["pr_auc"]
            row["test_logloss"] = test_metrics["logloss"]
        records.append(row)

    table = pd.DataFrame.from_records(records)
    if "test_auc" in table and not table.empty:
        baseline = table.loc[table["variant"] == "full", "test_auc"]
        if not baseline.empty:
            table["test_auc_delta"] = table["test_auc"] - float(baseline.iloc[0])
    return table
