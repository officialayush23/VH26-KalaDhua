"""Online logistic model for the cold-start and online-learning path.

The GBDT is the accurate model, but it is useless in the first minutes of a
deployment: it has to be trained, exported and loaded, and until then the cache
has no learned signal at all. This module produces the model that fills that
gap.

Three properties matter:

1. **It works with zero project-specific data.** ``cold_start_model()`` returns
   a hand-specified logistic model whose weights encode what every cache paper
   already knows -- recent and frequent keys get reused, big keys are worth less
   per byte, expensive keys are worth more to keep. It is not fitted to
   anything, so it can ship inside the binary.
2. **It improves online.** ``partial_fit`` consumes labelled events as the
   server observes them, so the same artifact that shipped as a prior becomes a
   fitted model within a few thousand requests.
3. **It is cheap to evaluate.** One dot product over 16 standardised features.

The engine blends the three predictors by confidence; see the "Cold start"
section of ``README.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .dataset import SPLIT_TRAIN, SPLIT_VAL, xy
from .features import FEATURE_NAMES, N_FEATURES

LOG = logging.getLogger(__name__)

# Hand-specified prior, in contract feature order. Units are "per standardised
# unit", so these are directly comparable to each other.
COLD_START_WEIGHTS: dict[str, float] = {
    "log_age_ms": -0.55,          # long since last touched -> unlikely to come back
    "log_inter_arrival_ms": -0.40,
    "freq_1m": 0.70,
    "freq_5m": 0.45,
    "freq_1h": 0.25,
    "ewma_fast": 0.35,
    "ewma_slow": 0.20,
    "trend": 0.30,                # accelerating keys are the ones worth holding
    "acceleration": 0.10,
    "log_size_bytes": -0.12,
    "log_regen_p50_ms": 0.08,
    "cost_variance_ratio": 0.05,
    "regen_cost_usd": 0.10,
    "ttl_remaining_frac": 0.22,
    "cache_pressure": -0.05,
    "app_id": 0.0,                # never a prior; only a fitted model may use it
}
COLD_START_INTERCEPT = -0.4

# Standardisation used by the prior when no data has been seen. These are
# order-of-magnitude values for the AURA workloads, not fitted statistics; the
# point is only that no single feature dominates the dot product.
PRIOR_MEAN: dict[str, float] = {
    "log_age_ms": 8.0,
    "log_inter_arrival_ms": 8.0,
    "freq_1m": 2.0,
    "freq_5m": 4.0,
    "freq_1h": 8.0,
    "ewma_fast": 1.0,
    "ewma_slow": 3.0,
    "trend": -0.8,
    "acceleration": 0.0,
    "log_size_bytes": 11.0,
    "log_regen_p50_ms": 5.0,
    "cost_variance_ratio": 1.2,
    "regen_cost_usd": 1e-5,
    "ttl_remaining_frac": 0.5,
    "cache_pressure": 0.6,
    "app_id": 1.0,
}
PRIOR_SCALE: dict[str, float] = {
    "log_age_ms": 3.0,
    "log_inter_arrival_ms": 3.0,
    "freq_1m": 2.0,
    "freq_5m": 4.0,
    "freq_1h": 8.0,
    "ewma_fast": 1.5,
    "ewma_slow": 3.0,
    "trend": 1.0,
    "acceleration": 0.5,
    "log_size_bytes": 2.0,
    "log_regen_p50_ms": 2.0,
    "cost_variance_ratio": 0.8,
    "regen_cost_usd": 2e-5,
    "ttl_remaining_frac": 0.35,
    "cache_pressure": 0.25,
    "app_id": 1.0,
}


@dataclass
class LinearModel:
    """Standardise, dot, sigmoid. Serialises straight into the bundle."""

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    horizon_ms: int = 60_000
    fitted: bool = False
    n_train: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, array in (("coef", self.coef), ("mean", self.mean), ("scale", self.scale)):
            if len(array) != len(self.feature_names):
                raise ValueError(
                    f"{name} has {len(array)} entries, "
                    f"expected {len(self.feature_names)}"
                )
        self.scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale)

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        return self.standardise(x) @ self.coef + self.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.raw_score(x), -60.0, 60.0)))


def cold_start_model(horizon_ms: int = 60_000) -> LinearModel:
    """The zero-data model. Ships with the binary, needs no training run."""
    return LinearModel(
        coef=np.array([COLD_START_WEIGHTS[n] for n in FEATURE_NAMES], dtype=np.float64),
        intercept=float(COLD_START_INTERCEPT),
        mean=np.array([PRIOR_MEAN[n] for n in FEATURE_NAMES], dtype=np.float64),
        scale=np.array([PRIOR_SCALE[n] for n in FEATURE_NAMES], dtype=np.float64),
        horizon_ms=horizon_ms,
        fitted=False,
    )


def _standardiser(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return mean, scale


def train_linear(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    horizon_ms: int | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    epochs: int = 20,
    warm_start_from_prior: bool = True,
    calibrate: bool = True,
) -> LinearModel:
    """Fit the online logistic model with SGD, warm-started from the prior.

    Warm-starting matters: it means an under-trained model degrades towards the
    heuristic rather than towards noise, which is exactly the behaviour we want
    from the artifact that runs during the first minute of a deployment.
    """
    from sklearn.linear_model import SGDClassifier

    horizon = horizon_ms or cfg.primary_horizon_ms
    label_column = cfg.label_column(horizon)
    censored_column = cfg.censored_column(horizon)
    names = tuple(feature_names)

    usable = frame.loc[frame[censored_column] == 0] if censored_column in frame else frame
    train_frame = usable.loc[usable["split"] == SPLIT_TRAIN]
    val_frame = usable.loc[usable["split"] == SPLIT_VAL]
    if train_frame.empty:
        raise ValueError("training split is empty")

    x_train, y_train = xy(train_frame, label_column, names)
    mean, scale = _standardiser(x_train)
    z_train = (x_train - mean) / scale

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        max_iter=max(epochs, 1),
        tol=None,
        shuffle=True,
        average=True,
        random_state=cfg.seed,
    )
    coef_init = None
    intercept_init = None
    if warm_start_from_prior and len(names) == N_FEATURES:
        prior = cold_start_model(horizon)
        coef_init = prior.coef.reshape(1, -1).copy()
        intercept_init = np.array([prior.intercept], dtype=np.float64)
    clf.fit(z_train, y_train, coef_init=coef_init, intercept_init=intercept_init)

    model = LinearModel(
        coef=clf.coef_.ravel().astype(np.float64),
        intercept=float(np.asarray(clf.intercept_).ravel()[0]),
        mean=mean,
        scale=scale,
        feature_names=names,
        horizon_ms=horizon,
        fitted=True,
        n_train=int(len(y_train)),
    )
    if not val_frame.empty:
        from .train_gbdt import _binary_metrics

        x_val, y_val = xy(val_frame, label_column, names)
        if calibrate:
            calibrate_scores(model, x_val, y_val)
        model.metrics = _binary_metrics(y_val, model.predict(x_val))
        LOG.info(
            "linear h=%dms val_auc=%.4f logloss=%.4f",
            horizon,
            model.metrics.get("auc", float("nan")),
            model.metrics.get("logloss", float("nan")),
        )
    return model


def calibrate_scores(model: LinearModel, x: np.ndarray, y: np.ndarray) -> LinearModel:
    """Platt-scale the raw score and fold the scaling back into the weights.

    Averaged SGD ranks well but is badly scaled -- its probabilities are close
    to a step function, which is fine for AUC and useless for the value-density
    arithmetic the engine does with the number. Fitting ``a * score + b`` and
    folding ``a`` into the coefficients keeps the artifact a plain linear model
    while making the output an actual probability.
    """
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return model
    raw = model.raw_score(x).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    platt.fit(raw, y)
    a = float(platt.coef_.ravel()[0])
    b = float(np.asarray(platt.intercept_).ravel()[0])
    model.coef = model.coef * a
    model.intercept = a * model.intercept + b
    LOG.debug("calibrated linear model with a=%.4f b=%.4f", a, b)
    return model


def partial_fit(
    model: LinearModel,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
) -> LinearModel:
    """One online SGD step over a batch of freshly labelled events.

    This is the update the server applies as reuse outcomes are observed. It is
    written out longhand rather than via scikit-learn so the Rust side can run
    the identical update without an ML runtime.
    """
    z = model.standardise(x)
    prob = 1.0 / (1.0 + np.exp(-np.clip(z @ model.coef + model.intercept, -60.0, 60.0)))
    error = prob - np.asarray(y, dtype=np.float64)
    n = max(len(error), 1)
    model.coef = model.coef - learning_rate * (z.T @ error) / n
    model.intercept = float(model.intercept - learning_rate * error.mean())
    model.n_train += n
    model.fitted = True
    return model


# --------------------------------------------------------------------------
# Confidence blending
# --------------------------------------------------------------------------


def heuristic_probability(x: np.ndarray) -> np.ndarray:
    """The zero-model fallback: a monotone function of recency and frequency.

    Present here so the blend can be simulated offline and unit-tested; the
    engine has the same expression.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    log_age = x[:, FEATURE_NAMES.index("log_age_ms")]
    freq_5m = x[:, FEATURE_NAMES.index("freq_5m")]
    score = 0.6 * np.log1p(freq_5m) - 0.25 * log_age + 1.2
    return 1.0 / (1.0 + np.exp(-np.clip(score, -60.0, 60.0)))


def blend(
    heuristic: np.ndarray,
    linear: np.ndarray | None,
    gbdt: np.ndarray | None,
    linear_confidence: float,
    gbdt_confidence: float,
    confidence_floor: float = 0.20,
) -> np.ndarray:
    """Confidence-weighted blend: heuristic -> linear -> GBDT.

    ``*_confidence`` is in [0, 1] and is produced by the engine from sample
    count and observed calibration error. A predictor below
    ``confidence_floor`` (``[engine] ml_confidence_floor`` in the engine config)
    contributes nothing, which is what makes the transition safe: a freshly
    loaded bundle cannot take over the cache before it has proved itself.
    """
    result = np.asarray(heuristic, dtype=np.float64).copy()
    weight = 1.0
    if linear is not None and linear_confidence >= confidence_floor:
        w = min(max(linear_confidence, 0.0), 1.0)
        result = (1.0 - w) * result + w * np.asarray(linear, dtype=np.float64)
        weight = w
    if gbdt is not None and gbdt_confidence >= confidence_floor:
        w = min(max(gbdt_confidence, 0.0), 1.0)
        result = (1.0 - w) * result + w * np.asarray(gbdt, dtype=np.float64)
        weight = w
    LOG.debug("blend effective weight %.3f", weight)
    return result


def blend_frame(
    x: np.ndarray,
    linear: LinearModel | None = None,
    gbdt_probability: np.ndarray | None = None,
    linear_confidence: float = 0.0,
    gbdt_confidence: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper used by the notebook's cold-start demonstration."""
    heuristic = heuristic_probability(x)
    linear_probability = linear.predict(x) if linear is not None else None
    return blend(
        heuristic,
        linear_probability,
        gbdt_probability,
        linear_confidence,
        gbdt_confidence,
    )


def stream_partial_fit(
    model: LinearModel,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    learning_rate: float = 0.01,
) -> LinearModel:
    """Apply :func:`partial_fit` over an iterable of batches."""
    for x, y in batches:
        model = partial_fit(model, x, y, learning_rate=learning_rate)
    return model
