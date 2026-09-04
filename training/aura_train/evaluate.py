"""Evaluation.

Two kinds of number come out of here and they answer different questions.

*Statistical metrics* (AUC, PR-AUC, log loss, calibration) answer "does the model
rank future reuse correctly?". They are reported **per regime**, on regimes the
model never trained on, because a single pooled AUC hides exactly the failure
this project cares about: a model that is excellent on steady traffic and
useless during a flash crowd.

*The counterfactual replay* answers the question that actually matters: "if this
model had been driving the cache, what would the bill have been?" It replays the
held-out trace through a cache simulator once per policy and compares total
regeneration cost in USD. A model can win on AUC and lose here -- ranking reuse
well is worth nothing if the keys it ranks highly are cheap to regenerate.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .dataset import SPLIT_TEST, xy
from .features import FEATURE_NAMES

LOG = logging.getLogger(__name__)

# Categorical palette, fixed order, never cycled. Validated for CVD separation
# and contrast against a light surface.
PALETTE: tuple[str, ...] = ("#3b6ff5", "#d97706", "#0f9488", "#7c3aed")
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SURFACE = "#fcfcfb"

POLICY_COLORS: dict[str, str] = {
    "aura": PALETTE[0],
    "gdsf": PALETTE[1],
    "lfu": PALETTE[2],
    "lru": PALETTE[3],
    "belady": MUTED,
}


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    from .train_gbdt import _binary_metrics

    return _binary_metrics(y_true, prob)


# --------------------------------------------------------------------------
# Per-regime metrics
# --------------------------------------------------------------------------


def per_regime_metrics(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    predict: Callable[[np.ndarray], np.ndarray],
    horizon_ms: int | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    splits: Iterable[str] = (SPLIT_TEST,),
) -> pd.DataFrame:
    """AUC / PR-AUC / log loss for each regime, plus a pooled row.

    This is the table the project's claim rests on, so it is emitted verbatim to
    CSV and to a figure rather than summarised in prose.
    """
    horizon = horizon_ms or cfg.primary_horizon_ms
    label_column = cfg.label_column(horizon)
    censored_column = cfg.censored_column(horizon)
    usable = frame.loc[frame[censored_column] == 0] if censored_column in frame else frame
    subset = usable.loc[usable["split"].isin(list(splits))]
    if subset.empty:
        raise ValueError(f"no rows in splits {list(splits)}")

    records: list[dict[str, object]] = []
    for regime, group in subset.groupby("regime", sort=True):
        x, y = xy(group, label_column, feature_names)
        row = _metrics(y, predict(x))
        records.append(
            {
                "regime": str(regime),
                "seen_in_training": str(regime) in cfg.split.train_regimes,
                **row,
            }
        )
    x_all, y_all = xy(subset, label_column, feature_names)
    pooled = _metrics(y_all, predict(x_all))
    records.append({"regime": "ALL", "seen_in_training": False, **pooled})

    table = pd.DataFrame.from_records(records)
    table["horizon_ms"] = horizon
    return table


def calibration_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    bins: int = 12,
) -> pd.DataFrame:
    """Equal-width reliability bins: mean predicted vs observed frequency."""
    prob = np.asarray(prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, bins - 1)
    records = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        if n == 0:
            continue
        records.append(
            {
                "bin": b,
                "predicted": float(prob[mask].mean()),
                "observed": float(y_true[mask].mean()),
                "count": n,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        weights = frame["count"] / frame["count"].sum()
        frame.attrs["ece"] = float((weights * (frame["predicted"] - frame["observed"]).abs()).sum())
    return frame


# --------------------------------------------------------------------------
# Counterfactual cache replay
# --------------------------------------------------------------------------


@dataclass
class ReplayResult:
    policy: str
    requests: int
    hits: int
    byte_hits: int
    total_bytes: int
    regen_cost_usd: float
    evictions: int

    @property
    def object_hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def byte_hit_rate(self) -> float:
        return self.byte_hits / self.total_bytes if self.total_bytes else 0.0


@dataclass
class _Resident:
    size_bytes: int
    cost_usd: float
    last_access: int
    frequency: int
    score: float


class CacheReplay:
    """Sampled-eviction cache simulator.

    It evicts the worst of ``candidate_sample`` randomly chosen residents rather
    than the global worst, which is what the engine does (``[engine]
    candidate_sample = 32``) and what makes the comparison fair: every policy
    here pays the same approximation.
    """

    def __init__(
        self,
        capacity_bytes: int,
        value_fn: Callable[[_Resident, int], float],
        candidate_sample: int = 32,
        seed: int = 42,
    ) -> None:
        self.capacity_bytes = max(int(capacity_bytes), 1)
        self.value_fn = value_fn
        self.candidate_sample = candidate_sample
        self.rng = random.Random(seed)
        self.entries: dict[int, _Resident] = {}
        self.used_bytes = 0
        self.evictions = 0

    def _evict_until(self, needed: int, clock: int) -> None:
        while self.used_bytes + needed > self.capacity_bytes and self.entries:
            keys = list(self.entries)
            if len(keys) > self.candidate_sample:
                keys = self.rng.sample(keys, self.candidate_sample)
            victim = min(keys, key=lambda k: self.value_fn(self.entries[k], clock))
            self.used_bytes -= self.entries.pop(victim).size_bytes
            self.evictions += 1

    def access(
        self,
        clock: int,
        key_id: int,
        size_bytes: int,
        cost_usd: float,
        score: float,
    ) -> bool:
        entry = self.entries.get(key_id)
        if entry is not None:
            entry.last_access = clock
            entry.frequency += 1
            entry.score = score
            return True
        if size_bytes <= self.capacity_bytes:
            self._evict_until(size_bytes, clock)
            self.entries[key_id] = _Resident(size_bytes, cost_usd, clock, 1, score)
            self.used_bytes += size_bytes
        return False


def _value_lru(entry: _Resident, clock: int) -> float:
    return float(entry.last_access)


def _value_lfu(entry: _Resident, clock: int) -> float:
    return float(entry.frequency)


def _value_gdsf(entry: _Resident, clock: int) -> float:
    # Greedy-Dual-Size-Frequency with the usual aging term folded in as the
    # access clock, so old objects decay without a separate inflation counter.
    return entry.frequency * max(entry.cost_usd, 1e-12) / max(entry.size_bytes, 1) + (
        entry.last_access / 1e12
    )


def _value_aura(entry: _Resident, clock: int) -> float:
    # Predicted economic value density: probability of reuse times the cost we
    # avoid, per byte held.
    return entry.score * max(entry.cost_usd, 1e-12) / max(entry.size_bytes, 1)


POLICIES: dict[str, Callable[[_Resident, int], float]] = {
    "lru": _value_lru,
    "lfu": _value_lfu,
    "gdsf": _value_gdsf,
    "aura": _value_aura,
}


def replay(
    frame: pd.DataFrame,
    scores: np.ndarray,
    capacity_bytes: int,
    policies: Iterable[str] = ("lru", "lfu", "gdsf", "aura"),
    candidate_sample: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Replay one trace under several policies and price the misses.

    ``scores`` is the model's reuse probability for each row; only the ``aura``
    policy consumes it, the baselines are blind to it. The cost charged on a
    miss is ``regen_cost_usd``, the same priced cost vector the model sees as a
    feature, so the two halves of the system agree on what money is.
    """
    key_ids = frame["key_id"].to_numpy(dtype=np.int64)
    sizes = frame["size_bytes"].to_numpy(dtype=np.int64)
    costs = frame["regen_cost_usd"].to_numpy(dtype=np.float64)
    probabilities = np.asarray(scores, dtype=np.float64)
    if len(probabilities) != len(key_ids):
        raise ValueError("scores and frame must be the same length")

    records = []
    for policy in policies:
        sim = CacheReplay(capacity_bytes, POLICIES[policy], candidate_sample, seed)
        hits = 0
        byte_hits = 0
        cost = 0.0
        for i in range(len(key_ids)):
            hit = sim.access(i, int(key_ids[i]), int(sizes[i]), float(costs[i]),
                             float(probabilities[i]))
            if hit:
                hits += 1
                byte_hits += int(sizes[i])
            else:
                cost += float(costs[i])
        result = ReplayResult(
            policy=policy,
            requests=len(key_ids),
            hits=hits,
            byte_hits=byte_hits,
            total_bytes=int(sizes.sum()),
            regen_cost_usd=cost,
            evictions=sim.evictions,
        )
        records.append(
            {
                "policy": result.policy,
                "object_hit_rate": result.object_hit_rate,
                "byte_hit_rate": result.byte_hit_rate,
                "regen_cost_usd": result.regen_cost_usd,
                "evictions": result.evictions,
                "requests": result.requests,
            }
        )
        LOG.info(
            "replay %-5s hit_rate=%.4f cost=$%.4f",
            policy,
            result.object_hit_rate,
            result.regen_cost_usd,
        )

    table = pd.DataFrame.from_records(records)
    baseline = table.loc[table["policy"] == "lru", "regen_cost_usd"]
    if not baseline.empty and float(baseline.iloc[0]) > 0:
        table["saving_vs_lru"] = 1.0 - table["regen_cost_usd"] / float(baseline.iloc[0])
    return table


def replay_by_regime(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    predict: Callable[[np.ndarray], np.ndarray],
    capacity_bytes: int | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Counterfactual replay for each held-out regime separately."""
    capacity = capacity_bytes or cfg.features.sim_capacity_bytes
    test = frame.loc[frame["split"] == SPLIT_TEST]
    if test.empty:
        raise ValueError("no test rows to replay")
    tables = []
    for regime, group in test.groupby("regime", sort=True):
        ordered = group.sort_values("ts_ms")
        x = ordered[list(feature_names)].to_numpy(dtype=np.float64)
        table = replay(ordered, predict(x), capacity)
        table.insert(0, "regime", str(regime))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_per_regime(table: pd.DataFrame, path: Path, metric: str = "auc") -> Path:
    """One bar per regime, direct-labelled, single axis, sorted by value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = table.loc[table["regime"] != "ALL"].sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * max(len(data), 3) + 1.6))
    colors = [PALETTE[1] if seen else PALETTE[0] for seen in data["seen_in_training"]]
    bars = ax.barh(data["regime"], data[metric], color=colors, height=0.62)
    for bar, value in zip(bars, data[metric], strict=True):
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    pooled = table.loc[table["regime"] == "ALL", metric]
    if not pooled.empty:
        ax.axvline(float(pooled.iloc[0]), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(
            float(pooled.iloc[0]),
            len(data) - 0.4,
            " pooled",
            color=MUTED,
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Held-out reuse {metric.upper()} by regime", fontsize=12, loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)
    return path


def plot_calibration(curve: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="perfect calibration")
    ax.plot(
        curve["predicted"],
        curve["observed"],
        color=PALETTE[0],
        linewidth=2.0,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="model",
    )
    ece = curve.attrs.get("ece")
    title = "Calibration on held-out regimes"
    if ece is not None:
        title += f"  (ECE {ece:.3f})"
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("predicted reuse probability")
    ax.set_ylabel("observed reuse frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)
    return path


def plot_importance(importance: dict[str, float], path: Path, top: int = 16) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(importance.items())[:top][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.0, 0.38 * max(len(items), 3) + 1.6))
    ax.barh(names, values, color=PALETTE[0], height=0.62)
    for y, value in enumerate(values):
        ax.text(value + max(values) * 0.01, y, f"{value:.3f}", va="center",
                fontsize=9, color=INK)
    ax.set_xlabel("share of total split gain")
    ax.set_title("Feature importance (split gain)", fontsize=12, loc="left", pad=12)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)
    return path


def plot_replay(table: pd.DataFrame, path: Path) -> Path:
    """Cost per policy, grouped by regime. One axis: USD."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = list(dict.fromkeys(table["regime"])) if "regime" in table else ["all"]
    policies = list(dict.fromkeys(table["policy"]))
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(1.6 * max(len(regimes), 3) + 3.0, 4.6))
    for i, policy in enumerate(policies):
        rows = table.loc[table["policy"] == policy]
        if "regime" in table:
            values = [
                float(rows.loc[rows["regime"] == r, "regen_cost_usd"].sum()) for r in regimes
            ]
        else:
            values = [float(rows["regen_cost_usd"].sum())]
        offsets = np.arange(len(regimes)) + i * width - 0.4 + width / 2
        ax.bar(offsets, values, width=width * 0.88, label=policy,
               color=POLICY_COLORS.get(policy, PALETTE[i % len(PALETTE)]))
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("regeneration cost (USD)")
    ax.set_title("Counterfactual replay: cost of the misses each policy allows",
                 fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncols=len(policies))
    _style_axes(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------
# Whole-run report
# --------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    per_regime: pd.DataFrame
    calibration: pd.DataFrame
    importance: dict[str, float]
    replay: pd.DataFrame
    ablations: pd.DataFrame | None = None
    paths: dict[str, Path] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        lines = ["per-regime metrics:", self.per_regime.to_string(index=False)]
        lines += ["", "counterfactual replay:", self.replay.to_string(index=False)]
        if self.ablations is not None:
            lines += ["", "ablations:", self.ablations.to_string(index=False)]
        return lines


def evaluate(
    cfg: TrainingConfig,
    frame: pd.DataFrame,
    predict: Callable[[np.ndarray], np.ndarray],
    importance: dict[str, float] | None = None,
    horizon_ms: int | None = None,
    ablations: pd.DataFrame | None = None,
    write_figures: bool = True,
) -> EvaluationReport:
    """Produce every table and figure for one model."""
    horizon = horizon_ms or cfg.primary_horizon_ms
    cfg.ensure_dirs()
    label_column = cfg.label_column(horizon)
    censored_column = cfg.censored_column(horizon)

    table = per_regime_metrics(cfg, frame, predict, horizon)
    usable = frame.loc[frame[censored_column] == 0] if censored_column in frame else frame
    test = usable.loc[usable["split"] == SPLIT_TEST]
    x_test, y_test = xy(test, label_column)
    probabilities = predict(x_test)
    curve = calibration_curve(y_test, probabilities)
    replay_table = replay_by_regime(cfg, usable, predict)

    paths: dict[str, Path] = {}
    tag = cfg.horizon_label(horizon)
    csv_path = cfg.report_dir / f"per_regime_{tag}.csv"
    table.to_csv(csv_path, index=False)
    paths["per_regime_csv"] = csv_path
    replay_csv = cfg.report_dir / f"replay_{tag}.csv"
    replay_table.to_csv(replay_csv, index=False)
    paths["replay_csv"] = replay_csv
    if ablations is not None:
        ablation_csv = cfg.report_dir / f"ablations_{tag}.csv"
        ablations.to_csv(ablation_csv, index=False)
        paths["ablations_csv"] = ablation_csv

    if write_figures:
        paths["per_regime_png"] = plot_per_regime(table, cfg.report_dir / f"per_regime_{tag}.png")
        paths["calibration_png"] = plot_calibration(
            curve, cfg.report_dir / f"calibration_{tag}.png"
        )
        if importance:
            paths["importance_png"] = plot_importance(
                importance, cfg.report_dir / f"importance_{tag}.png"
            )
        paths["replay_png"] = plot_replay(replay_table, cfg.report_dir / f"replay_{tag}.png")

    return EvaluationReport(
        per_regime=table,
        calibration=curve,
        importance=importance or {},
        replay=replay_table,
        ablations=ablations,
        paths=paths,
    )
