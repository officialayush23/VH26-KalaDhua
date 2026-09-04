"""Future-reuse labels.

For an access to key ``k`` at time ``t`` and a horizon ``h``::

    label_h = 1 if k is accessed again at some t' with 0 < t' - t <= h

Computed by a single reverse pass over the trace, which needs only the
"most recently seen (in reverse) timestamp per key", i.e. the *next* access
time in forward order. That is O(unique keys) memory and one pass, same as the
feature builder.

Right censoring matters and is easy to get wrong. If ``t + h`` is beyond the end
of the trace, a zero label does not mean "not reused", it means "we could not
observe it". Those rows are flagged and dropped from training and evaluation.
Keeping them makes the tail of every trace look like a stream of negatives and
teaches the model that late-arriving traffic is worthless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelResult:
    """Per-row labels and censoring flags, one list per horizon."""

    horizons_ms: tuple[int, ...]
    labels: dict[int, list[int]]
    censored: dict[int, list[int]]
    next_access_ms: list[float]

    def positive_rate(self, horizon_ms: int) -> float:
        values = [
            y
            for y, c in zip(self.labels[horizon_ms], self.censored[horizon_ms], strict=True)
            if not c
        ]
        return sum(values) / len(values) if values else 0.0


def next_access_times(key_ids: Sequence[int], ts_ms: Sequence[float]) -> list[float]:
    """For every row, the timestamp of the next access to the same key.

    ``inf`` when the key is never accessed again.
    """
    if len(key_ids) != len(ts_ms):
        raise ValueError("key_ids and ts_ms must be the same length")
    out = [float("inf")] * len(key_ids)
    seen: dict[int, float] = {}
    for i in range(len(key_ids) - 1, -1, -1):
        key = key_ids[i]
        nxt = seen.get(key)
        if nxt is not None:
            out[i] = nxt
        seen[key] = ts_ms[i]
    return out


def build_labels(
    key_ids: Sequence[int],
    ts_ms: Sequence[float],
    horizons_ms: Sequence[int],
    trace_end_ms: float | None = None,
) -> LabelResult:
    """Label every row at every horizon, flagging right-censored rows."""
    nxt = next_access_times(key_ids, ts_ms)
    end_ms = float(trace_end_ms) if trace_end_ms is not None else (max(ts_ms) if ts_ms else 0.0)

    labels: dict[int, list[int]] = {}
    censored: dict[int, list[int]] = {}
    for horizon in horizons_ms:
        h = float(horizon)
        row_labels: list[int] = []
        row_censored: list[int] = []
        for i, t in enumerate(ts_ms):
            reused = (nxt[i] - t) <= h
            row_labels.append(1 if reused else 0)
            # A positive is always observable; only a zero can be censored.
            row_censored.append(1 if (not reused and t + h > end_ms) else 0)
        labels[horizon] = row_labels
        censored[horizon] = row_censored
        n_cens = sum(row_censored)
        LOG.debug(
            "horizon %d ms: %.4f positive, %d rows censored",
            horizon,
            sum(row_labels) / max(len(row_labels), 1),
            n_cens,
        )
    return LabelResult(
        horizons_ms=tuple(int(h) for h in horizons_ms),
        labels=labels,
        censored=censored,
        next_access_ms=nxt,
    )
