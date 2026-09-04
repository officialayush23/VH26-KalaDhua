"""A small but genuine recommender.

Item-item cosine similarity over the synthetic interaction matrix, followed by a
ranking pass. The arithmetic is real: a request burns tens to hundreds of
milliseconds of CPU in BLAS, and the cost vector the application reports is that
measurement, not a sleep.

Shape of the work:

1. *profile* - fold the user's interaction history into a single embedding,
   weighted by recency. Stands in for a feature-store read, and is reported as
   database time because that is what it is in production.
2. *retrieval* - score every catalogue item against the profile and keep the top
   `candidates`. One matrix-vector product; reported as accelerator time.
3. *similarity* - the expensive step. A candidate x catalogue cosine similarity
   block, which is what an item-item recommender actually computes, run once per
   ensemble shard.
4. *ranking* - diversity-aware re-ranking in Python over the candidate slice.

The expensive tail scales the number of ensemble shards, so a tail key runs the
same retrieval and produces the same-sized object while burning 20-50x the CPU.
That is the workload a frequency-only policy cannot see.
"""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass

import numpy as np

from common.costing import CostMeter
from recommendation.data import CATEGORIES, Catalogue, catalogue

BASE_CANDIDATES = 2_000
SIMILARITY_CHUNK = 512
MAX_PASSES = 64

MIN_OBJECT_BYTES = 524_288
MAX_OBJECT_BYTES = 4_194_304
FACTOR_BYTES = 2.75  # float16 in base64, including the JSON quoting overhead
ITEM_OVERHEAD_BYTES = 170


@dataclass
class Ranking:
    """The regenerated object served to the caller."""

    user_id: int
    items: list[dict[str, object]]
    candidates: int
    passes: int
    coherence: float
    embedding_ms: float
    similarity_ms: float
    ranking_ms: float
    index_ms: float

    def as_dict(self) -> dict[str, object]:
        """JSON body for `/work/{id}`."""
        return {
            "user_id": self.user_id,
            "items": self.items,
            "candidates_considered": self.candidates,
            "ensemble_passes": self.passes,
            "coherence": round(self.coherence, 6),
            "stage_ms": {
                "embedding": round(self.embedding_ms, 3),
                "similarity": round(self.similarity_ms, 3),
                "ranking": round(self.ranking_ms, 3),
                "index_lookup": round(self.index_ms, 3),
            },
        }


def recommend(
    user_id: int,
    *,
    top_k: int = 60,
    work_factor: float = 1.0,
    explain_dim: int = 256,
    meter: CostMeter | None = None,
    cat: Catalogue | None = None,
) -> Ranking:
    """Produce a ranking for `user_id`.

    `meter`, when supplied, receives the accelerator and feature-store timings
    that the caller cannot infer from wall clock alone.
    """
    cat = cat or catalogue()
    passes = int(max(1, min(MAX_PASSES, round(work_factor))))
    explain_dim = int(max(16, min(cat.n_items, explain_dim)))
    section = _Section()

    with section:
        history = cat.history(user_id)
        history_ts = cat.history_timestamps(user_id)
        recency = np.exp(-(history_ts.max() - history_ts) / (14 * 86_400.0)).astype(np.float32)
        seen = np.unique(history)
    index_ms = section.ms

    with section:
        profile = (cat.embeddings[history] * recency[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(profile))
        if norm > 0.0:
            profile /= norm
        affinity = cat.embeddings @ profile
        affinity += 0.12 * cat.popularity * float(cat.n_items)
        affinity[seen] -= 1.5
        candidates = min(BASE_CANDIDATES, cat.n_items - 1)
        candidate_ids = np.argpartition(-affinity, candidates)[:candidates]
        candidate_ids = candidate_ids[np.argsort(-affinity[candidate_ids])]
        block = np.ascontiguousarray(cat.embeddings[candidate_ids])
    embedding_ms = section.ms

    # Item-item similarity, once per ensemble shard, in chunks so peak memory
    # stays bounded regardless of how many shards a tail key asks for.
    with section:
        neighbour_strength = np.zeros(candidates, dtype=np.float32)
        contributions = np.zeros((candidates, explain_dim), dtype=np.float32)
        shards = _shard_weights(cat.embeddings.shape[1], passes)
        for shard in range(passes):
            weights = shards[shard]
            for start in range(0, candidates, SIMILARITY_CHUNK):
                stop = min(candidates, start + SIMILARITY_CHUNK)
                similarity = (block[start:stop] * weights) @ cat.embeddings.T
                neighbour_strength[start:stop] += np.partition(similarity, -16, axis=1)[:, -16:].mean(axis=1)
                if shard == 0:
                    contributions[start:stop] = similarity[:, :explain_dim]
        neighbour_strength /= float(passes)
    similarity_ms = section.ms

    with section:
        scores = affinity[candidate_ids] * (0.75 + 0.25 * neighbour_strength)
        order = np.argsort(-scores)
        per_category: dict[int, int] = {}
        chosen: list[int] = []
        cap = max(3, top_k // 6)
        for position in order:
            item = int(candidate_ids[position])
            category = int(cat.categories[item])
            taken = per_category.get(category, 0)
            if taken >= cap:
                continue
            per_category[category] = taken + 1
            chosen.append(int(position))
            if len(chosen) >= top_k:
                break
        if len(chosen) < top_k:
            chosen.extend(int(p) for p in order[: top_k - len(chosen)])
    ranking_ms = section.ms

    packed = contributions.astype(np.float16)
    items: list[dict[str, object]] = []
    for rank, position in enumerate(chosen[:top_k]):
        item = int(candidate_ids[position])
        items.append(
            {
                "rank": rank + 1,
                "item_id": item,
                "score": round(float(scores[position]), 6),
                "category": CATEGORIES[int(cat.categories[item]) % len(CATEGORIES)],
                "unit_price": float(cat.prices[item]),
                "neighbour_strength": round(float(neighbour_strength[position]), 6),
                "factors_f16_b64": base64.b64encode(packed[position].tobytes()).decode("ascii"),
            }
        )

    if meter is not None:
        meter.add_gpu_ms(embedding_ms)
        meter.add_db_ms(index_ms)

    return Ranking(
        user_id=user_id,
        items=items,
        candidates=candidates,
        passes=passes,
        coherence=float(neighbour_strength.mean()),
        embedding_ms=embedding_ms,
        similarity_ms=similarity_ms,
        ranking_ms=ranking_ms,
        index_ms=index_ms,
    )


def _shard_weights(dim: int, passes: int) -> np.ndarray:
    """Deterministic per-shard feature weights for the ensemble."""
    rng = np.random.default_rng(20260904)
    return (0.85 + 0.3 * rng.random((max(1, passes), dim))).astype(np.float32)


def payload_dimensions(key_id: int, target_bytes: int) -> tuple[int, int]:
    """Pick `top_k` and explanation width so the object lands near `target_bytes`.

    The size is carried by real content - more ranked items, each shipping more
    of its similarity row - rather than by padding.
    """
    explain_dim = 600 + (key_id % 5) * 600
    per_item = explain_dim * FACTOR_BYTES + ITEM_OVERHEAD_BYTES
    top_k = int(max(40, min(BASE_CANDIDATES, target_bytes / per_item)))
    return top_k, explain_dim


def target_bytes_for(key_id: int) -> int:
    """Deterministic object size in [0.5 MB, 4 MB] for this key."""
    spread = (math.sin(key_id * 0.7391) + 1.0) / 2.0
    return int(MIN_OBJECT_BYTES + spread * (MAX_OBJECT_BYTES - MIN_OBJECT_BYTES))


class _Section:
    """Reusable wall-clock timer for one pipeline stage."""

    def __init__(self) -> None:
        self.ms = 0.0
        self._t0 = 0.0

    def __enter__(self) -> _Section:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
