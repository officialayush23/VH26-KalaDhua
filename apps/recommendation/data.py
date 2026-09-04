"""Synthetic catalogue and interaction matrix for the recommendation app.

Built once at start-up from a fixed seed, so two processes on two machines hold
the same corpus and the benchmark numbers are comparable. Nothing here is
random at request time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from common.settings import get_settings

N_ITEMS = 6_000
N_USERS = 20_000
EMBED_DIM = 384
AVG_INTERACTIONS = 48
CATEGORIES = (
    "electronics",
    "apparel",
    "grocery",
    "home",
    "beauty",
    "sports",
    "books",
    "toys",
    "automotive",
    "garden",
)


@dataclass
class Catalogue:
    """Item embeddings, popularity and the user interaction index."""

    embeddings: np.ndarray  # (N_ITEMS, EMBED_DIM) float32, L2 normalised
    popularity: np.ndarray  # (N_ITEMS,) float32, sums to 1
    categories: np.ndarray  # (N_ITEMS,) int16
    prices: np.ndarray  # (N_ITEMS,) float32
    interaction_items: np.ndarray  # flat item ids, grouped by user
    interaction_offsets: np.ndarray  # (N_USERS + 1,) index into interaction_items
    interaction_ts: np.ndarray  # (len(interaction_items),) int64, sorted per user

    @property
    def n_items(self) -> int:
        """Catalogue size."""
        return int(self.embeddings.shape[0])

    @property
    def n_users(self) -> int:
        """Number of known users."""
        return int(self.interaction_offsets.shape[0] - 1)

    def history(self, user_id: int) -> np.ndarray:
        """Item ids this user interacted with."""
        u = user_id % self.n_users
        lo, hi = int(self.interaction_offsets[u]), int(self.interaction_offsets[u + 1])
        return self.interaction_items[lo:hi]

    def history_timestamps(self, user_id: int) -> np.ndarray:
        """Interaction timestamps for this user, ascending."""
        u = user_id % self.n_users
        lo, hi = int(self.interaction_offsets[u]), int(self.interaction_offsets[u + 1])
        return self.interaction_ts[lo:hi]


def _build(seed: int) -> Catalogue:
    rng = np.random.default_rng(seed)

    # Items live on a handful of latent topics, so cosine similarity is
    # structured rather than uniform noise - the ranking has something to find.
    n_topics = 24
    topics = rng.normal(size=(n_topics, EMBED_DIM)).astype(np.float32)
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    assignment = rng.integers(0, n_topics, size=N_ITEMS)
    noise = rng.normal(scale=0.55, size=(N_ITEMS, EMBED_DIM)).astype(np.float32)
    embeddings = topics[assignment] + noise
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    ranks = np.arange(1, N_ITEMS + 1, dtype=np.float64)
    popularity = 1.0 / np.power(ranks, 0.9)
    rng.shuffle(popularity)
    popularity = (popularity / popularity.sum()).astype(np.float64)

    categories = (assignment % len(CATEGORIES)).astype(np.int16)
    prices = np.round(rng.gamma(shape=2.2, scale=18.0, size=N_ITEMS), 2).astype(np.float32)

    counts = rng.poisson(AVG_INTERACTIONS, size=N_USERS).clip(4, 400)
    offsets = np.zeros(N_USERS + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    items = rng.choice(N_ITEMS, size=total, p=popularity).astype(np.int32)
    timestamps = np.sort(rng.integers(0, 90 * 86_400, size=total)).astype(np.int64)

    return Catalogue(
        embeddings=embeddings,
        popularity=popularity.astype(np.float32),
        categories=categories,
        prices=prices,
        interaction_items=items,
        interaction_offsets=offsets,
        interaction_ts=timestamps,
    )


@lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    """Process-wide catalogue singleton."""
    return _build(get_settings().seed)
