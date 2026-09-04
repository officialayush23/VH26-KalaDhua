"""Synthetic media and document objects.

The content workload is the mirror image of analytics: the objects are large
(100 KB to 20 MB) and cost almost nothing in CPU to produce, but every miss
moves real bytes. Cache value here is dominated by bandwidth and by the
occasional object that carries a third-party API charge.

Generation is deliberately cheap - a seeded 64 KB block tiled to the requested
size, then salted so two objects never share content. What it burns is memory
bandwidth, not arithmetic, which is exactly the cost shape being modelled.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

BLOCK_BYTES = 65_536

KB = 1_024
MB = 1_024 * 1_024


@dataclass(frozen=True)
class ObjectKind:
    """One family of content objects."""

    name: str
    min_bytes: int
    max_bytes: int
    ttl_ms: int
    sla_class: str
    priced: bool = False


KINDS: tuple[ObjectKind, ...] = (
    ObjectKind("image_variant", 100 * KB, 2 * MB, ttl_ms=3_600_000, sla_class="normal"),
    ObjectKind("document_render", 200 * KB, 3 * MB, ttl_ms=1_800_000, sla_class="normal"),
    ObjectKind("video_segment", 2 * MB, 20 * MB, ttl_ms=600_000, sla_class="low"),
    ObjectKind("syndicated_article", 8 * KB, 120 * KB, ttl_ms=90_000, sla_class="high", priced=True),
)

BY_NAME: dict[str, ObjectKind] = {kind.name: kind for kind in KINDS}

# The priced object type is rarer than the media types, but it is the one that
# makes the cost-spike scenario bite.
_MIX = ("image_variant", "image_variant", "document_render", "video_segment", "syndicated_article")


def kind_for(key_id: int) -> ObjectKind:
    """Deterministic object family for a key id."""
    return BY_NAME[_MIX[key_id % len(_MIX)]]


def size_for(key_id: int, kind: ObjectKind) -> int:
    """Deterministic object size within the family's band."""
    spread = (math.sin(key_id * 1.2371 + len(kind.name)) + 1.0) / 2.0
    return int(kind.min_bytes + spread * (kind.max_bytes - kind.min_bytes))


@lru_cache(maxsize=1)
def _base_block() -> bytes:
    return np.random.default_rng(20260904).bytes(BLOCK_BYTES)


def generate(key_id: int, size_bytes: int) -> bytes:
    """Produce `size_bytes` of deterministic content for `key_id`."""
    block = _base_block()
    salt = hashlib.blake2b(str(key_id).encode("utf-8"), digest_size=32).digest()
    repeats = size_bytes // BLOCK_BYTES + 1
    body = bytearray(block * repeats)
    # Salt at page boundaries so the object is unique without paying to
    # randomise every byte.
    for offset in range(0, len(body) - len(salt), 4_096):
        body[offset : offset + len(salt)] = salt
    header = f"aura-content:{key_id}:{size_bytes}:".encode("ascii")
    body[: len(header)] = header
    return bytes(body[:size_bytes])


def digest(payload: bytes) -> str:
    """Short content digest, so a caller can verify a hit matches the origin."""
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
