"""The in-process L1 cache.

This is a real cache, unlike the admission window inside the engine that the telemetry
used to label ``l1``. It holds values, it is bounded in both bytes and entries, and it sits
in front of the network call to AURA.

Why a second tier at all
------------------------
L1 and L2 protect different things. L1 removes a network round trip; L2 removes a rebuild
and protects the origin across the whole application fleet. Only L2 can reason about value,
because only L2 sees every process's demand at once. So L1 is deliberately stupid: LRU,
byte-capped, short TTL, no scoring. Putting a second adaptive policy here would create two
systems competing over the same objects and make every measurement ambiguous.

The correctness decision that matters
-------------------------------------
**An object is only eligible for L1 if it can tolerate being wrong for a few seconds.**

Invalidating L2 is one message to one service. Invalidating L1 means reaching every
application process that might hold a copy, and a process that was starting up, or briefly
partitioned, or simply slow to read the broadcast, will keep serving the old value. That is
a much weaker guarantee, and pretending otherwise is how caches serve stale prices.

So eligibility is decided per object by its freshness class:

===================  ==================  ==========================================
freshness class      L1 eligible         reasoning
===================  ==================  ==========================================
``immutable``        yes, long TTL       cannot go stale; a rendered thumbnail
``time_bound``       yes, short TTL      already tolerates staleness by design
``user_state``       yes, short TTL      key changes when the user acts, so a stale
                                         copy is unreachable rather than wrong
``write_bound``      **no**              a price, a balance, a permission. L2 only,
                                         where invalidation is a single hop.
===================  ==================  ==========================================

Nothing here is clever. It is the boundary that stops the fast path from quietly becoming
the wrong path.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

FreshnessClass = Literal["immutable", "time_bound", "user_state", "write_bound"]

#: Freshness classes that may be held in a per-process cache. See the module docstring.
L1_ELIGIBLE: frozenset[str] = frozenset({"immutable", "time_bound", "user_state"})

#: Ceiling on how long anything may live in L1, whatever TTL the object asks for.
#: A short ceiling is what bounds the blast radius of a missed invalidation.
MAX_L1_TTL_MS: float = 5_000.0


@dataclass(slots=True)
class _Entry:
    value: Any
    size_bytes: int
    expires_at: float
    tags: tuple[str, ...]
    encoding: str


@dataclass(slots=True)
class L1Stats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    evicted: int = 0
    invalidated: int = 0
    admitted: int = 0
    refused_class: int = 0
    refused_size: int = 0
    used_bytes: int = 0
    entries: int = 0
    capacity_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "expired": self.expired,
            "evicted": self.evicted,
            "invalidated": self.invalidated,
            "admitted": self.admitted,
            "refused_class": self.refused_class,
            "refused_size": self.refused_size,
            "used_bytes": self.used_bytes,
            "entries": self.entries,
            "capacity_bytes": self.capacity_bytes,
        }


class L1Cache:
    """A small, thread-safe, byte-bounded LRU cache with tag-based invalidation.

    Sizes are the caller's measurement of the object, not ``sys.getsizeof``. The point of
    the byte cap is to bound the process's memory in the same units the engine reasons
    about, so the two tiers can be compared honestly.
    """

    def __init__(
        self,
        max_bytes: int = 32 * 1024 * 1024,
        max_entries: int = 10_000,
        max_ttl_ms: float = MAX_L1_TTL_MS,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._tags: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._clock = clock
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.max_ttl_ms = float(max_ttl_ms)
        self._used = 0
        self.stats = L1Stats(capacity_bytes=self.max_bytes)

    # -- reads -----------------------------------------------------------

    def get(self, key: str) -> _Entry | None:
        """Return the entry for ``key``, or ``None`` on a miss or an expiry."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if entry.expires_at <= now:
                self._drop_locked(key)
                self.stats.expired += 1
                self.stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return entry

    def get_value(self, key: str, default: Any = None) -> Any:
        entry = self.get(key)
        return default if entry is None else entry.value

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    # -- writes ----------------------------------------------------------

    def eligible(self, freshness_class: str) -> bool:
        return freshness_class in L1_ELIGIBLE

    def put(
        self,
        key: str,
        value: Any,
        *,
        size_bytes: int,
        ttl_ms: float,
        freshness_class: FreshnessClass = "time_bound",
        tags: Iterable[str] = (),
        encoding: str = "json",
    ) -> bool:
        """Offer an object to L1. Returns whether it was admitted.

        Refusal is normal, not an error: a ``write_bound`` object is *supposed* to live
        only in L2, and an object larger than a quarter of the cache would evict most of
        the working set to hold one thing.
        """
        if not self.eligible(freshness_class):
            with self._lock:
                self.stats.refused_class += 1
            return False

        size = max(int(size_bytes), 1)
        if size > self.max_bytes // 4:
            with self._lock:
                self.stats.refused_size += 1
            return False

        # The ceiling applies whatever the object asks for. An object with no TTL is not
        # eligible for an unbounded stay in a tier that cannot be reliably invalidated.
        effective_ttl = self.max_ttl_ms if ttl_ms <= 0 else min(float(ttl_ms), self.max_ttl_ms)
        expires_at = self._clock() + effective_ttl / 1000.0
        tag_tuple = tuple(tags)

        with self._lock:
            if key in self._entries:
                self._drop_locked(key)
            while (
                self._entries
                and (self._used + size > self.max_bytes or len(self._entries) >= self.max_entries)
            ):
                oldest, _ = next(iter(self._entries.items()))
                self._drop_locked(oldest)
                self.stats.evicted += 1

            self._entries[key] = _Entry(value, size, expires_at, tag_tuple, encoding)
            self._used += size
            for tag in tag_tuple:
                self._tags.setdefault(tag, set()).add(key)
            self.stats.admitted += 1
            self._sync_locked()
        return True

    # -- removal ---------------------------------------------------------

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._drop_locked(key)
            self.stats.invalidated += 1
            self._sync_locked()
        return True

    def invalidate_tags(self, tags: Iterable[str]) -> int:
        """Drop every entry carrying any of these tags. Returns how many were removed."""
        removed = 0
        with self._lock:
            for tag in tags:
                for key in list(self._tags.get(tag, ())):
                    if key in self._entries:
                        self._drop_locked(key)
                        removed += 1
            self.stats.invalidated += removed
            self._sync_locked()
        return removed

    def invalidate_prefix(self, prefix: str) -> int:
        """Drop every key starting with ``prefix``.

        This is how a namespace version bump reaches L1: after ``recommendation`` moves to
        v8, everything under ``rec:...:v7:`` is unreachable anyway, and clearing it early
        just reclaims the bytes sooner.
        """
        with self._lock:
            keys = [k for k in self._entries if k.startswith(prefix)]
            for key in keys:
                self._drop_locked(key)
            self.stats.invalidated += len(keys)
            self._sync_locked()
        return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._tags.clear()
            self._used = 0
            self._sync_locked()

    def purge_expired(self) -> int:
        now = self._clock()
        with self._lock:
            stale = [k for k, e in self._entries.items() if e.expires_at <= now]
            for key in stale:
                self._drop_locked(key)
            self.stats.expired += len(stale)
            self._sync_locked()
        return len(stale)

    # -- internals -------------------------------------------------------

    def _drop_locked(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._used -= entry.size_bytes
        if self._used < 0:
            self._used = 0
        for tag in entry.tags:
            holders = self._tags.get(tag)
            if holders is None:
                continue
            holders.discard(key)
            if not holders:
                self._tags.pop(tag, None)

    def _sync_locked(self) -> None:
        self.stats.used_bytes = self._used
        self.stats.entries = len(self._entries)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            self._sync_locked()
            return self.stats.as_dict()


__all__ = ["L1Cache", "L1Stats", "FreshnessClass", "L1_ELIGIBLE", "MAX_L1_TTL_MS"]
