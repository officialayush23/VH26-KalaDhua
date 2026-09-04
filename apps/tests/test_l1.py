"""Tests for the in-process L1 cache.

The behaviour worth protecting is not "does an LRU work" — it is the boundary that keeps a
per-process cache from serving values it cannot reliably be told are wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.l1 import MAX_L1_TTL_MS, L1Cache  # noqa: E402


class FakeClock:
    """Monotonic seconds under test control, so nothing sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def make(**kw: object) -> tuple[L1Cache, FakeClock]:
    clock = FakeClock()
    cache = L1Cache(clock=clock, **kw)  # type: ignore[arg-type]
    return cache, clock


def test_hit_and_miss():
    c, _ = make(max_bytes=10_000)
    assert c.get("a") is None
    c.put("a", {"v": 1}, size_bytes=100, ttl_ms=1_000)
    entry = c.get("a")
    assert entry is not None and entry.value == {"v": 1}
    assert c.stats.hits == 1
    assert c.stats.misses == 1


def test_write_bound_objects_are_never_admitted():
    """The correctness boundary. A price must not live in a tier we cannot invalidate."""
    c, _ = make(max_bytes=10_000)
    admitted = c.put(
        "product:42:price",
        {"price": 40},
        size_bytes=50,
        ttl_ms=60_000,
        freshness_class="write_bound",
    )
    assert admitted is False
    assert c.get("product:42:price") is None
    assert c.stats.refused_class == 1


def test_ttl_is_capped_regardless_of_what_the_object_asks_for():
    c, clock = make(max_bytes=10_000)
    c.put("a", 1, size_bytes=10, ttl_ms=600_000)  # asks for ten minutes
    clock.advance_ms(MAX_L1_TTL_MS + 1)
    assert c.get("a") is None, "L1 honoured a TTL longer than its own ceiling"
    assert c.stats.expired == 1


def test_zero_ttl_does_not_mean_forever():
    c, clock = make(max_bytes=10_000)
    c.put("a", 1, size_bytes=10, ttl_ms=0)
    clock.advance_ms(MAX_L1_TTL_MS + 1)
    assert c.get("a") is None


def test_expiry_is_reported_as_a_miss_not_a_hit():
    c, clock = make(max_bytes=10_000)
    c.put("a", 1, size_bytes=10, ttl_ms=100)
    clock.advance_ms(200)
    assert c.get("a") is None
    assert c.stats.hits == 0
    assert c.stats.misses == 1


def test_byte_budget_is_never_exceeded():
    c, _ = make(max_bytes=1_000, max_entries=1_000)
    for i in range(100):
        c.put(f"k{i}", i, size_bytes=90, ttl_ms=5_000)
        assert c.stats.used_bytes <= 1_000
    assert c.stats.evicted > 0


def test_entry_count_is_bounded():
    c, _ = make(max_bytes=10_000_000, max_entries=5)
    for i in range(50):
        c.put(f"k{i}", i, size_bytes=10, ttl_ms=5_000)
    assert c.stats.entries <= 5


def test_eviction_is_least_recently_used():
    # 100-byte objects need a cache of at least 400 bytes to clear the quarter rule.
    c, _ = make(max_bytes=400, max_entries=10)
    for k in ("a", "b", "c", "d"):
        c.put(k, k, size_bytes=100, ttl_ms=5_000)
    assert c.stats.used_bytes == 400, "the cache should now be exactly full"
    c.get("a")  # a becomes the most recently used, so b is now the coldest
    c.put("e", "e", size_bytes=100, ttl_ms=5_000)
    assert c.get("b") is None, "the least recently used entry should have gone"
    assert c.get("a") is not None
    assert c.get("e") is not None


def test_objects_larger_than_a_quarter_of_the_cache_are_refused():
    c, _ = make(max_bytes=1_000)
    assert c.put("big", "x", size_bytes=400, ttl_ms=1_000) is False
    assert c.stats.refused_size == 1
    # and refusing it does not disturb what is already there
    c.put("small", "y", size_bytes=100, ttl_ms=1_000)
    assert c.put("big2", "x", size_bytes=999, ttl_ms=1_000) is False
    assert c.get("small") is not None


def test_tag_invalidation_removes_every_dependent_entry():
    c, _ = make(max_bytes=10_000)
    c.put("rollup:a", 1, size_bytes=10, ttl_ms=5_000, tags=["row:product:42", "table:orders"])
    c.put("rollup:b", 2, size_bytes=10, ttl_ms=5_000, tags=["row:product:42"])
    c.put("rollup:c", 3, size_bytes=10, ttl_ms=5_000, tags=["row:product:77"])
    removed = c.invalidate_tags(["row:product:42"])
    assert removed == 2
    assert c.get("rollup:a") is None and c.get("rollup:b") is None
    assert c.get("rollup:c") is not None


def test_tag_index_does_not_leak_after_eviction():
    """An evicted entry must take its tag records with it, or the index grows forever."""
    c, _ = make(max_bytes=1_000, max_entries=4)
    for i in range(50):
        c.put(f"k{i}", i, size_bytes=100, ttl_ms=5_000, tags=[f"t{i}"])
    # At most `max_entries` keys are resident, so at most that many tags may remain.
    assert len(c._tags) <= 4, f"tag index leaked: {len(c._tags)} tags for {c.stats.entries} entries"
    for tag, holders in c._tags.items():
        assert holders, f"empty tag bucket left behind: {tag}"
        for key in holders:
            assert key in c._entries, f"tag {tag} points at evicted key {key}"


def test_prefix_invalidation_handles_a_version_bump():
    c, _ = make(max_bytes=10_000)
    c.put("rec:user:1:v7:home", 1, size_bytes=10, ttl_ms=5_000)
    c.put("rec:user:2:v7:home", 2, size_bytes=10, ttl_ms=5_000)
    c.put("rec:user:1:v8:home", 3, size_bytes=10, ttl_ms=5_000)
    assert c.invalidate_prefix("rec:user:1:v7") == 1
    assert c.get("rec:user:1:v7:home") is None
    assert c.get("rec:user:1:v8:home") is not None


def test_reinserting_a_key_replaces_it_without_double_counting_bytes():
    c, _ = make(max_bytes=10_000)
    c.put("a", 1, size_bytes=100, ttl_ms=5_000)
    c.put("a", 2, size_bytes=300, ttl_ms=5_000)
    assert c.stats.used_bytes == 300
    assert c.stats.entries == 1
    assert c.get_value("a") == 2


def test_purge_expired_reclaims_bytes():
    c, clock = make(max_bytes=10_000)
    for i in range(10):
        c.put(f"k{i}", i, size_bytes=100, ttl_ms=1_000)
    clock.advance_ms(2_000)
    assert c.purge_expired() == 10
    assert c.stats.used_bytes == 0


def test_snapshot_is_serialisable():
    c, _ = make(max_bytes=10_000)
    c.put("a", 1, size_bytes=10, ttl_ms=1_000)
    c.get("a")
    snap = c.snapshot()
    assert snap["hits"] == 1
    assert snap["capacity_bytes"] == 10_000
    assert 0.0 <= snap["hit_rate"] <= 1.0


def test_thread_safety_under_concurrent_writers():
    import threading

    c, _ = make(max_bytes=100_000, max_entries=500)

    def worker(n: int) -> None:
        for i in range(500):
            c.put(f"k{n}-{i}", i, size_bytes=100, ttl_ms=5_000)
            c.get(f"k{n}-{i // 2}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert c.stats.used_bytes <= 100_000
    assert c.stats.entries <= 500
    assert c.stats.used_bytes == sum(e.size_bytes for e in c._entries.values())


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
