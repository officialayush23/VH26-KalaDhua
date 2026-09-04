"""Request generators used by `POST /load` and by the universe driver.

Four shapes, chosen because they stress different parts of a cache policy:

``zipf``
    Classic skewed popularity. Frequency-based policies do well here.
``scan``
    Every request is a new key. Frequency-based policies pollute themselves.
``burst``
    A small hot set that flips to a *different* small hot set at intervals -
    flash-crowd behaviour.
``popularity_shift``
    The Zipf ranking drifts continuously, so yesterday's head becomes today's
    tail. Punishes policies with long memories.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass

import numpy as np

PATTERNS: tuple[str, ...] = ("zipf", "scan", "burst", "popularity_shift")


@dataclass
class LoadSpec:
    """One `POST /load` job."""

    rps: float
    duration_s: float
    pattern: str = "zipf"
    key_space: int = 5_000
    alpha: float = 1.05
    concurrency: int = 32
    seed: int | None = None

    def validate(self) -> None:
        """Reject nonsense before a job is scheduled."""
        if self.pattern not in PATTERNS:
            raise ValueError(f"unknown pattern {self.pattern!r}; expected one of {PATTERNS}")
        if self.rps <= 0 or self.rps > 20_000:
            raise ValueError("rps must be in (0, 20000]")
        if self.duration_s <= 0 or self.duration_s > 3_600:
            raise ValueError("duration_s must be in (0, 3600]")
        if self.key_space < 1:
            raise ValueError("key_space must be >= 1")


class KeyGenerator:
    """Produces the integer key ids for one workload pattern."""

    def __init__(self, spec: LoadSpec) -> None:
        self._spec = spec
        self._rng = np.random.default_rng(spec.seed)
        self._n = spec.key_space
        self._cursor = 0
        self._started = time.monotonic()
        self._weights = _zipf_weights(self._n, spec.alpha)
        self._ranking = np.arange(self._n)
        self._hot_offset = 0
        self._hot_width = max(4, self._n // 200)

    def next_key(self) -> int:
        """Draw the next key id."""
        pattern = self._spec.pattern
        if pattern == "scan":
            self._cursor += 1
            return self._cursor % max(1, self._n * 8)
        if pattern == "burst":
            elapsed = time.monotonic() - self._started
            epoch = int(elapsed // 8.0)
            if epoch != self._hot_offset:
                self._hot_offset = epoch
            base = (epoch * 977) % self._n
            if self._rng.random() < 0.85:
                return int((base + self._rng.integers(0, self._hot_width)) % self._n)
            return int(self._rng.integers(0, self._n))
        if pattern == "popularity_shift":
            elapsed = time.monotonic() - self._started
            drift = int(elapsed * self._n / 60.0)
            rank = int(self._rng.choice(self._n, p=self._weights))
            return int((self._ranking[rank] + drift) % self._n)
        rank = int(self._rng.choice(self._n, p=self._weights))
        return int(self._ranking[rank])

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.next_key()


def _zipf_weights(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


@dataclass
class LoadReport:
    """Result of a completed load job."""

    pattern: str
    requested: int
    completed: int
    failed: int
    duration_s: float
    achieved_rps: float

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form."""
        return {
            "pattern": self.pattern,
            "requested": self.requested,
            "completed": self.completed,
            "failed": self.failed,
            "duration_s": round(self.duration_s, 3),
            "achieved_rps": round(self.achieved_rps, 2),
        }


async def drive(spec: LoadSpec, handler: Callable[[int], Awaitable[None]]) -> LoadReport:
    """Run `handler` against generated keys at roughly `spec.rps` for `duration_s`.

    Pacing is open-loop with a bounded worker pool: if the origin cannot keep
    up, the achieved rate in the report drops rather than the queue growing
    without limit.
    """
    spec.validate()
    generator = KeyGenerator(spec)
    semaphore = asyncio.Semaphore(spec.concurrency)
    started = time.perf_counter()
    deadline = started + spec.duration_s
    interval = 1.0 / spec.rps

    completed = 0
    failed = 0
    issued = 0
    pending: set[asyncio.Task[None]] = set()

    async def one(key_id: int) -> None:
        nonlocal completed, failed
        async with semaphore:
            try:
                await handler(key_id)
                completed += 1
            except Exception:
                failed += 1

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        target = started + issued * interval
        if target > now:
            await asyncio.sleep(min(target - now, deadline - now))
        task = asyncio.create_task(one(generator.next_key()))
        pending.add(task)
        task.add_done_callback(pending.discard)
        issued += 1

    if pending:
        await asyncio.gather(*list(pending), return_exceptions=True)

    elapsed = time.perf_counter() - started
    return LoadReport(
        pattern=spec.pattern,
        requested=issued,
        completed=completed,
        failed=failed,
        duration_s=elapsed,
        achieved_rps=completed / elapsed if elapsed > 0 else 0.0,
    )
