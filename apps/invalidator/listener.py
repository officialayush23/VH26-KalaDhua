"""Postgres LISTEN → AURA invalidation.

The bridge that makes a hand-edited row visible to the cache. Someone opens the Supabase
SQL editor and runs ``update app_products set unit_price = 20 where id = 1292``. That write
never touches our application code, so no SDK hook can see it. The trigger installed by
``training/sql/005_consistency.sql`` emits a NOTIFY; this process is what turns that into
``POST /v1/invalidate``.

Three properties it has to have, in order of importance:

**It must never sit on the request path.** An invalidation arriving 50 ms late is fine. A
GET blocked on a database round trip is not. So this is a separate process with its own
connection, and the cache does not wait for it.

**It must survive the database going away.** A listener that dies on a dropped connection
is worse than no listener, because the cache keeps serving happily while nobody is watching
for changes. Reconnection is exponential with jitter, and the process reports how long it
has been disconnected so the dashboard can say so out loud.

**It must not amplify.** A bulk update of ten thousand rows produces ten thousand
notifications. Sending ten thousand HTTP requests would turn a database write into a
denial-of-service against our own engine, so tags are batched over a short window and
deduplicated before they are sent.

Run it with::

    python -m invalidator.listener

It needs ``AURA_APPS_SUPABASE_DIRECT_CONNECTION_URL`` (the *direct* connection, not the
transaction pooler — pgbouncer in transaction mode does not support LISTEN) and
``AURA_APPS_AURA_BASE_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

LOG = logging.getLogger("aura.invalidator")

CHANNEL = "aura_invalidate"

#: Tags are collected for this long before being sent, so a bulk update becomes one call.
BATCH_WINDOW_S = 0.15

#: Ceiling on tags per request. A larger batch is split rather than dropped.
MAX_TAGS_PER_CALL = 500


@dataclass
class Notification:
    """One decoded NOTIFY payload."""

    tag: str
    entity: str = ""
    row_id: str = ""
    op: str = ""
    version: int | None = None
    mode: str = "hard"
    table: str = ""
    at: float = 0.0

    @classmethod
    def parse(cls, payload: str) -> Notification | None:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            # A bare tag is also accepted, so the mechanism can be driven by hand:
            #   select pg_notify('aura_invalidate', 'row:product:1292');
            payload = payload.strip()
            return cls(tag=payload) if payload else None
        tag = raw.get("tag")
        if not tag:
            return None
        return cls(
            tag=str(tag),
            entity=str(raw.get("entity", "")),
            row_id=str(raw.get("id", "")),
            op=str(raw.get("op", "")),
            version=raw.get("version"),
            mode=str(raw.get("mode", "hard")),
            table=str(raw.get("table", "")),
            at=float(raw.get("at", 0.0) or 0.0),
        )


@dataclass
class Stats:
    received: int = 0
    sent_batches: int = 0
    tags_sent: int = 0
    keys_invalidated: int = 0
    version_bumps: int = 0
    send_failures: int = 0
    reconnects: int = 0
    connected_since: float = 0.0
    disconnected_since: float | None = None
    last_tag: str = ""
    lag_ms_recent: list[float] = field(default_factory=list)

    def note_lag(self, ms: float) -> None:
        self.lag_ms_recent.append(ms)
        if len(self.lag_ms_recent) > 200:
            self.lag_ms_recent.pop(0)

    @property
    def mean_lag_ms(self) -> float:
        return sum(self.lag_ms_recent) / len(self.lag_ms_recent) if self.lag_ms_recent else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "sent_batches": self.sent_batches,
            "tags_sent": self.tags_sent,
            "keys_invalidated": self.keys_invalidated,
            "version_bumps": self.version_bumps,
            "send_failures": self.send_failures,
            "reconnects": self.reconnects,
            "connected": self.disconnected_since is None,
            "mean_lag_ms": round(self.mean_lag_ms, 2),
            "last_tag": self.last_tag,
        }


class InvalidationForwarder:
    """Batches tags and posts them to the engine."""

    def __init__(self, base_url: str, stats: Stats, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.stats = stats
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(maxsize=100_000)

    async def aclose(self) -> None:
        await self._client.aclose()

    def submit(self, note: Notification) -> None:
        try:
            self._queue.put_nowait(note)
        except asyncio.QueueFull:
            # Dropping is the right failure here: the alternative is unbounded memory in a
            # process whose whole job is to stay alive. Say so loudly.
            LOG.error("invalidation queue full, dropping tag %s", note.tag)

    async def run(self) -> None:
        """Drain the queue in short batches, forever."""
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.monotonic() + BATCH_WINDOW_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            await self._flush(batch)

    async def _flush(self, batch: list[Notification]) -> None:
        versions = [n for n in batch if n.mode == "version"]
        hard = _dedupe(n.tag for n in batch if n.mode == "hard")
        soft = _dedupe(n.tag for n in batch if n.mode == "soft")

        for note in versions:
            await self._bump_namespace(note)
        for chunk in _chunks(hard, MAX_TAGS_PER_CALL):
            await self._post_invalidate(chunk, "hard", batch)
        for chunk in _chunks(soft, MAX_TAGS_PER_CALL):
            await self._post_invalidate(chunk, "soft", batch)

    async def _post_invalidate(self, tags: list[str], mode: str, batch: list[Notification]) -> None:
        body = {"tags": tags, "mode": mode, "source": "postgres"}
        try:
            resp = await self._client.post(f"{self.base_url}/v1/invalidate", json=body)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - a listener must not die on a bad response
            self.stats.send_failures += 1
            LOG.warning("invalidate failed for %d %s tag(s): %s", len(tags), mode, exc)
            return

        matched = int(data.get("matched", 0))
        self.stats.sent_batches += 1
        self.stats.tags_sent += len(tags)
        self.stats.keys_invalidated += matched
        self.stats.last_tag = tags[-1]

        now = time.time()
        for note in batch:
            if note.at:
                self.stats.note_lag((now - note.at) * 1000.0)

        LOG.info(
            "invalidated %d cache object(s) from %d %s tag(s), first=%s",
            matched,
            len(tags),
            mode,
            tags[0],
        )

    async def _bump_namespace(self, note: Notification) -> None:
        body = {"namespace": note.row_id, "version": note.version}
        try:
            resp = await self._client.post(f"{self.base_url}/v1/version/bump", json=body)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self.stats.send_failures += 1
            LOG.warning("namespace bump failed for %s: %s", note.row_id, exc)
            return
        self.stats.version_bumps += 1
        LOG.info("namespace %s retired to version %s", note.row_id, note.version)


def _dedupe(tags: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for tag in tags:
        seen.setdefault(tag, None)
    return list(seen)


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def listen_forever(dsn: str, forwarder: InvalidationForwarder, stats: Stats) -> None:
    """Hold a LISTEN connection open, reconnecting with backoff when it drops."""
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "asyncpg is required for the invalidation listener: pip install asyncpg"
        ) from exc

    backoff = 0.5
    while True:
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            await conn.add_listener(
                CHANNEL,
                lambda _c, _pid, _ch, payload: _on_notify(payload, forwarder, stats),
            )
            stats.connected_since = time.time()
            if stats.disconnected_since is not None:
                LOG.info(
                    "reconnected after %.1fs offline",
                    time.time() - stats.disconnected_since,
                )
            stats.disconnected_since = None
            backoff = 0.5
            LOG.info("listening on channel %s", CHANNEL)

            # asyncpg delivers notifications on its own reader task; this loop only has to
            # notice when the connection has gone.
            while not conn.is_closed():
                await asyncio.sleep(1.0)
            raise ConnectionError("connection closed by server")

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if stats.disconnected_since is None:
                stats.disconnected_since = time.time()
            stats.reconnects += 1
            # Jitter matters: several listeners restarting in lockstep after a database
            # restart is how a recovery turns into a second outage.
            delay = min(backoff, 30.0) * (0.5 + random.random())
            LOG.warning("listener disconnected (%s); retrying in %.1fs", exc, delay)
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, 30.0)
        finally:
            if conn is not None and not conn.is_closed():
                await conn.close()


def _on_notify(payload: str, forwarder: InvalidationForwarder, stats: Stats) -> None:
    note = Notification.parse(payload)
    if note is None:
        LOG.warning("ignoring unparseable notification: %r", payload[:200])
        return
    stats.received += 1
    forwarder.submit(note)


async def report_forever(stats: Stats, every_s: float) -> None:
    while True:
        await asyncio.sleep(every_s)
        LOG.info("listener status %s", json.dumps(stats.as_dict()))


async def main_async(args: argparse.Namespace) -> int:
    dsn = args.dsn or os.environ.get("AURA_APPS_SUPABASE_DIRECT_CONNECTION_URL") or os.environ.get(
        "SUPABASE_DIRECT_CONNECTION_URL"
    )
    if not dsn:
        LOG.error(
            "no database URL. Set AURA_APPS_SUPABASE_DIRECT_CONNECTION_URL or pass --dsn. "
            "It must be the direct connection, not the transaction pooler: pgbouncer in "
            "transaction mode does not support LISTEN."
        )
        return 2

    stats = Stats()
    forwarder = InvalidationForwarder(args.aura_url, stats)

    stop = asyncio.Event()

    def _handle_signal() -> None:
        LOG.info("shutting down")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    tasks = [
        asyncio.create_task(listen_forever(dsn, forwarder, stats), name="listen"),
        asyncio.create_task(forwarder.run(), name="forward"),
        asyncio.create_task(report_forever(stats, args.report_every_s), name="report"),
    ]
    await stop.wait()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await forwarder.aclose()
    LOG.info("final status %s", json.dumps(stats.as_dict()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward Postgres change events to AURA")
    parser.add_argument("--dsn", default=None, help="Postgres direct connection URL")
    parser.add_argument(
        "--aura-url",
        default=os.environ.get("AURA_APPS_AURA_BASE_URL", "http://localhost:8080"),
    )
    parser.add_argument("--report-every-s", type=float, default=30.0)
    parser.add_argument("--log-level", default=os.environ.get("AURA_APPS_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
