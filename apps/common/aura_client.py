"""Client SDK for aura-server.

This is the whole application-agnosticism claim in one file. An application says
what it wants (`key`, `object_type`, `ttl_ms`, `sla_class`) and how to rebuild it
(`regen`); the SDK measures what the rebuild actually cost and hands that
measurement to the cache. The cache never learns what a "recommendation" or a
"dashboard query" is - it only ever sees an `ObjectContext`.

Usage::

    async with AuraClient("http://localhost:8080", application="analytics") as client:
        value = await client.get_or_regen(
            key="analytics:revenue:mumbai:30d",
            object_type="dashboard_query",
            ttl_ms=600_000,
            regen=load_revenue,
        )

`regen` may be sync or async, may take zero arguments or a single `CostMeter`,
and may return either a value or a `(value, CostVector)` pair. Anything it
reports is merged into what the SDK measured.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal

import httpx

from .costing import CostMeter, CostVector, ObjectContext
from .settings import Pricing, get_settings

log = logging.getLogger("aura.client")

Encoding = Literal["json", "base64"]
RegenCallable = Callable[..., Any | Awaitable[Any]]

_COST_MEMO_CAPACITY = 20_000


class AuraUnavailable(RuntimeError):
    """Raised internally when the cache cannot be reached; never surfaces."""


@dataclass
class CacheEntry:
    """A cache hit as returned by `GET /v1/cache/{key}`."""

    value: Any
    age_ms: float = 0.0
    layer: str = "unknown"
    latency_us: float = 0.0


@dataclass
class PutResult:
    """The engine's admission decision for a `PUT /v1/cache/{key}`."""

    admitted: bool
    reason_code: str = ""
    evicted: list[str] = field(default_factory=list)
    used_bytes: int = 0


# What a cached "this does not exist" looks like on the wire.
#
# Negative caching is the edge case a value-scored cache is *most* likely to get wrong,
# because a miss that produces nothing looks worthless by every measure the scorer has: no
# bytes, no reuse history, and a rebuild cost that came back empty. Yet a key that does not
# exist and is asked for constantly -- a deleted product still linked from somewhere, a
# probing scanner, a bad client retrying -- is precisely the traffic that reaches the origin
# every single time and costs the most in aggregate.
#
# So an absent value is cached deliberately, as a tiny object with a short lifetime. Short,
# because the cost of being wrong is asymmetric: holding a stale "exists" serves bad data,
# while holding a stale "missing" only serves an unnecessary 404 for a few seconds.
_ABSENT_MARKER = "__aura_absent__"
NEGATIVE_TTL_MS = 30_000


def is_absent(value: Any) -> bool:
    """True when the cache is holding a remembered absence rather than a value."""
    return isinstance(value, dict) and value.get(_ABSENT_MARKER) is True


@dataclass
class RegenOutcome:
    """What a `get_or_regen` call did, for callers that need the detail."""

    value: Any
    hit: bool
    cost: CostVector
    cost_usd: float
    size_bytes: int
    admitted: bool | None
    reason_code: str
    served_from: Literal["cache", "origin"]
    serve_ms: float
    # The object genuinely does not exist, and the cache is remembering that rather than
    # asking the origin again. Callers turn this into their own 404.
    absent: bool = False


def _headers(application: str, api_key: str | None) -> dict[str, str]:
    """Headers every call carries.

    The key is the application's identity as far as the engine is concerned: it attributes
    requests to the application the key was issued to and ignores any name in the body. An
    engine running open accepts calls without one, which is what keeps the local demo a
    single command.
    """
    settings = get_settings()
    key = api_key if api_key is not None else settings.aura_api_key
    headers = {"user-agent": f"aura-app/{application}"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    return headers


class CircuitBreaker:
    """Trips after N consecutive failures, half-opens after a cooldown.

    While open the SDK does not call the cache at all: the application serves
    from its own origin path and keeps working. That is the behaviour the demo
    depends on when aura-server is killed mid-run.
    """

    def __init__(self, failure_threshold: int, reset_s: float) -> None:
        self._threshold = max(1, failure_threshold)
        self._reset_s = reset_s
        self._failures = 0
        self._opened_at = 0.0
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.trips = 0

    def allow(self) -> bool:
        """True when a call to the cache should be attempted."""
        if self.state == "closed":
            return True
        if time.monotonic() - self._opened_at >= self._reset_s:
            self.state = "half_open"
            return True
        return False

    def record_success(self) -> None:
        """Reset after any successful call."""
        if self.state != "closed":
            log.info(
                "aura cache reachable again",
                extra={"event": "breaker_close", "trips": self.trips},
            )
        self._failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Count a failure and open the circuit once the threshold is hit."""
        self._failures += 1
        if self.state == "half_open" or self._failures >= self._threshold:
            if self.state != "open":
                self.trips += 1
            self.state = "open"
            self._opened_at = time.monotonic()


class AuraClient:
    """Async client for the AURA cache data plane."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        application: str,
        default_sla: str = "normal",
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        pricing: Pricing | None = None,
        failure_threshold: int | None = None,
        breaker_reset_s: float | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.aura_base_url).rstrip("/")
        self.application = application
        # Kept so the service can report whether it is authenticated without re-reading the
        # environment, and so an explicitly-passed key is not silently ignored later.
        self.api_key = api_key if api_key is not None else settings.aura_api_key
        self.default_sla = default_sla
        self.pricing = pricing or settings.pricing

        timeout = httpx.Timeout(
            timeout_s if timeout_s is not None else settings.aura_timeout_s,
            connect=connect_timeout_s if connect_timeout_s is not None else settings.aura_connect_timeout_s,
        )
        limits = httpx.Limits(
            max_connections=settings.aura_max_connections,
            max_keepalive_connections=settings.aura_max_connections,
        )
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            transport=transport,
            headers=_headers(application, api_key),
        )
        self._breaker = CircuitBreaker(
            failure_threshold if failure_threshold is not None else settings.breaker_failure_threshold,
            breaker_reset_s if breaker_reset_s is not None else settings.breaker_reset_s,
        )
        self._outage_logged = False
        self._closed = False

        # Last measured regeneration cost per key, so a hit can be credited with
        # the spend it actually avoided rather than an average.
        self._cost_memo: OrderedDict[str, tuple[CostVector, float]] = OrderedDict()

        self._counters: dict[str, float] = {
            "get_calls": 0,
            "hits": 0,
            "misses": 0,
            "put_calls": 0,
            "admitted": 0,
            "rejected": 0,
            "regens": 0,
            "cache_errors": 0,
            "breaker_skips": 0,
            # Origin calls avoided for keys that do not exist. Counted separately from
            # ordinary hits because they are the ones a value-scored cache is most likely
            # to have thrown away.
            "negative_hits": 0,
            "saved_usd": 0.0,
            "spent_usd": 0.0,
            "get_latency_ms_total": 0.0,
        }

    # ---------------------------------------------------------------- plumbing

    async def __aenter__(self) -> AuraClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if not self._closed:
            self._closed = True
            await self._http.aclose()

    @property
    def breaker_state(self) -> str:
        """Current circuit-breaker state."""
        return self._breaker.state

    def _note_cache_failure(self, op: str, exc: Exception) -> None:
        self._counters["cache_errors"] += 1
        self._breaker.record_failure()
        if not self._outage_logged:
            self._outage_logged = True
            log.warning(
                "aura cache unreachable, serving from origin",
                extra={"event": "cache_outage", "op": op, "error": repr(exc), "application": self.application},
            )

    def _note_cache_success(self) -> None:
        self._breaker.record_success()
        self._outage_logged = False

    # -------------------------------------------------------------- data plane

    async def get(self, key: str, *, encoding: Encoding = "json") -> CacheEntry | None:
        """`GET /v1/cache/{key}`. Returns None on miss, outage or open breaker."""
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return None
        self._counters["get_calls"] += 1
        started = time.perf_counter()
        try:
            response = await self._http.get(
                f"/v1/cache/{_quote(key)}",
                params={"application": self.application},
            )
        except Exception as exc:  # network, timeout, DNS - all serve-from-origin
            self._note_cache_failure("get", exc)
            return None
        self._counters["get_latency_ms_total"] += (time.perf_counter() - started) * 1000.0
        self._note_cache_success()

        if response.status_code == 404:
            self._counters["misses"] += 1
            return None
        if response.status_code >= 500:
            self._note_cache_failure("get", AuraUnavailable(f"status {response.status_code}"))
            return None
        if response.status_code != 200:
            self._counters["misses"] += 1
            return None

        try:
            body = response.json()
        except ValueError as exc:
            self._note_cache_failure("get", exc)
            return None
        if not body.get("hit"):
            self._counters["misses"] += 1
            return None

        self._counters["hits"] += 1
        return CacheEntry(
            value=_decode(body.get("value"), encoding),
            age_ms=float(body.get("age_ms") or 0.0),
            layer=str(body.get("layer") or "unknown"),
            latency_us=float(body.get("latency_us") or 0.0),
        )

    async def put(
        self,
        key: str,
        value: Any,
        context: ObjectContext,
        *,
        encoding: Encoding = "json",
    ) -> PutResult:
        """`PUT /v1/cache/{key}` with the measured context.

        A rejection is a normal answer, not an error: the object simply is not
        worth its space right now.
        """
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return PutResult(admitted=False, reason_code="cache_unavailable")
        self._counters["put_calls"] += 1
        payload = {
            "value": _encode(value, encoding),
            "context": context.model_dump(mode="json"),
        }
        try:
            response = await self._http.put(f"/v1/cache/{_quote(key)}", json=payload)
        except Exception as exc:
            self._note_cache_failure("put", exc)
            return PutResult(admitted=False, reason_code="cache_unavailable")
        self._note_cache_success()

        if response.status_code >= 500:
            self._note_cache_failure("put", AuraUnavailable(f"status {response.status_code}"))
            return PutResult(admitted=False, reason_code="cache_error")
        if response.status_code >= 400:
            return PutResult(admitted=False, reason_code=f"http_{response.status_code}")

        try:
            body = response.json()
        except ValueError:
            body = {}
        result = PutResult(
            admitted=bool(body.get("admitted", False)),
            reason_code=str(body.get("reason_code") or ""),
            evicted=list(body.get("evicted") or []),
            used_bytes=int(body.get("used_bytes") or 0),
        )
        self._counters["admitted" if result.admitted else "rejected"] += 1
        return result

    async def delete(self, key: str) -> bool:
        """`DELETE /v1/cache/{key}`."""
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return False
        try:
            response = await self._http.delete(f"/v1/cache/{_quote(key)}")
        except Exception as exc:
            self._note_cache_failure("delete", exc)
            return False
        self._note_cache_success()
        if response.status_code != 200:
            return False
        try:
            return bool(response.json().get("removed", False))
        except ValueError:
            return False

    async def invalidate(
        self, tags: list[str], *, mode: str = "hard", source: str = "application"
    ) -> dict[str, Any] | None:
        """`POST /v1/invalidate` -- drop every object built from these tags.

        The application side of dependency invalidation. `hard` removes immediately, which
        is what a price or a permission needs; `soft` marks stale so the next reader gets
        the old value once while a rebuild runs behind them, which is far cheaper than a
        stampede for a derived rollup.

        Returns `None` rather than raising when the cache cannot be reached. A failed
        invalidation is serious -- it means stale objects survive -- so it is reported to the
        caller instead of being swallowed, but it must not take the application down with
        it: the TTL is still there as a backstop.
        """
        if not tags:
            return {"matched": 0, "keys_hard": 0, "keys_soft": 0}
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return None
        try:
            response = await self._http.post(
                "/v1/invalidate", json={"tags": tags, "mode": mode, "source": source}
            )
        except Exception as exc:
            self._note_cache_failure("invalidate", exc)
            return None
        self._note_cache_success()
        if response.status_code != 200:
            log.warning("invalidate refused: %s %s", response.status_code, response.text[:200])
            return None
        try:
            return dict(response.json())
        except ValueError:
            return None

    async def refresh(self, key: str) -> bool:
        """`POST /v1/cache/{key}/refresh` - ask the engine to re-warm a key."""
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return False
        try:
            response = await self._http.post(f"/v1/cache/{_quote(key)}/refresh")
        except Exception as exc:
            self._note_cache_failure("refresh", exc)
            return False
        self._note_cache_success()
        try:
            return bool(response.json().get("queued", False))
        except ValueError:
            return False

    async def batch_get(self, keys: list[str], *, encoding: Encoding = "json") -> dict[str, CacheEntry]:
        """`POST /v1/cache/batch/get`. Missing keys are simply absent."""
        if not keys or not self._breaker.allow():
            if keys:
                self._counters["breaker_skips"] += 1
            return {}
        try:
            response = await self._http.post("/v1/cache/batch/get", json={"keys": keys})
        except Exception as exc:
            self._note_cache_failure("batch_get", exc)
            return {}
        self._note_cache_success()
        if response.status_code != 200:
            return {}
        try:
            results = response.json().get("results") or {}
        except ValueError:
            return {}

        out: dict[str, CacheEntry] = {}
        for key, item in results.items():
            if not isinstance(item, dict) or not item.get("hit"):
                self._counters["misses"] += 1
                continue
            self._counters["hits"] += 1
            out[key] = CacheEntry(
                value=_decode(item.get("value"), encoding),
                age_ms=float(item.get("age_ms") or 0.0),
                layer=str(item.get("layer") or "unknown"),
                latency_us=float(item.get("latency_us") or 0.0),
            )
        return out

    # ---------------------------------------------------------- explainability

    async def bump_namespace(self, namespace: str) -> dict[str, Any] | None:
        """`POST /v1/version/bump` -- retire a generation without deleting anything.

        What a model redeploy should do. New requests carry the new version and miss
        cleanly; the old generation ages out under ordinary eviction pressure. Flushing
        instead would empty a large part of the cache at once and send the whole miss
        stream at the origin, which is the cache causing the outage it exists to prevent.
        """
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return None
        try:
            response = await self._http.post("/v1/version/bump", json={"namespace": namespace})
        except Exception as exc:
            self._note_cache_failure("version_bump", exc)
            return None
        self._note_cache_success()
        if response.status_code != 200:
            return None
        try:
            return dict(response.json())
        except ValueError:
            return None

    async def identity(self) -> dict[str, Any]:
        """What this process knows about its own connection to the cache.

        The whole integration is a URL and a key, so this is the whole of what there is to
        check -- and checking it is the difference between "the cache is doing nothing" and
        "we never authenticated".
        """
        reachable = False
        detail = ""
        try:
            response = await self._http.get("/healthz")
            reachable = response.status_code == 200
            if not reachable:
                detail = f"engine answered {response.status_code}"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"[:160]
        key = self.api_key
        return {
            "engine": self.base_url,
            "application": self.application,
            # Never the key itself. A demo page that prints its own credential is a demo
            # that leaks one the first time somebody screenshots it.
            "key_present": bool(key),
            "key_hint": (key[:11] + "..." if key else ""),
            "reachable": reachable,
            "detail": detail,
            "breaker": self.breaker_state,
        }

    async def explain(self, key: str) -> dict[str, Any] | None:
        """`GET /v1/explain/{key}` - why this object was kept, evicted or rejected."""
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return None
        try:
            response = await self._http.get(f"/v1/explain/{_quote(key)}")
        except Exception as exc:
            self._note_cache_failure("explain", exc)
            return None
        self._note_cache_success()
        if response.status_code != 200:
            return None
        try:
            return dict(response.json())
        except ValueError:
            return None

    async def explain_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """`GET /v1/explain/recent` - the engine's latest decisions."""
        if not self._breaker.allow():
            self._counters["breaker_skips"] += 1
            return []
        try:
            response = await self._http.get("/v1/explain/recent", params={"limit": limit})
        except Exception as exc:
            self._note_cache_failure("explain_recent", exc)
            return []
        self._note_cache_success()
        if response.status_code != 200:
            return []
        try:
            return list(response.json().get("decisions") or [])
        except ValueError:
            return []

    # ------------------------------------------------------------- the SDK core

    async def get_or_regen(
        self,
        key: str,
        *,
        object_type: str,
        ttl_ms: int,
        regen: RegenCallable,
        sla_class: str | None = None,
        encoding: Encoding = "json",
        force_fresh: bool = False,
        cost_hint: CostVector | None = None,
        depends_on: list[str] | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Serve `key` from the cache, or regenerate it and report the real cost."""
        outcome = await self.get_or_regen_detailed(
            key,
            object_type=object_type,
            ttl_ms=ttl_ms,
            regen=regen,
            sla_class=sla_class,
            encoding=encoding,
            force_fresh=force_fresh,
            cost_hint=cost_hint,
            depends_on=depends_on,
            namespace=namespace,
        )
        return outcome.value

    async def get_or_regen_detailed(
        self,
        key: str,
        *,
        object_type: str,
        ttl_ms: int,
        regen: RegenCallable,
        sla_class: str | None = None,
        encoding: Encoding = "json",
        force_fresh: bool = False,
        cost_hint: CostVector | None = None,
        depends_on: list[str] | None = None,
        namespace: str | None = None,
    ) -> RegenOutcome:
        """`get_or_regen`, but returning the full accounting for the call.

        `depends_on` is what makes a write in the database able to reach this object. The
        tags travel with the admission, so a later `POST /v1/invalidate` naming any of them
        removes exactly the objects built from that row -- and nothing else.
        """
        started = time.perf_counter()

        if not force_fresh:
            entry = await self.get(key, encoding=encoding)
            if entry is not None:
                saved, size_bytes = self._recall_cost(key)
                self._counters["saved_usd"] += saved
                if is_absent(entry.value):
                    # A remembered absence still saved an origin call, which is the whole
                    # point of storing it, so it counts as a hit and reports the cost it
                    # avoided. What it must not do is hand the caller the marker.
                    self._counters["negative_hits"] += 1
                    return RegenOutcome(
                        value=None,
                        hit=True,
                        cost=CostVector(),
                        cost_usd=saved,
                        size_bytes=size_bytes,
                        admitted=None,
                        reason_code="negative_hit",
                        served_from="cache",
                        serve_ms=(time.perf_counter() - started) * 1000.0,
                        absent=True,
                    )
                return RegenOutcome(
                    value=entry.value,
                    hit=True,
                    cost=CostVector(),
                    cost_usd=saved,
                    size_bytes=size_bytes,
                    admitted=None,
                    reason_code="hit",
                    served_from="cache",
                    serve_ms=(time.perf_counter() - started) * 1000.0,
                )

        meter = CostMeter()
        meter.__enter__()
        try:
            produced = await _call_regen(regen, meter)
        finally:
            meter.stop()

        value, reported = _split_regen_result(produced)
        payload = _encode(value, encoding)
        # For a base64 object the meaningful size is the object itself, not the
        # 4/3 inflation the JSON transport adds on the way to the server.
        if encoding == "base64" and isinstance(value, (bytes, bytearray)):
            size_bytes = len(value)
        else:
            size_bytes = _payload_bytes(payload)
        cost = meter.finish(size_bytes=size_bytes)
        if reported is not None:
            cost = _overlay(cost, reported)
        if cost_hint is not None:
            cost = cost.merged(cost_hint)
        cost.network_bytes = max(cost.network_bytes, size_bytes)

        cost_usd = cost.usd(self.pricing)
        if value is None:
            # Charged for what it actually occupies. Reporting the size of the thing that
            # was not there would make the cache price a few bytes as though they were a
            # megabyte, and refuse to keep them.
            size_bytes = 64
        self._counters["regens"] += 1
        self._counters["spent_usd"] += cost_usd
        self._remember_cost(key, cost, cost_usd, size_bytes)

        # The origin says this does not exist. Remember that rather than asking again on
        # every request: a key that is absent and popular is the traffic that reaches the
        # origin one hundred percent of the time.
        absent = value is None
        stored: Any = {_ABSENT_MARKER: True} if absent else value
        effective_ttl = min(ttl_ms, NEGATIVE_TTL_MS) if absent else ttl_ms

        context = ObjectContext(
            application=self.application,
            object_type=object_type,
            size_bytes=size_bytes,
            ttl_ms=effective_ttl,
            sla_class=sla_class or self.default_sla,
            regen=cost,
            depends_on=list(depends_on or ()),
            namespace=namespace,
        )
        result = await self.put(key, stored, context, encoding=encoding)

        return RegenOutcome(
            value=value,
            hit=False,
            absent=absent,
            cost=cost,
            cost_usd=cost_usd,
            size_bytes=size_bytes,
            admitted=result.admitted,
            reason_code=result.reason_code,
            served_from="origin",
            serve_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------- memo

    def _remember_cost(self, key: str, cost: CostVector, cost_usd: float, size_bytes: int) -> None:
        cost.network_bytes = max(cost.network_bytes, size_bytes)
        self._cost_memo[key] = (cost, cost_usd)
        self._cost_memo.move_to_end(key)
        while len(self._cost_memo) > _COST_MEMO_CAPACITY:
            self._cost_memo.popitem(last=False)

    def _recall_cost(self, key: str) -> tuple[float, int]:
        entry = self._cost_memo.get(key)
        if entry is None:
            return 0.0, 0
        cost, usd = entry
        self._cost_memo.move_to_end(key)
        return usd, cost.network_bytes

    def last_cost(self, key: str) -> CostVector | None:
        """The most recent measured regeneration cost for `key`, if known."""
        entry = self._cost_memo.get(key)
        return entry[0] if entry else None

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        """Snapshot of what this client has done."""
        counters = dict(self._counters)
        lookups = counters["hits"] + counters["misses"]
        get_calls = max(1.0, counters["get_calls"])
        return {
            "application": self.application,
            "base_url": self.base_url,
            "hits": int(counters["hits"]),
            "misses": int(counters["misses"]),
            "hit_rate": round(counters["hits"] / lookups, 4) if lookups else 0.0,
            "regens": int(counters["regens"]),
            "put_calls": int(counters["put_calls"]),
            "admitted": int(counters["admitted"]),
            "rejected": int(counters["rejected"]),
            "cache_errors": int(counters["cache_errors"]),
            "breaker_skips": int(counters["breaker_skips"]),
            "breaker_state": self._breaker.state,
            "breaker_trips": self._breaker.trips,
            "avg_get_latency_ms": round(counters["get_latency_ms_total"] / get_calls, 4),
            "spent_usd": round(counters["spent_usd"], 8),
            "saved_usd": round(counters["saved_usd"], 8),
            "tracked_keys": len(self._cost_memo),
        }


class SyncAuraClient:
    """Blocking facade over :class:`AuraClient` for scripts and notebooks.

    Runs its own event loop on a background thread, so it is safe to use from
    ordinary synchronous code but must not be used from inside a running loop.
    """

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="aura-sync", daemon=True)
        self._thread.start()
        self._client = AuraClient(base_url, **kwargs)

    def _run(self, coro: Awaitable[Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def get(self, key: str, *, encoding: Encoding = "json") -> CacheEntry | None:
        """Blocking `GET /v1/cache/{key}`."""
        return self._run(self._client.get(key, encoding=encoding))

    def put(self, key: str, value: Any, context: ObjectContext, *, encoding: Encoding = "json") -> PutResult:
        """Blocking `PUT /v1/cache/{key}`."""
        return self._run(self._client.put(key, value, context, encoding=encoding))

    def get_or_regen(self, key: str, **kwargs: Any) -> Any:
        """Blocking `get_or_regen`."""
        return self._run(self._client.get_or_regen(key, **kwargs))

    def explain(self, key: str) -> dict[str, Any] | None:
        """Blocking `explain`."""
        return self._run(self._client.explain(key))

    def stats(self) -> dict[str, Any]:
        """Client statistics snapshot."""
        return self._client.stats()

    def close(self) -> None:
        """Shut down the client and its background loop."""
        self._run(self._client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def __enter__(self) -> SyncAuraClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ------------------------------------------------------------------ internals


def _quote(key: str) -> str:
    from urllib.parse import quote

    return quote(key, safe="")


def _encode(value: Any, encoding: Encoding) -> Any:
    if encoding == "base64":
        raw = value if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8")
        return base64.b64encode(bytes(raw)).decode("ascii")
    return value


def _decode(value: Any, encoding: Encoding) -> Any:
    if encoding == "base64" and isinstance(value, str):
        return base64.b64decode(value.encode("ascii"))
    return value


def _payload_bytes(payload: Any) -> int:
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return len(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))


async def _call_regen(regen: RegenCallable, meter: CostMeter) -> Any:
    takes_meter = _accepts_meter(regen)
    result = regen(meter) if takes_meter else regen()
    if inspect.isawaitable(result):
        result = await result
    return result


def _accepts_meter(regen: RegenCallable) -> bool:
    try:
        signature = inspect.signature(regen)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            return True
        if parameter.kind is parameter.VAR_POSITIONAL:
            return True
    return False


def _overlay(measured: CostVector, reported: CostVector) -> CostVector:
    """Reported non-zero dimensions win over the SDK's own measurement.

    A regeneration that knows its database or accelerator time exactly should be
    believed; anything it leaves at zero keeps the measured value. Sizes and
    latency always take the larger of the two.
    """
    return CostVector(
        cpu_ms=reported.cpu_ms or measured.cpu_ms,
        gpu_ms=reported.gpu_ms or measured.gpu_ms,
        db_ms=reported.db_ms or measured.db_ms,
        network_bytes=max(reported.network_bytes, measured.network_bytes),
        api_calls=max(reported.api_calls, measured.api_calls),
        api_cost_usd=reported.api_cost_usd or measured.api_cost_usd,
        latency_ms=max(reported.latency_ms, measured.latency_ms),
    )


def _split_regen_result(produced: Any) -> tuple[Any, CostVector | None]:
    if isinstance(produced, tuple) and len(produced) == 2 and isinstance(produced[1], CostVector):
        return produced[0], produced[1]
    return produced, None
