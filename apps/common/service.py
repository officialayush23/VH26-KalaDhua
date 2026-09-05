"""Shared HTTP scaffolding for the three example applications.

Every app is the same shape: measure a regeneration, hand the measurement to
AURA, expose `/health`, `/work/{id}`, `/stats`, `/load`, `/metrics` and an
expensive-tail switch. Only the *economics* differ, and those live in each app's
own modules. This file keeps the plumbing in one place so the differences stay
visible.

The apps are built directly on Starlette, which is the ASGI layer FastAPI itself
is built on; request bodies are validated with pydantic v2 models.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .aura_client import AuraClient
from .loadgen import LoadSpec, drive
from .settings import Settings, get_settings
from .telemetry import AppTelemetry

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


# --------------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """Structured log lines. One JSON object per record, no multi-line output."""

    _SKIP = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a single-line JSON object."""
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, echoes it, and emits one access log line."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Wrap one request."""
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logging.getLogger("aura.http").exception(
                "request failed",
                extra={"event": "request_error", "path": request.url.path, "method": request.method},
            )
            response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = rid
        response.headers["cache-control"] = "no-store"
        logging.getLogger("aura.http").info(
            "request",
            extra={
                "event": "access",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 3),
            },
        )
        request_id_var.reset(token)
        return response


# ---------------------------------------------------------------- request models


class LoadRequest(BaseModel):
    """Body of `POST /load` (contract section 7)."""

    rps: float = 200.0
    duration_s: float = 60.0
    pattern: str = "zipf"
    key_space: int = 5_000
    alpha: float = 1.05
    concurrency: int = 32
    seed: int | None = None
    fresh: bool = False


class TailRequest(BaseModel):
    """Body of `POST /expensive-tail`."""

    enabled: bool = True
    fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    multiplier: float = Field(default=30.0, ge=1.0, le=200.0)


# ------------------------------------------------------------------- the service


class ExpensiveTail:
    """The workload that breaks frequency-only policies.

    A stable ~`fraction` slice of the key space costs `multiplier` times more to
    regenerate than everything else, while being requested exactly as often. A
    policy that ranks by frequency alone cannot see the difference; a policy
    that ranks by cost-per-byte can.
    """

    def __init__(self, fraction: float, multiplier: float) -> None:
        self.enabled = False
        self.fraction = fraction
        self.multiplier = multiplier

    def contains(self, key_id: int | str) -> bool:
        """Membership is a stable hash, so it never correlates with frequency."""
        if not self.enabled or self.fraction <= 0.0:
            return False
        digest = hashlib.blake2b(str(key_id).encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") / float(1 << 64)
        return bucket < self.fraction

    def factor(self, key_id: int | str) -> float:
        """Cost multiplier for this key."""
        return self.multiplier if self.contains(key_id) else 1.0

    def state(self) -> dict[str, Any]:
        """Current configuration."""
        return {"enabled": self.enabled, "fraction": self.fraction, "multiplier": self.multiplier}


WorkFn = Callable[[int | str, bool, dict[str, str]], Awaitable[dict[str, Any]]]


class AppService:
    """Everything an example application shares."""

    def __init__(
        self,
        *,
        application: str,
        default_sla: str,
        work: WorkFn,
        settings: Settings | None = None,
        extra_metrics: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.application = application
        self.telemetry = AppTelemetry(application)
        self.client = AuraClient(
            self.settings.aura_base_url,
            application=application,
            default_sla=default_sla,
        )
        self.tail = ExpensiveTail(self.settings.tail_fraction, self.settings.tail_multiplier)
        self.log = logging.getLogger(f"aura.app.{application}")
        self._work = work
        self._extra_metrics = extra_metrics
        self._load_task: asyncio.Task[Any] | None = None
        self._load_report: dict[str, Any] | None = None
        self._shutdown = asyncio.Event()

    # ------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        """Cancel background work and release the connection pool."""
        self._shutdown.set()
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
            try:
                await self._load_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.client.aclose()

    # ---------------------------------------------------------------- routes

    def standard_routes(self) -> list[Route]:
        """The endpoints every example app must expose."""
        return [
            Route("/health", self.health, methods=["GET"]),
            Route("/connection", self.connection_endpoint, methods=["GET"]),
            Route("/retire", self.retire_endpoint, methods=["POST"]),
            Route("/work/{key_id}", self.work_endpoint, methods=["GET"]),
            Route("/stats", self.stats_endpoint, methods=["GET"]),
            Route("/metrics", self.metrics_endpoint, methods=["GET"]),
            Route("/load", self.load_endpoint, methods=["POST"]),
            Route("/expensive-tail", self.tail_get, methods=["GET"]),
            Route("/expensive-tail", self.tail_post, methods=["POST"]),
            Route("/explain/{key_id}", self.explain_endpoint, methods=["GET"]),
        ]

    async def health(self, request: Request) -> JSONResponse:
        """Liveness plus the cache's reachability as this app sees it."""
        return JSONResponse(
            {
                "ok": True,
                "application": self.application,
                "uptime_s": round(time.time() - self.telemetry.started_at, 2),
                "aura_base_url": self.client.base_url,
                "cache_breaker": self.client.breaker_state,
                "expensive_tail": self.tail.state(),
            }
        )

    async def work_endpoint(self, request: Request) -> JSONResponse:
        """`GET /work/{id}?fresh=` - the expensive endpoint, via AURA."""
        key_id = request.path_params["key_id"]
        fresh = _truthy(request.query_params.get("fresh"))
        options = dict(request.query_params)
        try:
            body = await self._work(key_id, fresh, options)
        except Exception as exc:
            self.telemetry.record_error()
            self.log.exception("work failed", extra={"event": "work_error", "key_id": key_id})
            return JSONResponse({"error": "regeneration_failed", "detail": str(exc)}, status_code=500)
        return JSONResponse(body)

    async def connection_endpoint(self, request: Request) -> JSONResponse:
        """`GET /connection` -- am I actually plugged into the cache?

        The entire integration is a base URL and a key, which means the entire set of things
        that can be wrong with it is: wrong URL, missing key, rejected key, engine down.
        This answers all four in one call, so "the cache is not helping" can be separated
        from "we never reached the cache" without reading two sets of logs.
        """
        _ = request
        identity = await self.client.identity()
        identity["objects_admitted"] = int(self.client.stats().get("admitted", 0))
        identity["objects_refused"] = int(self.client.stats().get("rejected", 0))
        return JSONResponse(identity, status_code=200 if identity["reachable"] else 503)

    async def retire_endpoint(self, request: Request) -> JSONResponse:
        """`POST /retire` -- redeploy the model, the way it should be done.

        Bumps this application's namespace. Nothing is deleted: requests after the bump
        carry the new version and miss cleanly, while the previous generation ages out under
        ordinary pressure. The alternative -- flushing -- turns a routine deploy into a
        thundering herd against the origin.
        """
        _ = request
        result = await self.client.bump_namespace(self.application)
        if result is None:
            return JSONResponse({"error": "the cache did not answer"}, status_code=502)
        return JSONResponse(result)

    async def stats_endpoint(self, request: Request) -> JSONResponse:
        """`GET /stats` - contract section 7 payload plus SDK counters."""
        payload = self.telemetry.stats()
        payload["client"] = self.client.stats()
        payload["expensive_tail"] = self.tail.state()
        payload["load_job"] = self.load_state()
        return JSONResponse(payload)

    async def metrics_endpoint(self, request: Request) -> PlainTextResponse:
        """`GET /metrics` - Prometheus text exposition."""
        extra = self._extra_metrics() if self._extra_metrics else {}
        extra.setdefault(
            "aura_app_cache_breaker_open",
            1.0 if self.client.breaker_state == "open" else 0.0,
        )
        body = self.telemetry.prometheus(extra)
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    async def load_endpoint(self, request: Request) -> JSONResponse:
        """`POST /load` - drive synthetic traffic through this app's own path."""
        try:
            payload = LoadRequest.model_validate(await _json_body(request))
        except ValidationError as exc:
            return JSONResponse({"error": "invalid_body", "detail": exc.errors()}, status_code=422)

        if self._load_task and not self._load_task.done():
            return JSONResponse({"error": "load_already_running", "job": self.load_state()}, status_code=409)

        spec = LoadSpec(
            rps=payload.rps,
            duration_s=payload.duration_s,
            pattern=payload.pattern,
            key_space=payload.key_space,
            alpha=payload.alpha,
            concurrency=payload.concurrency,
            seed=payload.seed if payload.seed is not None else self.settings.seed,
        )
        try:
            spec.validate()
        except ValueError as exc:
            return JSONResponse({"error": "invalid_spec", "detail": str(exc)}, status_code=422)

        async def handler(key_id: int) -> None:
            await self._work(key_id, payload.fresh, {})

        self._load_report = None
        self._load_task = asyncio.create_task(self._run_load(spec, handler))
        return JSONResponse(
            {
                "started": True,
                "application": self.application,
                "spec": {
                    "rps": spec.rps,
                    "duration_s": spec.duration_s,
                    "pattern": spec.pattern,
                    "key_space": spec.key_space,
                },
            },
            status_code=202,
        )

    async def _run_load(self, spec: LoadSpec, handler: Callable[[int], Awaitable[None]]) -> None:
        self.log.info("load job started", extra={"event": "load_start", "pattern": spec.pattern, "rps": spec.rps})
        try:
            report = await drive(spec, handler)
            self._load_report = report.as_dict()
            self.log.info("load job finished", extra={"event": "load_done", **self._load_report})
        except asyncio.CancelledError:
            self._load_report = {"pattern": spec.pattern, "cancelled": True}
            raise

    def load_state(self) -> dict[str, Any]:
        """Status of the current or last `/load` job."""
        running = bool(self._load_task and not self._load_task.done())
        return {"running": running, "last_report": self._load_report}

    async def tail_get(self, request: Request) -> JSONResponse:
        """`GET /expensive-tail`."""
        return JSONResponse(self.tail.state())

    async def tail_post(self, request: Request) -> JSONResponse:
        """`POST /expensive-tail` - switch the pathological workload on or off."""
        try:
            payload = TailRequest.model_validate(await _json_body(request))
        except ValidationError as exc:
            return JSONResponse({"error": "invalid_body", "detail": exc.errors()}, status_code=422)
        self.tail.enabled = payload.enabled
        self.tail.fraction = payload.fraction
        self.tail.multiplier = payload.multiplier
        self.log.info("expensive tail updated", extra={"event": "tail_update", **self.tail.state()})
        return JSONResponse(self.tail.state())

    async def explain_endpoint(self, request: Request) -> JSONResponse:
        """`GET /explain/{id}` - ask AURA why this app's object was kept or evicted."""
        key_id = request.path_params["key_id"]
        key = key_id if str(key_id).startswith(f"{self.application}:") else self.cache_key(key_id)
        explanation = await self.client.explain(key)
        if explanation is None:
            return JSONResponse(
                {"key": key, "available": False, "reason": "cache_unreachable_or_unknown_key"},
                status_code=404,
            )
        return JSONResponse(explanation)

    def cache_key(self, key_id: int | str) -> str:
        """Namespace a key id. Overridden by apps with richer key shapes."""
        return f"{self.application}:{key_id}"

    # ------------------------------------------------------------ accounting

    def typical_regen_ms(self, object_type: str | None = None) -> float:
        """Median observed rebuild time, in milliseconds.

        Used to say what a cache hit saved in *waiting*, not just in money. It is the median
        of what this process has actually measured rather than a configured estimate, so on
        a cold start it is zero and the page shows nothing rather than a number nobody
        earned.
        """
        _ = object_type  # one distribution per service today; the argument keeps the door open
        return float(self.telemetry.regen_ms.quantile(0.50))

    def account(
        self,
        *,
        object_type: str,
        outcome: Any,
        key_id: int | str,
    ) -> None:
        """Fold one `RegenOutcome` into this application's telemetry."""
        tail = self.tail.contains(key_id)
        self.telemetry.record_request(object_type, outcome.serve_ms, outcome.size_bytes, tail)
        if outcome.hit:
            self.telemetry.record_hit(outcome.cost_usd)
            return
        if outcome.reason_code == "cache_unavailable":
            # An unreachable cache is not a miss: counting it as one would make
            # the hit rate look bad for a reason that has nothing to do with the
            # policy under study.
            self.telemetry.record_cache_unavailable()
        else:
            self.telemetry.record_miss()
        penalty = outcome.cost.sla_penalty_usd(self.settings.pricing)
        self.telemetry.record_regen(
            outcome.cost.latency_ms,
            outcome.cost_usd,
            penalty,
            outcome.cost.api_cost_usd,
        )
        if outcome.admitted is not None and outcome.reason_code != "cache_unavailable":
            self.telemetry.record_admission(outcome.admitted)


def build_app(service: AppService, routes: list[Route]) -> Starlette:
    """Assemble the Starlette application with logging, ids and clean shutdown.

    Application-specific routes are matched first, so an app that needs a richer
    `/health` than the generic one simply provides its own.
    """

    async def report_tier1() -> None:
        """Publish this process's L1 counters to the engine every few seconds.

        Five seconds, not five hundred milliseconds: these are counters for a chart, not a
        control loop, and a fleet of application processes each posting at request rate
        would be a self-inflicted load test of the thing being measured.
        """
        while True:
            try:
                await asyncio.sleep(5.0)
                await service.client.report_l1()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Reporting is decoration. Never let it end the loop, and never let it
                # surface as an application error.
                continue

    async def lifespan(app: Starlette):  # noqa: ANN202 - Starlette's own protocol
        service.log.info(
            "application started",
            extra={"event": "startup", "application": service.application, "aura": service.client.base_url},
        )
        reporter = asyncio.create_task(report_tier1())
        try:
            yield
        finally:
            reporter.cancel()
            await service.aclose()
            service.log.info("application stopped", extra={"event": "shutdown"})

    return Starlette(
        routes=list(routes) + service.standard_routes(),
        middleware=[Middleware(RequestContextMiddleware)],
        lifespan=asynccontextmanager(lifespan),
    )


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}
