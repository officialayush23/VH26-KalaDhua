"""Content service - port 8103.

Economics of this application:

* objects are large (100 KB to 20 MB) and cheap to produce in CPU terms;
* a miss costs bandwidth, not compute, so the value of a cached object scales
  with its size rather than inversely to it;
* interest is flash-shaped and short-lived: an object is hot for minutes and
  then worthless, so long TTLs and long memories are wasted here;
* one object family is bought from a priced third party. Its regeneration
  carries a real USD charge, and that charge can change at runtime through
  `POST /price` - the cost-spike scenario.

Between them, analytics and content are the reason a single scalar "value" per
object does not work: the cheapest object to keep here is the most expensive one
to keep there.
"""

from __future__ import annotations

import hashlib
from typing import Any

import uvicorn
from pydantic import BaseModel, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from common.costing import CostMeter, CostVector
from common.service import AppService, build_app, configure_logging
from common.settings import get_settings
from content import objects
from content.external_api import ExternalApiClient

APPLICATION = "content"
PORT = 8103


class PriceRequest(BaseModel):
    """Body of `POST /price`."""

    price_usd: float = Field(ge=0.0, le=10.0)
    median_latency_ms: float | None = Field(default=None, ge=0.0, le=5_000.0)
    sigma: float | None = Field(default=None, ge=0.0, le=3.0)
    reason: str = "manual"


class ContentService(AppService):
    """Object delivery wired to AURA."""

    def __init__(self) -> None:
        super().__init__(
            application=APPLICATION,
            default_sla="normal",
            work=self.produce,
            extra_metrics=self._extra,
        )
        self.api = ExternalApiClient(seed=self.settings.seed)

    def _extra(self) -> dict[str, float]:
        return {
            "aura_app_external_api_price_usd": self.api.price_usd,
            "aura_app_external_api_calls": float(self.api.calls),
            "aura_app_external_api_spend_usd": round(self.api.spend_usd, 8),
        }

    def cache_key(self, key_id: int | str) -> str:
        """Content keys carry the object family, so the cache can see the mix."""
        numeric = _numeric(key_id)
        kind = objects.kind_for(numeric)
        return f"{APPLICATION}:{kind.name}:{numeric}"

    async def produce(self, key_id: int | str, fresh: bool, options: dict[str, str] | None = None) -> dict[str, Any]:
        """Serve one content object, through the cache."""
        numeric = _numeric(key_id)
        kind = objects.kind_for(numeric)
        size_bytes = objects.size_for(numeric, kind)
        # The expensive tail buys larger, pricier documents from the provider
        # while keeping the object size and request rate unchanged.
        factor = self.tail.factor(key_id)
        key = self.cache_key(key_id)

        async def regen(meter: CostMeter) -> tuple[bytes, CostVector]:
            if kind.priced:
                calls = max(1, int(round(factor)))
                payload = b""
                total_latency = 0.0
                for _ in range(calls):
                    response = await self.api.fetch(numeric, size_bytes)
                    payload = response.body
                    total_latency += response.latency_ms
                    meter.add_api_call(response.price_usd)
                return payload, CostVector(latency_ms=total_latency)
            with meter.section() as section:
                payload = objects.generate(numeric, size_bytes)
            return payload, CostVector(cpu_ms=section.cpu_ms)

        outcome = await self.client.get_or_regen_detailed(
            key,
            object_type=kind.name,
            ttl_ms=kind.ttl_ms,
            regen=regen,
            sla_class=kind.sla_class,
            encoding="base64",
            force_fresh=fresh,
        )
        self.account(object_type=kind.name, outcome=outcome, key_id=key_id)

        payload = outcome.value if isinstance(outcome.value, (bytes, bytearray)) else b""
        return {
            "key": key,
            "application": APPLICATION,
            "object_type": kind.name,
            "served_from": outcome.served_from,
            "expensive_tail": self.tail.contains(key_id),
            "size_bytes": len(payload),
            "wire_bytes": outcome.size_bytes,
            "digest": objects.digest(payload) if payload else None,
            "serve_ms": round(outcome.serve_ms, 3),
            "regen": outcome.cost.model_dump(),
            "regen_cost_usd": round(outcome.cost_usd, 8),
            "priced_object": kind.priced,
            "api_price_usd": self.api.price_usd if kind.priced else 0.0,
            "admitted": outcome.admitted,
            "reason_code": outcome.reason_code,
            "ttl_ms": kind.ttl_ms,
        }


def _numeric(key_id: int | str) -> int:
    try:
        return abs(int(key_id))
    except (TypeError, ValueError):
        digest = hashlib.blake2b(str(key_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (1 << 31)


def create_app():  # noqa: ANN201 - Starlette application factory
    """Build the ASGI application."""
    configure_logging(get_settings().log_level)
    service = ContentService()

    async def price_get(request: Request) -> JSONResponse:
        """Current provider economics."""
        return JSONResponse(service.api.stats())

    async def price_post(request: Request) -> JSONResponse:
        """Change the third-party price at runtime - the cost-spike lever."""
        raw = await request.body()
        try:
            payload = PriceRequest.model_validate_json(raw or b"{}")
        except ValidationError as exc:
            return JSONResponse({"error": "invalid_body", "detail": exc.errors()}, status_code=422)
        change = service.api.set_price(payload.price_usd, reason=payload.reason)
        if payload.median_latency_ms is not None or payload.sigma is not None:
            service.api.set_latency(
                payload.median_latency_ms if payload.median_latency_ms is not None else service.api.median_latency_ms,
                payload.sigma,
            )
        service.log.info("external api price changed", extra={"event": "price_change", **change})
        return JSONResponse({"changed": change, "state": service.api.stats()})

    async def profile(request: Request) -> JSONResponse:
        """Cost profile and the object catalogue."""
        return JSONResponse(
            {
                "application": APPLICATION,
                "cost_profile": "bandwidth_heavy",
                "traffic_shape": "flash_popularity",
                "object_bytes": [objects.KINDS[0].min_bytes, max(k.max_bytes for k in objects.KINDS)],
                "regen_ms_range": [1, 1500],
                "kinds": [
                    {
                        "name": kind.name,
                        "min_bytes": kind.min_bytes,
                        "max_bytes": kind.max_bytes,
                        "ttl_ms": kind.ttl_ms,
                        "sla_class": kind.sla_class,
                        "priced": kind.priced,
                    }
                    for kind in objects.KINDS
                ],
                "external_api": service.api.stats(),
            }
        )

    routes = [
        Route("/price", price_get, methods=["GET"]),
        Route("/price", price_post, methods=["POST"]),
        Route("/profile", profile, methods=["GET"]),
    ]
    return build_app(service, routes)


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None, timeout_graceful_shutdown=10)
