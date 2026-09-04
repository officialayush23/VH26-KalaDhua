"""Recommendation service - port 8101.

Economics of this application:

* objects are large (0.5-4 MB personalised ranking payloads);
* regeneration is CPU and accelerator heavy, 80-2000 ms, with high variance
  because it depends on the length of the user's history;
* the key space is personalised, so it has a very long tail, and popularity
  shifts quickly as segments trend;
* the value of a cached object decays fast - a 5 minute TTL.

That combination is what makes a size-blind, cost-blind policy expensive here:
a single admitted object can cost as much space as a hundred analytics results,
and evicting the wrong one costs a second of CPU to rebuild.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from common.costing import CostMeter, CostVector
from common.service import AppService, build_app, configure_logging
from common.settings import get_settings
from recommendation import data, model

APPLICATION = "recommendation"
OBJECT_TYPE = "ranking_result"
TTL_MS = 300_000
PORT = 8101

SEGMENTS = ("default", "mobile", "web", "loyalty")


class RecommendationService(AppService):
    """Ranking regeneration wired to AURA."""

    def __init__(self) -> None:
        super().__init__(
            application=APPLICATION,
            default_sla="high",
            work=self.produce,
            extra_metrics=self._extra,
        )
        self.catalogue = data.catalogue()

    def _extra(self) -> dict[str, float]:
        return {
            "aura_app_catalogue_items": float(self.catalogue.n_items),
            "aura_app_catalogue_users": float(self.catalogue.n_users),
        }

    def cache_key(self, key_id: int | str) -> str:
        """Personalised keys: one object per user and segment."""
        user_id = _user_id(key_id)
        segment = SEGMENTS[user_id % len(SEGMENTS)]
        return f"{APPLICATION}:user:{user_id}:{segment}"

    async def produce(self, key_id: int | str, fresh: bool, options: dict[str, str] | None = None) -> dict[str, Any]:
        """Serve one ranking, through the cache."""
        user_id = _user_id(key_id)
        key = self.cache_key(key_id)
        work_factor = self.tail.factor(key_id)
        target_bytes = model.target_bytes_for(user_id)
        top_k, explain_dim = model.payload_dimensions(user_id, target_bytes)

        async def regen(meter: CostMeter) -> tuple[dict[str, Any], CostVector]:
            ranking = await asyncio.to_thread(
                model.recommend,
                user_id,
                top_k=top_k,
                work_factor=work_factor,
                explain_dim=explain_dim,
                meter=meter,
                cat=self.catalogue,
            )
            body = ranking.as_dict()
            body["segment"] = SEGMENTS[user_id % len(SEGMENTS)]
            return body, CostVector(gpu_ms=ranking.embedding_ms, db_ms=ranking.index_ms)

        outcome = await self.client.get_or_regen_detailed(
            key,
            object_type=OBJECT_TYPE,
            ttl_ms=TTL_MS,
            regen=regen,
            sla_class="high",
            force_fresh=fresh,
        )
        self.account(object_type=OBJECT_TYPE, outcome=outcome, key_id=key_id)

        body: dict[str, Any] = {
            "key": key,
            "application": APPLICATION,
            "object_type": OBJECT_TYPE,
            "served_from": outcome.served_from,
            "expensive_tail": self.tail.contains(key_id),
            "size_bytes": outcome.size_bytes,
            "serve_ms": round(outcome.serve_ms, 3),
            "regen": outcome.cost.model_dump(),
            "regen_cost_usd": round(outcome.cost_usd, 8),
            "admitted": outcome.admitted,
            "reason_code": outcome.reason_code,
        }
        value = outcome.value if isinstance(outcome.value, dict) else {}
        # Multi-megabyte rankings are only echoed on request; the load path does
        # not need to pay for serialising them twice.
        if _wants_value(options):
            body["value"] = value
        else:
            body["preview"] = {
                "segment": value.get("segment"),
                "candidates_considered": value.get("candidates_considered"),
                "ensemble_passes": value.get("ensemble_passes"),
                "stage_ms": value.get("stage_ms"),
                "top_items": [
                    {k: item[k] for k in ("rank", "item_id", "score", "category") if k in item}
                    for item in (value.get("items") or [])[:5]
                ],
                "items_total": len(value.get("items") or []),
            }
        return body


def _wants_value(options: dict[str, str] | None) -> bool:
    return str((options or {}).get("value", "")).lower() in {"1", "true", "yes", "on"}


def _user_id(key_id: int | str) -> int:
    try:
        return abs(int(key_id)) % data.N_USERS
    except (TypeError, ValueError):
        digest = hashlib.blake2b(str(key_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % data.N_USERS


def create_app():  # noqa: ANN201 - Starlette application factory
    """Build the ASGI application."""
    configure_logging(get_settings().log_level)
    service = RecommendationService()

    async def profile(request: Request) -> JSONResponse:
        """The cost profile this app advertises, for the dashboard and the README."""
        return JSONResponse(
            {
                "application": APPLICATION,
                "object_type": OBJECT_TYPE,
                "cost_profile": "cpu_gpu_heavy",
                "traffic_shape": "personalised_long_tail",
                "object_bytes": [model.MIN_OBJECT_BYTES, model.MAX_OBJECT_BYTES],
                "regen_ms_range": [80, 2500],
                "ttl_ms": TTL_MS,
                "sla_class": "high",
                "catalogue_items": service.catalogue.n_items,
                "users": service.catalogue.n_users,
            }
        )

    return build_app(service, [Route("/profile", profile, methods=["GET"])])


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None, timeout_graceful_shutdown=10)
