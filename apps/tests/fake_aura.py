"""A minimal stand-in for aura-server implementing contract section 2.

Only the data plane, the explain endpoint and enough introspection for the SDK
tests. It exists so the client can be developed and verified before the Rust
engine is ready, and so the contract can be exercised in CI without it.

`POST /_control` switches admission on and off, which is how the tests drive the
"the engine said no" path.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route


class FakeAura:
    """In-memory cache with the AURA HTTP surface."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.admit = True
        self.puts: list[dict[str, Any]] = []
        self.gets = 0
        self.capacity_bytes = 512 * 1024 * 1024

    @property
    def used_bytes(self) -> int:
        """Bytes currently held."""
        return sum(int(entry["context"]["size_bytes"]) for entry in self.store.values())

    def app(self) -> Starlette:
        """Build the ASGI application."""
        return Starlette(
            routes=[
                Route("/v1/cache/batch/get", self.batch_get, methods=["POST"]),
                Route("/v1/cache/{key:path}/refresh", self.refresh, methods=["POST"]),
                Route("/v1/cache/{key:path}", self.get, methods=["GET"]),
                Route("/v1/cache/{key:path}", self.put, methods=["PUT"]),
                Route("/v1/cache/{key:path}", self.delete, methods=["DELETE"]),
                Route("/v1/explain/recent", self.explain_recent, methods=["GET"]),
                Route("/v1/explain/{key:path}", self.explain, methods=["GET"]),
                Route("/v1/stats", self.stats, methods=["GET"]),
                Route("/healthz", self.healthz, methods=["GET"]),
                Route("/metrics", self.metrics, methods=["GET"]),
                Route("/_control", self.control, methods=["POST"]),
            ]
        )

    async def get(self, request: Request) -> JSONResponse:
        """`GET /v1/cache/{key}`."""
        self.gets += 1
        key = request.path_params["key"]
        entry = self.store.get(key)
        if entry is None:
            return JSONResponse({"hit": False, "reason": "miss"}, status_code=404)
        return JSONResponse(
            {
                "hit": True,
                "value": entry["value"],
                "age_ms": (time.time() - entry["stored_at"]) * 1000.0,
                "layer": "L2",
                "latency_us": 41,
            }
        )

    async def put(self, request: Request) -> JSONResponse:
        """`PUT /v1/cache/{key}`."""
        key = request.path_params["key"]
        body = json.loads(await request.body())
        context = body["context"]
        self.puts.append({"key": key, "context": context})
        if not self.admit:
            return JSONResponse(
                {
                    "admitted": False,
                    "reason_code": "density_below_threshold",
                    "evicted": [],
                    "used_bytes": self.used_bytes,
                }
            )
        self.store[key] = {"value": body["value"], "context": context, "stored_at": time.time()}
        return JSONResponse(
            {
                "admitted": True,
                "reason_code": "density_above_threshold",
                "evicted": [],
                "used_bytes": self.used_bytes,
            }
        )

    async def delete(self, request: Request) -> JSONResponse:
        """`DELETE /v1/cache/{key}`."""
        key = request.path_params["key"]
        removed = self.store.pop(key, None) is not None
        return JSONResponse({"removed": removed})

    async def refresh(self, request: Request) -> JSONResponse:
        """`POST /v1/cache/{key}/refresh`."""
        return JSONResponse({"queued": True})

    async def batch_get(self, request: Request) -> JSONResponse:
        """`POST /v1/cache/batch/get`."""
        body = json.loads(await request.body())
        results: dict[str, Any] = {}
        for key in body.get("keys", []):
            entry = self.store.get(key)
            if entry is None:
                results[key] = {"hit": False, "reason": "miss"}
            else:
                results[key] = {
                    "hit": True,
                    "value": entry["value"],
                    "age_ms": (time.time() - entry["stored_at"]) * 1000.0,
                    "layer": "L2",
                    "latency_us": 38,
                }
        return JSONResponse({"results": results})

    async def explain(self, request: Request) -> JSONResponse:
        """`GET /v1/explain/{key}`."""
        key = request.path_params["key"]
        entry = self.store.get(key)
        return JSONResponse(
            {
                "key": key,
                "present": entry is not None,
                "action": "Keep" if entry else "Evict",
                "reuse_probability": {"h10s": 0.41, "h60s": 0.87, "h600s": 0.93},
                "economic_value_usd": 0.00231,
                "value_density": 4.72,
                "eviction_threshold": 1.11,
                "features": {"size_bytes": entry["context"]["size_bytes"] if entry else 0},
                "contributions": [{"feature": "trend", "weight": 0.31}],
                "reasons": ["fake server"],
                "predictor": "fake",
                "predictor_confidence": 1.0,
            }
        )

    async def explain_recent(self, request: Request) -> JSONResponse:
        """`GET /v1/explain/recent`."""
        limit = int(request.query_params.get("limit", 50))
        decisions = [{"key": key, "action": "Keep"} for key in list(self.store)[:limit]]
        return JSONResponse({"decisions": decisions})

    async def stats(self, request: Request) -> JSONResponse:
        """`GET /v1/stats`."""
        return JSONResponse({"keys": len(self.store), "used_bytes": self.used_bytes, "gets": self.gets})

    async def healthz(self, request: Request) -> JSONResponse:
        """`GET /healthz`."""
        return JSONResponse({"ok": True, "version": "fake", "uptime_s": 0})

    async def metrics(self, request: Request) -> PlainTextResponse:
        """`GET /metrics`."""
        return PlainTextResponse(f"aura_fake_keys {len(self.store)}\n")

    async def control(self, request: Request) -> JSONResponse:
        """Test-only switch for admission behaviour."""
        body = json.loads(await request.body() or b"{}")
        if "admit" in body:
            self.admit = bool(body["admit"])
        if body.get("flush"):
            self.store.clear()
        return JSONResponse({"admit": self.admit, "keys": len(self.store)})


class FakeServer:
    """Runs a :class:`FakeAura` under uvicorn on a background thread."""

    def __init__(self) -> None:
        self.fake = FakeAura()
        self.port = _free_port()
        config = uvicorn.Config(
            self.fake.app(),
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="fake-aura", daemon=True)

    @property
    def base_url(self) -> str:
        """Where the fake listens."""
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float = 15.0) -> None:
        """Start serving and wait until the socket is accepting."""
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                return
            time.sleep(0.02)
        raise RuntimeError("fake aura server did not start")

    def stop(self) -> None:
        """Stop serving."""
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    """Serve the fake on port 8080, for demoing the apps without the engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a stand-in AURA server (contract section 2).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(FakeAura().app(), host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
