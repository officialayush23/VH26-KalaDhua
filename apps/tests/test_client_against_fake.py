"""Behavioural tests for the AURA client SDK.

The engine does not have to exist for these to run: `tests/fake_aura.py` is a
small server implementing contract section 2, served over real HTTP by uvicorn
so the client exercises its actual transport, timeouts and pool.

Covered: hit, miss, admission, rejection, outage, base64 objects, explain,
batch get, delete, and the synchronous facade.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aura_client import AuraClient, SyncAuraClient  # noqa: E402
from common.costing import CostMeter, CostVector  # noqa: E402
from tests.fake_aura import FakeServer  # noqa: E402


@pytest.fixture(scope="module")
def server() -> Any:
    """A running fake AURA server."""
    fake = FakeServer()
    fake.start()
    try:
        yield fake
    finally:
        fake.stop()


def run(coro: Any) -> Any:
    """Run one coroutine to completion (no pytest-asyncio dependency)."""
    return asyncio.run(coro)


def burn_cpu(iterations: int = 60_000) -> float:
    """Do enough arithmetic that `process_time` registers a non-zero delta."""
    total = 0.0
    for i in range(iterations):
        total += (i % 7) ** 0.5
    return total


# ------------------------------------------------------------------ hit / miss


def test_miss_regenerates_and_reports_measured_cost(server: Any) -> None:
    """A miss runs regen once and PUTs the measured ObjectContext."""

    async def scenario() -> None:
        run_count = {"n": 0}

        async def regen(meter: CostMeter) -> dict[str, Any]:
            run_count["n"] += 1
            burn_cpu()
            meter.add_db_ms(12.5)
            meter.add_gpu_ms(3.0)
            return {"answer": 42, "padding": "x" * 500}

        async with AuraClient(server.base_url, application="analytics", default_sla="high") as client:
            value = await client.get_or_regen(
                "analytics:revenue:mumbai:30d",
                object_type="dashboard_query",
                ttl_ms=600_000,
                regen=regen,
            )
            assert value == {"answer": 42, "padding": "x" * 500}
            assert run_count["n"] == 1

            stats = client.stats()
            assert stats["misses"] == 1
            assert stats["regens"] == 1
            assert stats["admitted"] == 1
            assert stats["spent_usd"] > 0.0

    server.fake.admit = True
    run(scenario())

    put = server.fake.puts[-1]
    context = put["context"]
    assert put["key"] == "analytics:revenue:mumbai:30d"
    assert context["application"] == "analytics"
    assert context["object_type"] == "dashboard_query"
    assert context["ttl_ms"] == 600_000
    assert context["sla_class"] == "high"
    assert context["size_bytes"] > 500
    assert context["regen"]["cpu_ms"] > 0.0, "cpu time must be measured, not guessed"
    assert context["regen"]["db_ms"] == 12.5
    assert context["regen"]["gpu_ms"] == 3.0
    assert context["regen"]["latency_ms"] > 0.0
    assert context["regen"]["network_bytes"] == context["size_bytes"]


def test_hit_skips_regeneration_and_credits_the_saving(server: Any) -> None:
    """A second request for the same key is served from the cache."""

    async def scenario() -> None:
        calls = {"n": 0}

        async def regen() -> dict[str, Any]:
            calls["n"] += 1
            burn_cpu()
            return {"rows": list(range(50))}

        async with AuraClient(server.base_url, application="analytics") as client:
            first = await client.get_or_regen(
                "analytics:hit-path", object_type="dashboard_query", ttl_ms=60_000, regen=regen
            )
            second = await client.get_or_regen(
                "analytics:hit-path", object_type="dashboard_query", ttl_ms=60_000, regen=regen
            )
            assert first == second
            assert calls["n"] == 1

            stats = client.stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 0.5
            assert stats["saved_usd"] > 0.0, "a hit must be credited with the cost it avoided"

    server.fake.admit = True
    run(scenario())


def test_force_fresh_bypasses_the_cache(server: Any) -> None:
    """`force_fresh` regenerates even when the object is present."""

    async def scenario() -> None:
        calls = {"n": 0}

        async def regen() -> dict[str, Any]:
            calls["n"] += 1
            return {"v": calls["n"]}

        async with AuraClient(server.base_url, application="analytics") as client:
            await client.get_or_regen("analytics:fresh", object_type="q", ttl_ms=1_000, regen=regen)
            await client.get_or_regen("analytics:fresh", object_type="q", ttl_ms=1_000, regen=regen, force_fresh=True)
            assert calls["n"] == 2

    server.fake.admit = True
    run(scenario())


# ------------------------------------------------------------------- admission


def test_rejection_is_a_normal_answer(server: Any) -> None:
    """`admitted: false` must not raise, and the value is still served."""

    async def scenario() -> None:
        async def regen() -> dict[str, Any]:
            return {"large": "y" * 1_000}

        async with AuraClient(server.base_url, application="content") as client:
            outcome = await client.get_or_regen_detailed(
                "content:rejected-object", object_type="video_segment", ttl_ms=60_000, regen=regen
            )
            assert outcome.value == {"large": "y" * 1_000}
            assert outcome.admitted is False
            assert outcome.reason_code == "density_below_threshold"
            assert outcome.served_from == "origin"

            stats = client.stats()
            assert stats["rejected"] == 1
            assert stats["cache_errors"] == 0, "a rejection is not a failure"
            assert stats["breaker_state"] == "closed"

    server.fake.admit = False
    try:
        run(scenario())
    finally:
        server.fake.admit = True


# ---------------------------------------------------------------------- outage


def test_outage_serves_from_origin_and_opens_the_breaker() -> None:
    """With the cache unreachable the application keeps working."""
    port = _closed_port()

    async def scenario() -> None:
        calls = {"n": 0}

        async def regen() -> dict[str, Any]:
            calls["n"] += 1
            return {"origin": calls["n"]}

        async with AuraClient(
            f"http://127.0.0.1:{port}",
            application="recommendation",
            failure_threshold=2,
            breaker_reset_s=60.0,
            connect_timeout_s=0.25,
            timeout_s=0.5,
        ) as client:
            for i in range(6):
                value = await client.get_or_regen(
                    f"recommendation:user:{i}", object_type="ranking_result", ttl_ms=1_000, regen=regen
                )
                assert value == {"origin": i + 1}

            stats = client.stats()
            assert calls["n"] == 6, "every request must still be served"
            assert stats["cache_errors"] >= 2
            assert stats["breaker_state"] == "open"
            assert stats["breaker_trips"] == 1
            assert stats["breaker_skips"] > 0, "an open breaker must stop calling the cache"

    run(scenario())


def test_breaker_recovers_when_the_cache_returns(server: Any) -> None:
    """After the cooldown the client tries again and closes the breaker."""

    async def scenario() -> None:
        async def regen() -> dict[str, Any]:
            return {"ok": True}

        async with AuraClient(
            server.base_url,
            application="analytics",
            failure_threshold=1,
            breaker_reset_s=0.05,
        ) as client:
            client._breaker.record_failure()  # simulate a prior outage
            assert client.breaker_state == "open"
            await asyncio.sleep(0.1)
            await client.get_or_regen("analytics:recovered", object_type="dashboard_query", ttl_ms=1_000, regen=regen)
            assert client.breaker_state == "closed"

    server.fake.admit = True
    run(scenario())


# ------------------------------------------------------------------- encodings


def test_base64_objects_round_trip_unchanged(server: Any) -> None:
    """Large binary objects survive the cache byte for byte."""
    blob = bytes(range(256)) * 400  # 102 400 bytes

    async def scenario() -> None:
        async def regen() -> bytes:
            return blob

        async with AuraClient(server.base_url, application="content") as client:
            first = await client.get_or_regen(
                "content:image_variant:7",
                object_type="image_variant",
                ttl_ms=60_000,
                regen=regen,
                encoding="base64",
            )
            second = await client.get_or_regen(
                "content:image_variant:7",
                object_type="image_variant",
                ttl_ms=60_000,
                regen=regen,
                encoding="base64",
            )
            assert first == blob
            assert second == blob

    server.fake.admit = True
    run(scenario())

    context = next(p["context"] for p in reversed(server.fake.puts) if p["key"] == "content:image_variant:7")
    assert context["size_bytes"] == len(blob), "size must be the object, not its base64 inflation"


def test_regen_may_report_its_own_cost_vector(server: Any) -> None:
    """A reported non-zero dimension overrides the SDK's own measurement."""

    async def scenario() -> None:
        async def regen() -> tuple[dict[str, Any], CostVector]:
            return {"x": 1}, CostVector(db_ms=333.0, api_calls=2, api_cost_usd=0.004)

        async with AuraClient(server.base_url, application="content") as client:
            outcome = await client.get_or_regen_detailed(
                "content:priced-object", object_type="syndicated_article", ttl_ms=90_000, regen=regen
            )
            assert outcome.cost.db_ms == 333.0
            assert outcome.cost.api_cost_usd == 0.004
            assert outcome.cost_usd >= 0.004

    server.fake.admit = True
    run(scenario())


# --------------------------------------------------------- the rest of the API


def test_explain_delete_refresh_and_batch(server: Any) -> None:
    """The remaining data-plane and explainability calls."""

    async def scenario() -> None:
        async def regen() -> dict[str, Any]:
            return {"v": 1}

        async with AuraClient(server.base_url, application="analytics") as client:
            await client.get_or_regen("analytics:explain-me", object_type="q", ttl_ms=1_000, regen=regen)

            explanation = await client.explain("analytics:explain-me")
            assert explanation is not None
            assert explanation["present"] is True
            assert explanation["action"] == "Keep"

            recent = await client.explain_recent(limit=5)
            assert isinstance(recent, list)

            batch = await client.batch_get(["analytics:explain-me", "analytics:not-there"])
            assert "analytics:explain-me" in batch
            assert "analytics:not-there" not in batch

            assert await client.refresh("analytics:explain-me") is True
            assert await client.delete("analytics:explain-me") is True
            assert await client.get("analytics:explain-me") is None

    server.fake.admit = True
    run(scenario())


def test_sync_facade(server: Any) -> None:
    """The blocking wrapper works from ordinary synchronous code."""

    def regen() -> dict[str, Any]:
        return {"sync": True}

    with SyncAuraClient(server.base_url, application="analytics") as client:
        first = client.get_or_regen("analytics:sync", object_type="q", ttl_ms=1_000, regen=regen)
        second = client.get_or_regen("analytics:sync", object_type="q", ttl_ms=1_000, regen=regen)
        assert first == second == {"sync": True}
        assert client.stats()["hits"] == 1


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
