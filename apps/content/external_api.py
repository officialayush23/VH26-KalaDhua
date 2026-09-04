"""A simulated priced third-party API.

Some content objects are not generated locally - they are bought. This client
models a syndication provider that charges per call and answers with variable
latency. Two properties matter for the research:

* the price is adjustable at runtime (`POST /price`), which is what drives the
  cost-spike scenario: the same objects, the same traffic, ten times the cost of
  a miss;
* the latency is heavy-tailed, so an eviction can cost a second of user-visible
  wait as well as money.

The wait is a real `await`, not a busy loop: this models network time, and the
application reports it as latency rather than as CPU.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

DEFAULT_PRICE_USD = 0.0025


@dataclass
class ApiResponse:
    """One purchased document."""

    body: bytes
    latency_ms: float
    price_usd: float
    provider: str


class ExternalApiClient:
    """Priced provider with runtime-adjustable economics."""

    def __init__(
        self,
        *,
        price_usd: float = DEFAULT_PRICE_USD,
        median_latency_ms: float = 45.0,
        sigma: float = 0.7,
        seed: int = 42,
        provider: str = "syndication-partner",
    ) -> None:
        self.price_usd = price_usd
        self.median_latency_ms = median_latency_ms
        self.sigma = sigma
        self.provider = provider
        self.calls = 0
        self.spend_usd = 0.0
        self.latency_ms_total = 0.0
        self._rng = np.random.default_rng(seed)
        self._price_history: list[dict[str, float | str]] = []
        self.set_price(price_usd, reason="initial")

    def set_price(self, price_usd: float, *, reason: str = "manual") -> dict[str, object]:
        """Change the per-call price. This is the cost-spike lever."""
        previous = self.price_usd
        self.price_usd = max(0.0, float(price_usd))
        entry = {
            "ts": time.time(),
            "from_usd": previous,
            "to_usd": self.price_usd,
            "reason": reason,
        }
        self._price_history.append(entry)
        del self._price_history[:-32]
        return entry

    def set_latency(self, median_latency_ms: float, sigma: float | None = None) -> None:
        """Adjust the provider's latency distribution."""
        self.median_latency_ms = max(0.0, float(median_latency_ms))
        if sigma is not None:
            self.sigma = max(0.0, float(sigma))

    async def fetch(self, key_id: int, size_bytes: int) -> ApiResponse:
        """Buy one document. Charges the current price and waits for the provider."""
        latency_ms = float(self._rng.lognormal(mean=np.log(max(1e-3, self.median_latency_ms)), sigma=self.sigma))
        latency_ms = min(latency_ms, 5_000.0)
        started = time.perf_counter()
        await asyncio.sleep(latency_ms / 1000.0)
        observed_ms = (time.perf_counter() - started) * 1000.0

        payload = _document(key_id, size_bytes)
        self.calls += 1
        self.spend_usd += self.price_usd
        self.latency_ms_total += observed_ms
        return ApiResponse(
            body=payload,
            latency_ms=observed_ms,
            price_usd=self.price_usd,
            provider=self.provider,
        )

    def stats(self) -> dict[str, object]:
        """Provider counters for `/stats` and `/price`."""
        return {
            "provider": self.provider,
            "price_usd_per_call": self.price_usd,
            "median_latency_ms": self.median_latency_ms,
            "sigma": self.sigma,
            "calls": self.calls,
            "spend_usd": round(self.spend_usd, 8),
            "avg_latency_ms": round(self.latency_ms_total / self.calls, 3) if self.calls else 0.0,
            "price_history": self._price_history[-8:],
        }


def _document(key_id: int, size_bytes: int) -> bytes:
    """A syndicated article body of the requested size."""
    from content.objects import generate

    return generate(key_id ^ 0x5EED, size_bytes)
