"""Measurement and pricing of regeneration work.

The research claim of AURA rests on the applications reporting what a
regeneration *actually* cost, not what someone guessed it would cost. Everything
in this module is a measurement: wall clock via `time.perf_counter`, CPU via
`time.process_time`, database time reported by the query layer, payload size
from the serialised bytes, and third-party spend from the API client.
"""

from __future__ import annotations

import time
from types import TracebackType

from pydantic import BaseModel, Field

from .settings import Pricing

SlaClass = str
SLA_CLASSES: tuple[str, ...] = ("critical", "high", "normal", "low")


class CostVector(BaseModel):
    """Cost of regenerating one object once (contract section 1.1)."""

    cpu_ms: float = 0.0
    gpu_ms: float = 0.0
    db_ms: float = 0.0
    network_bytes: int = 0
    api_calls: int = 0
    api_cost_usd: float = 0.0
    latency_ms: float = 0.0

    def merged(self, other: CostVector) -> CostVector:
        """Sum two cost vectors; latency takes the larger of the two."""
        return CostVector(
            cpu_ms=self.cpu_ms + other.cpu_ms,
            gpu_ms=self.gpu_ms + other.gpu_ms,
            db_ms=self.db_ms + other.db_ms,
            network_bytes=self.network_bytes + other.network_bytes,
            api_calls=self.api_calls + other.api_calls,
            api_cost_usd=self.api_cost_usd + other.api_cost_usd,
            latency_ms=max(self.latency_ms, other.latency_ms),
        )

    def scaled(self, factor: float) -> CostVector:
        """Scale every resource dimension by `factor`."""
        return CostVector(
            cpu_ms=self.cpu_ms * factor,
            gpu_ms=self.gpu_ms * factor,
            db_ms=self.db_ms * factor,
            network_bytes=int(self.network_bytes * factor),
            api_calls=self.api_calls,
            api_cost_usd=self.api_cost_usd * factor,
            latency_ms=self.latency_ms * factor,
        )

    def usd(self, pricing: Pricing) -> float:
        """Price this cost vector with the engine's pricing table."""
        return (
            self.cpu_ms * pricing.cpu_ms_usd
            + self.gpu_ms * pricing.gpu_ms_usd
            + self.db_ms * pricing.db_ms_usd
            + (self.network_bytes / 1_073_741_824.0) * pricing.network_gb_usd
            + self.api_cost_usd
        )

    def sla_penalty_usd(self, pricing: Pricing) -> float:
        """Penalty owed for exceeding the latency objective, in USD."""
        over = max(0.0, self.latency_ms - pricing.slo_p95_ms)
        return over * pricing.sla_penalty_per_ms_over_slo_usd


class ObjectContext(BaseModel):
    """What an application tells AURA about an object (contract section 1.2)."""

    application: str
    object_type: str
    size_bytes: int
    ttl_ms: int
    sla_class: SlaClass = "normal"
    regen: CostVector = Field(default_factory=CostVector)


class CostMeter:
    """Measures one regeneration.

    Wall latency and CPU time are taken by the meter itself. Everything the
    process cannot observe from the outside - database service time, accelerator
    time, third-party API spend - is contributed by the code being measured::

        with CostMeter() as meter:
            rows, db_ms = await run_query()
            meter.add_db_ms(db_ms)
        cost = meter.finish(size_bytes=len(payload))

    `time.process_time` counts CPU burned by the whole process. Under
    concurrency a regeneration that awaits will absorb some of its neighbours'
    CPU; the applications therefore measure the CPU-bound section directly with
    `section()` rather than wrapping long awaits.
    """

    def __init__(self) -> None:
        self._t0 = 0.0
        self._c0 = 0.0
        self._wall_ms = 0.0
        self._cpu_ms = 0.0
        self._closed = False
        self.gpu_ms = 0.0
        self.db_ms = 0.0
        self.api_calls = 0
        self.api_cost_usd = 0.0
        self.network_bytes = 0

    def __enter__(self) -> CostMeter:
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        """Freeze the wall and CPU readings. Idempotent."""
        if self._closed:
            return
        self._wall_ms = (time.perf_counter() - self._t0) * 1000.0
        self._cpu_ms = (time.process_time() - self._c0) * 1000.0
        self._closed = True

    def add_db_ms(self, ms: float) -> None:
        """Record database service time measured by the query layer."""
        self.db_ms += ms

    def add_gpu_ms(self, ms: float) -> None:
        """Record accelerator time measured by the model layer."""
        self.gpu_ms += ms

    def add_api_call(self, cost_usd: float, calls: int = 1) -> None:
        """Record a priced third-party call."""
        self.api_calls += calls
        self.api_cost_usd += cost_usd

    def add_network_bytes(self, n: int) -> None:
        """Record bytes moved that are not the payload itself."""
        self.network_bytes += n

    def section(self) -> CpuSection:
        """Measure the CPU of one block explicitly, excluding awaits."""
        return CpuSection(self)

    @property
    def wall_ms(self) -> float:
        """Wall time so far, in milliseconds."""
        if self._closed:
            return self._wall_ms
        return (time.perf_counter() - self._t0) * 1000.0

    def finish(self, size_bytes: int) -> CostVector:
        """Close the meter and produce the measured cost vector."""
        self.stop()
        return CostVector(
            cpu_ms=round(max(0.0, self._cpu_ms), 4),
            gpu_ms=round(self.gpu_ms, 4),
            db_ms=round(self.db_ms, 4),
            network_bytes=self.network_bytes + size_bytes,
            api_calls=self.api_calls,
            api_cost_usd=round(self.api_cost_usd, 8),
            latency_ms=round(self._wall_ms, 4),
        )


class CpuSection:
    """CPU timer for a single synchronous block, attributed to a meter."""

    def __init__(self, meter: CostMeter) -> None:
        self._meter = meter
        self._c0 = 0.0
        self.cpu_ms = 0.0

    def __enter__(self) -> CpuSection:
        self._c0 = time.process_time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cpu_ms = (time.process_time() - self._c0) * 1000.0


class Timer:
    """Small wall-clock timer used where only elapsed milliseconds matter."""

    def __init__(self) -> None:
        self._t0 = 0.0
        self.ms = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
