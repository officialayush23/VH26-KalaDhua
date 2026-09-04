"""Per-application counters, exposed on `/stats` and `/metrics`."""

from __future__ import annotations

import threading
import time
from bisect import insort
from collections import defaultdict

_MAX_SAMPLES = 4096


class Histogram:
    """Reservoir of recent samples, good enough for p50/p95 on a demo path."""

    def __init__(self, capacity: int = _MAX_SAMPLES) -> None:
        self._capacity = capacity
        self._sorted: list[float] = []
        self._order: list[float] = []
        self.count = 0
        self.total = 0.0

    def observe(self, value: float) -> None:
        """Record one sample."""
        self.count += 1
        self.total += value
        if len(self._order) >= self._capacity:
            oldest = self._order.pop(0)
            idx = _index_of(self._sorted, oldest)
            if idx is not None:
                self._sorted.pop(idx)
        self._order.append(value)
        insort(self._sorted, value)

    def quantile(self, q: float) -> float:
        """Interpolation-free quantile over the retained window."""
        if not self._sorted:
            return 0.0
        idx = min(len(self._sorted) - 1, max(0, int(q * len(self._sorted))))
        return self._sorted[idx]

    @property
    def mean(self) -> float:
        """Mean over every sample ever observed."""
        return self.total / self.count if self.count else 0.0


def _index_of(values: list[float], target: float) -> int | None:
    from bisect import bisect_left

    i = bisect_left(values, target)
    if i < len(values) and values[i] == target:
        return i
    return None


class AppTelemetry:
    """Counters for one example application.

    Thread-safe because `/load` drives work from a background task while the
    HTTP handlers run on the same loop and the driver polls `/stats`.
    """

    def __init__(self, application: str) -> None:
        self.application = application
        self.started_at = time.time()
        self._lock = threading.Lock()

        self.requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.regens = 0
        self.errors = 0
        self.admitted = 0
        self.rejected = 0
        self.cache_unavailable = 0
        self.bytes_served = 0
        self.tail_requests = 0

        self.regen_ms = Histogram()
        self.serve_ms = Histogram()
        self.object_bytes = Histogram()

        self.regen_cost_usd = 0.0
        self.saved_cost_usd = 0.0
        self.sla_penalty_usd = 0.0
        self.api_cost_usd = 0.0

        self.by_object_type: dict[str, int] = defaultdict(int)

    def record_request(self, object_type: str, serve_ms: float, size_bytes: int, tail: bool) -> None:
        """One `/work` request completed."""
        with self._lock:
            self.requests += 1
            self.by_object_type[object_type] += 1
            self.serve_ms.observe(serve_ms)
            self.object_bytes.observe(float(size_bytes))
            self.bytes_served += size_bytes
            if tail:
                self.tail_requests += 1

    def record_hit(self, saved_usd: float) -> None:
        """A cache hit, with the regeneration cost it avoided."""
        with self._lock:
            self.cache_hits += 1
            self.saved_cost_usd += saved_usd

    def record_miss(self) -> None:
        """A cache miss."""
        with self._lock:
            self.cache_misses += 1

    def record_regen(self, latency_ms: float, cost_usd: float, penalty_usd: float, api_usd: float) -> None:
        """A regeneration ran, with its measured cost."""
        with self._lock:
            self.regens += 1
            self.regen_ms.observe(latency_ms)
            self.regen_cost_usd += cost_usd
            self.sla_penalty_usd += penalty_usd
            self.api_cost_usd += api_usd

    def record_admission(self, admitted: bool) -> None:
        """The engine's answer to a PUT."""
        with self._lock:
            if admitted:
                self.admitted += 1
            else:
                self.rejected += 1

    def record_error(self) -> None:
        """A request failed."""
        with self._lock:
            self.errors += 1

    def record_cache_unavailable(self) -> None:
        """A cache call was skipped or failed; the app served from origin."""
        with self._lock:
            self.cache_unavailable += 1

    def hit_rate(self) -> float:
        """Object hit rate over cache lookups."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def stats(self) -> dict[str, object]:
        """The `/stats` payload (contract section 7, plus research extras)."""
        return {
            "application": self.application,
            "uptime_s": round(time.time() - self.started_at, 2),
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate(), 4),
            "regens": self.regens,
            "avg_regen_ms": round(self.regen_ms.mean, 3),
            "p50_regen_ms": round(self.regen_ms.quantile(0.50), 3),
            "p95_regen_ms": round(self.regen_ms.quantile(0.95), 3),
            "p50_serve_ms": round(self.serve_ms.quantile(0.50), 3),
            "p95_serve_ms": round(self.serve_ms.quantile(0.95), 3),
            "avg_object_bytes": int(self.object_bytes.mean),
            "p95_object_bytes": int(self.object_bytes.quantile(0.95)),
            "bytes_served": self.bytes_served,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "cache_unavailable": self.cache_unavailable,
            "errors": self.errors,
            "tail_requests": self.tail_requests,
            "cost_usd": round(self.regen_cost_usd, 8),
            "saved_cost_usd": round(self.saved_cost_usd, 8),
            "sla_penalty_usd": round(self.sla_penalty_usd, 8),
            "api_cost_usd": round(self.api_cost_usd, 8),
            "by_object_type": dict(self.by_object_type),
        }

    def prometheus(self, extra: dict[str, float] | None = None) -> str:
        """Prometheus text exposition for this application."""
        app = self.application
        lines: list[str] = []

        def emit(name: str, kind: str, help_text: str, value: float, labels: str = "") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            label = f'{{application="{app}"{labels}}}'
            lines.append(f"{name}{label} {value}")

        emit("aura_app_requests_total", "counter", "Requests served by the application.", self.requests)
        emit("aura_app_cache_hits_total", "counter", "Cache hits.", self.cache_hits)
        emit("aura_app_cache_misses_total", "counter", "Cache misses.", self.cache_misses)
        emit("aura_app_regens_total", "counter", "Regenerations executed.", self.regens)
        emit("aura_app_admitted_total", "counter", "Objects admitted by AURA.", self.admitted)
        emit("aura_app_rejected_total", "counter", "Objects rejected by AURA.", self.rejected)
        emit(
            "aura_app_cache_unavailable_total",
            "counter",
            "Cache calls skipped or failed; served from origin.",
            self.cache_unavailable,
        )
        emit("aura_app_errors_total", "counter", "Failed requests.", self.errors)
        emit("aura_app_bytes_served_total", "counter", "Payload bytes served.", self.bytes_served)
        emit("aura_app_tail_requests_total", "counter", "Requests hitting the expensive tail.", self.tail_requests)
        emit("aura_app_hit_rate", "gauge", "Object hit rate.", round(self.hit_rate(), 6))
        emit("aura_app_regen_cost_usd_total", "counter", "Measured regeneration spend.", round(self.regen_cost_usd, 8))
        emit(
            "aura_app_saved_cost_usd_total",
            "counter",
            "Regeneration spend avoided by cache hits.",
            round(self.saved_cost_usd, 8),
        )
        emit("aura_app_sla_penalty_usd_total", "counter", "SLA penalty accrued.", round(self.sla_penalty_usd, 8))
        emit("aura_app_api_cost_usd_total", "counter", "Third-party API spend.", round(self.api_cost_usd, 8))

        for quantile, value in (("0.5", self.regen_ms.quantile(0.5)), ("0.95", self.regen_ms.quantile(0.95))):
            lines.append("# HELP aura_app_regen_latency_ms Regeneration latency in milliseconds.")
            lines.append("# TYPE aura_app_regen_latency_ms summary")
            lines.append(f'aura_app_regen_latency_ms{{application="{app}",quantile="{quantile}"}} {round(value, 3)}')
        for quantile, value in (("0.5", self.serve_ms.quantile(0.5)), ("0.95", self.serve_ms.quantile(0.95))):
            lines.append("# HELP aura_app_serve_latency_ms End-to-end serve latency in milliseconds.")
            lines.append("# TYPE aura_app_serve_latency_ms summary")
            lines.append(f'aura_app_serve_latency_ms{{application="{app}",quantile="{quantile}"}} {round(value, 3)}')

        for object_type, count in sorted(self.by_object_type.items()):
            lines.append("# HELP aura_app_object_type_total Requests by object type.")
            lines.append("# TYPE aura_app_object_type_total counter")
            lines.append(f'aura_app_object_type_total{{application="{app}",object_type="{object_type}"}} {count}')

        for name, value in sorted((extra or {}).items()):
            lines.append(f"# HELP {name} Application-specific gauge.")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f'{name}{{application="{app}"}} {value}')

        return "\n".join(lines) + "\n"
