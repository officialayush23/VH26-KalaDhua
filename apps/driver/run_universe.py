"""Mixed-workload driver for the three example applications.

This is the "no dashboard needed" demo path: it drives a realistic mix of
traffic across recommendation, analytics and content, and prints a live table of
what each application is paying and what the cache is saving it.

    python -m driver.run_universe --duration 120
    python -m driver.run_universe --spawn --expensive-tail --duration 180

`--spawn` starts the three services itself; without it they are expected to be
running already. `--expensive-tail` switches on the pathological workload on all
three, and `--price-spike` raises the content provider's price part way through
so the cost-spike scenario is visible in the table.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from common.loadgen import KeyGenerator, LoadSpec

REPO_APPS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AppTarget:
    """One application under load."""

    name: str
    port: int
    module: str
    pattern: str
    key_space: int
    share: float
    concurrency: int = 12
    errors: int = 0
    first_error: str = ""
    process: subprocess.Popen[bytes] | None = None
    generator: KeyGenerator | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        """Where the service listens."""
        return f"http://127.0.0.1:{self.port}"


# The two workloads the argument rests on: one CPU-bound, one database-bound. Content is a
# third cost shape (large objects, bandwidth-dominated, a priced third party) and is real,
# but it is opt-in with --with-content so a demo run starts two services rather than three.
TARGETS = [
    AppTarget("recommendation", 8101, "recommendation.main", "popularity_shift", 300, 0.20, concurrency=4),
    AppTarget("analytics", 8102, "analytics.main", "zipf", 240, 0.50, concurrency=16),
]

OPTIONAL_TARGETS = [
    AppTarget("content", 8103, "content.main", "burst", 1_500, 0.30, concurrency=8),
]


async def wait_healthy(client: httpx.AsyncClient, target: AppTarget, timeout_s: float = 120.0) -> bool:
    """Poll `/health` until the service answers or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{target.base_url}/health", timeout=3.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


def log_path(target: AppTarget) -> str:
    """Where a spawned service's output goes."""
    return os.path.join(REPO_APPS, "runs", f"{target.name}.log")


def spawn(target: AppTarget) -> subprocess.Popen[bytes]:
    """Start one service as a child process, keeping its output.

    The output used to go to DEVNULL, which meant a service that died on its first line
    reported nothing but "did not become healthy" - true, useless, and the reason a
    five-minute problem took an evening.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", REPO_APPS)
    os.makedirs(os.path.join(REPO_APPS, "runs"), exist_ok=True)
    handle = open(log_path(target), "wb")
    return subprocess.Popen(
        [sys.executable, "-m", target.module],
        cwd=REPO_APPS,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def tail(path: str, lines: int = 20) -> str:
    """The last few lines of a log file, for a failure message."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:]).rstrip()
    except OSError:
        return "(no output captured)"


async def configure_tail(client: httpx.AsyncClient, target: AppTarget, enabled: bool) -> None:
    """Switch the expensive tail on or off for one application."""
    try:
        await client.post(
            f"{target.base_url}/expensive-tail",
            json={"enabled": enabled, "fraction": 0.05, "multiplier": 30.0},
            timeout=5.0,
        )
    except Exception:
        pass


async def drive_target(
    client: httpx.AsyncClient,
    target: AppTarget,
    rps: float,
    duration_s: float,
    stop: asyncio.Event,
) -> None:
    """Issue requests to one application at a fixed rate."""
    spec = LoadSpec(rps=max(0.5, rps), duration_s=duration_s, pattern=target.pattern, key_space=target.key_space)
    generator = KeyGenerator(spec)
    target.generator = generator
    interval = 1.0 / spec.rps
    started = time.perf_counter()
    issued = 0
    inflight: set[asyncio.Task[Any]] = set()
    semaphore = asyncio.Semaphore(target.concurrency)

    async def one(key_id: int) -> None:
        async with semaphore:
            try:
                await client.get(f"{target.base_url}/work/{key_id}", timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                # Counted and remembered rather than swallowed. A load generator that hides
                # its own failures reports a flat table and lets you conclude the cache is
                # doing nothing, when in fact nothing was ever asked of it.
                target.errors += 1
                if not target.first_error:
                    target.first_error = f"{type(exc).__name__}: {exc}"[:160]

    while not stop.is_set() and time.perf_counter() - started < duration_s:
        due = started + issued * interval
        now = time.perf_counter()
        if due > now:
            try:
                await asyncio.wait_for(stop.wait(), timeout=due - now)
                break
            # asyncio.TimeoutError only became an alias of the built-in TimeoutError in
            # Python 3.11. On 3.10 the bare `except TimeoutError` here caught nothing, the
            # exception escaped a task nobody awaited, and this loop died after issuing
            # exactly one request -- which is why the table showed reqs=1 for the whole run
            # and never moved.
            except (asyncio.TimeoutError, TimeoutError):
                pass
        task = asyncio.create_task(one(generator.next_key()))
        inflight.add(task)
        task.add_done_callback(inflight.discard)
        issued += 1

    if inflight:
        await asyncio.gather(*list(inflight), return_exceptions=True)


async def poll_stats(client: httpx.AsyncClient, target: AppTarget) -> dict[str, Any]:
    """Fetch one application's `/stats`."""
    try:
        response = await client.get(f"{target.base_url}/stats", timeout=5.0)
        if response.status_code == 200:
            return dict(response.json())
    except Exception:
        pass
    return {}


def render(rows: list[tuple[AppTarget, dict[str, Any]]], elapsed: float) -> str:
    """Format the live table."""
    header = (
        f"{'application':<16}{'reqs':>8}{'hit%':>7}{'regens':>8}"
        f"{'p50ms':>8}{'p95ms':>9}{'avg KB':>9}{'spent $':>11}{'saved $':>11}{'cache':>10}"
    )
    lines = [f"t={elapsed:6.1f}s", header, "-" * len(header)]
    total_spent = 0.0
    total_saved = 0.0
    total_errors = 0
    for target, stats in rows:
        total_errors += target.errors
        if not stats:
            lines.append(f"{target.name:<16}{'(unreachable)':>73}")
            continue
        client_stats = stats.get("client") or {}
        spent = float(stats.get("cost_usd", 0.0))
        saved = float(stats.get("saved_cost_usd", 0.0))
        total_spent += spent
        total_saved += saved
        lines.append(
            f"{target.name:<16}"
            f"{int(stats.get('requests', 0)):>8}"
            f"{100 * float(stats.get('hit_rate', 0.0)):>7.1f}"
            f"{int(stats.get('regens', 0)):>8}"
            f"{float(stats.get('p50_regen_ms', 0.0)):>8.1f}"
            f"{float(stats.get('p95_regen_ms', 0.0)):>9.1f}"
            f"{float(stats.get('avg_object_bytes', 0)) / 1024.0:>9.1f}"
            f"{spent:>11.6f}"
            f"{saved:>11.6f}"
            f"{str(client_stats.get('breaker_state', '?')):>10}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'total':<16}{'':>8}{'':>7}{'':>8}{'':>8}{'':>9}{'':>9}{total_spent:>11.6f}{total_saved:>11.6f}")
    if total_errors:
        # Said plainly. A table of zeroes with a silent failure count underneath it reads as
        # "the cache did nothing", when the truth is that nothing ever reached the cache.
        detail = next((t.first_error for t, _ in rows if t.first_error), "")
        lines.append(f"{total_errors} request(s) failed. First: {detail}")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    """Drive the universe and print the live table."""
    available = TARGETS + (OPTIONAL_TARGETS if args.with_content else [])
    targets = [t for t in available if args.only is None or t.name in args.only]

    if args.spawn:
        for target in targets:
            target.process = spawn(target)
            print(f"started {target.name} (pid {target.process.pid}) on port {target.port}", flush=True)

    async with httpx.AsyncClient() as client:
        for target in targets:
            if not await wait_healthy(client, target, timeout_s=args.startup_timeout):
                print(f"{target.name} did not become healthy at {target.base_url}", file=sys.stderr)
                if args.spawn:
                    print(f"--- last output from {target.name} " + "-" * 30, file=sys.stderr)
                    print(tail(log_path(target)), file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                await shutdown(targets)
                return 1
            await configure_tail(client, target, args.expensive_tail)

        stop = asyncio.Event()

        def request_stop(*_: object) -> None:
            stop.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except NotImplementedError:
                pass

        drivers = [
            asyncio.create_task(drive_target(client, target, args.rps * target.share, args.duration, stop))
            for target in targets
        ]

        started = time.perf_counter()
        spiked = False
        try:
            while not stop.is_set() and time.perf_counter() - started < args.duration:
                await asyncio.sleep(args.interval)
                elapsed = time.perf_counter() - started
                if args.price_spike and not spiked and elapsed >= args.duration / 2:
                    spiked = True
                    try:
                        await client.post(
                            "http://127.0.0.1:8103/price",
                            json={"price_usd": 0.025, "reason": "cost_spike_scenario"},
                            timeout=5.0,
                        )
                        print("\n*** content provider price raised to $0.025/call ***\n", flush=True)
                    except Exception:
                        pass
                rows = [(target, await poll_stats(client, target)) for target in targets]
                print("\n" + render(rows, elapsed), flush=True)
        finally:
            stop.set()
            await asyncio.gather(*drivers, return_exceptions=True)
            rows = [(target, await poll_stats(client, target)) for target in targets]
            print("\nfinal:\n" + render(rows, time.perf_counter() - started), flush=True)

    await shutdown(targets)
    return 0


async def shutdown(targets: list[AppTarget]) -> None:
    """Stop any child processes this driver started."""
    for target in targets:
        if target.process is not None and target.process.poll() is None:
            target.process.terminate()
            try:
                target.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                target.process.kill()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description="Drive a mixed workload across the AURA example applications.")
    parser.add_argument("--rps", type=float, default=20.0, help="aggregate requests per second across all apps")
    parser.add_argument("--duration", type=float, default=60.0, help="run time in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between table refreshes")
    parser.add_argument("--spawn", action="store_true", help="start the services as child processes")
    parser.add_argument("--expensive-tail", action="store_true", help="enable the expensive-tail workload")
    parser.add_argument("--price-spike", action="store_true", help="raise the content API price mid-run")
    parser.add_argument("--startup-timeout", type=float, default=120.0, help="seconds to wait for /health")
    parser.add_argument(
        "--with-content",
        action="store_true",
        help="also run the content service, the bandwidth-dominated third cost shape",
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to named applications")
    return parser.parse_args(argv)


def main() -> int:
    """Entry point."""
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
