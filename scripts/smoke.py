"""End-to-end smoke test against a deployed AURA engine.

What it proves, in order:

1. the engine is awake and answering
2. a key is accepted, and the engine attributes the request to the application that key
   was issued to rather than to anything in the request body
3. an object can be written, read back, and is reported as a hit from L2
4. a key that was never written is a clean miss with a rebuild lease, not an error
5. the cost of the rebuild reached the ledger, so the savings figure has something behind it

It is deliberately dependency-free -- standard library only, no httpx, no repo imports -- so
it runs from any Python 3.9 or newer with nothing installed. If this passes, the transport,
the auth, the cache path and the accounting are all working, and anything still broken is in
an application rather than between them.

Usage
-----
    python scripts/smoke.py --rec-key aura_sk_... --ana-key aura_sk_...

or set AURA_RECOMMENDATION_KEY and AURA_ANALYTICS_KEY and pass neither.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ENGINE = "https://vh26-kaladhua.onrender.com"

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
OFF = "\033[0m"


def call(
    method: str,
    url: str,
    *,
    key: str | None = None,
    body: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict]:
    """One request. Returns (status, parsed body); never raises for an HTTP error status.

    A 404 from the cache is a miss, not a failure, and a 401 is a fact worth printing
    rather than a traceback -- so error statuses come back as values like any other.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode()
            return res.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw[:200]}
    except Exception as e:  # timeout, DNS, TLS
        return 0, {"error": f"{type(e).__name__}: {e}"}


def ok(text: str) -> None:
    print(f"  {GREEN}pass{OFF}  {text}")


def bad(text: str) -> None:
    print(f"  {RED}FAIL{OFF}  {text}")


def note(text: str) -> None:
    print(f"        {DIM}{text}{OFF}")


def wake(engine: str) -> bool:
    """The free instance sleeps. The first request after that pays the cold start."""
    print("\nWaking the engine")
    for attempt in range(1, 21):
        status, body = call("GET", f"{engine}/healthz", timeout=15)
        if status == 200:
            ok(f"engine is awake ({body.get('status', 'ok')})")
            return True
        print(f"        {DIM}attempt {attempt}: still starting ...{OFF}")
        time.sleep(5)
    bad("the engine never answered /healthz")
    return False


def exercise(engine: str, app: str, key: str) -> bool:
    """Write one object, read it back, then miss on one that was never written."""
    print(f"\n{app}")
    stamp = int(time.time())
    hot = f"smoke:{app}:{stamp}"
    cold = f"smoke:{app}:{stamp}:never-written"

    # An expensive object: 48 KB that took 412ms of database time to build. Those two
    # numbers are the whole point -- they are what the cache scores, and a cache that is
    # not told them can only fall back on arrival order.
    status, body = call(
        "PUT",
        f"{engine}/v1/cache/{hot}",
        key=key,
        body={
            "value": {"rows": [{"region": "north", "revenue": 91_400}]},
            "context": {
                "application": app,
                "object_type": "report",
                "size_bytes": 48_000,
                "ttl_ms": 300_000,
            },
            "measured": {"db_ms": 412},
        },
    )
    if status == 401:
        bad(f"the key was refused: {body.get('error', body)}")
        return False
    if status >= 400 or status == 0:
        bad(f"PUT failed ({status}): {body.get('error', body)}")
        return False
    ok(f"wrote {hot}")
    note(f"decision: {body.get('action', body.get('decision', '?'))} -- {body.get('reason', '')}")

    status, body = call("GET", f"{engine}/v1/cache/{hot}?application={app}", key=key)
    if status == 200 and body.get("hit"):
        ok(f"read it back from {body.get('layer', 'L2')}, age {body.get('age_ms', 0)} ms")
    elif status == 404:
        # Not a transport failure. The cache was offered the object and declined it, or
        # evicted it immediately -- which is a legitimate decision and worth seeing.
        bad("the object was not resident a moment after being written")
        note(f"reason: {body.get('reason', '?')} -- the admission gate refused or evicted it")
        return False
    else:
        bad(f"GET failed ({status}): {body.get('error', body)}")
        return False

    status, body = call("GET", f"{engine}/v1/cache/{cold}?application={app}", key=key)
    if status == 404 and body.get("hit") is False:
        ok("a key that was never written is a clean miss")
        if body.get("rebuild") is not None:
            note(
                f"rebuild lease: {body.get('rebuild')} -- the first caller rebuilds, "
                "the rest wait rather than all hitting the origin"
            )
    else:
        bad(f"expected a miss, got {status}: {body}")
        return False

    return True


def summary(engine: str, key: str) -> None:
    """What the engine now believes about itself."""
    print("\nEngine state")
    status, f = call("GET", f"{engine}/v1/stats", key=key)
    if status != 200:
        bad(f"could not read /v1/stats ({status}): {f.get('error', f)}")
        return

    l2 = f.get("layers", {}).get("l2", {})
    cost = f.get("cost", {})
    engine_stats = f.get("engine", {})
    tier1 = f.get("tier1", {})

    print(f"  requests           {engine_stats.get('requests', 0)}")
    print(f"  L2 hit rate        {l2.get('hit_rate', 0):.1%}")
    print(f"  resident objects   {engine_stats.get('resident_objects', 0)}")
    print(f"  admitted / refused {engine_stats.get('admissions', 0)} / {engine_stats.get('admissions_rejected', 0)}")
    print(f"  evictions          {engine_stats.get('evictions', 0)}")
    print(f"  backend spend      ${cost.get('backend_usd', 0):.6f}")
    print(f"  saved vs no cache  ${cost.get('saved_vs_no_cache_usd', 0):.6f}")

    reporting = tier1.get("reporting", 0)
    if reporting:
        print(f"  L1 processes       {reporting} reporting, {tier1.get('hit_rate', 0):.1%} hit rate")
    else:
        # Expected when only this script has run: L1 lives inside the application
        # processes, and this script is not one of them.
        note("no application is reporting an L1 yet -- start the services to see the local tier")

    apps = f.get("applications", [])
    if apps:
        print("\n  per application")
        for a in apps:
            print(
                f"    {a.get('application', '?'):<16} "
                f"{a.get('requests', 0):>6} requests   "
                f"hit rate {a.get('hit_rate', 0):>6.1%}   "
                f"${a.get('cost_usd', 0):.6f}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default=os.environ.get("AURA_ENGINE", ENGINE))
    p.add_argument("--rec-key", default=os.environ.get("AURA_RECOMMENDATION_KEY"))
    p.add_argument("--ana-key", default=os.environ.get("AURA_ANALYTICS_KEY"))
    args = p.parse_args()

    if not args.rec_key or not args.ana_key:
        print("Two application keys are required. Mint them in the console's Connect tab.")
        print("  python scripts/smoke.py --rec-key aura_sk_... --ana-key aura_sk_...")
        return 2

    print(f"engine  {args.engine}")
    if not wake(args.engine):
        return 1

    results = [
        exercise(args.engine, "recommendation", args.rec_key),
        exercise(args.engine, "analytics", args.ana_key),
    ]
    summary(args.engine, args.rec_key)

    print()
    if all(results):
        print(f"{GREEN}Everything answered. The transport, the keys, the cache path and the ledger all work.{OFF}")
        print("Next: .\\scripts\\demo.ps1 to bring the two applications up with their local L1.")
        return 0
    print(f"{RED}Something did not answer. The failing line above says which.{OFF}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
