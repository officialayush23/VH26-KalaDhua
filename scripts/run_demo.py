"""Bring the entire demo up with one command, and say what it is doing at every step.

    python scripts/run_demo.py --rec-key aura_sk_... --ana-key aura_sk_...

What it starts, in order, checking each before moving to the next:

    1. dependencies          the interpreter can import what the services need
    2. the engine            woken, because a free instance sleeps and the first
                             request after that pays about a minute of cold start
    3. the keys              each one is accepted and attributed to its application
    4. the services          recommendation on 8101, analytics on 8102, local
    5. registration          the engine has now seen both of them call it
    6. traffic               a synthetic customer population, optional
    7. the local tier        both processes are reporting their in-process L1
    8. the pages             printed, and opened if this machine has a browser

Nothing here is a mock. The services are the real services, the engine is the deployed
one, and the requests between them cross a real network -- which is the only arrangement
in which the difference between the local tier and the shared one is visible rather than
asserted.

Ctrl+C shuts everything down. The two service windows write their logs to runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPS = ROOT / "apps"
RUNS = ROOT / "runs"

ENGINE = "https://vh26-kaladhua.onrender.com"
CONSOLE = "https://universe-ten-iota.vercel.app"

SERVICES = [
    ("recommendation", "recommendation.main", 8101),
    ("analytics", "analytics.main", 8102),
]

# Windows terminals only understand ANSI once someone asks them to.
if os.name == "nt":
    os.system("")

G, R, Y, D, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

_step = 0
TOTAL = 8


def step(title: str) -> None:
    global _step
    _step += 1
    print(f"\n{D}[{_step}/{TOTAL}]{OFF} {title}")


def ok(t: str) -> None:
    print(f"      {G}ok{OFF}    {t}")


def fail(t: str) -> None:
    print(f"      {R}fail{OFF}  {t}")


def warn(t: str) -> None:
    print(f"      {Y}note{OFF}  {t}")


def dim(t: str) -> None:
    print(f"            {D}{t}{OFF}")


def call(method: str, url: str, *, key: str | None = None, body: dict | None = None,
         timeout: float = 20.0) -> tuple[int, dict]:
    """One request. An error status is a value, not an exception: a 404 from the cache is
    a miss and a 401 is a fact, and both are worth printing rather than raising."""
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
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def until(predicate, *, seconds: float, every: float = 1.0, tick=None):
    """Poll until the predicate returns something truthy, or give up. Returns its value."""
    deadline = time.monotonic() + seconds
    n = 0
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        n += 1
        if tick and n % max(1, int(5 / every)) == 0:
            tick(int(deadline - time.monotonic()))
        time.sleep(every)
    return None


# ----------------------------------------------------------------------------- the steps


def check_dependencies(python: str) -> bool:
    step("Checking the interpreter can run the services")
    probe = "import httpx, starlette, uvicorn, pydantic_settings; print('ok')"
    try:
        out = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=60)
    except Exception as e:
        fail(f"could not run {python}: {e}")
        return False
    if out.returncode != 0:
        fail("a package the services need is missing")
        dim((out.stderr or "").strip().splitlines()[-1] if out.stderr else "")
        dim(f'fix it with:  "{python}" -m pip install -r "{APPS / "requirements.txt"}"')
        return False
    ok(f"{python}")
    return True


def wake_engine(engine: str) -> bool:
    step("Waking the engine")
    dim("a sleeping free instance takes up to a minute; better to pay it now than on stage")
    got = until(
        lambda: call("GET", f"{engine}/healthz", timeout=12)[0] == 200,
        seconds=150,
        every=5,
        tick=lambda left: dim(f"still starting, {left}s before giving up"),
    )
    if got:
        ok(f"{engine} is awake")
        return True
    fail("the engine never answered /healthz")
    return False


def check_key(engine: str, app: str, key: str) -> bool:
    """Write one expensive object and read it back, as that application."""
    stamp = int(time.time())
    k = f"startup:{app}:{stamp}"
    status, body = call(
        "PUT",
        f"{engine}/v1/cache/{k}",
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
        fail(f"{app}: the key was refused -- {body.get('error', body)}")
        return False
    if status >= 400 or status == 0:
        fail(f"{app}: write failed ({status}) -- {body.get('error', body)}")
        return False
    status, body = call("GET", f"{engine}/v1/cache/{k}?application={app}", key=key)
    if status == 200 and body.get("hit"):
        ok(f"{app}: wrote 48 KB / 412 ms, read it back from {body.get('layer', 'L2')}")
        return True
    if status == 404:
        # A legitimate decision rather than a transport failure, and worth naming.
        warn(f"{app}: the cache declined to keep it -- {body.get('reason', 'no reason given')}")
        dim("the key works; the admission gate simply did not want that object")
        return True
    fail(f"{app}: read failed ({status}) -- {body.get('error', body)}")
    return False


def check_keys(engine: str, keys: dict[str, str]) -> bool:
    step("Checking both keys against the engine")
    return all(check_key(engine, app, key) for app, key in keys.items())


def tail(path: Path, lines: int = 12) -> None:
    """The end of a log, which is where the reason a process died actually is."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except Exception:
        return
    for line in text[-lines:]:
        dim(line[:160])


def start_services(
    python: str, engine: str, keys: dict[str, str]
) -> tuple[list[subprocess.Popen], int]:
    step("Starting the two applications on this machine")
    RUNS.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []
    for app, module, port in SERVICES:
        env = {
            **os.environ,
            "AURA_APPS_AURA_BASE_URL": engine,
            "AURA_API_KEY": keys[app],
            "PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
        log = open(RUNS / f"{app}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(
            [python, "-m", module],
            cwd=str(APPS),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        ok(f"{app} starting on port {port}  {D}(log: runs/{app}.log){OFF}")

    healthy = 0
    for (app, _module, port), proc in zip(SERVICES, procs):
        # A process that has already exited will never answer, so stop waiting for it. The
        # difference between reading the reason in one second and in forty-five is the
        # difference between fixing it and losing patience with it.
        def ready(p=port, proc=proc):
            if proc.poll() is not None:
                return "dead"
            return call("GET", f"http://localhost:{p}/health", timeout=3)[0] == 200

        alive = until(ready, seconds=45, every=1)
        if alive == "dead":
            alive = False
        if alive:
            ok(f"{app} is answering on http://localhost:{port}")
            healthy += 1
        else:
            fail(f"{app} did not come up. The end of runs/{app}.log:")
            tail(RUNS / f"{app}.log")
    return procs, healthy


def check_registration(engine: str, key: str) -> bool:
    step("Asking the engine who is talking to it")
    dim("nothing was registered: the key is the identity, so a service appears by calling")

    def seen():
        status, body = call("GET", f"{engine}/v1/connections", key=key)
        if status != 200:
            return None
        live = [k for k in body.get("keys", []) if k.get("connected")]
        return live if len(live) >= 2 else None

    live = until(seen, seconds=40, every=2)
    if not live:
        warn("the engine has not seen both applications yet; they register on first call")
        return False
    for k in live:
        traffic = k.get("traffic") or {}
        ok(f"{k['application']}  {traffic.get('requests', 0)} requests, key {k.get('hint', '')}...")
    return True


def check_local_tier(engine: str, key: str) -> bool:
    step("Waiting for the local tier to report")
    dim("a request served from a local copy never reaches the engine, so L1 is posted, not observed")

    def reporting():
        status, body = call("GET", f"{engine}/v1/stats", key=key)
        if status != 200:
            return None
        t1 = body.get("tier1") or {}
        return t1 if t1.get("reporting", 0) >= 1 else None

    t1 = until(reporting, seconds=30, every=3)
    if not t1:
        warn("no L1 report yet -- the services post theirs every five seconds")
        return False
    ok(f"{t1['reporting']} process(es) reporting, {t1.get('entries', 0)} objects held locally")
    return True


def start_traffic(python: str, users: int, rps: float, duration: float, scenario: str | None):
    step("Starting the customer population")
    if users <= 0:
        warn("skipped (--users 0); drive it yourself from the control panel")
        return None
    argv = [python, "-m", "simulator.driver", "--users", str(users),
            "--rps", str(rps), "--duration", str(duration)]
    if scenario and scenario != "none":
        argv += ["--scenario", scenario]
    RUNS.mkdir(exist_ok=True)
    log = open(RUNS / "traffic.log", "w", encoding="utf-8")
    proc = subprocess.Popen(argv, cwd=str(APPS), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            stdout=log, stderr=subprocess.STDOUT)
    ok(f"{users} simulated customers at {rps:g} req/s for {duration:g}s  {D}(log: runs/traffic.log){OFF}")
    dim("stop or reshape it any time from the control panel")
    return proc


def show(engine: str, open_pages: bool) -> None:
    step("Everything is up")
    rows = [
        ("console (the dashboard)", CONSOLE),
        ("storefront - recommendation", "http://localhost:8101/"),
        ("storefront - analytics", "http://localhost:8102/"),
        ("traffic control panel", "http://localhost:8101/control"),
        ("engine", engine),
    ]
    print()
    for label, url in rows:
        print(f"      {label:<30} {url}")

    print(f"""
      {D}The order that tells the story{OFF}

      1. console, Connect tab      both applications listed as connected
      2. a storefront              click products; served_from goes origin, then l1
      3. console, Evidence tab     six charts filling as the traffic runs
      4. control panel             flash crowd, then price change, then model redeploy
      5. console, Decisions tab    the sentence explaining any single decision

      {D}Ctrl+C stops the services. Logs are in runs/.{OFF}
""")
    if open_pages:
        for _label, url in rows[:4]:
            try:
                webbrowser.open_new_tab(url)
                time.sleep(0.4)
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default=os.environ.get("AURA_ENGINE", ENGINE))
    p.add_argument("--rec-key", default=os.environ.get("AURA_RECOMMENDATION_KEY"))
    p.add_argument("--ana-key", default=os.environ.get("AURA_ANALYTICS_KEY"))
    p.add_argument("--users", type=int, default=4000, help="0 to start no traffic")
    p.add_argument("--rps", type=float, default=40.0)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--scenario", default=None,
                   help="flash_crowd, price_change, model_redeploy, popularity_shift")
    p.add_argument("--no-open", action="store_true", help="do not open browser tabs")
    args = p.parse_args()

    if not args.rec_key or not args.ana_key:
        print("Two application keys are required. Mint them in the console's Connect tab.\n")
        print("  python scripts/run_demo.py --rec-key aura_sk_... --ana-key aura_sk_...")
        return 2

    keys = {"recommendation": args.rec_key, "analytics": args.ana_key}
    # The virtual environment if there is one, this interpreter otherwise.
    venv = ROOT / "aura" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = str(venv) if venv.exists() else sys.executable

    print(f"\nAURA demo\n{D}engine {args.engine}{OFF}")

    if not check_dependencies(python):
        return 1
    if not wake_engine(args.engine):
        return 1
    if not check_keys(args.engine, keys):
        fail("the keys did not work, so the applications would not either")
        return 1

    procs, healthy = start_services(python, args.engine, keys)
    if healthy == 0:
        # Carrying on would print a page of URLs that all lead nowhere, which is a worse
        # answer than stopping and saying which log to read.
        fail("neither service started, so there is nothing to demonstrate")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        return 1

    check_registration(args.engine, args.rec_key)
    traffic = start_traffic(python, args.users, args.rps, args.duration, args.scenario)
    if traffic:
        procs.append(traffic)
    check_local_tier(args.engine, args.rec_key)
    show(args.engine, open_pages=not args.no_open)

    try:
        while True:
            time.sleep(1)
            for proc in list(procs):
                if proc.poll() is not None and proc is not traffic:
                    warn(f"a service exited with code {proc.returncode} -- check runs/")
                    procs.remove(proc)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        for proc in procs:
            if proc.poll() is None:
                # Terminate rather than kill: the driver flushes its per-request log on the
                # way out, and that file is the evidence for everything the run claims.
                try:
                    proc.send_signal(signal.SIGTERM if os.name != "nt" else signal.CTRL_BREAK_EVENT)
                except Exception:
                    proc.terminate()
        time.sleep(1.5)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
