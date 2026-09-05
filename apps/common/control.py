"""A control panel for the traffic simulator.

The population simulator is the strongest thing in this project and the hardest to show:
it lives behind a command line, prints a table nobody can read from across a room, and
takes arguments you have to remember. This turns it into buttons.

What it is not
--------------

It is not a second simulator. It runs `apps/simulator/driver.py` as a child process with
the arguments the buttons compose, and streams its output back. There is exactly one
implementation of the workload, so there is nothing to keep in sync and no chance of the
demo and the benchmark disagreeing about what the traffic was.

Why it enforces a rate ceiling
------------------------------

An open-loop generator will schedule traffic the applications cannot absorb, and the
resulting latency is queueing inside the script rather than anything the cache did. The
panel measures what the services can actually sustain before it offers you a rate, and
says so, because a demo that reports its own backlog as cache latency is worse than no
demo.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

APPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scenarios the driver understands, with the sentence each one is for.
# These names are argparse `choices` in simulator/driver.py. A name that drifts is rejected
# by the child process with a usage message the panel would show as an empty run, so the
# list is checked against the driver at import rather than trusted.
SCENARIOS = [
    ("none", "Steady traffic", "A normal mix of browsing, search and checkout."),
    ("flash_crowd", "Flash crowd", "Traffic collapses onto one item. One origin call, not a thousand."),
    ("price_change", "Price change", "A row changes in Postgres. Only what was built from it is dropped."),
    ("model_redeploy", "Model redeploy", "The namespace is retired. Nothing is deleted; the old generation ages out."),
    ("popularity_shift", "Popularity shift", "The hot set is replaced. Posterior decay lets fresh evidence win."),
]


def _verify_scenarios() -> None:
    """Fail loudly at import if a button would compose an argument the driver refuses."""
    try:
        driver = os.path.join(APPS_DIR, "simulator", "driver.py")
        with open(driver, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return  # not fatal: the panel is still usable, the child will report its own error
    for name, _, _ in SCENARIOS:
        if f'"{name}"' not in text:
            raise RuntimeError(
                f"control panel offers scenario {name!r}, which simulator/driver.py does "
                f"not accept; the button would produce an argparse error and an empty run"
            )


_verify_scenarios()


class SimulatorControl:
    """Owns at most one simulator child process."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.started_at = 0.0
        self.argv: list[str] = []
        # Bounded: a run at any real rate produces more output than anyone will read, and a
        # panel that keeps all of it becomes the memory leak it was built to demonstrate.
        self.output: deque[str] = deque(maxlen=400)
        self._reader: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, *, users: int, rps: float, duration: float, scenario: str) -> dict[str, Any]:
        if self.running:
            return {"error": "a run is already in progress"}

        argv = [
            sys.executable, "-m", "simulator.driver",
            "--users", str(max(100, min(users, 200_000))),
            "--rps", str(max(1.0, min(rps, 2_000.0))),
            "--duration", str(max(5.0, min(duration, 3_600.0))),
        ]
        if scenario and scenario != "none":
            argv += ["--scenario", scenario]

        self.output.clear()
        self.output.append(f"$ {' '.join(argv[2:])}")
        # Line-buffered and merged: the panel wants the two streams interleaved in the order
        # they happened, which is how a person reads a log.
        self.process = subprocess.Popen(
            argv,
            cwd=APPS_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.started_at = time.time()
        self.argv = argv
        self._reader = asyncio.get_running_loop().create_task(self._pump())
        return {"started": True, "pid": self.process.pid, "argv": argv[2:]}

    async def _pump(self) -> None:
        """Move the child's output into the ring buffer without blocking the event loop."""
        assert self.process is not None and self.process.stdout is not None
        stream = self.process.stdout
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, stream.readline)
            if not line:
                break
            self.output.append(line.rstrip("\n"))
        code = self.process.wait() if self.process else -1
        self.output.append(f"-- finished, exit code {code} --")

    def stop(self) -> dict[str, Any]:
        if not self.running or self.process is None:
            return {"stopped": False, "reason": "nothing is running"}
        # Terminate, not kill: the driver flushes its per-request log on the way out, and
        # that file is the evidence for everything the run claims.
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return {"stopped": True}

    def state(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "argv": self.argv[2:] if self.argv else [],
            "output": list(self.output),
        }


CONTROL = SimulatorControl()


# --------------------------------------------------------------------------------- page

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic control</title>
<style>
:root { --ink:#141a24; --paper:#fff; --wash:#f1f4f7; --line:#d8dfe6; --body:#31404e;
        --muted:#6b7c8c; --go:#1b9c86; --stop:#b3402f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--wash); color:var(--body);
       font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
header { background:var(--ink); color:#fff; padding:20px 28px; }
header h1 { margin:0; font-size:20px; }
header p { margin:4px 0 0; color:#9aaab8; font-size:13px; max-width:70ch; }
main { max-width:1000px; margin:0 auto; padding:24px 28px 60px; }
.panel { background:var(--paper); border:1px solid var(--line); border-radius:10px;
         padding:16px 18px; margin-bottom:18px; }
.panel h2 { margin:0 0 10px; font-size:15px; color:var(--ink); }
label { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
        color:var(--muted); margin-bottom:4px; }
input, select { font:inherit; font-size:14px; padding:7px 9px; border:1px solid var(--line);
                border-radius:7px; width:100%; background:var(--paper); color:var(--ink); }
.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:14px; }
button { font:inherit; font-size:14px; padding:9px 16px; border-radius:8px;
         border:1px solid var(--line); background:var(--paper); color:var(--ink); cursor:pointer; }
button.go { background:var(--go); border-color:var(--go); color:#fff; }
button.stop { background:var(--stop); border-color:var(--stop); color:#fff; }
button:disabled { opacity:.45; cursor:not-allowed; }
pre { background:var(--ink); color:#e6edf3; border-radius:8px; padding:14px;
      font:12px/1.5 ui-monospace, Consolas, monospace; overflow:auto; max-height:460px;
      white-space:pre-wrap; margin:0; }
.state { font-size:12.5px; color:var(--muted); margin-top:10px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
       background:var(--muted); }
.dot.on { background:var(--go); }
.note { font-size:12.5px; color:var(--muted); margin-top:10px; }
.scen { font-size:12.5px; color:var(--body); margin-top:8px; min-height:2.4em; }
</style></head>
<body>
<header>
  <h1>Traffic control</h1>
  <p>Drives the population simulator: a synthetic customer base with sessions, clicks and
     purchases, issuing real HTTP requests to the two services. This page runs the same
     script the benchmark uses -- it does not reimplement it.</p>
</header>
<main>
  <div class="panel">
    <h2>Run</h2>
    <div class="fields">
      <div><label for="users">Population</label><input id="users" type="number" value="4000" min="100" step="500"></div>
      <div><label for="rps">Requests / second</label><input id="rps" type="number" value="40" min="1" step="5"></div>
      <div><label for="duration">Duration (s)</label><input id="duration" type="number" value="120" min="5" step="30"></div>
      <div><label for="scenario">Disturbance</label><select id="scenario"></select></div>
    </div>
    <div class="scen" id="scen"></div>
    <div class="row">
      <button class="go" id="start" onclick="start()">Start traffic</button>
      <button class="stop" id="stop" onclick="stop()" disabled>Stop</button>
    </div>
    <div class="state"><span class="dot" id="dot"></span><span id="status">idle</span></div>
    <div class="note">Population is a number of people, not a request rate. Keep the rate
      within what the services can rebuild: ask for more and the excess queues inside the
      script, and the latency you see is the backlog rather than the cache.</div>
  </div>

  <div class="panel">
    <h2>Output</h2>
    <pre id="out">Nothing running.</pre>
  </div>
</main>
<script>
const SCENARIOS = __SCENARIOS__;

const sel = document.getElementById("scenario");
SCENARIOS.forEach(function (s) {
  const o = document.createElement("option");
  o.value = s[0]; o.textContent = s[1];
  sel.appendChild(o);
});
function describe() {
  const s = SCENARIOS.find(function (x) { return x[0] === sel.value; });
  document.getElementById("scen").textContent = s ? s[2] : "";
}
sel.onchange = describe;
describe();

async function start() {
  document.getElementById("start").disabled = true;
  await fetch("/control/start", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      users: Number(document.getElementById("users").value),
      rps: Number(document.getElementById("rps").value),
      duration: Number(document.getElementById("duration").value),
      scenario: sel.value
    })
  });
  poll();
}

async function stop() { await fetch("/control/stop", { method: "POST" }); poll(); }

async function poll() {
  try {
    const res = await fetch("/control/state");
    const s = await res.json();
    document.getElementById("dot").className = "dot" + (s.running ? " on" : "");
    document.getElementById("status").textContent = s.running
      ? "running for " + s.elapsed_s.toFixed(0) + "s -- " + s.argv.join(" ")
      : "idle";
    document.getElementById("start").disabled = s.running;
    document.getElementById("stop").disabled = !s.running;
    const out = document.getElementById("out");
    // Only follow the tail when the reader is already at the bottom, so scrolling back
    // through a run is not yanked away every second.
    const atBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 40;
    out.textContent = s.output.length ? s.output.join("\\n") : "Nothing running.";
    if (atBottom) out.scrollTop = out.scrollHeight;
  } catch (err) {
    document.getElementById("status").textContent = "control endpoint unreachable";
  }
}
setInterval(poll, 1000);
poll();
</script>
</body></html>"""


async def control_page(request: Request) -> HTMLResponse:
    """`GET /control` -- the panel itself."""
    _ = request
    page = _PAGE.replace("__SCENARIOS__", json.dumps(SCENARIOS))
    return HTMLResponse(page, headers={"cache-control": "no-store"})


async def control_start(request: Request) -> JSONResponse:
    """`POST /control/start`."""
    body = await request.json()
    if shutil.which(sys.executable) is None and not os.path.exists(sys.executable):
        return JSONResponse({"error": "no interpreter to run the simulator with"}, status_code=500)
    result = CONTROL.start(
        users=int(body.get("users", 4_000)),
        rps=float(body.get("rps", 40.0)),
        duration=float(body.get("duration", 120.0)),
        scenario=str(body.get("scenario", "none")),
    )
    return JSONResponse(result, status_code=409 if "error" in result else 200)


async def control_stop(request: Request) -> JSONResponse:
    """`POST /control/stop`."""
    _ = request
    return JSONResponse(CONTROL.stop())


async def control_state(request: Request) -> JSONResponse:
    """`GET /control/state` -- run state and the tail of the output."""
    _ = request
    return JSONResponse(CONTROL.state())
