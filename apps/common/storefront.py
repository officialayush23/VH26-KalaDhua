"""A browsable storefront for the two example applications.

Why this exists
---------------

The engine's own console shows the cache's internal state: hit rates, policy weights, the
audit log. That answers "what is the cache doing" but not "what does it do *for me*", and
those are different questions. A judge watching a dashboard has to take the numbers on
trust. A judge who clicks a product, sees the page take 900 ms, clicks back, and sees the
same page return in 3 ms with a line saying *"served from cache, saved $0.000251 of GPU
time"* has verified it themselves.

So each application serves a small storefront of its own, and every response carries the
cache's verdict for that exact request: hit or miss, where it came from, what the rebuild
cost, and what keeping it saved. Nothing here is a mock-up -- the page calls the same
`/work/{id}` endpoint the load generators call, through the same cache, and the numbers
shown are the ones the engine returned.

The page is deliberately one file with no build step and no framework. It is served by the
application it belongs to, so there is nothing extra to deploy, nothing to keep in sync, and
opening the app in a browser is the whole setup.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse

# --------------------------------------------------------------------------- shared shell

_STYLE = """
:root {
  --ink: #141a24; --ink-soft: #1e2933; --paper: #ffffff; --wash: #f1f4f7;
  --line: #d8dfe6; --body: #31404e; --muted: #6b7c8c;
  --hit: #1b9c86; --hit-soft: #d9f0eb; --miss: #d97a16; --miss-soft: #fbe9d4;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--wash); color: var(--body);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
header {
  background: var(--ink); color: #fff; padding: 20px 28px;
  display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
}
header h1 { margin: 0; font-size: 20px; letter-spacing: .3px; }
header .sub { color: #9aaab8; font-size: 13px; }
header a { color: var(--hit); text-decoration: none; font-size: 13px; margin-left: auto; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 28px 60px; }
.bar {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 22px;
}
.stat { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
.stat .v { font-size: 22px; font-weight: 650; color: var(--ink); margin-top: 2px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }
.card {
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px; cursor: pointer; transition: border-color .12s, transform .12s;
}
.card:hover { border-color: var(--ink-soft); transform: translateY(-1px); }
.card h3 { margin: 0 0 4px; font-size: 14px; color: var(--ink); }
.card .meta { font-size: 12px; color: var(--muted); }
.verdict { margin-top: 10px; font-size: 12px; border-radius: 6px; padding: 6px 8px; display: none; }
.verdict.hit { background: var(--hit-soft); color: #0f6b5c; display: block; }
.verdict.miss { background: var(--miss-soft); color: #8a4c0b; display: block; }
.verdict.err { background: #f7e2de; color: #8c3222; display: block; }
.panel {
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 22px;
}
.panel h2 { margin: 0 0 4px; font-size: 15px; color: var(--ink); }
.panel p { margin: 0 0 12px; font-size: 13px; color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; color: var(--muted); font-size: 11px;
     text-transform: uppercase; letter-spacing: .5px; padding: 6px 8px; }
td { padding: 7px 8px; border-top: 1px solid var(--line); }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
.tag { font-size: 11px; padding: 2px 7px; border-radius: 20px; font-weight: 600; }
.tag.hit { background: var(--hit-soft); color: #0f6b5c; }
.tag.miss { background: var(--miss-soft); color: #8a4c0b; }
.hint { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
button {
  font: inherit; font-size: 13px; padding: 7px 13px; border-radius: 7px;
  border: 1px solid var(--line); background: var(--paper); color: var(--ink); cursor: pointer;
}
button:hover { border-color: var(--ink-soft); }
button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
"""

_SCRIPT = """
// Every figure on this page comes from the response to the request the click made. Nothing
// is simulated in the browser, and nothing is averaged over a window the viewer cannot see.
const log = [];
let served = 0, hits = 0, savedUsd = 0, spentUsd = 0, msSaved = 0;

function usd(v) {
  if (!v) return "$0";
  if (v < 0.0001) return (v * 100).toFixed(4) + "c";
  if (v < 0.01) return (v * 100).toFixed(2) + "c";
  return "$" + v.toFixed(4);
}

function paintStats() {
  document.getElementById("served").textContent = served;
  document.getElementById("hitrate").textContent =
    served ? (100 * hits / served).toFixed(0) + "%" : "--";
  document.getElementById("saved").textContent = usd(savedUsd);
  document.getElementById("spent").textContent = usd(spentUsd);
  document.getElementById("timesaved").textContent =
    msSaved > 1000 ? (msSaved / 1000).toFixed(1) + "s" : Math.round(msSaved) + "ms";
}

function paintLog() {
  const rows = log.slice(0, 12).map(function (e) {
    return "<tr><td>" + e.what + "</td>" +
      "<td><span class='tag " + (e.hit ? "hit" : "miss") + "'>" +
        (e.hit ? "cache" : "rebuilt") + "</span></td>" +
      "<td class='n'>" + e.ms.toFixed(1) + " ms</td>" +
      "<td class='n'>" + (e.kb ? e.kb.toFixed(0) + " KB" : "--") + "</td>" +
      "<td class='n'>" + (e.hit ? usd(e.saved) : "-" + usd(e.cost)) + "</td></tr>";
  }).join("");
  document.getElementById("log").innerHTML = rows ||
    "<tr><td colspan='5' style='color:var(--muted)'>Nothing yet. Click something above.</td></tr>";
}

async function ask(url, label, el) {
  const t0 = performance.now();
  try {
    const res = await fetch(url);
    const body = await res.json();
    const ms = performance.now() - t0;
    const hit = body.served_from === "cache";
    const cost = body.regen_cost_usd || 0;

    served += 1;
    if (hit) {
      hits += 1;
      // What the rebuild would have cost, had the cache not had it. The application
      // reports this per object, so it is the real figure for this key rather than an
      // average over the run.
      savedUsd += cost;
      msSaved += Math.max(0, (body.regen_ms_if_missed || 0) - ms);
    } else {
      spentUsd += cost;
    }

    log.unshift({
      what: label, hit: hit, ms: ms,
      kb: (body.size_bytes || 0) / 1024, saved: cost, cost: cost
    });
    paintStats(); paintLog();

    if (el) {
      el.className = "verdict " + (hit ? "hit" : "miss");
      el.textContent = hit
        ? "Served from cache in " + ms.toFixed(0) + " ms - saved " + usd(cost) + " of rebuild"
        : "Rebuilt in " + ms.toFixed(0) + " ms, cost " + usd(cost) + " - now cached";
    }
    return body;
  } catch (err) {
    if (el) { el.className = "verdict err"; el.textContent = "Request failed: " + err.message; }
    return null;
  }
}
"""


def _shell(title: str, subtitle: str, body: str, extra_script: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{_STYLE}</style></head>
<body>
<header>
  <h1>{title}</h1>
  <span class="sub">{subtitle}</span>
  <a href="/stats">raw stats JSON</a>
</header>
<main>
  <div class="bar">
    <div class="stat"><div class="k">Requests</div><div class="v" id="served">0</div></div>
    <div class="stat"><div class="k">Served from cache</div><div class="v" id="hitrate">--</div></div>
    <div class="stat"><div class="k">Backend work avoided</div><div class="v" id="saved">$0</div></div>
    <div class="stat"><div class="k">Backend work paid for</div><div class="v" id="spent">$0</div></div>
    <div class="stat"><div class="k">Waiting avoided</div><div class="v" id="timesaved">0ms</div></div>
  </div>
  {body}
  <div class="panel">
    <h2>What just happened</h2>
    <p>One line per request you made, newest first. The verdict is the cache's, not this page's.</p>
    <table>
      <thead><tr><th>request</th><th>outcome</th><th>latency</th><th>size</th><th>money</th></tr></thead>
      <tbody id="log"></tbody>
    </table>
  </div>
</main>
<script>{_SCRIPT}
{extra_script}
paintStats(); paintLog();
</script>
</body></html>"""


# ----------------------------------------------------------------------- recommendation

_SEGMENTS = ("default", "mobile", "web", "loyalty")


def recommendation_page(users: int) -> str:
    """A shopfront where each shopper's ranking is an expensive, cacheable object."""
    cards = "".join(
        f"""<div class="card" onclick="open_user({u})">
              <h3>Shopper #{u}</h3>
              <div class="meta">segment: {_SEGMENTS[u % len(_SEGMENTS)]}</div>
              <div class="verdict" id="v{u}"></div>
            </div>"""
        for u in (7, 42, 108, 256, 512, 1024, 2048, 4096)
    )
    body = f"""
  <div class="panel">
    <h2>Personalised recommendations</h2>
    <p>Each shopper's ranking is rebuilt by a real model pass -- hundreds of milliseconds of
       CPU and accelerator time, producing a payload of a megabyte or more. Click the same
       shopper twice: the first click pays for it, the second does not.</p>
    <div class="row">
      <button class="primary" onclick="browse()">Simulate 20 shoppers browsing</button>
      <button onclick="click_through(42)">Shopper #42 clicks a product</button>
    </div>
    <div class="hint">A click advances that shopper's epoch, which changes their cache key.
      Their old ranking is not deleted -- it simply stops being asked for, and ages out.</div>
  </div>
  <div class="grid">{cards}</div>"""

    script = f"""
const EPOCH = {{}};
const USERS = {users};

async function open_user(u) {{
  await ask("/work/" + u + "?epoch=" + (EPOCH[u] || 0), "shopper #" + u, document.getElementById("v" + u));
}}

async function click_through(u) {{
  // A click is a write. The epoch moves, so the next read is a guaranteed miss on a new
  // key -- invalidation by construction rather than by deletion.
  EPOCH[u] = (EPOCH[u] || 0) + 1;
  const el = document.getElementById("v" + u);
  if (el) {{ el.className = "verdict miss"; el.textContent = "clicked - epoch now " + EPOCH[u]; }}
  await open_user(u);
}}

async function browse() {{
  // Zipf-ish: a few shoppers are far more active than the rest, which is what gives the
  // cache something worth keeping.
  const hot = [7, 42, 108, 256];
  for (let i = 0; i < 20; i++) {{
    const u = Math.random() < 0.7
      ? hot[Math.floor(Math.random() * hot.length)]
      : Math.floor(Math.random() * USERS);
    await ask("/work/" + u + "?epoch=" + (EPOCH[u] || 0), "shopper #" + u, document.getElementById("v" + u));
  }}
}}"""
    return _shell(
        "Storefront",
        "recommendation service - large objects, expensive to rebuild",
        body,
        script,
    )


# ---------------------------------------------------------------------------- analytics


def analytics_page(regions: int) -> str:
    """A dashboard where each tile is a genuine SQL aggregate over the order book."""
    tiles = "".join(
        f"""<div class="card" onclick="open_tile({i})">
              <h3>{name}</h3>
              <div class="meta">{detail}</div>
              <div class="verdict" id="v{i}"></div>
            </div>"""
        for i, (name, detail) in enumerate(
            [
                ("Revenue by region", "all regions, 30 days"),
                ("Top products", "region 1, 7 days"),
                ("Daily trend", "region 2, 30 days"),
                ("Category matrix", "all regions, 90 days"),
                ("Cohort retention", "region 3, 365 days"),
                ("Revenue by region", "all regions, 1 day"),
                ("Top products", "region 4, 30 days"),
                ("Daily trend", "region 5, 7 days"),
            ]
        )
    )
    body = f"""
  <div class="panel">
    <h2>Operations dashboard</h2>
    <p>Every tile is a real aggregate over the order book -- joins across orders, order
       lines, products and regions. The database does the work; the cache decides whether it
       has to do it again.</p>
    <div class="row">
      <button class="primary" onclick="refresh_all()">Open the whole dashboard</button>
      <button onclick="price_change()">A price changes in the database</button>
    </div>
    <div class="hint">Each tile declares the rows and tables it was computed from. A price
      change invalidates exactly the tiles built from that region -- not the whole cache,
      and not nothing.</div>
  </div>
  <div class="grid">{tiles}</div>"""

    script = f"""
const REGIONS = {regions};

async function open_tile(i) {{
  await ask("/work/" + i, "tile " + i, document.getElementById("v" + i));
}}

async function refresh_all() {{
  for (let i = 0; i < 8; i++) await open_tile(i);
}}

async function price_change() {{
  // Posts the tag the database trigger would emit. Everything downstream of that row is
  // dropped; everything else keeps its place.
  try {{
    const res = await fetch("/invalidate", {{
      method: "POST",
      headers: {{ "content-type": "application/json" }},
      body: JSON.stringify({{ tags: ["row:region:1", "table:app_products"], mode: "hard" }})
    }});
    const body = await res.json();
    const n = body.keys_hard === undefined ? "?" : body.keys_hard;
    log.unshift({{ what: "price change in region 1", hit: false, ms: 0, kb: 0, saved: 0, cost: 0 }});
    paintLog();
    alert("Invalidated " + n + " cached tile(s) built from region 1.\\n\\n" +
          "Re-open the dashboard: the tiles that depended on it rebuild, the rest are still cached.");
  }} catch (err) {{
    alert("Invalidation failed: " + err.message);
  }}
}}"""
    return _shell(
        "Operations dashboard",
        "analytics service - small objects, expensive queries",
        body,
        script,
    )


# ------------------------------------------------------------------------------- routes


def storefront_route(render, **kwargs):  # noqa: ANN001, ANN201
    """Build the Starlette handler for one application's page."""

    async def page(request: Request) -> HTMLResponse:
        # No caching of the shell itself: the whole point is that the numbers on it are
        # this request's, and a browser-cached page would quietly show someone else's.
        return HTMLResponse(
            render(**kwargs),
            headers={"cache-control": "no-store"},
        )

    return page
