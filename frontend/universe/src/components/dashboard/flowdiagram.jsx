import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { MotionPathPlugin } from "gsap/MotionPathPlugin"

import { bytes, cn, ms, pct, usd } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

gsap.registerPlugin(MotionPathPlugin)

const W = 1200
const H = 600

/// The request path as it is actually built, which is not the diagram most cache talks
/// draw. Three things here are easy to get wrong and all three are wrong on purpose in the
/// version this replaced:
///
/// * **L1 lives in the application, not in the cache.** `apps/common/l1.py` is a real
///   byte-capped LRU holding values inside each app process. The engine's own recency
///   window admits objects; it stores nothing. Drawing that window as a tier claims a
///   hierarchy the code does not have.
/// * **AURA never calls the origin.** Only the application knows how to rebuild an object.
///   On a miss the cache hands out a lease saying who should rebuild, and everyone else
///   waits. That is why the miss arrow turns around and goes back to the application.
/// * **Supabase is on two different planes.** The analytics application queries it on the
///   request path; the engine reads model bundles and writes benchmark results off it. One
///   box for both would imply the cache sits in front of its own control plane.
const N = {
  client: { x: 40, y: 262, w: 132, h: 70, label: "Customer", sub: "simulator or user" },
  app: { x: 232, y: 262, w: 166, h: 70, label: "Application", sub: "+ in-process L1" },
  aura: { x: 470, y: 262, w: 170, h: 70, label: "AURA L2", sub: "scored, shared pool" },
  lease: { x: 470, y: 96, w: 170, h: 66, label: "Single flight", sub: "one rebuilder, rest wait" },
  decide: { x: 470, y: 440, w: 170, h: 70, label: "Decision engine", sub: "admit, refuse, evict" },
  db: { x: 940, y: 150, w: 176, h: 70, label: "Supabase Postgres", sub: "analytics queries" },
  compute: { x: 940, y: 278, w: 176, h: 70, label: "CPU / GPU work", sub: "ranking, media" },
  invalidator: { x: 712, y: 452, w: 156, h: 66, label: "Invalidator", sub: "writes to versions" },
  registry: { x: 940, y: 452, w: 176, h: 66, label: "Supabase control", sub: "models, results" },
}

const c = (n) => ({ x: n.x + n.w / 2, y: n.y + n.h / 2 })
const rightOf = (n) => ({ x: n.x + n.w, y: n.y + n.h / 2 })
const leftOf = (n) => ({ x: n.x, y: n.y + n.h / 2 })
const topOf = (n) => ({ x: n.x + n.w / 2, y: n.y })
const bottomOf = (n) => ({ x: n.x + n.w / 2, y: n.y + n.h })

/// Cubic between two anchors with horizontal control points, so every edge reads as a flow
/// rather than a corner.
function curve(a, b, bow = 0.5) {
  const dx = (b.x - a.x) * bow
  return `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`
}

const PATHS = {
  clientApp: curve(rightOf(N.client), leftOf(N.app)),
  // The L1 answer never leaves the application process. Short arc, and deliberately the
  // only edge that does not touch the network.
  l1Return: `M${topOf(N.app).x + 24},${N.app.y} C${N.app.x + 150},${N.app.y - 66} ${c(N.client).x + 40},${N.client.y - 66} ${c(N.client).x},${N.client.y}`,
  appAura: curve(rightOf(N.app), leftOf(N.aura)),
  hitReturn: `M${topOf(N.aura).x - 30},${N.aura.y} C${N.aura.x - 40},${N.aura.y - 110} ${c(N.app).x - 30},${N.app.y - 116} ${c(N.client).x + 34},${N.client.y}`,
  auraLease: `M${topOf(N.aura).x + 40},${N.aura.y} L${topOf(N.lease).x + 40},${N.lease.y + N.lease.h}`,
  appDb: curve(rightOf(N.app), leftOf(N.db), 0.45),
  appCompute: curve(rightOf(N.app), leftOf(N.compute), 0.45),
  appDecide: `M${bottomOf(N.app).x},${bottomOf(N.app).y} C${bottomOf(N.app).x},${bottomOf(N.app).y + 120} ${N.decide.x - 120},${c(N.decide).y} ${N.decide.x},${c(N.decide).y}`,
  decideAura: `M${topOf(N.decide).x},${N.decide.y} L${bottomOf(N.aura).x},${N.aura.y + N.aura.h}`,
  dbInvalidator: `M${bottomOf(N.db).x - 40},${N.db.y + N.db.h} C${N.db.x + 20},${N.db.y + 300} ${N.invalidator.x + 240},${c(N.invalidator).y} ${N.invalidator.x + N.invalidator.w},${c(N.invalidator).y}`,
  invalidatorAura: curve(leftOf(N.invalidator), { x: N.decide.x + N.decide.w, y: c(N.decide).y }, 0.6),
  auraRegistry: `M${N.aura.x + N.aura.w},${N.aura.y + N.aura.h - 10} C${N.aura.x + 320},${N.aura.y + 260} ${N.registry.x - 150},${c(N.registry).y} ${N.registry.x},${c(N.registry).y}`,
}

export function FlowDiagram({ frame, status }) {
  const svgRef = useRef(null)
  const hostRef = useRef(null)
  const [paused, setPaused] = useState(false)

  const l1 = frame?.layers?.l1 ?? {}
  const l2 = frame?.layers?.l2 ?? {}
  const hits = l2.hits ?? 0
  const misses = l2.misses ?? 0
  const totalReq = hits + misses
  const hitFrac = totalReq ? hits / totalReq : 0
  const l1Frac = l1.hit_rate ?? 0
  const rps = frame?.sim?.rps ?? 0
  const fidelity = frame?.fidelity ?? {}
  const backend = fidelity.backend ?? {}
  const engine = frame?.engine ?? {}
  const latency = frame?.latency ?? {}
  const cost = frame?.cost ?? {}
  const consistency = frame?.consistency ?? {}
  const flight = consistency.single_flight ?? {}
  const liveTraffic = fidelity.traffic === "applications"

  // The spawn loop runs on an interval outside React's render cycle, so it reads the
  // proportions through refs. They are written in an effect rather than during render:
  // a render can be thrown away and re-run, and the animation must not see a value the
  // page never committed.
  const hitRef = useRef(hitFrac)
  const l1Ref = useRef(l1Frac)
  const liveRef = useRef(status === "live" || status === "polling")
  useEffect(() => {
    hitRef.current = hitFrac
    l1Ref.current = l1Frac
    liveRef.current = status === "live" || status === "polling"
  }, [hitFrac, l1Frac, status])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    let cancelled = false
    const ctx = gsap.context(() => {
      const spawn = () => {
        if (cancelled || paused || !liveRef.current) return

        const roll = Math.random()
        const isL1 = roll < l1Ref.current
        const isHit = !isL1 && Math.random() < hitRef.current

        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle")
        dot.setAttribute("r", isHit || isL1 ? "4.5" : "5")
        dot.setAttribute("class", isL1 ? "fill-sky-400" : isHit ? "fill-primary" : "fill-amber-400")
        host.appendChild(dot)

        const legs = isL1
          ? [
              { path: PATHS.clientApp, dur: 0.4 },
              { path: PATHS.l1Return, dur: 0.5 },
            ]
          : isHit
            ? [
                { path: PATHS.clientApp, dur: 0.4 },
                { path: PATHS.appAura, dur: 0.4 },
                { path: PATHS.hitReturn, dur: 0.8 },
              ]
            : [
                { path: PATHS.clientApp, dur: 0.4 },
                { path: PATHS.appAura, dur: 0.4 },
                // The cache answers "you rebuild it", so the dot goes back to the
                // application before any origin is touched.
                { path: PATHS.appAura, dur: 0.35, reverse: true },
                { path: Math.random() < 0.5 ? PATHS.appDb : PATHS.appCompute, dur: 0.6 },
                { path: Math.random() < 0.5 ? PATHS.appDb : PATHS.appCompute, dur: 0.6, reverse: true },
                { path: PATHS.appDecide, dur: 0.7 },
                { path: PATHS.decideAura, dur: 0.35 },
              ]

        const tl = gsap.timeline({ onComplete: () => dot.remove() })
        legs.forEach((leg) => {
          tl.to(dot, {
            duration: leg.dur,
            ease: "none",
            motionPath: { path: leg.path, start: leg.reverse ? 1 : 0, end: leg.reverse ? 0 : 1 },
          })
        })
      }

      // Well below the real request rate on purpose: the picture shows the proportion
      // between the three paths, it is not a packet capture.
      const id = setInterval(spawn, 190)
      return () => clearInterval(id)
    }, svgRef)

    return () => {
      cancelled = true
      ctx.revert()
      while (host.firstChild) host.removeChild(host.firstChild)
    }
  }, [paused])

  const backendReal = backend.enabled && backend.measured
  const memoryReal = Boolean(fidelity.values_stored)

  return (
    <Panel
      title="Live request path"
      subtitle="Each dot is a request. Blue never leaves the application process. Lime is answered by the cache. Amber falls through, and the cache hands the rebuild back to the application, because only the application knows how to build the object."
      actions={
        <div className="flex items-center gap-2">
          <Pill tone="accent">{pct(hitFrac, 1)} cache hit</Pill>
          <button
            onClick={() => setPaused((p) => !p)}
            className="rounded-lg border border-border px-2.5 py-1 text-[13px] hover:bg-accent"
          >
            {paused ? "Animate" : "Freeze"}
          </button>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full min-w-[960px]"
          style={{ height: "clamp(360px, 44vw, 600px)" }}
        >
          <defs>
            <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" className="fill-muted-foreground" />
            </marker>
          </defs>

          <Zone x={214} y={214} w={202} h={168} label="Application process" tone="sky" />
          <Zone x={448} y={68} w={214} h={470} label="AURA engine" />
          <Zone x={916} y={104} w={222} h={274} label="Origins — the expensive part" tone="amber" />
          <Zone x={916} y={420} w={222} h={130} label="Control plane" tone="slate" />

          <Edge d={PATHS.clientApp} />
          <Edge d={PATHS.l1Return} tone="stroke-sky-400/60" dash label="L1 hit · no network" lx={214} ly={196} ltone="fill-sky-400" />
          <Edge d={PATHS.appAura} label="GET" lx={410} ly={286} />
          <Edge d={PATHS.hitReturn} tone="stroke-primary/60" dash label="HIT · nothing else is touched" lx={252} ly={150} ltone="fill-primary" />
          <Edge d={PATHS.auraLease} tone="stroke-amber-400/60" dash label="MISS · lease" lx={520} ly={224} ltone="fill-amber-400" />
          <Edge d={PATHS.appDb} tone="stroke-amber-400/55" label="rebuild" lx={700} ly={196} ltone="fill-amber-400" />
          <Edge d={PATHS.appCompute} tone="stroke-amber-400/55" />
          <Edge d={PATHS.appDecide} tone="stroke-amber-400/55" dash label="PUT with measured cost" lx={214} ly={432} ltone="fill-amber-400" />
          <Edge d={PATHS.decideAura} tone="stroke-primary/60" />
          <Edge d={PATHS.dbInvalidator} tone="stroke-orange-400/45" dash label="row changed" lx={860} ly={352} ltone="fill-orange-400" />
          <Edge d={PATHS.invalidatorAura} tone="stroke-orange-400/55" label="invalidate" lx={666} ly={430} ltone="fill-orange-400" />
          <Edge d={PATHS.auraRegistry} tone="stroke-border" dash label="off the request path" lx={700} ly={534} />

          <g ref={hostRef} />

          <Node n={N.client} stat={`${Number(rps).toLocaleString()}/s`} />
          <Node n={N.app} tone="sky" stat={`L1 ${pct(l1Frac, 0)}`} />
          <Node n={N.aura} accent stat={bytes(frame?.capacity?.used_bytes ?? 0)} />
          <Node
            n={N.lease}
            accent
            stat={`${Number(flight.origin_calls_suppressed ?? 0).toLocaleString()} calls stopped`}
          />
          <Node n={N.decide} accent stat={`${Number(engine.decision_overhead_us_p50 ?? 0).toFixed(0)} µs`} />
          <Node
            n={N.db}
            warn={backendReal}
            dim={!backendReal}
            stat={backendReal ? `${backend.measured_db_ms?.toFixed(0)} ms measured` : "not queried yet"}
          />
          <Node n={N.compute} warn stat={`${Number(misses).toLocaleString()} rebuilds`} />
          <Node
            n={N.invalidator}
            dim={!consistency.invalidations}
            stat={`${Number(consistency.keys_invalidated ?? 0).toLocaleString()} keys dropped`}
          />
          <Node n={N.registry} dim stat="models, results" />
        </svg>
      </div>

      {/* What is real and what is not, next to the picture rather than in a footnote. */}
      <div className="mt-4 grid gap-3 lg:grid-cols-3">
        <Fidelity
          ok={liveTraffic}
          title="Traffic"
          state={liveTraffic ? "live applications" : "engine generator"}
          body={
            liveTraffic
              ? "No scenario is running. Every request on this engine arrived over HTTP from an application that did the work itself."
              : "Requests come from the engine's own seeded generator, so runs repeat exactly. Stop the scenario and drive the applications to make this live."
          }
        />
        <Fidelity
          ok={memoryReal}
          title="Cached bytes"
          state={memoryReal ? "really held in memory" : "accounted, not held"}
          body={
            memoryReal
              ? `The pool is resident memory: ${bytes(frame?.capacity?.used_bytes ?? 0)} of payload is actually allocated.`
              : "Object sizes are tracked as a budget but no payload is stored. Start the engine with --real-values to hold the bytes for real."
          }
        />
        <Fidelity
          ok={backendReal}
          title="Rebuild cost"
          state={backendReal ? "measured from Supabase" : "synthetic"}
          body={
            backendReal
              ? `Analytics rebuild cost is a real aggregate against the seeded tables: ${backend.measured_db_ms?.toFixed(0)} ms mean, ${backend.p95_db_ms?.toFixed(0)} ms p95 over ${backend.samples} samples.`
              : "The generator supplies the cost figure. Start with --real-backend to calibrate it from live queries."
          }
        />
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <Foot label="Served from cache" value={Number(hits).toLocaleString()} />
        <Foot label="Fell through" value={Number(misses).toLocaleString()} />
        <Foot label="p95 on a miss" value={ms(latency.p95_ms ?? 0)} />
        <Foot label="Backend work avoided" value={usd(cost.saved_vs_no_cache_usd ?? 0)} />
      </div>
    </Panel>
  )
}

function Zone({ x, y, w, h, label, tone }) {
  const tones = {
    amber: "stroke-amber-400/35 fill-amber-400/[0.04]",
    sky: "stroke-sky-400/35 fill-sky-400/[0.04]",
    slate: "stroke-border fill-muted/25",
  }
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx="16"
        className={cn("stroke-primary/30 fill-primary/[0.04]", tones[tone])}
        strokeDasharray="4 5"
      />
      <text x={x + 12} y={y + 21} className="fill-muted-foreground text-[12.5px] font-medium">
        {label}
      </text>
    </g>
  )
}

function Edge({ d, tone = "stroke-border", dash, label, lx, ly, ltone = "fill-muted-foreground" }) {
  return (
    <g>
      <path
        d={d}
        className={cn("fill-none", tone)}
        strokeWidth="1.8"
        strokeDasharray={dash ? "5 5" : undefined}
        markerEnd="url(#ah)"
      />
      {label && (
        <text x={lx} y={ly} className={cn("text-[12.5px] font-semibold", ltone)}>
          {label}
        </text>
      )}
    </g>
  )
}

function Node({ n, accent, warn, dim, tone, stat }) {
  const tones = {
    sky: "fill-sky-400/12 stroke-sky-400/60",
  }
  return (
    <g>
      <rect
        x={n.x}
        y={n.y}
        width={n.w}
        height={n.h}
        rx="12"
        className={cn(
          "stroke-[1.6]",
          tones[tone] ??
            (accent
              ? "fill-primary/12 stroke-primary/70"
              : warn
                ? "fill-amber-400/12 stroke-amber-400/60"
                : dim
                  ? "fill-muted/40 stroke-border"
                  : "fill-card stroke-border")
        )}
      />
      <text x={n.x + 13} y={n.y + 25} className="fill-foreground text-[14px] font-semibold">
        {n.label}
      </text>
      <text x={n.x + 13} y={n.y + 42} className="fill-muted-foreground text-[12.5px]">
        {n.sub}
      </text>
      <text x={n.x + 13} y={n.y + 59} className="fill-muted-foreground font-mono text-[12.5px]">
        {stat}
      </text>
    </g>
  )
}

function Fidelity({ ok, title, state, body }) {
  return (
    <div
      className={cn(
        "rounded-xl border px-3.5 py-3",
        ok ? "border-primary/45 bg-primary/[0.07]" : "border-amber-400/45 bg-amber-400/[0.06]"
      )}
    >
      <div className="flex items-center gap-2">
        <span className="text-[13px] font-semibold">{title}</span>
        <span
          className={cn(
            "rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
            ok ? "bg-primary/20 text-primary" : "bg-amber-400/20 text-amber-300"
          )}
        >
          {state}
        </span>
      </div>
      <p className="mt-1.5 text-[12.5px] leading-snug text-muted-foreground">{body}</p>
    </div>
  )
}

function Foot({ label, value }) {
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2">
      <div className="text-[12px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-[15px] tabular-nums">{value}</div>
    </div>
  )
}
