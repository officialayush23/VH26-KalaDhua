import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { MotionPathPlugin } from "gsap/MotionPathPlugin"

import { bytes, cn, ms, pct, usd } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

gsap.registerPlugin(MotionPathPlugin)

const W = 1180
const H = 560

// Node geometry. Kept as data so the paths below and the boxes above can never drift
// apart: both are generated from the same numbers.
const N = {
  client: { x: 60, y: 250, w: 130, h: 66, label: "Client", sub: "GET key" },
  l1: { x: 265, y: 250, w: 130, h: 66, label: "L1 window", sub: "recency only" },
  l2: { x: 470, y: 250, w: 150, h: 66, label: "AURA L2", sub: "scored cache" },
  decide: { x: 470, y: 420, w: 150, h: 66, label: "Decision", sub: "keep or refuse" },
  app: { x: 725, y: 120, w: 160, h: 66, label: "App service", sub: "rebuilds object" },
  db: { x: 960, y: 120, w: 150, h: 66, label: "Supabase", sub: "Postgres" },
  gpu: { x: 960, y: 250, w: 150, h: 66, label: "GPU / API", sub: "content service" },
  registry: { x: 960, y: 420, w: 150, h: 66, label: "Supabase", sub: "model + results" },
}

const c = (n) => ({ x: n.x + n.w / 2, y: n.y + n.h / 2 })
const rightOf = (n) => ({ x: n.x + n.w, y: n.y + n.h / 2 })
const leftOf = (n) => ({ x: n.x, y: n.y + n.h / 2 })
const topOf = (n) => ({ x: n.x + n.w / 2, y: n.y })
const bottomOf = (n) => ({ x: n.x + n.w / 2, y: n.y + n.h })

/// Cubic between two anchors with horizontal control points, which keeps every edge
/// reading as a flow rather than a corner.
function curve(a, b, bow = 0.5) {
  const dx = (b.x - a.x) * bow
  return `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`
}

const PATHS = {
  clientL1: curve(rightOf(N.client), leftOf(N.l1)),
  l1l2: curve(rightOf(N.l1), leftOf(N.l2)),
  hitBack: `M${c(N.l2).x},${N.l2.y} C${c(N.l2).x},${N.l2.y - 90} ${c(N.client).x},${N.client.y - 90} ${c(N.client).x},${N.client.y}`,
  missApp: curve(rightOf(N.l2), leftOf(N.app)),
  appDb: curve(rightOf(N.app), leftOf(N.db)),
  appGpu: curve(rightOf(N.app), leftOf(N.gpu), 0.6),
  appDecide: `M${bottomOf(N.app).x},${bottomOf(N.app).y} C${bottomOf(N.app).x},${bottomOf(N.app).y + 120} ${c(N.decide).x + 120},${c(N.decide).y} ${N.decide.x + N.decide.w},${c(N.decide).y}`,
  decideL2: `M${c(N.decide).x},${N.decide.y} L${c(N.l2).x},${N.l2.y + N.l2.h}`,
  l2Registry: `M${N.l2.x + N.l2.w},${N.l2.y + N.l2.h - 8} C${N.l2.x + N.l2.w + 200},${N.l2.y + 200} ${N.registry.x - 120},${c(N.registry).y} ${N.registry.x},${c(N.registry).y}`,
}

/// The request path as a live diagram. Dots are emitted along the hit and miss edges in
/// proportion to the measured hit rate, so the picture is driven by the same numbers as
/// the rest of the page rather than being decorative.
export function FlowDiagram({ frame, status }) {
  const svgRef = useRef(null)
  const hostRef = useRef(null)
  const [paused, setPaused] = useState(false)

  const l2 = frame?.layers?.l2 ?? {}
  const hits = l2.hits ?? 0
  const misses = l2.misses ?? 0
  const totalReq = hits + misses
  const hitFrac = totalReq ? hits / totalReq : 0
  const rps = frame?.sim?.rps ?? 0
  const fidelity = frame?.fidelity ?? {}
  const backend = fidelity.backend ?? {}
  const engine = frame?.engine ?? {}
  const latency = frame?.latency ?? {}
  const cost = frame?.cost ?? {}

  const hitRef = useRef(hitFrac)
  hitRef.current = hitFrac
  const liveRef = useRef(status === "live" || status === "polling")
  liveRef.current = status === "live" || status === "polling"

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    let cancelled = false
    const ctx = gsap.context(() => {
      const spawn = () => {
        if (cancelled) return
        if (paused || !liveRef.current) return

        const isHit = Math.random() < hitRef.current
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle")
        dot.setAttribute("r", isHit ? "4.5" : "5")
        dot.setAttribute(
          "class",
          isHit ? "fill-primary" : "fill-amber-400"
        )
        host.appendChild(dot)

        const legs = isHit
          ? [
              { path: PATHS.clientL1, dur: 0.42 },
              { path: PATHS.l1l2, dur: 0.42 },
              { path: PATHS.hitBack, dur: 0.75 },
            ]
          : [
              { path: PATHS.clientL1, dur: 0.42 },
              { path: PATHS.l1l2, dur: 0.42 },
              { path: PATHS.missApp, dur: 0.55 },
              { path: Math.random() < 0.5 ? PATHS.appDb : PATHS.appGpu, dur: 0.6 },
              { path: Math.random() < 0.5 ? PATHS.appDb : PATHS.appGpu, dur: 0.6, reverse: true },
              { path: PATHS.appDecide, dur: 0.7 },
              { path: PATHS.decideL2, dur: 0.35 },
            ]

        const tl = gsap.timeline({
          onComplete: () => dot.remove(),
        })
        legs.forEach((leg) => {
          tl.to(dot, {
            duration: leg.dur,
            ease: "none",
            motionPath: { path: leg.path, start: leg.reverse ? 1 : 0, end: leg.reverse ? 0 : 1 },
          })
        })
      }

      // Emission rate is capped well below the real request rate: the point is to show the
      // proportion between the two paths, not to render four thousand dots a second.
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
      subtitle="Each dot is a request. Lime returns from the cache and touches nothing else. Amber falls through to an application service, which is the only thing that talks to a database. The ratio between them is the measured hit rate."
      actions={
        <div className="flex items-center gap-2">
          <Pill tone="accent">{pct(hitFrac, 1)} hit</Pill>
          <button
            onClick={() => setPaused((p) => !p)}
            className="rounded-lg border border-border px-2.5 py-1 text-[12px] hover:bg-accent"
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
          className="w-full min-w-[900px]"
          style={{ height: "clamp(340px, 42vw, 560px)" }}
        >
          <defs>
            <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" className="fill-border" />
            </marker>
          </defs>

          {/* Zones */}
          <Zone x={240} y={60} w={420} h={440} label="AURA engine — in process, no network" />
          <Zone x={700} y={60} w={210} h={200} label="Application tier" tone="amber" />
          <Zone x={935} y={60} w={205} h={440} label="Supabase" tone="slate" />

          {/* Edges */}
          <Edge d={PATHS.clientL1} />
          <Edge d={PATHS.l1l2} />
          <Edge d={PATHS.hitBack} tone="stroke-primary/45" dash label="HIT · returns here" lx={330} ly={148} ltone="fill-primary" />
          <Edge d={PATHS.missApp} tone="stroke-amber-400/50" label="MISS" lx={660} ly={205} ltone="fill-amber-400" />
          <Edge d={PATHS.appDb} tone="stroke-amber-400/40" />
          <Edge d={PATHS.appGpu} tone="stroke-amber-400/40" />
          <Edge d={PATHS.appDecide} tone="stroke-amber-400/40" dash label="measured cost" lx={840} ly={380} />
          <Edge d={PATHS.decideL2} tone="stroke-primary/45" />
          <Edge d={PATHS.l2Registry} tone="stroke-border" dash label="off the request path" lx={790} ly={478} />

          {/* Dots are appended here at runtime */}
          <g ref={hostRef} />

          {/* Nodes */}
          <Node n={N.client} stat={`${Number(rps).toLocaleString()}/s`} />
          <Node n={N.l1} stat={pct(frame?.layers?.l1?.hit_rate ?? 0, 0)} />
          <Node n={N.l2} accent stat={`${bytes(frame?.capacity?.used_bytes ?? 0)}`} />
          <Node n={N.decide} accent stat={`${Number(engine.decision_overhead_us_p50 ?? 0).toFixed(0)} µs`} />
          <Node n={N.app} warn stat={`${Number(misses).toLocaleString()} rebuilds`} />
          <Node
            n={N.db}
            warn={backendReal}
            stat={backendReal ? `${backend.measured_db_ms?.toFixed(0)} ms real` : "not queried"}
            dim={!backendReal}
          />
          <Node n={N.gpu} dim stat="simulated" />
          <Node n={N.registry} dim stat="control plane" />
        </svg>
      </div>

      {/* What is real and what is not. This belongs next to the picture, not in a footnote. */}
      <div className="mt-4 grid gap-3 lg:grid-cols-3">
        <Fidelity
          ok={false}
          title="Traffic"
          state="simulated"
          body="Requests come from the engine's own generator, seeded so runs repeat. No external client is connected."
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
              ? `Analytics rebuild cost is a real aggregate against your seeded tables: ${backend.measured_db_ms?.toFixed(0)} ms mean, ${backend.p95_db_ms?.toFixed(0)} ms p95 over ${backend.samples} samples.`
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
    amber: "stroke-amber-400/25 fill-amber-400/[0.03]",
    slate: "stroke-border fill-muted/20",
  }
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx="16"
        className={cn("stroke-primary/20 fill-primary/[0.03]", tones[tone])}
        strokeDasharray="4 5"
      />
      <text x={x + 12} y={y + 20} className="fill-muted-foreground text-[11px]">
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
        strokeWidth="1.6"
        strokeDasharray={dash ? "5 5" : undefined}
        markerEnd="url(#ah)"
      />
      {label && (
        <text x={lx} y={ly} className={cn("text-[11px] font-medium", ltone)}>
          {label}
        </text>
      )}
    </g>
  )
}

function Node({ n, accent, warn, dim, stat }) {
  return (
    <g>
      <rect
        x={n.x}
        y={n.y}
        width={n.w}
        height={n.h}
        rx="12"
        className={cn(
          "stroke-[1.5]",
          accent
            ? "fill-primary/10 stroke-primary/60"
            : warn
              ? "fill-amber-400/10 stroke-amber-400/50"
              : dim
                ? "fill-muted/30 stroke-border"
                : "fill-card stroke-border"
        )}
      />
      <text x={n.x + 12} y={n.y + 24} className="fill-foreground text-[13px] font-semibold">
        {n.label}
      </text>
      <text x={n.x + 12} y={n.y + 40} className="fill-muted-foreground text-[10.5px]">
        {n.sub}
      </text>
      <text x={n.x + 12} y={n.y + 56} className="fill-muted-foreground font-mono text-[10.5px]">
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
        ok ? "border-primary/40 bg-primary/[0.05]" : "border-amber-400/35 bg-amber-400/[0.04]"
      )}
    >
      <div className="flex items-center gap-2">
        <span className="text-[12px] font-semibold">{title}</span>
        <span
          className={cn(
            "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
            ok ? "bg-primary/15 text-primary" : "bg-amber-400/15 text-amber-400"
          )}
        >
          {state}
        </span>
      </div>
      <p className="mt-1.5 text-[11.5px] leading-snug text-muted-foreground">{body}</p>
    </div>
  )
}

function Foot({ label, value }) {
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2">
      <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-[15px] tabular-nums">{value}</div>
    </div>
  )
}
