import { useEffect, useMemo, useRef, useState } from "react"
import gsap from "gsap"
import { MotionPathPlugin } from "gsap/MotionPathPlugin"

import { bytes, cn, pct, usd } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

gsap.registerPlugin(MotionPathPlugin)

/// Where AURA sits in a system, as opposed to what happens to one request.
///
/// The shape is the ordinary one: every application process keeps a tiny local cache, and
/// behind them all sits one shared tier that the whole fleet talks to. That shared tier is
/// almost always Redis or Memcached, and this is the slot AURA occupies. Saying so plainly
/// is the point of the picture - it is not a new layer in the stack, it is the same layer
/// with a different question being asked inside it: not "what was touched least recently"
/// but "what is worth the memory it occupies".
///
/// The distinction the diagram has to carry is why the two tiers cannot be the same thing.
/// L1 removes a network round trip and nothing else; only the shared tier sees every
/// process's demand at once, which is the only vantage point from which value can be
/// judged. A per-process cache reasoning about value would have each process learning its
/// own contradictory answer.

const W = 1240
const H = 640

const NODES = {
  users: { x: 30, y: 288, w: 118, h: 64, label: "Users", sub: "requests arrive" },
  app1: { x: 208, y: 118, w: 190, h: 84, label: "Service A", sub: "recommendation" },
  app2: { x: 208, y: 278, w: 190, h: 84, label: "Service B", sub: "analytics" },
  app3: { x: 208, y: 438, w: 190, h: 84, label: "Service C", sub: "anything else" },
  aura: { x: 520, y: 236, w: 250, h: 168, label: "AURA", sub: "one shared pool" },
  pg: { x: 900, y: 112, w: 210, h: 78, label: "PostgreSQL", sub: "the rows themselves" },
  compute: { x: 900, y: 236, w: 210, h: 78, label: "CPU / GPU", sub: "ranking, media, models" },
  api: { x: 900, y: 360, w: 210, h: 78, label: "Paid APIs", sub: "billed per call" },
  control: { x: 900, y: 508, w: 210, h: 78, label: "Control plane", sub: "models, results, audit" },
}

const c = (n) => ({ x: n.x + n.w / 2, y: n.y + n.h / 2 })
const rightOf = (n) => ({ x: n.x + n.w, y: n.y + n.h / 2 })
const leftOf = (n) => ({ x: n.x, y: n.y + n.h / 2 })

function curve(a, b, bow = 0.5) {
  const dx = (b.x - a.x) * bow
  return `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`
}

const PATHS = {
  usersApp1: curve(rightOf(NODES.users), leftOf(NODES.app1), 0.6),
  usersApp2: curve(rightOf(NODES.users), leftOf(NODES.app2), 0.6),
  usersApp3: curve(rightOf(NODES.users), leftOf(NODES.app3), 0.6),
  app1Aura: curve(rightOf(NODES.app1), { x: NODES.aura.x, y: NODES.aura.y + 42 }, 0.55),
  app2Aura: curve(rightOf(NODES.app2), { x: NODES.aura.x, y: NODES.aura.y + 84 }, 0.55),
  app3Aura: curve(rightOf(NODES.app3), { x: NODES.aura.x, y: NODES.aura.y + 126 }, 0.55),
  auraPg: curve({ x: NODES.aura.x + NODES.aura.w, y: NODES.aura.y + 40 }, leftOf(NODES.pg), 0.5),
  auraCompute: curve(rightOf(NODES.aura), leftOf(NODES.compute), 0.5),
  auraApi: curve({ x: NODES.aura.x + NODES.aura.w, y: NODES.aura.y + 128 }, leftOf(NODES.api), 0.5),
  auraControl: `M${c(NODES.aura).x},${NODES.aura.y + NODES.aura.h} C${c(NODES.aura).x},${NODES.aura.y + 300} ${NODES.control.x - 140},${c(NODES.control).y} ${NODES.control.x},${c(NODES.control).y}`,
}

export function ArchitecturePanel({ frame, status }) {
  const svgRef = useRef(null)
  const hostRef = useRef(null)
  const [paused, setPaused] = useState(false)

  const l2 = frame?.layers?.l2 ?? {}
  const l1 = frame?.layers?.l1 ?? {}
  const cap = frame?.capacity ?? {}
  const engine = frame?.engine ?? {}
  const flight = frame?.consistency?.single_flight ?? {}
  const apps = frame?.applications ?? []
  const hitFrac = l2.hit_rate ?? 0

  const appLabels = useMemo(() => {
    const names = apps.map((a) => a.application)
    return {
      app1: names[0] ?? NODES.app1.sub,
      app2: names[1] ?? NODES.app2.sub,
      app3: names[2] ?? NODES.app3.sub,
    }
  }, [apps])

  const hitRef = useRef(hitFrac)
  const liveRef = useRef(false)
  useEffect(() => {
    hitRef.current = hitFrac
    liveRef.current = status === "live" || status === "polling"
  }, [hitFrac, status])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let cancelled = false

    const ctx = gsap.context(() => {
      const lanes = [
        [PATHS.usersApp1, PATHS.app1Aura],
        [PATHS.usersApp2, PATHS.app2Aura],
        [PATHS.usersApp3, PATHS.app3Aura],
      ]
      const origins = [PATHS.auraPg, PATHS.auraCompute, PATHS.auraApi]

      const spawn = () => {
        if (cancelled || paused || !liveRef.current) return
        const lane = lanes[Math.floor(Math.random() * lanes.length)]
        const isHit = Math.random() < hitRef.current

        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle")
        dot.setAttribute("r", "4.5")
        dot.setAttribute("class", isHit ? "fill-primary" : "fill-amber-400")
        host.appendChild(dot)

        const legs = isHit
          ? [
              { path: lane[0], dur: 0.45 },
              { path: lane[1], dur: 0.5 },
              { path: lane[1], dur: 0.45, reverse: true },
              { path: lane[0], dur: 0.4, reverse: true },
            ]
          : [
              { path: lane[0], dur: 0.45 },
              { path: lane[1], dur: 0.5 },
              { path: origins[Math.floor(Math.random() * origins.length)], dur: 0.6 },
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

      const id = setInterval(spawn, 220)
      return () => clearInterval(id)
    }, svgRef)

    return () => {
      cancelled = true
      ctx.revert()
      while (host.firstChild) host.removeChild(host.firstChild)
    }
  }, [paused])

  return (
    <Panel
      title="Where the cache sits"
      subtitle="Every process keeps a small local cache; one shared tier sits behind all of them. That shared tier is the slot Redis or Memcached normally fills, and it is the slot AURA occupies. Same layer, different question asked inside it."
      actions={
        <div className="flex items-center gap-2">
          <Pill tone="accent">{pct(hitFrac, 1)} served from the shared pool</Pill>
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
          className="w-full min-w-[980px]"
          style={{ height: "clamp(380px, 46vw, 640px)" }}
        >
          <defs>
            <marker id="arch-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" className="fill-muted-foreground" />
            </marker>
            <linearGradient id="arch-pool" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" className="text-primary" stopColor="currentColor" stopOpacity="0.22" />
              <stop offset="100%" className="text-primary" stopColor="currentColor" stopOpacity="0.06" />
            </linearGradient>
          </defs>

          <Zone x={186} y={86} w={236} h={470} label="Application tier — one process each" tone="sky" />
          <Zone x={496} y={196} w={300} h={252} label="Shared cache tier" />
          <Zone x={876} y={76} w={260} h={382} label="Origins — where the money goes" tone="amber" />

          {[PATHS.usersApp1, PATHS.usersApp2, PATHS.usersApp3].map((d, i) => (
            <Edge key={`u${i}`} d={d} />
          ))}
          {[PATHS.app1Aura, PATHS.app2Aura, PATHS.app3Aura].map((d, i) => (
            <Edge key={`a${i}`} d={d} />
          ))}
          <Edge d={PATHS.auraPg} tone="stroke-amber-400/55" label="miss" lx={800} ly={262} ltone="fill-amber-400" />
          <Edge d={PATHS.auraCompute} tone="stroke-amber-400/55" />
          <Edge d={PATHS.auraApi} tone="stroke-amber-400/55" />
          <Edge d={PATHS.auraControl} tone="stroke-border" dash label="off the request path" lx={620} ly={560} />

          <g ref={hostRef} />

          <Node n={NODES.users} stat={`${Number(frame?.sim?.rps ?? 0).toLocaleString()}/s`} />
          <AppNode n={NODES.app1} name={appLabels.app1} l1={l1.hit_rate ?? 0} />
          <AppNode n={NODES.app2} name={appLabels.app2} l1={l1.hit_rate ?? 0} />
          <AppNode n={NODES.app3} name={appLabels.app3} l1={l1.hit_rate ?? 0} />

          <AuraNode
            n={NODES.aura}
            used={cap.used_bytes ?? engine.used_bytes ?? 0}
            capacity={cap.logical_bytes ?? engine.capacity_bytes ?? 1}
            objects={engine.resident_objects ?? 0}
            suppressed={flight.origin_calls_suppressed ?? 0}
          />

          <Node n={NODES.pg} warn stat={`${Number(l2.misses ?? 0).toLocaleString()} fell through`} />
          <Node n={NODES.compute} warn stat={`${usd(frame?.cost?.backend_usd ?? 0)} spent rebuilding`} />
          <Node n={NODES.api} warn stat="billed per call" />
          <Node n={NODES.control} dim stat="models, benchmarks, audit" />
        </svg>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-3">
        <Note
          title="Why two tiers and not one"
          body="L1 removes a network round trip. Only the shared tier sees every process's demand at once, which is the only place value can be judged. A per-process cache reasoning about value would have each process reaching its own contradictory answer."
        />
        <Note
          title="Why this replaces Redis rather than sitting beside it"
          body="It is the same slot in the same stack, speaking the same shape of request. What changes is the question asked on eviction: not what was touched least recently, but what is worth the memory it occupies, priced with what the application measured."
        />
        <Note
          title="Why the cache never calls the origin"
          body="Only the application knows how to rebuild an object. On a miss the cache hands out a lease saying who should rebuild and makes everyone else wait, which is what turns a thundering herd into one origin call."
          stat={`${Number(flight.origin_calls_suppressed ?? 0).toLocaleString()} origin calls stopped so far`}
        />
      </div>
    </Panel>
  )
}

function Zone({ x, y, w, h, label, tone }) {
  const tones = {
    amber: "stroke-amber-400/35 fill-amber-400/[0.04]",
    sky: "stroke-sky-400/35 fill-sky-400/[0.04]",
  }
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx="18"
        className={cn("stroke-primary/30 fill-primary/[0.04]", tones[tone])}
        strokeDasharray="4 5"
      />
      <text x={x + 14} y={y + 22} className="fill-muted-foreground text-[12.5px] font-medium">
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
        markerEnd="url(#arch-ah)"
      />
      {label && (
        <text x={lx} y={ly} className={cn("text-[12.5px] font-semibold", ltone)}>
          {label}
        </text>
      )}
    </g>
  )
}

function Node({ n, warn, dim, stat }) {
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
          warn ? "fill-amber-400/12 stroke-amber-400/60" : dim ? "fill-muted/40 stroke-border" : "fill-card stroke-border"
        )}
      />
      <text x={n.x + 14} y={n.y + 26} className="fill-foreground text-[14px] font-semibold">
        {n.label}
      </text>
      <text x={n.x + 14} y={n.y + 44} className="fill-muted-foreground text-[12.5px]">
        {n.sub}
      </text>
      {stat && (
        <text x={n.x + 14} y={n.y + 62} className="fill-muted-foreground font-mono text-[12px]">
          {stat}
        </text>
      )}
    </g>
  )
}

/// An application process with its own small cache drawn inside it, because that is where
/// it lives: in the process, not on the network.
function AppNode({ n, name, l1 }) {
  return (
    <g>
      <rect
        x={n.x}
        y={n.y}
        width={n.w}
        height={n.h}
        rx="12"
        className="fill-sky-400/[0.10] stroke-sky-400/55 stroke-[1.6]"
      />
      <text x={n.x + 14} y={n.y + 26} className="fill-foreground text-[14px] font-semibold">
        {n.label}
      </text>
      <text x={n.x + 14} y={n.y + 43} className="fill-muted-foreground text-[12.5px]">
        {name}
      </text>
      <rect
        x={n.x + 12}
        y={n.y + 52}
        width={n.w - 24}
        height={22}
        rx="6"
        className="fill-sky-400/15 stroke-sky-400/45"
      />
      <text x={n.x + 20} y={n.y + 67} className="fill-sky-300 font-mono text-[11.5px]">
        L1 in-process · {pct(l1, 0)}
      </text>
    </g>
  )
}

/// The shared tier, drawn with its pool actually filling, because the one number an
/// operator wants from this picture is how full the thing is.
function AuraNode({ n, used, capacity, objects, suppressed }) {
  const frac = capacity > 0 ? Math.min(1, used / capacity) : 0
  const barX = n.x + 16
  const barY = n.y + 104
  const barW = n.w - 32
  return (
    <g>
      <rect
        x={n.x}
        y={n.y}
        width={n.w}
        height={n.h}
        rx="16"
        className="stroke-primary/70 stroke-[2]"
        fill="url(#arch-pool)"
      />
      <text x={n.x + 18} y={n.y + 32} className="fill-primary text-[19px] font-semibold">
        AURA
      </text>
      <text x={n.x + 18} y={n.y + 52} className="fill-muted-foreground text-[12.5px]">
        the shared tier, where Redis would be
      </text>
      <text x={n.x + 18} y={n.y + 74} className="fill-muted-foreground text-[12px]">
        admission · scored eviction · single flight
      </text>

      <rect x={barX} y={barY} width={barW} height={10} rx="5" className="fill-muted/60" />
      <rect
        x={barX}
        y={barY}
        width={Math.max(2, barW * frac)}
        height={10}
        rx="5"
        className="fill-primary"
      />
      <text x={barX} y={barY + 30} className="fill-foreground font-mono text-[12px]">
        {bytes(used)} of {bytes(capacity)} · {Number(objects).toLocaleString()} objects
      </text>
      <text x={barX} y={barY + 48} className="fill-muted-foreground font-mono text-[11.5px]">
        {Number(suppressed).toLocaleString()} origin calls stopped
      </text>
    </g>
  )
}

function Note({ title, body, stat }) {
  return (
    <div className="rounded-xl border border-border bg-background px-3.5 py-3">
      <div className="text-[13px] font-semibold">{title}</div>
      <p className="mt-1 text-[12.5px] leading-snug text-muted-foreground">{body}</p>
      {stat && <div className="mt-2 font-mono text-[12px] text-primary">{stat}</div>}
    </div>
  )
}
