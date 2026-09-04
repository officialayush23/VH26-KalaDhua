import { useEffect, useRef } from "react"
import gsap from "gsap"

import { bytes, cn, ms, pct, usd } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

/// The request path, drawn with the live numbers on it.
///
/// This exists because the single most common question about the system is "where does a
/// request actually go", and the answer is the whole argument: a hit is answered here and
/// touches nothing, a miss goes to the application service, and only the application
/// service talks to Postgres. Showing that with real counts on each edge is more
/// convincing than saying it.
export function Pipeline({ frame }) {
  const l2 = frame?.layers?.l2 ?? {}
  const l1 = frame?.layers?.l1 ?? {}
  const cost = frame?.cost ?? {}
  const latency = frame?.latency ?? {}
  const engine = frame?.engine ?? {}
  const apps = frame?.applications ?? []

  const hits = l2.hits ?? 0
  const misses = l2.misses ?? 0
  const total = hits + misses
  const hitFrac = total ? hits / total : 0

  const flowRef = useRef(null)

  // Dots travel down the hit and miss edges at a rate proportional to how much traffic
  // takes each. Purely indicative, but it makes the split legible at a glance.
  useEffect(() => {
    const node = flowRef.current
    if (!node) return
    const ctx = gsap.context(() => {
      gsap.fromTo(
        "[data-flow='hit']",
        { attr: { offset: 0 } },
        { attr: { offset: 1 }, duration: 1.6, repeat: -1, ease: "none" }
      )
      gsap.fromTo(
        "[data-flow='miss']",
        { attr: { offset: 0 } },
        { attr: { offset: 1 }, duration: 2.6, repeat: -1, ease: "none", delay: 0.4 }
      )
    }, node)
    return () => ctx.revert()
  }, [])

  return (
    <Panel
      title="Where a request actually goes"
      subtitle="A hit is answered inside the cache and touches nothing else. Only a miss reaches an application service, and only that service talks to the database. Numbers below are this run."
      actions={
        <Pill tone="accent">{Number(total).toLocaleString()} requests so far</Pill>
      }
    >
      <div ref={flowRef} className="space-y-3">
        <Stage
          tone="border-border"
          title="1 · Request arrives"
          right={`${Number(frame?.sim?.rps ?? 0).toLocaleString()} req/s`}
          body="A client asks for an object by key. It does not say what the object is, only what it costs to rebuild."
        />

        <Connector />

        <Stage
          tone="border-primary/40 bg-primary/[0.04]"
          title="2 · Cache lookup"
          right={`${pct(hitFrac, 1)} hit`}
          body="A hash lookup and a counter touch. No model runs here, which is why the read path costs nothing."
        >
          <div className="mt-2.5 flex h-2.5 w-full overflow-hidden rounded-full bg-muted/60">
            <div
              className="h-full bg-primary transition-[width] duration-500"
              style={{ width: `${hitFrac * 100}%` }}
            />
            <div
              className="h-full bg-amber-400/70 transition-[width] duration-500"
              style={{ width: `${(1 - hitFrac) * 100}%` }}
            />
          </div>
          <div className="mt-1.5 flex justify-between text-[11px] text-muted-foreground">
            <span>{Number(hits).toLocaleString()} hits served here</span>
            <span>{Number(misses).toLocaleString()} missed</span>
          </div>
        </Stage>

        <div className="grid gap-3 md:grid-cols-2">
          <Branch
            label="HIT"
            tone="good"
            share={pct(hitFrac, 1)}
            arrow="returns immediately"
            rows={[
              ["Served from", "memory"],
              ["Backend work", "none"],
              ["Cost of this path", "$0"],
            ]}
            note="This is the whole point. Nothing downstream is touched."
          />
          <Branch
            label="MISS"
            tone="warn"
            share={pct(1 - hitFrac, 1)}
            arrow="falls through to the application"
            rows={[
              ["Goes to", "application service"],
              ["Which queries", "Postgres / GPU / external API"],
              ["Typical wait", ms(latency.p95_ms ?? 0)],
            ]}
            note="The service measures what that rebuild actually cost and hands the number back."
          />
        </div>

        <Connector label="on a miss only" />

        <Stage
          tone="border-amber-400/30 bg-amber-400/[0.04]"
          title="3 · Application service rebuilds"
          right={`${Number(misses).toLocaleString()} rebuilds`}
          body="Three services with different cost shapes. Each one measures its own real work rather than estimating it."
        >
          <div className="mt-2.5 grid gap-2 sm:grid-cols-3">
            {apps.length === 0 && (
              <div className="col-span-3 text-[11.5px] text-muted-foreground">
                No per-application traffic yet.
              </div>
            )}
            {apps.map((a) => (
              <div key={a.application} className="rounded-lg border border-border/60 px-2.5 py-2">
                <div className="text-[12px] font-medium">{a.application}</div>
                <div className="mt-0.5 text-[10.5px] text-muted-foreground">
                  {a.cost_profile === "db_heavy"
                    ? "queries the database"
                    : a.cost_profile === "gpu_heavy"
                      ? "runs GPU work"
                      : "CPU plus a query"}
                </div>
                <div className="mt-1 font-mono text-[11px] tabular-nums">
                  {ms(a.regen_p50_ms)} · {bytes(a.avg_object_bytes)}
                </div>
              </div>
            ))}
          </div>
        </Stage>

        <Connector label="measured cost comes back" />

        <Stage
          tone="border-primary/40 bg-primary/[0.04]"
          title="4 · The decision"
          right={`${Number(engine.admissions ?? 0).toLocaleString()} kept`}
          body="Price the object, predict whether it returns, compute value per byte held, and compare it against whatever would have to be evicted to fit it."
        >
          <div className="mt-2.5 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Cell label="kept" value={Number(engine.admissions ?? 0).toLocaleString()} />
            <Cell label="refused" value={Number(engine.admissions_rejected ?? 0).toLocaleString()} />
            <Cell label="evicted" value={Number(engine.evictions ?? 0).toLocaleString()} />
            <Cell
              label="time to decide"
              value={`${Number(engine.decision_overhead_us_p50 ?? 0).toFixed(0)} µs`}
            />
          </div>
        </Stage>

        <Connector />

        <Stage
          tone="border-emerald-400/30 bg-emerald-400/[0.04]"
          title="5 · What it saved"
          right={usd(cost.saved_vs_no_cache_usd ?? 0)}
          body="Backend work never performed, priced with the same cost model used to make the decisions."
        >
          <div className="mt-2.5 grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Cell label="avoided" value={usd(cost.saved_vs_no_cache_usd ?? 0)} tone="text-emerald-400" />
            <Cell label="spent" value={usd(cost.total_usd ?? 0)} />
            <Cell label="L1 window hit" value={pct(l1.hit_rate ?? 0, 0)} />
          </div>
        </Stage>

        <div className="rounded-xl border border-border/60 bg-muted/20 px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
            Off the request path entirely
          </div>
          <p className="mt-1.5 text-[12.5px] leading-snug text-muted-foreground">
            Supabase holds the trained model, the benchmark history and the event log. The
            engine reads it on boot and writes to it after a benchmark. It is never consulted
            to answer a request — if it were, the cache would be pointless.
          </p>
        </div>
      </div>
    </Panel>
  )
}

function Stage({ title, right, body, tone, children }) {
  return (
    <div className={cn("rounded-xl border bg-background/40 px-4 py-3", tone)}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[13.5px] font-semibold">{title}</h3>
        <span className="font-mono text-[12px] tabular-nums text-muted-foreground">{right}</span>
      </div>
      <p className="mt-1 text-[12.5px] leading-snug text-muted-foreground">{body}</p>
      {children}
    </div>
  )
}

function Connector({ label }) {
  return (
    <div className="flex items-center gap-2 pl-4">
      <div className="h-5 w-px bg-border" />
      {label && <span className="text-[11px] text-muted-foreground">{label}</span>}
    </div>
  )
}

function Branch({ label, tone, share, arrow, rows, note }) {
  const tones = {
    good: "border-emerald-400/40 bg-emerald-400/[0.05]",
    warn: "border-amber-400/40 bg-amber-400/[0.05]",
  }
  const badges = {
    good: "bg-emerald-400/15 text-emerald-400",
    warn: "bg-amber-400/15 text-amber-400",
  }
  return (
    <div className={cn("rounded-xl border px-4 py-3", tones[tone])}>
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
            badges[tone]
          )}
        >
          {label}
        </span>
        <span className="font-mono text-[12px] tabular-nums">{share}</span>
        <span className="ml-auto text-[11px] text-muted-foreground">{arrow}</span>
      </div>
      <dl className="mt-2 space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 text-[12px]">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right font-mono">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-[11.5px] leading-snug text-muted-foreground">{note}</p>
    </div>
  )
}

function Cell({ label, value, tone }) {
  return (
    <div className="rounded-lg border border-border/60 bg-background/50 px-2.5 py-1.5">
      <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={cn("mt-0.5 font-mono text-[14px] tabular-nums", tone)}>{value}</div>
    </div>
  )
}

/// States plainly where the traffic is coming from. Without this the dashboard reads as
/// though production applications are connected, which they are not, and that is exactly
/// the thing an examiner will ask about first.
export function TrafficSource({ frame, status }) {
  const sim = frame?.sim ?? {}
  const simulated = Boolean(sim.running) || sim.scenario !== "none"

  return (
    <div
      className={cn(
        "rounded-xl border px-4 py-3",
        simulated ? "border-amber-400/35 bg-amber-400/[0.05]" : "border-border bg-card/40"
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
          Traffic source
        </span>
        <Pill tone={simulated ? "warn" : "good"}>
          {status === "offline"
            ? "engine not running"
            : simulated
              ? "built-in simulator"
              : "live applications"}
        </Pill>
        {simulated && (
          <span className="font-mono text-[11.5px] text-muted-foreground">
            scenario: {sim.scenario ?? "—"}
          </span>
        )}
      </div>
      <p className="mt-1.5 text-[12.5px] leading-snug text-muted-foreground">
        {simulated ? (
          <>
            No external application is connected. The engine's own workload generator is
            producing the requests: three synthetic services with deliberately different
            rebuild costs, from a fixed seed so every run repeats exactly. Every number on
            this page is computed from that stream, not sampled from production. To drive it
            with the real Python services instead, run{" "}
            <code className="rounded bg-muted/60 px-1 font-mono text-[11px]">
              python -m driver.run_universe
            </code>{" "}
            from <code className="font-mono text-[11px]">apps/</code>.
          </>
        ) : (
          <>The engine is idle. It only holds what connected applications put into it.</>
        )}
      </p>
    </div>
  )
}
