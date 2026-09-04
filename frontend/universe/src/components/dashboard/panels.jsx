import { useEffect, useMemo, useRef, useState } from "react"
import gsap from "gsap"

import { Button } from "@/components/ui/button"
import { bytes, cn, ms, pct, usd } from "@/lib/utils"
import { get, post } from "@/hooks/useLiveFeed"
import { Bar, Panel, Sparkline, Stat, Ticker } from "./primitives"

const POLICY_TONES = {
  lru: "bg-slate-400",
  lfu: "bg-sky-400",
  gdsf: "bg-violet-400",
  tiny_lfu: "bg-teal-400",
  cost_aware: "bg-amber-400",
  trend_aware: "bg-rose-400",
}

export function CostPanel({ frame }) {
  const cost = frame?.cost ?? {}
  const baselines = cost.baselines ?? {}
  const rows = Object.entries(baselines).map(([policy, v]) => ({
    policy,
    total: typeof v === "object" ? v.total_usd : v,
    hit: typeof v === "object" ? v.hit_rate : null,
  }))
  rows.push({ policy: "aura", total: cost.total_usd ?? 0, hit: frame?.layers?.l2?.hit_rate ?? 0 })

  const max = Math.max(...rows.map((r) => r.total || 0), 1e-9)

  return (
    <Panel
      title="Cost against the baselines"
      subtitle="Every policy is replayed on the identical request stream, priced with the same cost model."
    >
      <div className="space-y-2.5">
        {rows.map((r) => (
          <div key={r.policy} className="grid grid-cols-[5.5rem_1fr_5rem] items-center gap-3">
            <span
              className={cn(
                "font-mono text-xs",
                r.policy === "aura" ? "font-semibold text-primary" : "text-muted-foreground"
              )}
            >
              {r.policy}
            </span>
            <Bar
              fraction={(r.total || 0) / max}
              tone={r.policy === "aura" ? "bg-primary" : "bg-muted-foreground/40"}
            />
            <span className="text-right font-mono text-xs tabular-nums">
              {usd(r.total, 2)}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Total spend"
          value={<Ticker value={cost.total_usd} format={(v) => usd(v)} />}
        />
        <Stat
          label="Avoided"
          tone="good"
          value={<Ticker value={cost.saved_vs_no_cache_usd} format={(v) => usd(v)} />}
          hint="versus no cache at all"
        />
        <Stat
          label="SLA penalty"
          tone={cost.sla_penalty_usd > 0 ? "warn" : "default"}
          value={<Ticker value={cost.sla_penalty_usd} format={(v) => usd(v)} />}
        />
        <Stat
          label="Burn rate"
          value={<Ticker value={cost.burn_rate_usd_per_hour} format={(v) => `${usd(v)}/hr`} />}
        />
      </div>
    </Panel>
  )
}

export function LayerPanel({ frame }) {
  const l2 = frame?.layers?.l2 ?? {}
  const l1 = frame?.layers?.l1 ?? {}
  const latency = frame?.latency ?? {}
  const engine = frame?.engine ?? {}

  return (
    <Panel title="Cache and latency" subtitle="Read path is a lookup; no model runs on a hit.">
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat
          label="L2 hit rate"
          tone="good"
          value={<Ticker value={l2.hit_rate} format={(v) => pct(v)} />}
          hint={`${(l2.hits ?? 0).toLocaleString()} hits`}
        />
        <Stat
          label="Byte hit rate"
          value={<Ticker value={l2.byte_hit_rate} format={(v) => pct(v)} />}
        />
        <Stat label="L1 window" value={<Ticker value={l1.hit_rate} format={(v) => pct(v)} />} />
        <Stat
          label="Backend calls"
          value={<Ticker value={l2.misses} format={(v) => Math.round(v).toLocaleString()} />}
        />
        <Stat label="p50" value={<Ticker value={latency.p50_ms} format={ms} />} />
        <Stat label="p95" value={<Ticker value={latency.p95_ms} format={ms} />} />
        <Stat label="p99" value={<Ticker value={latency.p99_ms} format={ms} />} />
        <Stat
          label="Decision cost"
          value={<Ticker value={engine.decision_overhead_us_p50} format={(v) => `${v.toFixed(1)} µs`} />}
          hint="p50, write path only"
        />
      </div>
    </Panel>
  )
}

export function PolicyPanel({ frame }) {
  const policy = frame?.policy ?? {}
  const mixture = policy.mixture ?? {}
  const workload = frame?.workload ?? {}
  const entries = Object.entries(mixture)

  return (
    <Panel
      title="Policy mixture"
      subtitle="Thompson sampling over six experts. The blend moves as the workload does."
    >
      <div className="mb-3 flex h-3 w-full overflow-hidden rounded-full">
        {entries.map(([name, weight]) => (
          <MixtureSegment key={name} name={name} weight={weight} />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
        {entries.map(([name, weight]) => (
          <div key={name} className="flex items-center gap-2 text-xs">
            <span className={cn("h-2 w-2 shrink-0 rounded-full", POLICY_TONES[name] ?? "bg-muted")} />
            <span className="truncate text-muted-foreground">{name}</span>
            <span className="ml-auto font-mono tabular-nums">{pct(weight, 0)}</span>
          </div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Regime" value={<span className="text-lg">{workload.regime ?? "—"}</span>}
          hint={`confidence ${pct(workload.confidence ?? 0, 0)}`} />
        <Stat label="Predictor" value={<span className="text-lg">{policy.predictor ?? "—"}</span>}
          hint={`confidence ${pct(policy.predictor_confidence ?? 0, 0)}`} />
        <Stat label="Scan score"
          value={<Ticker value={workload.features?.scan_score} format={(v) => v.toFixed(3)} />} />
        <Stat label="Bandit regret"
          value={<Ticker value={policy.bandit_regret} format={(v) => v.toFixed(4)} />} />
      </div>
    </Panel>
  )
}

function MixtureSegment({ name, weight }) {
  const ref = useRef(null)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    const tween = gsap.to(node, {
      width: `${(weight || 0) * 100}%`,
      duration: 0.55,
      ease: "power2.out",
    })
    return () => tween.kill()
  }, [weight])
  return <div ref={ref} className={cn("h-full w-0", POLICY_TONES[name] ?? "bg-muted")} />
}

export function CapacityPanel({ frame }) {
  const cap = frame?.capacity ?? {}
  const marginal = cap.marginal?.[0] ?? {}
  const mrc = (cap.mrc ?? []).map((p) => ({ x: p.bytes, y: p.hit_rate }))

  const decisionTone =
    cap.decision === "ScaleUp" ? "good" : cap.decision === "ScaleDown" ? "warn" : "default"

  return (
    <Panel
      title="Capacity control"
      subtitle="Memory is bought only when the next block pays for itself."
    >
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="Pool" value={bytes(cap.logical_bytes)} hint={`${bytes(cap.used_bytes)} used`} />
        <Stat label="Pressure" value={<Ticker value={cap.pressure} format={(v) => pct(v, 0)} />} />
        <Stat label="Decision" tone={decisionTone} value={<span className="text-lg">{cap.decision ?? "—"}</span>} />
        <Stat
          label="Net of next step"
          tone={marginal.net_usd_hr > 0 ? "good" : "bad"}
          value={<Ticker value={marginal.net_usd_hr} format={(v) => `${usd(v)}/hr`} />}
          hint={`+${pct(marginal.delta_hit_rate ?? 0, 1)} hit rate`}
        />
      </div>
      <p className="mt-3 text-xs text-muted-foreground">{cap.reason}</p>
      <div className="mt-3">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
          Miss ratio curve
        </div>
        <Sparkline points={mrc} tone="stroke-primary" />
      </div>
    </Panel>
  )
}

/// New decisions animate in from the top. Without that, a fast feed reads as a flicker
/// and it is impossible to see that anything changed.
export function DecisionFeed({ frame }) {
  const decisions = frame?.recent_decisions ?? []
  const listRef = useRef(null)
  const seen = useRef(new Set())

  useEffect(() => {
    const node = listRef.current
    if (!node) return
    const fresh = Array.from(node.children).filter((el) => !seen.current.has(el.dataset.k))
    fresh.forEach((el) => seen.current.add(el.dataset.k))
    if (seen.current.size > 400) seen.current = new Set()
    if (!fresh.length) return
    const tween = gsap.fromTo(
      fresh,
      { opacity: 0, x: -10 },
      { opacity: 1, x: 0, duration: 0.35, stagger: 0.03, ease: "power2.out" }
    )
    return () => tween.kill()
  }, [decisions])

  return (
    <Panel
      title="Decisions"
      subtitle="Why each object was admitted or refused, with the numbers behind it."
      className="h-full"
    >
      <div ref={listRef} className="max-h-[26rem] space-y-1.5 overflow-y-auto pr-1">
        {decisions.length === 0 && (
          <p className="text-xs text-muted-foreground">waiting for traffic</p>
        )}
        {decisions.map((d, i) => (
          <article
            key={`${d.key}-${d.t}-${i}`}
            data-k={`${d.key}-${d.t}`}
            className="rounded-lg border border-border/60 bg-background/40 p-2.5"
          >
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "rounded px-1.5 py-0.5 font-mono text-[10px] uppercase",
                  d.action === "Admit"
                    ? "bg-emerald-500/15 text-emerald-500"
                    : "bg-rose-500/15 text-rose-500"
                )}
              >
                {d.action}
              </span>
              <span className="truncate font-mono text-xs">{d.key}</span>
              <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground">
                {usd(d.economic_value_usd, 6)}
              </span>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
              <span>reuse 60s {pct(d.reuse_probability?.h60s ?? 0, 0)}</span>
              <span>density {Number(d.value_density ?? 0).toFixed(2)}</span>
              <span>bar {Number(d.eviction_threshold ?? 0).toFixed(2)}</span>
            </div>
            {d.reasons?.length > 0 && (
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                {d.reasons[0]}
              </p>
            )}
          </article>
        ))}
      </div>
    </Panel>
  )
}

export function ApplicationsPanel({ frame }) {
  const apps = frame?.applications ?? []
  return (
    <Panel title="Applications" subtitle="Different cost shapes, which is why one global policy underperforms.">
      <div className="space-y-2">
        {apps.length === 0 && <p className="text-xs text-muted-foreground">no traffic yet</p>}
        {apps.map((a) => (
          <div key={a.application} className="rounded-lg border border-border/60 p-2.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium">{a.application}</span>
              <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {a.cost_profile}
              </span>
              <span className="ml-auto font-mono text-xs tabular-nums">{pct(a.hit_rate)}</span>
            </div>
            <div className="mt-1.5">
              <Bar fraction={a.hit_rate} tone="bg-primary/70" />
            </div>
            <div className="mt-1.5 flex flex-wrap gap-x-3 text-[11px] text-muted-foreground">
              <span>{Number(a.requests ?? 0).toLocaleString()} req</span>
              <span>avg {bytes(a.avg_object_bytes)}</span>
              <span>regen {ms(a.regen_p50_ms)}</span>
              <span>{usd(a.cost_usd, 3)}</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

export function EventsPanel({ frame }) {
  const events = frame?.events ?? []
  return (
    <Panel title="Events" className="h-full">
      <div className="max-h-56 space-y-1 overflow-y-auto pr-1">
        {events.length === 0 && <p className="text-xs text-muted-foreground">quiet</p>}
        {events.map((e, i) => (
          <div key={`${e.t}-${i}`} className="flex gap-2 text-[11px]">
            <span className="shrink-0 font-mono text-muted-foreground">
              {(e.t / 1000).toFixed(1)}s
            </span>
            <span className="shrink-0 font-medium">{e.kind}</span>
            <span className="truncate text-muted-foreground">{e.detail}</span>
          </div>
        ))}
      </div>
    </Panel>
  )
}

const ATTACKS = [
  "FlashCrowd",
  "Scan",
  "PopularityShift",
  "CostSpike",
  "ExpensiveTail",
  "HotKeyEmergence",
  "WorkingSetExplosion",
  "MixedChaos",
]

export function Controls({ frame, send }) {
  const [scenarios, setScenarios] = useState([])
  const [busy, setBusy] = useState(null)
  const sim = frame?.sim ?? {}

  useEffect(() => {
    get("/v1/scenarios")
      .then((d) => setScenarios(d.scenarios ?? []))
      .catch(() => setScenarios([]))
  }, [])

  const fire = async (attack) => {
    setBusy(attack)
    if (!send({ type: "attack", attack, duration_s: 25 })) {
      await post("/v1/sim/attack", { attack, duration_s: 25 })
    }
    setTimeout(() => setBusy(null), 600)
  }

  return (
    <Panel
      title="Scenario"
      subtitle="Disturb the workload and watch the policy and capacity respond."
    >
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="h-8 rounded-md border border-border bg-background px-2 text-xs"
          value={sim.scenario ?? ""}
          onChange={(e) => post("/v1/sim/start", { scenario: e.target.value, speed: sim.speed ?? 1 })}
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
        <div className="flex items-center gap-1">
          {[1, 2, 4, 8].map((s) => (
            <Button
              key={s}
              size="sm"
              variant={sim.speed === s ? "default" : "outline"}
              onClick={() => {
                if (!send({ type: "speed", speed: s })) post("/v1/sim/speed", { speed: s })
              }}
            >
              {s}x
            </Button>
          ))}
        </div>
        <Button size="sm" variant="secondary" onClick={() => post("/v1/sim/stop")}>
          Pause
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {ATTACKS.map((a) => (
          <Button
            key={a}
            size="sm"
            variant={busy === a ? "default" : "outline"}
            onClick={() => fire(a)}
          >
            {a}
          </Button>
        ))}
      </div>
    </Panel>
  )
}

export function BenchPanel() {
  const [report, setReport] = useState(null)
  const [running, setRunning] = useState(false)
  const [scenario, setScenario] = useState("expensive_tail")

  const run = async () => {
    setRunning(true)
    try {
      const r = await post("/v1/bench/run", {
        scenario,
        policies: ["lru", "lfu", "gdsf", "cost_aware", "aura"],
        capacity_bytes: 134217728,
        requests: 60000,
        seed: 42,
      })
      setReport(r)
    } finally {
      setRunning(false)
    }
  }

  const best = useMemo(() => {
    if (!report?.rows?.length) return null
    return report.rows.reduce((a, b) => (a.total_cost_usd <= b.total_cost_usd ? a : b))
  }, [report])

  return (
    <Panel
      title="Benchmark"
      subtitle="Offline replay of one stream through every policy, plus the Belady ceiling."
      actions={
        <div className="flex items-center gap-2">
          <select
            className="h-8 rounded-md border border-border bg-background px-2 text-xs"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
          >
            {["expensive_tail", "scan_resistance", "flash_crowd", "shifting_popularity", "mixed_production", "steady_zipf"].map(
              (s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              )
            )}
          </select>
          <Button size="sm" onClick={run} disabled={running}>
            {running ? "Running…" : "Run"}
          </Button>
        </div>
      }
    >
      {!report && (
        <p className="text-xs text-muted-foreground">
          Runs in process and takes a few seconds. Every policy sees the identical stream.
        </p>
      )}
      {report && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b border-border">
                <th className="py-1.5 text-left font-medium">policy</th>
                <th className="py-1.5 text-right font-medium">hit</th>
                <th className="py-1.5 text-right font-medium">byte hit</th>
                <th className="py-1.5 text-right font-medium">p95</th>
                <th className="py-1.5 text-right font-medium">cost</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {report.rows.map((r) => (
                <tr
                  key={r.policy}
                  className={cn(
                    "border-b border-border/40",
                    r.policy === best?.policy && "bg-primary/5 font-semibold text-primary"
                  )}
                >
                  <td className="py-1.5 text-left">{r.policy}</td>
                  <td className="py-1.5 text-right">{pct(r.object_hit_rate)}</td>
                  <td className="py-1.5 text-right">{pct(r.byte_hit_rate)}</td>
                  <td className="py-1.5 text-right">{ms(r.p95_latency_ms)}</td>
                  <td className="py-1.5 text-right">{usd(r.total_cost_usd)}</td>
                </tr>
              ))}
              <tr className="text-muted-foreground">
                <td className="py-1.5 text-left">belady</td>
                <td className="py-1.5 text-right">{pct(report.belady_upper_bound?.object_hit_rate ?? 0)}</td>
                <td className="py-1.5 text-right">—</td>
                <td className="py-1.5 text-right">—</td>
                <td className="py-1.5 text-right">{usd(report.belady_upper_bound?.total_cost_usd ?? 0)}</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-muted-foreground">
            Winner <span className="font-medium text-foreground">{report.winner}</span>.{" "}
            {Object.entries(report.improvement_vs ?? {})
              .map(([k, v]) => `${pct(v, 1)} cheaper than ${k}`)
              .join(", ")}
          </p>
        </div>
      )}
    </Panel>
  )
}
