import { useEffect, useMemo, useRef, useState } from "react"
import gsap from "gsap"

import { Button } from "@/components/ui/button"
import { bytes, cn, ms, pct, usd } from "@/lib/utils"
import { get, post } from "@/hooks/useLiveFeed"
import { Bar, Gauge, Legend, LineChart, Metric, Panel, Pill } from "./primitives"

const POLICY_TONES = {
  lru: "bg-slate-400",
  lfu: "bg-sky-400",
  gdsf: "bg-violet-400",
  tiny_lfu: "bg-teal-400",
  cost_aware: "bg-amber-400",
  trend_aware: "bg-rose-400",
}

const POLICY_MEANING = {
  lru: "keeps whatever was touched most recently",
  lfu: "keeps whatever is touched most often",
  gdsf: "weighs frequency against cost and size",
  tiny_lfu: "frequency sketch, resists one-off keys",
  cost_aware: "keeps whatever is most expensive per byte",
  trend_aware: "keeps whatever is rising fastest",
}

/// The headline row. Four numbers that answer, in order: is it working, how much did it
/// save, how fast is it, and is it about to run out of room.
export function Headline({ frame, history }) {
  const l2 = frame?.layers?.l2 ?? {}
  const cost = frame?.cost ?? {}
  const cap = frame?.capacity ?? {}
  const latency = frame?.latency ?? {}

  const savings = cost.savings_vs ?? {}
  const best = Object.entries(savings).reduce(
    (acc, [k, v]) => (v > acc.v ? { k, v } : acc),
    { k: null, v: -Infinity }
  )

  const spark = history.slice(-80).map((h) => ({ x: h.t, y: h.hitRate }))
  const costSpark = history.slice(-80).map((h) => ({ x: h.t, y: h.savedUsd }))

  return (
    <div className="grid gap-4 xl:grid-cols-4">
      <Panel
        title="Requests served from cache"
        subtitle="Share of requests answered without touching a backend."
        className="xl:col-span-1"
      >
        <div className="flex items-center justify-center">
          <Gauge
            fraction={l2.hit_rate ?? 0}
            value={pct(l2.hit_rate ?? 0, 1)}
            label="object hit rate"
            tone="stroke-emerald-400"
          />
        </div>
        <div className="grid grid-cols-2 gap-3 text-[12px]">
          <div>
            <div className="text-muted-foreground">Hits</div>
            <div className="font-mono tabular-nums">{(l2.hits ?? 0).toLocaleString()}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Backend calls</div>
            <div className="font-mono tabular-nums">{(l2.misses ?? 0).toLocaleString()}</div>
          </div>
        </div>
      </Panel>

      <Panel
        title="Money not spent"
        subtitle="Backend work the cache avoided, priced with the cost model."
        className="xl:col-span-2"
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <Metric
            label="Avoided so far"
            value={usd(cost.saved_vs_no_cache_usd ?? 0)}
            tone="good"
            size="lg"
            explain="What running with no cache at all would have cost, minus what this run cost."
          />
          <Metric
            label={best.k ? `Cheaper than ${best.k}` : "Versus baselines"}
            value={best.k && best.v > -Infinity ? pct(best.v, 1) : "—"}
            tone={best.v > 0 ? "good" : "bad"}
            size="lg"
            explain="Same request stream replayed through the classical policy, same prices."
          />
        </div>
        <div className="mt-4">
          <LineChart
            points={costSpark}
            height={120}
            tone="stroke-emerald-400"
            fill="fill-emerald-400/10"
            yFormat={(v) => `$${v.toFixed(1)}`}
            xFormat={(v) => `${v.toFixed(0)}s`}
            yLabel="dollars avoided"
            xLabel="simulated seconds"
          />
        </div>
      </Panel>

      <Panel
        title="Speed and headroom"
        subtitle="What users feel, and how close the pool is to full."
        className="xl:col-span-1"
      >
        <div className="space-y-3">
          <Metric
            label="p95 response"
            value={ms(latency.p95_ms ?? 0)}
            explain="95 of every 100 requests are faster than this."
            tone={(latency.p95_ms ?? 0) > 150 ? "warn" : "good"}
          />
          <div className="rounded-xl border border-border bg-background/80 px-4 py-3">
            <div className="flex items-baseline justify-between">
              <span className="text-[12px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
                Memory in use
              </span>
              <span className="font-mono text-sm tabular-nums">{pct(cap.pressure ?? 0, 0)}</span>
            </div>
            <div className="mt-2">
              <Bar
                fraction={cap.pressure ?? 0}
                tone={(cap.pressure ?? 0) > 0.9 ? "bg-amber-400" : "bg-primary"}
              />
            </div>
            <div className="mt-2 text-[12px] text-muted-foreground">
              {bytes(cap.used_bytes ?? 0)} of {bytes(cap.logical_bytes ?? 0)}
            </div>
          </div>
        </div>
        <div className="mt-3">
          <LineChart
            points={spark}
            height={80}
            yFormat={(v) => pct(v, 0)}
            xFormat={(v) => `${v.toFixed(0)}s`}
            yLabel="hit rate over time"
          />
        </div>
      </Panel>
    </div>
  )
}

export function CostPanel({ frame }) {
  const cost = frame?.cost ?? {}
  const baselines = cost.baselines ?? {}
  const rows = Object.entries(baselines).map(([policy, v]) => ({
    policy,
    total: typeof v === "object" ? v.total_usd : v,
    hit: typeof v === "object" ? v.hit_rate : null,
    regen: typeof v === "object" ? v.regen_usd : null,
    penalty: typeof v === "object" ? v.penalty_usd : null,
  }))
  rows.push({
    policy: "aura",
    total: cost.total_usd ?? 0,
    hit: frame?.layers?.l2?.hit_rate ?? 0,
    regen: cost.backend_usd ?? 0,
    penalty: cost.sla_penalty_usd ?? 0,
  })
  rows.sort((a, b) => (a.total ?? 0) - (b.total ?? 0))
  const max = Math.max(...rows.map((r) => r.total || 0), 1e-9)

  return (
    <Panel
      title="Total cost by policy"
      subtitle="Every policy replayed on the identical request stream and priced with the same model. Lower is better."
      footer="Cost is regeneration plus SLA penalty plus the rent on the memory held. A policy with a better hit rate can still lose if it kept the cheap objects."
    >
      <div className="space-y-3">
        {rows.map((r, i) => (
          <div key={r.policy}>
            <div className="mb-1 flex items-baseline justify-between gap-3">
              <span className="flex items-center gap-2">
                <span
                  className={cn(
                    "font-mono text-[13.5px]",
                    r.policy === "aura" ? "font-semibold text-primary" : "text-foreground"
                  )}
                >
                  {r.policy}
                </span>
                {i === 0 && <Pill tone="good">cheapest</Pill>}
                {r.policy === "aura" && i !== 0 && <Pill tone="accent">this engine</Pill>}
              </span>
              <span className="font-mono text-[13.5px] tabular-nums">{usd(r.total)}</span>
            </div>
            <Bar
              fraction={(r.total || 0) / max}
              height="h-2.5"
              tone={r.policy === "aura" ? "bg-primary" : "bg-muted-foreground/35"}
            />
            <div className="mt-1 flex flex-wrap gap-x-4 text-[12px] text-muted-foreground">
              <span>hit rate {r.hit !== null ? pct(r.hit) : "—"}</span>
              {r.regen !== null && <span>rebuild {usd(r.regen)}</span>}
              {r.penalty !== null && <span>SLA penalty {usd(r.penalty)}</span>}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-4">
        <Metric label="Rebuild cost" value={usd(cost.backend_usd ?? 0)} size="sm"
          explain="Backend work actually paid for." />
        <Metric label="Memory rent" value={usd(cost.cache_usd ?? 0)} size="sm"
          explain="What holding the cached bytes costs." />
        <Metric label="SLA penalty" value={usd(cost.sla_penalty_usd ?? 0)} size="sm"
          tone={(cost.sla_penalty_usd ?? 0) > 0 ? "warn" : "default"}
          explain="Charged when a response is slower than the target." />
        <Metric label="Burn rate" value={`${usd(cost.burn_rate_usd_per_hour ?? 0)}`} unit="/hr" size="sm"
          explain="Current spend extrapolated to an hour." />
      </div>
    </Panel>
  )
}

export function PolicyPanel({ frame }) {
  const policy = frame?.policy ?? {}
  const mixture = policy.mixture ?? {}
  const workload = frame?.workload ?? {}
  const f = workload.features ?? {}
  const entries = Object.entries(mixture).sort((a, b) => b[1] - a[1])
  const leader = entries[0]

  const REGIME_EXPLAIN = {
    Steady: "Traffic is skewed but stable. Nothing unusual to defend against.",
    FlashCrowd: "Traffic has collapsed onto a few keys. Keep them, evict everything else.",
    Scan: "A sweep of one-off keys is passing through. Admitting them would flush the working set.",
    Shifting: "The popular set is being replaced. Old favourites are going cold.",
    Expensive: "The costly objects are the rare ones. Hit rate and cost disagree here.",
    Growing: "The working set is outgrowing the pool.",
  }

  return (
    <Panel
      title="How the cache is deciding right now"
      subtitle="Six classical strategies run as competing experts. The engine blends them by how well each is paying off, and the blend shifts as traffic changes."
      actions={<Pill tone="accent">{policy.predictor ?? "—"} predictor</Pill>}
    >
      <div className="mb-2 flex h-4 w-full overflow-hidden rounded-lg">
        {Object.entries(mixture).map(([name, weight]) => (
          <MixtureSegment key={name} name={name} weight={weight} />
        ))}
      </div>
      <Legend
        items={Object.entries(mixture).map(([name, weight]) => ({
          label: name,
          tone: POLICY_TONES[name] ?? "bg-muted",
          value: pct(weight, 0),
        }))}
      />

      {leader && (
        <p className="mt-3 rounded-lg border border-border bg-background px-3 py-2 text-[13.5px] leading-snug">
          <span className="font-medium">Leaning on {leader[0]}</span>{" "}
          <span className="text-muted-foreground">
            ({pct(leader[1], 0)} of the blend) — {POLICY_MEANING[leader[0]]}.
          </span>
        </p>
      )}

      <div className="mt-4 rounded-xl border border-border bg-background/40 p-4">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
            Detected traffic pattern
          </span>
          <Pill tone={workload.regime === "Scan" ? "warn" : "default"}>
            {workload.regime ?? "—"} · {pct(workload.confidence ?? 0, 0)} confident
          </Pill>
        </div>
        <p className="mt-2 text-[13.5px] leading-snug text-muted-foreground">
          {REGIME_EXPLAIN[workload.regime] ?? "Waiting for enough traffic to classify."}
        </p>
        <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-2 sm:grid-cols-3">
          <Signal label="Burstiness" value={f.burstiness} explain="how concentrated traffic is" />
          <Signal label="Entropy" value={f.entropy} explain="how spread out the keys are" />
          <Signal label="Scan score" value={f.scan_score} explain="share of never-seen keys" />
          <Signal label="Popularity shift" value={f.popularity_shift} explain="how fast the hot set turns over" />
          <Signal label="Working set growth" value={f.working_set_growth} explain="is the key space expanding" />
          <Signal label="Reuse gap p50" value={f.reuse_distance_p50} explain="typical ms between repeats" fmt={(v) => `${v.toFixed(0)} ms`} />
        </div>
      </div>
    </Panel>
  )
}

function Signal({ label, value, explain, fmt }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] text-muted-foreground">{label}</span>
        <span className="font-mono text-[12px] tabular-nums">
          {value === undefined ? "—" : fmt ? fmt(value) : Number(value).toFixed(3)}
        </span>
      </div>
      <div className="text-[12px] leading-tight text-muted-foreground">{explain}</div>
    </div>
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
  const m = cap.marginal?.[0] ?? {}
  const mrc = (cap.mrc ?? []).map((p) => ({ x: p.bytes, y: p.hit_rate }))

  const tone =
    cap.decision === "ScaleUp" ? "good" : cap.decision === "ScaleDown" ? "warn" : "default"

  return (
    <Panel
      title="Should we buy more memory?"
      subtitle="The engine answers this in dollars, not in a utilisation threshold: it prices the hit rate the next block would buy against what that block costs to rent."
      actions={<Pill tone={tone}>{cap.decision ?? "—"}</Pill>}
      footer={cap.reason}
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Pool size now" value={bytes(cap.logical_bytes ?? 0)} size="sm"
          explain={`${bytes(cap.used_bytes ?? 0)} of it in use`} />
        <Metric label="Next step would be" value={bytes(m.to_bytes ?? 0)} size="sm"
          explain={`buys ${pct(m.delta_hit_rate ?? 0, 2)} more hit rate`} />
        <Metric label="That saves" value={`${usd(m.backend_savings_usd_hr ?? 0)}`} unit="/hr" size="sm"
          tone="good" explain="backend work no longer paid for" />
        <Metric label="Net result" value={`${usd(m.net_usd_hr ?? 0)}`} unit="/hr" size="sm"
          tone={(m.net_usd_hr ?? 0) > 0 ? "good" : "bad"}
          explain={`after ${usd(m.cache_cost_usd_hr ?? 0)}/hr of memory rent`} />
      </div>
      <div className="mt-4">
        <div className="mb-1 text-[12px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
          Hit rate against pool size
        </div>
        <LineChart
          points={mrc}
          height={150}
          yFormat={(v) => pct(v, 0)}
          xFormat={(v) => bytes(v)}
          yLabel="hit rate"
          xLabel="pool size"
        />
        <p className="mt-1 text-[12px] text-muted-foreground">
          The curve flattens as it goes right. Once the extra hit rate is worth less than
          the rent, buying more memory loses money.
        </p>
      </div>
    </Panel>
  )
}

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
      { opacity: 0, x: -8 },
      { opacity: 1, x: 0, duration: 0.3, stagger: 0.025, ease: "power2.out" }
    )
    return () => tween.kill()
  }, [decisions])

  return (
    <Panel
      title="Live decisions"
      subtitle="Each object the cache was offered, and why it was kept or refused."
      className="h-full"
      footer="Value density is expected value divided by what it costs to hold. The bar is the density of the object that would have to be evicted to make room."
    >
      <div ref={listRef} className="max-h-[30rem] space-y-2 overflow-y-auto pr-1">
        {decisions.length === 0 && (
          <p className="text-[13.5px] text-muted-foreground">Waiting for traffic.</p>
        )}
        {decisions.map((d, i) => (
          <article
            key={`${d.key}-${d.t}-${i}`}
            data-k={`${d.key}-${d.t}`}
            className="rounded-xl border border-border bg-background/40 p-3"
          >
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                  d.action === "Admit"
                    ? "bg-emerald-500/15 text-emerald-400"
                    : "bg-rose-500/15 text-rose-400"
                )}
              >
                {d.action === "Admit" ? "kept" : "refused"}
              </span>
              <span className="truncate font-mono text-[12px]">{d.key}</span>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2 text-[12px]">
              <Field label="rebuild cost" value={usd(d.economic_value_usd ?? 0, 6)} />
              <Field label="chance of reuse" value={pct(d.reuse_probability?.h60s ?? 0, 0)} />
              <Field
                label="value vs bar"
                value={`${Number(d.value_density ?? 0).toFixed(1)} / ${Number(d.eviction_threshold ?? 0).toFixed(1)}`}
              />
            </div>
            {d.reasons?.length > 0 && (
              <p className="mt-2 text-[12px] leading-snug text-muted-foreground">
                {d.reasons[0]}
              </p>
            )}
          </article>
        ))}
      </div>
    </Panel>
  )
}

function Field({ label, value }) {
  return (
    <div>
      <div className="text-muted-foreground">{label}</div>
      <div className="font-mono tabular-nums">{value}</div>
    </div>
  )
}

export function ApplicationsPanel({ frame }) {
  const apps = frame?.applications ?? []
  const PROFILE = {
    db_heavy: "rebuilding means a database query",
    gpu_heavy: "rebuilding means GPU work",
    mixed: "rebuilding means CPU plus a query",
  }
  return (
    <Panel
      title="Per application"
      subtitle="Three services with different cost shapes. This is why a single global policy underperforms: the right trade-off for a 40 KB query result is not the right one for a 2 MB media file."
    >
      <div className="space-y-3">
        {apps.length === 0 && <p className="text-[13.5px] text-muted-foreground">No traffic yet.</p>}
        {apps.map((a) => (
          <div key={a.application} className="rounded-xl border border-border bg-background/40 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13.5px] font-medium">{a.application}</span>
              <Pill>{PROFILE[a.cost_profile] ?? a.cost_profile}</Pill>
              <span className="ml-auto font-mono text-[13.5px] tabular-nums">{pct(a.hit_rate)}</span>
            </div>
            <div className="mt-2">
              <Bar fraction={a.hit_rate} tone="bg-primary/70" />
            </div>
            <div className="mt-2 grid grid-cols-4 gap-2 text-[12px]">
              <Field label="requests" value={Number(a.requests ?? 0).toLocaleString()} />
              <Field label="avg size" value={bytes(a.avg_object_bytes)} />
              <Field label="rebuild time" value={ms(a.regen_p50_ms)} />
              <Field label="spent" value={usd(a.cost_usd, 3)} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

export function EnginePanel({ frame }) {
  const e = frame?.engine ?? {}
  const offered = (e.admissions ?? 0) + (e.admissions_rejected ?? 0)
  return (
    <Panel
      title="Engine internals"
      subtitle="What the decision path is actually doing."
    >
      <div className="grid grid-cols-2 gap-3">
        <Metric label="Objects kept" value={Number(e.admissions ?? 0).toLocaleString()} size="sm"
          explain={offered ? `of ${offered.toLocaleString()} offered` : undefined} />
        <Metric label="Objects refused" value={Number(e.admissions_rejected ?? 0).toLocaleString()} size="sm"
          explain="not worth the space they would take" />
        <Metric label="Evictions" value={Number(e.evictions ?? 0).toLocaleString()} size="sm"
          explain="dropped to make room for something better" />
        <Metric label="Resident objects" value={Number(e.resident_objects ?? 0).toLocaleString()} size="sm"
          explain="currently in the cache" />
        <Metric label="Model calls" value={Number(e.inference_calls ?? 0).toLocaleString()} size="sm"
          explain="only on the write path, never on a read" />
        <Metric label="Decision cost" value={`${Number(e.decision_overhead_us_p50 ?? 0).toFixed(1)}`} unit="µs" size="sm"
          explain="median time to decide, write path only" />
      </div>
    </Panel>
  )
}

export function EventsPanel({ frame }) {
  const events = frame?.events ?? []
  return (
    <Panel title="Event log" subtitle="Capacity changes, policy shifts and injected disturbances.">
      <div className="max-h-64 space-y-1.5 overflow-y-auto pr-1">
        {events.length === 0 && <p className="text-[13.5px] text-muted-foreground">Quiet.</p>}
        {events.map((e, i) => (
          <div key={`${e.t}-${i}`} className="flex gap-2.5 text-[12px]">
            <span className="w-14 shrink-0 text-right font-mono text-muted-foreground">
              {(e.t / 1000).toFixed(1)}s
            </span>
            <span className="w-24 shrink-0 font-medium">{e.kind}</span>
            <span className="text-muted-foreground">{e.detail}</span>
          </div>
        ))}
      </div>
    </Panel>
  )
}

const ATTACKS = [
  { id: "FlashCrowd", label: "Flash crowd", hint: "traffic collapses onto a few keys" },
  { id: "Scan", label: "Scan", hint: "a sweep of one-off keys tries to flush the cache" },
  { id: "PopularityShift", label: "Popularity shift", hint: "the hot set is replaced" },
  { id: "CostSpike", label: "Cost spike", hint: "rebuilding gets much more expensive" },
  { id: "ExpensiveTail", label: "Expensive tail", hint: "the rare objects become the costly ones" },
  { id: "HotKeyEmergence", label: "Hot key appears", hint: "a cold key becomes the hottest" },
  { id: "WorkingSetExplosion", label: "Working set grows", hint: "more distinct keys than fit" },
  { id: "MixedChaos", label: "Mixed chaos", hint: "several at once" },
]

export function Controls({ frame, send, status }) {
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
    setTimeout(() => setBusy(null), 900)
  }

  const current = scenarios.find((s) => s.id === sim.scenario)

  return (
    <Panel
      title="Drive the demo"
      subtitle="Pick a traffic pattern, then throw a disturbance at it and watch the policy blend and pool size react."
      actions={
        <div className="flex items-center gap-2">
          <Pill tone={sim.running ? "good" : "warn"}>
            {sim.running ? "running" : "paused"}
          </Pill>
          <Pill>{Number(sim.rps ?? 0).toLocaleString()} req/s</Pill>
        </div>
      }
      footer={current?.description}
    >
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-[12px] uppercase tracking-wide text-muted-foreground">
            Traffic pattern
          </label>
          <select
            className="h-9 min-w-52 rounded-lg border border-border bg-background px-2.5 text-[13.5px]"
            value={sim.scenario ?? ""}
            onChange={(e) => post("/v1/sim/start", { scenario: e.target.value, speed: sim.speed ?? 1 })}
            disabled={status === "offline"}
          >
            {scenarios.length === 0 && <option>engine offline</option>}
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[12px] uppercase tracking-wide text-muted-foreground">
            Speed
          </label>
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
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[12px] uppercase tracking-wide text-muted-foreground">
            Traffic
          </label>
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={sim.running ? "outline" : "default"}
              disabled={status === "offline" || sim.running}
              onClick={() =>
                post("/v1/sim/start", {
                  scenario: sim.scenario && sim.scenario !== "none" ? sim.scenario : "mixed_production",
                  speed: sim.speed ?? 1,
                })
              }
            >
              Start
            </Button>
            <Button
              size="sm"
              variant={sim.running ? "secondary" : "outline"}
              disabled={status === "offline" || !sim.running}
              onClick={() => post("/v1/sim/stop")}
            >
              Stop
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={status === "offline"}
              title="Restart this scenario from zero with a fresh seed"
              onClick={() =>
                post("/v1/sim/start", {
                  scenario: sim.scenario && sim.scenario !== "none" ? sim.scenario : "mixed_production",
                  speed: sim.speed ?? 1,
                  seed: Math.floor(Math.random() * 100000),
                })
              }
            >
              Restart
            </Button>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <label className="text-[12px] uppercase tracking-wide text-muted-foreground">
          Throw a disturbance
        </label>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {ATTACKS.map((a) => (
            <button
              key={a.id}
              onClick={() => fire(a.id)}
              disabled={status === "offline"}
              className={cn(
                "rounded-xl border px-3 py-2 text-left transition-colors disabled:opacity-40",
                busy === a.id
                  ? "border-primary bg-primary/10"
                  : "border-border/70 bg-background/40 hover:border-primary/50 hover:bg-accent/40"
              )}
            >
              <div className="text-[13.5px] font-medium">{a.label}</div>
              <div className="mt-0.5 text-[12px] leading-tight text-muted-foreground">
                {a.hint}
              </div>
            </button>
          ))}
        </div>
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
      title="Head to head benchmark"
      subtitle="Replays one fixed request stream through every policy offline, so the comparison is not affected by timing or luck. Belady is the offline optimum: no online policy can beat it."
      actions={
        <div className="flex items-center gap-2">
          <select
            className="h-9 rounded-lg border border-border bg-background px-2.5 text-[13.5px]"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
          >
            {[
              "expensive_tail",
              "scan_resistance",
              "flash_crowd",
              "shifting_popularity",
              "mixed_production",
              "steady_zipf",
            ].map((s) => (
              <option key={s} value={s}>
                {s.replace(/_/g, " ")}
              </option>
            ))}
          </select>
          <Button size="sm" onClick={run} disabled={running}>
            {running ? "Running…" : "Run benchmark"}
          </Button>
        </div>
      }
      footer={
        report
          ? "Results are also published to Supabase when the engine has credentials."
          : "Takes a few seconds. 60,000 requests through five policies plus the optimal bound."
      }
    >
      {!report && (
        <div className="flex h-40 items-center justify-center rounded-xl border border-dashed border-border text-[13.5px] text-muted-foreground">
          Press run to compare the policies.
        </div>
      )}
      {report && (
        <div className="overflow-x-auto">
          <table className="w-full text-[13.5px]">
            <thead>
              <tr className="border-b border-border text-[12px] uppercase tracking-wide text-muted-foreground">
                <th className="py-2 text-left font-medium">Policy</th>
                <th className="py-2 text-right font-medium">Hit rate</th>
                <th className="py-2 text-right font-medium">Byte hit rate</th>
                <th className="py-2 text-right font-medium">p95</th>
                <th className="py-2 text-right font-medium">Backend calls</th>
                <th className="py-2 text-right font-medium">Total cost</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {report.rows.map((r) => (
                <tr
                  key={r.policy}
                  className={cn(
                    "border-b border-border/70",
                    r.policy === best?.policy && "bg-primary/5 font-semibold text-primary"
                  )}
                >
                  <td className="py-2 text-left">{r.policy}</td>
                  <td className="py-2 text-right">{pct(r.object_hit_rate)}</td>
                  <td className="py-2 text-right">{pct(r.byte_hit_rate)}</td>
                  <td className="py-2 text-right">{ms(r.p95_latency_ms)}</td>
                  <td className="py-2 text-right">{Number(r.backend_requests).toLocaleString()}</td>
                  <td className="py-2 text-right">{usd(r.total_cost_usd)}</td>
                </tr>
              ))}
              <tr className="text-muted-foreground">
                <td className="py-2 text-left">belady (optimal)</td>
                <td className="py-2 text-right">{pct(report.belady_upper_bound?.object_hit_rate ?? 0)}</td>
                <td className="py-2 text-right">—</td>
                <td className="py-2 text-right">—</td>
                <td className="py-2 text-right">—</td>
                <td className="py-2 text-right">{usd(report.belady_upper_bound?.total_cost_usd ?? 0)}</td>
              </tr>
            </tbody>
          </table>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Pill tone="good">winner: {report.winner}</Pill>
            {Object.entries(report.improvement_vs ?? {}).map(([k, v]) => (
              <Pill key={k} tone={v > 0 ? "accent" : "bad"}>
                {pct(v, 1)} vs {k}
              </Pill>
            ))}
          </div>
        </div>
      )}
    </Panel>
  )
}

/// Shows what Supabase is and is not doing. The distinction matters: people assume a
/// database behind a cache is on the request path, and here it deliberately is not.
export function SupabasePanel() {
  const [state, setState] = useState(null)

  useEffect(() => {
    const load = () => get("/v1/supabase").then(setState).catch(() => setState(null))
    load()
    const id = setInterval(load, 15000)
    return () => clearInterval(id)
  }, [])

  const tone = !state?.configured ? "warn" : state.reachable ? "good" : "bad"
  const label = !state?.configured
    ? "not configured"
    : state.reachable
      ? "connected"
      : "configured but unreachable"

  return (
    <Panel
      title="Supabase"
      subtitle="Model registry, benchmark history and events."
      actions={<Pill tone={tone}>{label}</Pill>}
    >
      <div className="space-y-2 text-[13.5px]">
        <Row label="Active models" value={
          state?.active_models?.length ? state.active_models.join(", ") : "none published yet"
        } />
        <Row label="Benchmark runs" value="published automatically after each run" />
        <Row label="Cache reads" value="never — the data path does not touch Postgres" />
      </div>
      <p className="mt-3 text-[12px] leading-snug text-muted-foreground">
        On a miss it is the application service that queries Supabase and reports what that
        query cost. The cache learns from that measured cost. Putting Postgres in front of
        every read would defeat the purpose of a cache.
      </p>
      {!state?.configured && (
        <p className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-[12px]">
          Set <code className="font-mono">SUPABASE_URL</code> and{" "}
          <code className="font-mono">SUPABASE_SERVICE_ROLE_SECRET_KEY</code> in{" "}
          <code className="font-mono">backend/.env</code>, then restart the engine.
        </p>
      )}
    </Panel>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border/70 pb-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-mono text-[12px]">{value}</span>
    </div>
  )
}


/// Answers "where is the model and where does it plug in". Until a bundle is published
/// the engine runs on a logistic model it trains online, which is why the dashboard works
/// on a fresh clone with nothing trained.
export function ModelPanel({ frame }) {
  const policy = frame?.policy ?? {}
  const engine = frame?.engine ?? {}
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)

  const kind = policy.predictor ?? "unknown"
  const trained = kind === "gbdt" || kind === "linear"

  const reload = async (source) => {
    setBusy(true)
    try {
      setResult(await post("/v1/model/reload", source === "supabase" ? { source } : { path: "models" }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="The reuse model"
      subtitle="Predicts whether a key will be asked for again at 10 s, 60 s and 600 s. That prediction is one input to the value calculation, never the whole decision."
      actions={<Pill tone={trained ? "good" : "warn"}>{trained ? kind : "untrained"}</Pill>}
      footer={
        trained
          ? "Loaded. Predictions now come from the trained bundle."
          : "No bundle published yet. The engine is using the logistic model it trains online, which is a working fallback, not a failure."
      }
    >
      <div className="grid grid-cols-2 gap-3">
        <Metric label="Predictor in use" value={kind} size="sm"
          explain={trained ? "loaded from a published bundle" : "learns online from realised outcomes"} />
        <Metric label="Confidence" value={pct(policy.predictor_confidence ?? 0, 0)} size="sm"
          explain="how much weight the engine gives it" />
        <Metric label="Predictions made" value={Number(engine.inference_calls ?? 0).toLocaleString()} size="sm"
          explain="write path only, never on a read" />
        <Metric label="Influence on decisions" value={pct(policy.ml_influence ?? 0, 0)} size="sm"
          explain="the rest is the classical policy blend" />
      </div>

      <ol className="mt-4 space-y-2 text-[13.5px]">
        <Step n="1" title="Train it in Colab"
          body="Open training/notebooks/aura_training_colab.ipynb, add your two Supabase secrets, run every cell." />
        <Step n="2" title="It uploads itself"
          body="The notebook writes model_bundle.json to Supabase Storage and registers the row in aura_models with is_active set." />
        <Step n="3" title="Plug it in"
          body="Press the button below, or drop the exported .json files into engine/models/ and restart. No rebuild needed either way." />
      </ol>

      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" onClick={() => reload("supabase")} disabled={busy}>
          {busy ? "Loading…" : "Load from Supabase"}
        </Button>
        <Button size="sm" variant="outline" onClick={() => reload("file")} disabled={busy}>
          Load from engine/models
        </Button>
      </div>
      {result && (
        <p className={cn("mt-2 text-[12px]", result.ok ? "text-primary" : "text-rose-400")}>
          {result.ok
            ? `Loaded ${result.kind}${result.horizons?.length ? ` (${result.horizons.join(", ")})` : ""}.`
            : `Nothing loaded: ${result.error ?? "no bundle found"}.`}
        </p>
      )}
    </Panel>
  )
}

function Step({ n, title, body }) {
  return (
    <li className="flex gap-2.5">
      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-muted font-mono text-[12px]">
        {n}
      </span>
      <span>
        <span className="font-medium">{title}</span>{" "}
        <span className="text-muted-foreground">{body}</span>
      </span>
    </li>
  )
}
