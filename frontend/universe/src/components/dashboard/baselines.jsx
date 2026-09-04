import { bytes, cn, pct, usd } from "@/lib/utils"
import { Bar, Panel, Pill } from "./primitives"

/// The comparison the brief asks for, running live rather than in a benchmark tab.
///
/// Every baseline here is the same implementation the offline benchmark uses, fed the same
/// request stream as the engine and priced with the same table, so the only difference
/// between the columns is the policy. Two things are shown per row and both matter: cost,
/// which is the claim, and hit rate, which is usually where a classical policy wins while
/// still costing more - because it kept many cheap objects instead of a few expensive ones.
/// That inversion is the entire argument, and it is only visible when both numbers are on
/// screen together.

const BLURB = {
  lru: "Evicts whatever was touched least recently. Knows nothing about size or cost.",
  lfu: "Evicts the least frequently used. Popular beats expensive, always.",
  gds: "Greedy-Dual Size: cost per byte, with an inflation term so old objects age out.",
  tinylfu: "W-TinyLFU: a frequency sketch guarding a small admission window.",
  s3fifo: "Three FIFO queues, one-hit objects filtered out early.",
  sieve: "A lazy promotion pointer over a single FIFO queue.",
  lecar: "Learns an LRU/LFU mixture with regret matching.",
}

export function BaselinesPanel({ frame }) {
  const cost = frame?.cost ?? {}
  const rows = Object.entries(cost.baselines ?? {})
  const savings = cost.savings_vs ?? {}
  const auraTotal = cost.total_usd ?? 0
  const auraHit = frame?.layers?.l2?.hit_rate ?? 0
  const requests = Number(frame?.engine?.requests ?? 0)

  if (rows.length === 0) {
    return (
      <Panel
        title="Against the classical policies"
        subtitle="LRU, LFU and Greedy-Dual Size run beside the engine on the same request stream."
      >
        <p className="text-[13.5px] text-muted-foreground">
          Nothing to compare yet. The baselines fill as soon as the engine serves traffic.
        </p>
      </Panel>
    )
  }

  const ordered = [...rows].sort((a, b) => (a[1].total_usd ?? 0) - (b[1].total_usd ?? 0))
  const worst = Math.max(auraTotal, ...ordered.map(([, v]) => v.total_usd ?? 0), 1e-9)
  const beaten = ordered.filter(([k]) => (savings[k] ?? 0) > 0).length

  return (
    <Panel
      title="Against the classical policies"
      subtitle="Each one is the real implementation, not an approximation, fed the identical request stream and priced with the identical table. The only difference between these rows is the policy."
      actions={
        <Pill tone={beaten === ordered.length ? "accent" : "warn"}>
          cheaper than {beaten} of {ordered.length}
        </Pill>
      }
      footer={
        requests > 0
          ? `Over ${requests.toLocaleString()} requests. Object hit rate is not the objective and AURA will often trail on it: a policy that keeps a thousand small cheap objects hits more often than one keeping a hundred expensive ones, and pays more. Bytes served from cache, and the bill, are the numbers that answer the question.`
          : undefined
      }
    >
      <div className="space-y-2.5">
        <Row
          name="aura"
          label="AURA"
          total={auraTotal}
          hit={auraHit}
          byteHit={frame?.layers?.l2?.byte_hit_rate}
          worst={worst}
          held={frame?.engine?.used_bytes}
          self
        />
        {ordered.map(([name, v]) => (
          <Row
            key={name}
            name={name}
            label={name.toUpperCase()}
            total={v.total_usd ?? 0}
            hit={v.hit_rate ?? 0}
            byteHit={v.byte_hit_rate}
            held={v.used_bytes}
            worst={worst}
            delta={savings[name] ?? 0}
          />
        ))}
      </div>
    </Panel>
  )
}

function Row({ name, label, total, hit, byteHit, worst, held, delta, self }) {
  const cheaper = (delta ?? 0) > 0
  return (
    <div
      className={cn(
        "rounded-xl border px-4 py-3",
        self ? "border-primary/50 bg-primary/[0.07]" : "border-border bg-background"
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className={cn("text-[14px] font-semibold", self && "text-primary")}>{label}</span>
          {!self && (
            <span
              className={cn(
                "font-mono text-[12.5px] tabular-nums",
                cheaper ? "text-emerald-400" : "text-rose-400"
              )}
            >
              {cheaper ? "AURA is " : "AURA is "}
              {pct(Math.abs(delta ?? 0), 1)}
              {cheaper ? " cheaper" : " dearer"}
            </span>
          )}
        </div>
        <div className="flex items-baseline gap-4">
          <span className="text-[12.5px] text-muted-foreground">
            objects <span className="font-mono tabular-nums text-foreground">{pct(hit, 1)}</span>
          </span>
          {byteHit != null && (
            <span className="text-[12.5px] text-muted-foreground">
              bytes{" "}
              <span className="font-mono tabular-nums text-foreground">{pct(byteHit, 1)}</span>
            </span>
          )}
          {held != null && (
            <span className="text-[12.5px] text-muted-foreground">
              holding{" "}
              <span className="font-mono tabular-nums text-foreground">{bytes(held)}</span>
            </span>
          )}
          <span className="font-mono text-[15px] tabular-nums">{usd(total)}</span>
        </div>
      </div>
      <div className="mt-2">
        <Bar
          fraction={worst > 0 ? total / worst : 0}
          tone={self ? "bg-primary" : cheaper ? "bg-muted-foreground/50" : "bg-rose-400/70"}
          height="h-1.5"
        />
      </div>
      {!self && (
        <p className="mt-1.5 text-[12px] leading-snug text-muted-foreground">{BLURB[name]}</p>
      )}
    </div>
  )
}
