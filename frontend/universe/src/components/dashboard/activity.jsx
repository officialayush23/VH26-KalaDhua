import { useEffect, useMemo, useRef, useState } from "react"

import { cn } from "@/lib/utils"

/// The console's centrepiece: the cache explaining itself, one sentence per decision.
///
/// The engine already writes these sentences - kind, severity, subject, prose and the
/// numbers behind the prose travel together in every audit entry - so nothing here invents
/// a reason or reformats one. This component's whole job is to make several thousand of
/// them a minute readable: group them by what a person is looking for, keep the newest in
/// view unless the reader is scrolling, and put the money on the right where it can be
/// scanned down a column.

const GROUPS = [
  { id: "all", label: "Everything", kinds: null },
  { id: "admission", label: "Admissions", kinds: ["admit", "reject"] },
  { id: "eviction", label: "Evictions", kinds: ["evict", "expire"] },
  { id: "correctness", label: "Correctness", kinds: ["invalidate", "version_bump", "refresh"] },
  { id: "capacity", label: "Capacity", kinds: ["scale_up", "scale_down", "scale_hold", "pressure"] },
  { id: "brain", label: "Model & policy", kinds: ["policy_shift", "model_load", "regime_change"] },
]

/// Colour carries meaning and nothing else. Kept deliberately narrow: green for what the
/// cache chose to keep, rose for what it threw away or got wrong, amber for money and
/// memory decisions, violet for the parts that learn.
const KIND_TONE = {
  admit: "text-emerald-400 border-emerald-400/25 bg-emerald-400/10",
  reject: "text-zinc-400 border-zinc-400/25 bg-zinc-400/10",
  evict: "text-rose-400 border-rose-400/25 bg-rose-400/10",
  expire: "text-zinc-400 border-zinc-400/25 bg-zinc-400/10",
  invalidate: "text-orange-400 border-orange-400/25 bg-orange-400/10",
  version_bump: "text-orange-400 border-orange-400/25 bg-orange-400/10",
  refresh: "text-sky-400 border-sky-400/25 bg-sky-400/10",
  scale_up: "text-amber-400 border-amber-400/25 bg-amber-400/10",
  scale_down: "text-amber-400 border-amber-400/25 bg-amber-400/10",
  scale_hold: "text-zinc-400 border-zinc-400/25 bg-zinc-400/10",
  pressure: "text-amber-400 border-amber-400/25 bg-amber-400/10",
  policy_shift: "text-violet-400 border-violet-400/25 bg-violet-400/10",
  model_load: "text-violet-400 border-violet-400/25 bg-violet-400/10",
  regime_change: "text-violet-400 border-violet-400/25 bg-violet-400/10",
}

const SEVERITY_RAIL = {
  info: "bg-transparent",
  notice: "bg-amber-400/60",
  warning: "bg-rose-400/70",
}

function clockOf(entry) {
  // `at` is wall clock from the engine. Fall back to the engine clock so a line is never
  // blank, and never invent the browser's own time: the two can be seconds apart.
  const raw = entry?.at
  if (typeof raw === "string" && raw.length >= 19) return raw.slice(11, 19)
  const t = Number(entry?.t_ms ?? 0) / 1000
  return `${Math.floor(t / 60)}m${String(Math.floor(t % 60)).padStart(2, "0")}s`
}

export function ActivityLog({ log = [], status, className }) {
  const [group, setGroup] = useState("all")
  const [query, setQuery] = useState("")
  const [noticeOnly, setNoticeOnly] = useState(false)
  const [pinned, setPinned] = useState(true)
  const listRef = useRef(null)

  const rows = useMemo(() => {
    const kinds = GROUPS.find((g) => g.id === group)?.kinds
    const needle = query.trim().toLowerCase()
    return log.filter((e) => {
      if (kinds && !kinds.includes(e.kind)) return false
      if (noticeOnly && e.severity === "info") return false
      if (!needle) return true
      return (
        String(e.subject ?? "").toLowerCase().includes(needle) ||
        String(e.application ?? "").toLowerCase().includes(needle) ||
        String(e.message ?? "").toLowerCase().includes(needle)
      )
    })
  }, [log, group, query, noticeOnly])

  // Follow the tail only while the reader is at the top. Yanking the viewport back while
  // someone is reading an entry from thirty seconds ago is the fastest way to make a live
  // log useless.
  useEffect(() => {
    if (!pinned) return
    const node = listRef.current
    if (node) node.scrollTop = 0
  }, [rows, pinned])

  const onScroll = () => {
    const node = listRef.current
    if (node) setPinned(node.scrollTop < 24)
  }

  const counts = useMemo(() => {
    const out = {}
    for (const e of log) out[e.kind] = (out[e.kind] ?? 0) + 1
    return out
  }, [log])

  return (
    <section
      className={cn(
        "flex min-h-0 flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-sm",
        className
      )}
    >
      <header className="border-b border-border/80 px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="text-[15px] font-semibold tracking-tight">Decision log</h2>
            <p className="mt-1 text-[13.5px] leading-snug text-muted-foreground">
              Every choice the cache made, in the words it made it in, with the numbers that
              produced it.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by key, app or words"
              className="h-8 w-56 rounded-lg border border-border bg-background px-2.5 text-[13.5px] outline-none placeholder:text-muted-foreground focus:border-primary/50"
            />
            <button
              type="button"
              onClick={() => setNoticeOnly((v) => !v)}
              className={cn(
                "h-8 rounded-lg border px-2.5 text-[12px] font-medium transition-colors",
                noticeOnly
                  ? "border-amber-400/40 bg-amber-400/10 text-amber-300"
                  : "border-border bg-background text-muted-foreground hover:text-foreground"
              )}
            >
              Only what matters
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {GROUPS.map((g) => {
            const n = g.kinds
              ? g.kinds.reduce((acc, k) => acc + (counts[k] ?? 0), 0)
              : log.length
            return (
              <button
                key={g.id}
                type="button"
                onClick={() => setGroup(g.id)}
                className={cn(
                  "rounded-full border px-3 py-1 text-[12px] font-medium transition-colors",
                  group === g.id
                    ? "border-primary/40 bg-primary/10 text-foreground"
                    : "border-border bg-background text-muted-foreground hover:text-foreground"
                )}
              >
                {g.label}
                <span className="ml-1.5 font-mono text-[12px] tabular-nums opacity-60">
                  {n}
                </span>
              </button>
            )
          })}
          <span className="ml-auto flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                status === "live" ? "animate-pulse bg-emerald-400" : "bg-muted-foreground"
              )}
            />
            {pinned ? "following" : "scrolled back"}
          </span>
        </div>
      </header>

      <div
        ref={listRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 divide-y divide-border/70 overflow-y-auto"
      >
        {rows.length === 0 && (
          <div className="px-5 py-10 text-center">
            <p className="text-[13.5px] font-medium">Nothing to show yet.</p>
            <p className="mt-1 text-[13.5px] text-muted-foreground">
              {log.length === 0
                ? "The cache writes a line every time it keeps, refuses, evicts or invalidates something. Start traffic and this fills."
                : "No entries match this filter."}
            </p>
          </div>
        )}
        {rows.map((e) => (
          <LogRow key={e.seq} entry={e} />
        ))}
      </div>

      <footer className="flex items-center justify-between border-t border-border/80 px-5 py-2.5 text-[12px] text-muted-foreground">
        <span>
          {rows.length.toLocaleString()} shown, {log.length.toLocaleString()} in scrollback
        </span>
        {!pinned && (
          <button
            type="button"
            onClick={() => {
              setPinned(true)
              if (listRef.current) listRef.current.scrollTop = 0
            }}
            className="rounded-md border border-border px-2 py-0.5 font-medium text-foreground hover:bg-muted/40"
          >
            Jump to newest
          </button>
        )}
      </footer>
    </section>
  )
}

function LogRow({ entry }) {
  const tone = KIND_TONE[entry.kind] ?? "text-zinc-400 border-zinc-400/25 bg-zinc-400/10"
  const usd = Number(entry.usd_impact ?? 0)
  return (
    <article className="relative flex gap-3 px-5 py-3 transition-colors hover:bg-muted/25">
      <span
        aria-hidden
        className={cn(
          "absolute inset-y-0 left-0 w-[2px]",
          SEVERITY_RAIL[entry.severity] ?? SEVERITY_RAIL.info
        )}
      />
      <time className="mt-[3px] w-[62px] shrink-0 font-mono text-[12px] tabular-nums text-muted-foreground">
        {clockOf(entry)}
      </time>
      <span
        className={cn(
          "mt-[1px] h-[21px] shrink-0 rounded-md border px-2 text-[12px] font-semibold uppercase leading-[19px] tracking-[0.06em]",
          tone
        )}
      >
        {entry.label}
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className="truncate font-mono text-[13.5px] text-foreground/90">
            {entry.subject}
          </span>
          {entry.application && entry.application !== "engine" && (
            <span className="rounded border border-border px-1.5 text-[12px] text-muted-foreground">
              {entry.application}
            </span>
          )}
        </div>
        <p className="mt-1 text-[13.5px] leading-snug text-muted-foreground">
          {entry.message}
        </p>
        {Array.isArray(entry.facts) && entry.facts.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {entry.facts.map((f, i) => (
              <span
                key={`${entry.seq}-${f.name}-${i}`}
                className="rounded bg-muted/50 px-1.5 py-0.5 text-[12px] text-muted-foreground"
              >
                <span className="opacity-60">{f.name}</span>{" "}
                <span className="font-mono tabular-nums text-foreground/80">{f.value}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {usd !== 0 && (
        <div
          className={cn(
            "mt-[2px] shrink-0 font-mono text-[12px] tabular-nums",
            usd > 0 ? "text-emerald-400" : "text-rose-400"
          )}
          title={usd > 0 ? "money this decision saved" : "money this decision spent"}
        >
          {usd > 0 ? "+" : "-"}
          {formatUsd(Math.abs(usd))}
        </div>
      )}
    </article>
  )
}

/// Matches the engine's own money formatting. A cache decision is often worth millionths of
/// a dollar, and printing $0.00 for all of them hides the entire point.
function formatUsd(v) {
  if (v < 0.0001) return `${(v * 100).toFixed(4)}c`
  if (v < 0.01) return `${(v * 100).toFixed(2)}c`
  if (v < 100) return `$${v.toFixed(2)}`
  return `$${v.toFixed(0)}`
}
