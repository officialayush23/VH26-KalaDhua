import { useMemo } from "react"

import { cn } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

/// The evidence tab.
///
/// Every other tab answers "what is the cache doing right now". This one answers "what did
/// it do, and can you show me". The difference matters because every claim this project
/// makes -- that it adapts, that it costs less than LRU, that it grows the pool only when
/// the growth pays for itself -- is a claim about a sequence of decisions over time, and a
/// dial showing an instantaneous value cannot support or refute any of them.
///
/// The engine keeps no history. It sends a complete snapshot every frame and forgets it, so
/// the series charted here are accumulated in the browser by `useLiveFeed`. That is a
/// deliberate trade: the engine stays a cache rather than becoming a time-series database,
/// and the cost is that these charts start empty and fill as you watch. A chart that says
/// "collecting" for ten seconds is honest; one that invents a backfill is not.

const AXIS = { padL: 48, padR: 10, padT: 10, padB: 22, width: 520 }

/// Shared plot frame: gridlines, y labels, x labels. Every chart in this file draws inside
/// one, so they can be read against each other without re-learning the axes each time.
function Frame({ height, minX, maxX, minY, maxY, yFormat, xFormat, children }) {
  const { padL, padR, padT, padB, width } = AXIS
  const plotH = height - padT - padB
  const ticks = [minY, minY + (maxY - minY) / 2, maxY]
  const xTicks = [minX, minX + (maxX - minX) / 2, maxX]
  const py = (y) => padT + plotH - ((y - minY) / (maxY - minY || 1)) * plotH
  const px = (x) => padL + ((x - minX) / (maxX - minX || 1)) * (width - padL - padR)

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={width - padR}
            y1={py(t)}
            y2={py(t)}
            className="stroke-border/50"
            strokeWidth="1"
          />
          <text x={padL - 6} y={py(t) + 3.5} textAnchor="end" className="fill-muted-foreground text-[9px]">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {children({ px, py })}
      {xTicks.map((t, i) => (
        <text
          key={i}
          x={px(t)}
          y={height - 6}
          textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
          className="fill-muted-foreground text-[9px]"
        >
          {xFormat(t)}
        </text>
      ))}
    </svg>
  )
}

function Empty({ height, note }) {
  return (
    <div
      style={{ height }}
      className="flex items-center justify-center rounded-lg border border-dashed border-border px-4 text-center text-[12px] text-muted-foreground"
    >
      {note}
    </div>
  )
}

/// Several series against one shared y axis.
///
/// One axis and not one per series: the whole point of putting AURA's cost beside LRU's is
/// that the gap between the lines is the claim. Independent axes would make any two curves
/// look similar and the comparison would be decoration.
export function MultiLine({
  series,
  height = 190,
  yFormat = (v) => v.toFixed(2),
  xFormat = (v) => `${v.toFixed(0)}s`,
  zeroBased = true,
  note = "collecting data",
}) {
  const points = series.flatMap((s) => s.values)
  if (points.length < 4) return <Empty height={height} note={note} />

  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = zeroBased ? Math.min(0, ...ys) : Math.min(...ys)
  const maxY = Math.max(...ys, minY + 1e-9)

  return (
    <div>
      <Frame
        height={height}
        minX={minX}
        maxX={maxX}
        minY={minY}
        maxY={maxY}
        yFormat={yFormat}
        xFormat={xFormat}
      >
        {({ px, py }) =>
          series.map((s) => (
            <path
              key={s.key}
              d={s.values
                .map((p, i) => `${i === 0 ? "M" : "L"}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`)
                .join(" ")}
              className={cn("fill-none", s.tone)}
              strokeWidth={s.emphasis ? 2.4 : 1.6}
              strokeDasharray={s.dashed ? "4 3" : undefined}
              strokeLinejoin="round"
            />
          ))
        }
      </Frame>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <span className={cn("h-0.5 w-4 rounded", s.swatch)} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  )
}

/// Counts per tick, drawn as grouped bars.
///
/// Admissions, refusals and evictions are cumulative counters in the telemetry, and a
/// cumulative counter can only ever go up: plotted raw, a flash crowd looks exactly like a
/// quiet minute. Differencing consecutive frames turns them back into events, which is the
/// shape the question was asked in.
export function RateBars({ rows, keys, height = 190, xFormat = (v) => `${v.toFixed(0)}s` }) {
  if (rows.length < 4) return <Empty height={height} note="collecting data" />
  const { padL, padR, padT, padB, width } = AXIS
  const plotH = height - padT - padB
  const maxY = Math.max(1, ...rows.flatMap((r) => keys.map((k) => r[k.key] ?? 0)))
  const minX = rows[0].t
  const maxX = rows[rows.length - 1].t
  const slot = (width - padL - padR) / rows.length
  const bw = Math.max(1, (slot - 1) / keys.length)

  return (
    <div>
      <Frame
        height={height}
        minX={minX}
        maxX={maxX}
        minY={0}
        maxY={maxY}
        yFormat={(v) => v.toFixed(0)}
        xFormat={xFormat}
      >
        {({ py }) =>
          rows.flatMap((r, i) =>
            keys.map((k, j) => {
              const v = r[k.key] ?? 0
              const y = py(v)
              return (
                <rect
                  key={`${i}-${k.key}`}
                  x={padL + i * slot + j * bw}
                  y={y}
                  width={bw}
                  height={Math.max(0, padT + plotH - y)}
                  className={k.tone}
                />
              )
            })
          )
        }
      </Frame>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
        {keys.map((k) => (
          <span key={k.key} className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <span className={cn("h-2.5 w-2.5 rounded-sm", k.swatch)} />
            {k.label}
          </span>
        ))}
      </div>
    </div>
  )
}

/// Proportions that sum to one, over time.
///
/// The bandit's posterior is six numbers that always add to 100%, and the interesting thing
/// about it is not any single weight but which expert is taking share from which. Stacked
/// bands show that directly; six separate lines would make the reader do the addition.
export function StackedArea({ rows, bands, height = 190, xFormat = (v) => `${v.toFixed(0)}s` }) {
  if (rows.length < 4) return <Empty height={height} note="the bandit has not been asked anything yet" />
  const { padL, padR, padT, padB, width } = AXIS
  const plotH = height - padT - padB
  const minX = rows[0].t
  const maxX = rows[rows.length - 1].t
  const px = (x) => padL + ((x - minX) / (maxX - minX || 1)) * (width - padL - padR)

  // Cumulative offsets, band by band, so each polygon is the ribbon between two running
  // totals rather than a shape drawn from the baseline and overpainted.
  const below = rows.map(() => 0)
  const shapes = bands.map((b) => {
    const top = rows.map((r, i) => below[i] + (r.mixture?.[b.key] ?? 0))
    const d = [
      ...rows.map((r, i) => `${i === 0 ? "M" : "L"}${px(r.t).toFixed(1)},${(padT + plotH - top[i] * plotH).toFixed(1)}`),
      ...rows
        .map((r, i) => [r, i])
        .reverse()
        .map(([r, i]) => `L${px(r.t).toFixed(1)},${(padT + plotH - below[i] * plotH).toFixed(1)}`),
      "Z",
    ].join(" ")
    top.forEach((v, i) => {
      below[i] = v
    })
    return { key: b.key, d, tone: b.tone }
  })

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        {shapes.map((s) => (
          <path key={s.key} d={s.d} className={s.tone} />
        ))}
        {[0, 0.5, 1].map((f) => (
          <text
            key={f}
            x={padL - 6}
            y={padT + plotH - f * plotH + 3.5}
            textAnchor="end"
            className="fill-muted-foreground text-[9px]"
          >
            {`${(f * 100).toFixed(0)}%`}
          </text>
        ))}
        {[minX, maxX].map((t, i) => (
          <text
            key={t}
            x={px(t)}
            y={height - 6}
            textAnchor={i === 0 ? "start" : "end"}
            className="fill-muted-foreground text-[9px]"
          >
            {xFormat(t)}
          </text>
        ))}
      </svg>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
        {bands.map((b) => (
          <span key={b.key} className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <span className={cn("h-2.5 w-2.5 rounded-sm", b.swatch)} />
            {b.label}
          </span>
        ))}
      </div>
    </div>
  )
}

const EXPERTS = [
  { key: "lru", label: "LRU", tone: "fill-primary/80", swatch: "bg-primary/80" },
  { key: "lfu", label: "LFU", tone: "fill-sky-400/70", swatch: "bg-sky-400/70" },
  { key: "gdsf", label: "GDSF", tone: "fill-emerald-400/70", swatch: "bg-emerald-400/70" },
  { key: "tiny_lfu", label: "TinyLFU", tone: "fill-violet-400/70", swatch: "bg-violet-400/70" },
  { key: "cost_aware", label: "cost aware", tone: "fill-amber-400/70", swatch: "bg-amber-400/70" },
  { key: "trend_aware", label: "trend aware", tone: "fill-rose-400/70", swatch: "bg-rose-400/70" },
]

const MB = 1024 * 1024

export function EvidencePanel({ history, frame }) {
  const rows = useMemo(() => history.slice(-160), [history])

  // Differences between consecutive frames. The first row has no predecessor and is dropped
  // rather than being shown as a spike of its own total.
  const deltas = useMemo(
    () =>
      rows.slice(1).map((r, i) => {
        const prev = rows[i]
        const nz = (v) => (v > 0 ? v : 0)
        return {
          t: r.t,
          admitted: nz(r.admissions - prev.admissions),
          refused: nz(r.rejections - prev.rejections),
          evicted: nz(r.evictions - prev.evictions),
        }
      }),
    [rows]
  )

  const scaleEvents = useMemo(
    () => rows.filter((r) => r.decision && r.decision !== "hold" && r.decision !== "Hold"),
    [rows]
  )

  const line = (key, get) => ({ key, values: rows.map((r) => ({ x: r.t, y: get(r) })) })

  return (
    <div className="space-y-4">
      <Panel
        title="Offered load, and the load that actually arrived"
        subtitle="The rate you asked for against the rate the generator produced. A disturbance is exactly the gap between the two lines: nothing in the configuration changed, the traffic did."
        actions={<Pill tone={frame?.workload?.regime ? "accent" : "default"}>{frame?.workload?.regime ?? "no traffic"}</Pill>}
      >
        <MultiLine
          series={[
            { ...line("rps", (r) => r.rps), label: "arriving", tone: "stroke-primary", swatch: "bg-primary", emphasis: true },
            { ...line("base", (r) => r.baseRps), label: "configured", tone: "stroke-muted-foreground/70", swatch: "bg-muted-foreground/70", dashed: true },
          ]}
          yFormat={(v) => `${v.toFixed(0)}/s`}
          note="start a scenario, or point applications at the engine"
        />
      </Panel>

      <Panel
        title="What the cache did with it, per tick"
        subtitle="Admissions, refusals and evictions as events rather than running totals. Refusals rising while evictions stay flat is the admission gate working: the cache is declining objects instead of throwing better ones out to make room for them."
      >
        <RateBars
          rows={deltas}
          keys={[
            { key: "admitted", label: "admitted", tone: "fill-primary/80", swatch: "bg-primary/80" },
            { key: "refused", label: "refused at the door", tone: "fill-amber-400/80", swatch: "bg-amber-400/80" },
            { key: "evicted", label: "evicted", tone: "fill-rose-400/80", swatch: "bg-rose-400/80" },
          ]}
        />
      </Panel>

      <Panel
        title="Running cost, against the same traffic through LRU, LFU and GDS"
        subtitle="Three shadow caches see every request this one sees and are charged the same prices. They are not a table from a paper: they are running now, on this traffic, and the distance between the lines is the money the decision made."
        footer="Cost is rebuild cost plus SLA penalty plus what the memory itself costs to hold. A policy that keeps everything wins on hit rate and still loses here."
      >
        <MultiLine
          series={[
            { ...line("aura", (r) => r.totalUsd), label: "AURA", tone: "stroke-primary", swatch: "bg-primary", emphasis: true },
            { ...line("lru", (r) => r.baselineUsd.lru), label: "LRU", tone: "stroke-rose-400/80", swatch: "bg-rose-400/80" },
            { ...line("lfu", (r) => r.baselineUsd.lfu), label: "LFU", tone: "stroke-amber-400/80", swatch: "bg-amber-400/80" },
            { ...line("gds", (r) => r.baselineUsd.gds), label: "GDS", tone: "stroke-sky-400/80", swatch: "bg-sky-400/80" },
          ]}
          yFormat={(v) => `$${v.toFixed(2)}`}
          note="no traffic has been priced yet"
        />
      </Panel>

      <Panel
        title="Pool size, and every decision to change it"
        subtitle="The controller only grows the pool when the next block of memory is projected to save more than it costs to rent, and only shrinks it once it has seen enough traffic to tell an oversized pool from an idle one."
        actions={
          <Pill tone={scaleEvents.length ? "accent" : "default"}>
            {scaleEvents.length} change{scaleEvents.length === 1 ? "" : "s"} in view
          </Pill>
        }
      >
        <MultiLine
          series={[
            { ...line("cap", (r) => r.capacity / MB), label: "pool size", tone: "stroke-primary", swatch: "bg-primary", emphasis: true },
            { ...line("used", (r) => r.usedBytes / MB), label: "in use", tone: "stroke-emerald-400/80", swatch: "bg-emerald-400/80" },
          ]}
          yFormat={(v) => `${v.toFixed(0)} MB`}
          note="collecting data"
        />
        {scaleEvents.length > 0 && (
          <div className="mt-3 max-h-28 overflow-y-auto rounded-lg border border-border/70 bg-muted/20 px-3 py-2 font-mono text-[12px]">
            {scaleEvents
              .slice(-8)
              .reverse()
              .map((r, i) => (
                <div key={i} className="text-muted-foreground">
                  {r.t.toFixed(0)}s — {String(r.decision)} at {(r.capacity / MB).toFixed(0)} MB
                </div>
              ))}
          </div>
        )}
      </Panel>

      <Panel
        title="Which expert the bandit currently believes"
        subtitle="Six eviction heuristics, each with a Beta posterior updated from realised outcomes sixty seconds after the fact. The bands are the posterior means. When the workload changes, watch the share move — that shift is the adaptation, and nothing about it was configured."
        footer="Posteriors decay at 0.995 per tick, so evidence from a workload that has since ended stops outvoting evidence from the one running now."
      >
        <StackedArea rows={rows} bands={EXPERTS} />
      </Panel>

      <Panel
        title="Hit rate, both tiers"
        subtitle="L1 is the in-process cache inside each service and removes a network round trip. L2 is this engine and removes a rebuild. They are different jobs, so a single blended number would hide which one is working."
        actions={
          <Pill tone={(rows[rows.length - 1]?.l1Reporting ?? 0) > 0 ? "accent" : "warn"}>
            {rows[rows.length - 1]?.l1Reporting ?? 0} process{(rows[rows.length - 1]?.l1Reporting ?? 0) === 1 ? "" : "es"} reporting L1
          </Pill>
        }
        footer="L1 is measured inside the applications and posted here every five seconds, because a request served from a local copy never reaches the engine at all. With no application running, that line is honestly zero rather than quietly borrowed from something else."
      >
        <MultiLine
          series={[
            { ...line("l2", (r) => r.hitRate * 100), label: "L2 (engine)", tone: "stroke-primary", swatch: "bg-primary", emphasis: true },
            { ...line("l1", (r) => r.hitRateL1 * 100), label: "L1 (in process)", tone: "stroke-violet-400/80", swatch: "bg-violet-400/80" },
            { ...line("byte", (r) => r.byteHitRate * 100), label: "L2 by bytes", tone: "stroke-emerald-400/70", swatch: "bg-emerald-400/70", dashed: true },
          ]}
          yFormat={(v) => `${v.toFixed(0)}%`}
          note="collecting data"
        />
      </Panel>
    </div>
  )
}
