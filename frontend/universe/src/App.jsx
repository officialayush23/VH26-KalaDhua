import { useEffect, useRef } from "react"
import gsap from "gsap"

import { useLiveFeed } from "@/hooks/useLiveFeed"
import { cn } from "@/lib/utils"
import {
  ApplicationsPanel,
  BenchPanel,
  CapacityPanel,
  Controls,
  CostPanel,
  DecisionFeed,
  EnginePanel,
  EventsPanel,
  Headline,
  PolicyPanel,
  SupabasePanel,
} from "@/components/dashboard/panels"

const STATUS = {
  live: { tone: "bg-emerald-400", label: "live" },
  polling: { tone: "bg-amber-400", label: "polling" },
  reconnecting: { tone: "bg-amber-400", label: "reconnecting" },
  connecting: { tone: "bg-muted-foreground", label: "connecting" },
  offline: { tone: "bg-rose-400", label: "engine offline" },
}

export function App() {
  const { frame, history, status, send } = useLiveFeed()
  const rootRef = useRef(null)

  useEffect(() => {
    const node = rootRef.current
    if (!node) return
    const ctx = gsap.context(() => {
      gsap.from("[data-intro]", {
        opacity: 0,
        y: 16,
        duration: 0.5,
        stagger: 0.05,
        ease: "power3.out",
      })
    }, node)
    return () => ctx.revert()
  }, [])

  const s = STATUS[status] ?? STATUS.connecting

  return (
    <div className="min-h-svh bg-background text-foreground">
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 h-64 bg-gradient-to-b from-primary/[0.07] to-transparent"
      />

      <div ref={rootRef} className="relative mx-auto max-w-[1800px] px-5 py-7 lg:px-9">
        <header data-intro className="mb-7 flex flex-wrap items-start justify-between gap-5">
          <div className="max-w-3xl">
            <div className="flex items-center gap-3">
              <h1 className="text-[30px] font-semibold leading-none tracking-tight">AURA</h1>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card/60 px-2.5 py-1 text-[11px] font-medium">
                <span className={cn("h-1.5 w-1.5 rounded-full", s.tone)} />
                {s.label}
              </span>
            </div>
            <p className="mt-2 text-[14.5px] leading-relaxed text-muted-foreground">
              A cache that decides what to keep by what rebuilding it would actually cost.
              Conventional caches keep whatever was touched most recently, which throws away
              expensive objects to make room for cheap ones. This one prices every object and
              keeps the ones worth keeping.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <MiniStat
              label="Simulated time"
              value={`${Number(frame?.virtual_time_s ?? 0).toFixed(0)}s`}
            />
            <MiniStat
              label="Throughput"
              value={`${Number(frame?.sim?.rps ?? 0).toLocaleString()}/s`}
            />
            <MiniStat
              label="Requests"
              value={Number(
                (frame?.layers?.l2?.hits ?? 0) + (frame?.layers?.l2?.misses ?? 0)
              ).toLocaleString()}
            />
          </div>
        </header>

        {status === "offline" && (
          <div className="mb-6 rounded-2xl border border-rose-500/30 bg-rose-500/5 p-5">
            <h2 className="text-[15px] font-semibold">The engine is not running</h2>
            <p className="mt-1.5 text-[13.5px] text-muted-foreground">
              This page is fine. It just has nothing to talk to yet. Open a terminal in the
              repository and run:
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-muted/40 px-3.5 py-2.5 font-mono text-[12.5px]">
              cd engine{"\n"}cargo run --release -p aura-server -- --scenario mixed_production
            </pre>
            <p className="mt-2 text-[12.5px] text-muted-foreground">
              The first build takes a few minutes. When it prints{" "}
              <span className="font-mono">aura listening</span>, this page fills in on its own.
            </p>
          </div>
        )}

        <div data-intro className="mb-4">
          <Headline frame={frame} history={history} />
        </div>

        <div data-intro className="mb-4">
          <Controls frame={frame} send={send} status={status} />
        </div>

        <div className="grid gap-4 xl:grid-cols-12">
          <div data-intro className="space-y-4 xl:col-span-5">
            <CostPanel frame={frame} />
            <CapacityPanel frame={frame} />
          </div>

          <div data-intro className="space-y-4 xl:col-span-4">
            <PolicyPanel frame={frame} />
            <ApplicationsPanel frame={frame} />
            <EnginePanel frame={frame} />
          </div>

          <div data-intro className="space-y-4 xl:col-span-3">
            <DecisionFeed frame={frame} />
            <SupabasePanel />
            <EventsPanel frame={frame} />
          </div>
        </div>

        <div data-intro className="mt-4">
          <BenchPanel />
        </div>

        <footer className="mt-8 border-t border-border/50 pt-4 text-[12px] text-muted-foreground">
          Press <kbd className="rounded border border-border px-1">d</kbd> to switch between
          light and dark.
        </footer>
      </div>
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="rounded-xl border border-border/70 bg-card/50 px-3.5 py-2">
      <div className="text-[10.5px] uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-[17px] tabular-nums">{value}</div>
    </div>
  )
}

export default App
