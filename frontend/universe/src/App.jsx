import { useEffect, useRef } from "react"
import gsap from "gsap"

import { useLiveFeed } from "@/hooks/useLiveFeed"
import { cn, pct } from "@/lib/utils"
import {
  ApplicationsPanel,
  BenchPanel,
  CapacityPanel,
  Controls,
  CostPanel,
  DecisionFeed,
  EventsPanel,
  LayerPanel,
  PolicyPanel,
} from "@/components/dashboard/panels"

const STATUS_TONE = {
  live: "bg-emerald-500",
  polling: "bg-amber-500",
  reconnecting: "bg-amber-500",
  connecting: "bg-muted-foreground",
  offline: "bg-rose-500",
}

export function App() {
  const { frame, status, send } = useLiveFeed()
  const headerRef = useRef(null)

  useEffect(() => {
    const node = headerRef.current
    if (!node) return
    const ctx = gsap.context(() => {
      gsap.from("[data-intro]", {
        opacity: 0,
        y: 18,
        duration: 0.6,
        stagger: 0.06,
        ease: "power3.out",
      })
    }, node)
    return () => ctx.revert()
  }, [])

  const savings = frame?.cost?.savings_vs ?? {}
  const bestGain = Object.entries(savings).reduce(
    (acc, [k, v]) => (v > acc.v ? { k, v } : acc),
    { k: null, v: -Infinity }
  )

  return (
    <div className="min-h-svh bg-background text-foreground">
      <div ref={headerRef} className="mx-auto max-w-[1600px] px-4 py-6 lg:px-8">
        <header data-intro className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-2xl font-semibold tracking-tight">AURA</h1>
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-[11px]"
                )}
              >
                <span className={cn("h-1.5 w-1.5 rounded-full", STATUS_TONE[status])} />
                {status}
              </span>
            </div>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              An adaptive, utility- and runtime-aware cache. It decides what to keep by what
              rebuilding it would actually cost, not by how recently it was touched.
            </p>
          </div>
          {bestGain.k && bestGain.v > 0 && (
            <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-2.5">
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Cheaper than {bestGain.k}
              </div>
              <div className="font-mono text-2xl text-emerald-500 tabular-nums">
                {pct(bestGain.v, 1)}
              </div>
            </div>
          )}
        </header>

        {status === "offline" && (
          <div className="mb-6 rounded-lg border border-rose-500/30 bg-rose-500/5 p-4 text-sm">
            Cannot reach the engine. Start it with{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
              cargo run --release -p aura-server -- --scenario mixed_production
            </code>
            .
          </div>
        )}

        <div data-intro className="mb-4">
          <Controls frame={frame} send={send} />
        </div>

        <div data-intro className="mb-4">
          <LayerPanel frame={frame} />
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div data-intro className="lg:col-span-2 space-y-4">
            <CostPanel frame={frame} />
            <PolicyPanel frame={frame} />
            <CapacityPanel frame={frame} />
            <BenchPanel />
          </div>
          <div data-intro className="space-y-4">
            <DecisionFeed frame={frame} />
            <ApplicationsPanel frame={frame} />
            <EventsPanel frame={frame} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
