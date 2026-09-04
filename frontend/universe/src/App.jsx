import { useEffect, useRef, useState } from "react"
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
  ModelPanel,
  PolicyPanel,
  SupabasePanel,
} from "@/components/dashboard/panels"
import { ActivityLog } from "@/components/dashboard/activity"
import { ProfilesPanel } from "@/components/dashboard/profiles"
import { OnboardingPanel } from "@/components/dashboard/onboarding"
import { AuthNotice, SessionChip, useSession } from "@/components/dashboard/signin"
import { Pipeline, TrafficSource } from "@/components/dashboard/pipeline"
import { FlowDiagram } from "@/components/dashboard/flowdiagram"

const STATUS = {
  live: { tone: "bg-primary", label: "live" },
  polling: { tone: "bg-amber-400", label: "polling" },
  reconnecting: { tone: "bg-amber-400", label: "reconnecting" },
  connecting: { tone: "bg-muted-foreground", label: "connecting" },
  offline: { tone: "bg-rose-400", label: "engine offline" },
}

const TABS = [
  { id: "live", label: "Live" },
  { id: "overview", label: "Overview" },
  { id: "flow", label: "Request flow" },
  { id: "decisions", label: "Decisions" },
  { id: "tuning", label: "Tuning" },
  { id: "connect", label: "Connect" },
  { id: "benchmark", label: "Benchmark" },
  { id: "system", label: "Model & system" },
]

export function App() {
  const { frame, history, status, send, log } = useLiveFeed()
  const { session } = useSession()
  const [tab, setTab] = useState("live")
  const rootRef = useRef(null)

  // Animate the header once on mount only. Running a stagger across the whole page on
  // every tab change leaves elements mid-tween when React swaps them out.
  useEffect(() => {
    const node = rootRef.current
    if (!node) return
    const ctx = gsap.context(() => {
      gsap.from("[data-hero]", {
        opacity: 0,
        y: 14,
        duration: 0.5,
        stagger: 0.05,
        ease: "power3.out",
        clearProps: "all",
      })
    }, node)
    return () => ctx.revert()
  }, [])

  const paneRef = useRef(null)
  useEffect(() => {
    const node = paneRef.current
    if (!node) return
    const tween = gsap.fromTo(
      node,
      { opacity: 0, y: 8 },
      { opacity: 1, y: 0, duration: 0.32, ease: "power2.out", clearProps: "all" }
    )
    return () => tween.kill()
  }, [tab])

  const s = STATUS[status] ?? STATUS.connecting

  return (
    <div className="min-h-svh bg-background text-foreground">
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 h-72 bg-gradient-to-b from-primary/[0.06] to-transparent"
      />

      <div ref={rootRef} className="relative mx-auto max-w-[1760px] px-5 py-7 lg:px-8">
        <header data-hero className="mb-5">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div className="max-w-3xl">
              <div className="flex items-center gap-3">
                <h1 className="text-[30px] font-semibold leading-none tracking-tight">
                  AURA
                </h1>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-[12px] font-medium">
                  <span className={cn("h-1.5 w-1.5 rounded-full", s.tone)} />
                  {s.label}
                </span>
              </div>
              <p className="mt-2 text-[14.5px] leading-relaxed text-muted-foreground">
                An application-adaptive cache. Conventional caches keep whatever was touched
                most recently, which throws away an expensive database rollup to make room
                for a cheap lookup. This one measures what rebuilding each object actually
                cost and keeps the ones worth the space.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2.5">
              <SessionChip session={session} />
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
          </div>
        </header>

        <div data-hero className="mb-5">
          <TrafficSource frame={frame} status={status} />
        </div>

        <AuthNotice enforced={Boolean(frame?.auth?.enforced)} session={session} />

        {status === "offline" && (
          <div className="mb-5 rounded-2xl border border-rose-500/30 bg-rose-500/5 p-5">
            <h2 className="text-[15px] font-semibold">The engine is not running</h2>
            <p className="mt-1.5 text-[13.5px] text-muted-foreground">
              This page is fine, it just has nothing to talk to. In a terminal:
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-muted/40 px-3.5 py-2.5 font-mono text-[13.5px]">
              cd engine{"\n"}cargo run --release -p aura-server -- --scenario mixed_production
            </pre>
          </div>
        )}

        <nav
          data-hero
          className="mb-5 flex flex-wrap gap-1 rounded-xl border border-border bg-card p-1"
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "rounded-lg px-3.5 py-2 text-[13.5px] font-medium transition-colors",
                tab === t.id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div ref={paneRef}>
          {tab === "live" && (
            <div className="space-y-4">
              <Headline frame={frame} history={history} />
              <div className="grid items-start gap-4 xl:grid-cols-3">
                {/* The log is the page. Two thirds of the width and the full height of the
                    viewport, because reading a cache's reasoning is the demonstration; the
                    aggregates beside it are context for the sentence you are reading. */}
                <ActivityLog
                  log={log}
                  status={status}
                  className="h-[calc(100svh-330px)] min-h-[520px] xl:col-span-2"
                />
                <div className="space-y-4">
                  <CapacityPanel frame={frame} />
                  <ApplicationsPanel frame={frame} />
                </div>
              </div>
              <Controls frame={frame} send={send} status={status} />
            </div>
          )}

          {tab === "overview" && (
            <div className="space-y-4">
              <Headline frame={frame} history={history} />
              <Controls frame={frame} send={send} status={status} />
              <div className="grid items-start gap-4 xl:grid-cols-2">
                <CostPanel frame={frame} />
                <CapacityPanel frame={frame} />
              </div>
              <div className="grid items-start gap-4 xl:grid-cols-2">
                <PolicyPanel frame={frame} />
                <ApplicationsPanel frame={frame} />
              </div>
            </div>
          )}

          {tab === "flow" && (
            <div className="space-y-4">
              <FlowDiagram frame={frame} status={status} />
              <Pipeline frame={frame} />
              <div className="grid items-start gap-4 xl:grid-cols-2">
                <EnginePanel frame={frame} />
                <EventsPanel frame={frame} />
              </div>
            </div>
          )}

          {tab === "decisions" && (
            <div className="grid items-start gap-4 xl:grid-cols-3">
              <div className="space-y-4 xl:col-span-2">
                <DecisionFeed frame={frame} />
                <ActivityLog log={log} status={status} className="h-[560px]" />
              </div>
              <div className="space-y-4">
                <PolicyPanel frame={frame} />
                <EventsPanel frame={frame} />
              </div>
            </div>
          )}

          {tab === "tuning" && (
            <div className="space-y-4">
              <ProfilesPanel frame={frame} />
              <ApplicationsPanel frame={frame} />
            </div>
          )}

          {tab === "connect" && <OnboardingPanel frame={frame} />}

          {tab === "benchmark" && (
            <div className="space-y-4">
              <BenchPanel />
              <CostPanel frame={frame} />
            </div>
          )}

          {tab === "system" && (
            <div className="grid items-start gap-4 xl:grid-cols-2">
              <ModelPanel frame={frame} />
              <SupabasePanel />
              <EnginePanel frame={frame} />
              <ApplicationsPanel frame={frame} />
            </div>
          )}
        </div>

        <footer className="mt-8 border-t border-border/80 pt-4 text-[12px] text-muted-foreground">
          Press <kbd className="rounded border border-border px-1">d</kbd> to switch between
          light and dark.
        </footer>
      </div>
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="rounded-xl border border-border bg-card px-3.5 py-2">
      <div className="text-[12px] uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-[17px] tabular-nums">{value}</div>
    </div>
  )
}

export default App
