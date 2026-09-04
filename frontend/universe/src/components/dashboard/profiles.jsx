import { useEffect, useMemo, useState } from "react"

import { del, get, put } from "@/hooks/useLiveFeed"
import { bytes, cn, pct } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

/// Per-application tuning.
///
/// Two rules this panel follows and most config UIs do not:
///
/// * **Every control says what it does before you touch it.** The text comes from
///   `/v1/profiles/knobs`, which the engine serves from the same module that reads the
///   values, so a label cannot describe behaviour the engine does not have.
/// * **The engine's answer is the truth.** Values are clamped server-side; the panel renders
///   whatever comes back from the write rather than what it optimistically sent, so a knob
///   never shows a number the cache is not actually using.

async function putProfile(application, body) {
  return put(`/v1/applications/${encodeURIComponent(application)}/profile`, body)
}

async function resetProfile(application) {
  return del(`/v1/applications/${encodeURIComponent(application)}/profile`)
}

export function ProfilesPanel({ frame }) {
  const [data, setData] = useState(null)
  const [docs, setDocs] = useState(null)
  const [selected, setSelected] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  // The application list grows as applications connect, so this re-reads when that count
  // changes rather than polling a control-plane route four times a second.
  const appCount = frame?.applications?.length ?? 0
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const [profiles, knobs] = await Promise.all([
          get("/v1/profiles"),
          get("/v1/profiles/knobs"),
        ])
        if (!alive) return
        setData(profiles)
        setDocs(knobs)
        setError(null)
        setSelected((cur) => cur ?? profiles?.applications?.[0]?.application ?? null)
      } catch (err) {
        if (alive) setError(String(err))
      }
    })()
    return () => {
      alive = false
    }
  }, [appCount])

  const rows = data?.applications ?? []
  const current = rows.find((r) => r.application === selected) ?? null
  const knobs = docs?.knobs ?? []
  const objectives = docs?.objectives ?? []

  const apply = async (patch) => {
    if (!current) return
    setBusy(true)
    try {
      const next = await putProfile(current.application, patch)
      setData((prev) => ({
        ...prev,
        applications: (prev?.applications ?? []).map((r) =>
          r.application === current.application
            ? { ...r, profile: next.profile, customised: true }
            : r
        ),
      }))
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  const reset = async () => {
    if (!current) return
    setBusy(true)
    try {
      const next = await resetProfile(current.application)
      setData((prev) => ({
        ...prev,
        applications: (prev?.applications ?? []).map((r) =>
          r.application === current.application
            ? { ...r, profile: next.profile, customised: false }
            : r
        ),
      }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="Per-application tuning"
      subtitle="One cache serves applications whose objects cost different things to rebuild. These settings say what 'valuable' means for each one. They shape the arithmetic around the model's prediction and never the prediction itself."
      actions={
        current && (
          <div className="flex items-center gap-2">
            <Pill tone={current.customised ? "accent" : "default"}>
              {current.customised ? "customised" : "using defaults"}
            </Pill>
            {current.customised && (
              <button
                onClick={reset}
                disabled={busy}
                className="rounded-lg border border-border px-2.5 py-1 text-[13px] hover:bg-accent disabled:opacity-50"
              >
                Reset
              </button>
            )}
          </div>
        )
      }
    >
      {error && (
        <p className="mb-3 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[13px] text-amber-300">
          Could not reach the engine: {error}
        </p>
      )}

      {rows.length === 0 ? (
        <p className="text-[13.5px] text-muted-foreground">
          No applications yet. An application appears here the moment it makes its first call
          to the cache; nothing has to be registered first.
        </p>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[240px_minmax(0,1fr)]">
          <nav className="flex flex-col gap-1.5">
            {rows.map((r) => (
              <button
                key={r.application}
                onClick={() => setSelected(r.application)}
                className={cn(
                  "rounded-xl border px-3 py-2.5 text-left transition-colors",
                  selected === r.application
                    ? "border-primary/50 bg-primary/10"
                    : "border-border bg-background hover:bg-accent/50"
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[13.5px] font-semibold">{r.application}</span>
                  {r.customised && <span className="h-1.5 w-1.5 rounded-full bg-primary" />}
                </div>
                <div className="mt-1 text-[12px] text-muted-foreground">
                  {bytes(r.resident_bytes ?? 0)} held · {pct(r.pool_share ?? 0, 1)} of pool
                </div>
                <div className="mt-0.5 text-[12px] text-muted-foreground">
                  optimising for {r.profile?.objective ?? "cost"}
                </div>
              </button>
            ))}
          </nav>

          {current && (
            <div className="min-w-0 space-y-5">
              <div>
                <h3 className="text-[13.5px] font-semibold">What should this application optimise for?</h3>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  {objectives.map((o) => (
                    <button
                      key={o.id}
                      onClick={() => apply({ objective: o.id })}
                      disabled={busy || o.id === "custom"}
                      className={cn(
                        "rounded-xl border px-3.5 py-3 text-left transition-colors disabled:cursor-default",
                        current.profile?.objective === o.id
                          ? "border-primary/50 bg-primary/10"
                          : "border-border bg-background hover:bg-accent/40",
                        o.id === "custom" && current.profile?.objective !== "custom" && "opacity-45"
                      )}
                    >
                      <div className="text-[13.5px] font-semibold">{o.label}</div>
                      <p className="mt-1 text-[12.5px] leading-snug text-muted-foreground">
                        {o.effect}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                {knobs.map((k) =>
                  k.kind === "weights" ? (
                    <WeightKnob
                      key={k.id}
                      doc={k}
                      value={current.profile?.horizon_weights ?? [0.5, 0.35, 0.15]}
                      busy={busy}
                      onChange={(w) => apply({ horizon_weights: w })}
                    />
                  ) : (
                    <NumberKnob
                      key={k.id}
                      doc={k}
                      value={current.profile?.[k.id] ?? 0}
                      busy={busy}
                      onChange={(v) => apply({ [k.id]: v })}
                    />
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

/// A slider that explains itself in both directions. The two sentences under it are the
/// point: an operator should be able to predict what a change will do before making it.
function NumberKnob({ doc, value, busy, onChange }) {
  const [local, setLocal] = useState(value)
  const [applied, setApplied] = useState(value)
  if (applied !== value) {
    setApplied(value)
    setLocal(value)
  }

  return (
    <div className="rounded-xl border border-border bg-background px-4 py-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <span className="text-[13.5px] font-semibold">{doc.label}</span>
          <p className="mt-0.5 text-[12.5px] text-muted-foreground">{doc.what}</p>
        </div>
        <span className="font-mono text-[15px] tabular-nums">{Number(local).toFixed(2)}</span>
      </div>

      <input
        type="range"
        min={doc.min}
        max={doc.max}
        step={doc.step}
        value={local}
        disabled={busy}
        onChange={(e) => setLocal(Number(e.target.value))}
        onPointerUp={() => onChange(local)}
        onKeyUp={() => onChange(local)}
        className="mt-3 w-full accent-primary"
      />

      <div className="mt-2.5 grid gap-1.5 sm:grid-cols-2">
        <Effect arrow="↑" text={doc.raise} />
        <Effect arrow="↓" text={doc.lower} />
      </div>
    </div>
  )
}

function WeightKnob({ doc, value, busy, onChange }) {
  const [local, setLocal] = useState(value)
  const [applied, setApplied] = useState(value.join())
  if (applied !== value.join()) {
    setApplied(value.join())
    setLocal(value)
  }
  const total = useMemo(() => local.reduce((a, b) => a + b, 0) || 1, [local])

  const set = (i, v) => {
    const next = [...local]
    next[i] = v
    setLocal(next)
  }

  return (
    <div className="rounded-xl border border-border bg-background px-4 py-3.5">
      <div>
        <span className="text-[13.5px] font-semibold">{doc.label}</span>
        <p className="mt-0.5 text-[12.5px] text-muted-foreground">{doc.what}</p>
      </div>

      <div className="mt-3 space-y-2.5">
        {(doc.parts ?? []).map((part, i) => (
          <div key={part} className="flex items-center gap-3">
            <span className="w-[86px] shrink-0 text-[12.5px] text-muted-foreground">{part}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={local[i] ?? 0}
              disabled={busy}
              onChange={(e) => set(i, Number(e.target.value))}
              onPointerUp={() => onChange(local)}
              onKeyUp={() => onChange(local)}
              className="w-full accent-primary"
            />
            <span className="w-[52px] shrink-0 text-right font-mono text-[13px] tabular-nums">
              {pct((local[i] ?? 0) / total, 0)}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-2.5 grid gap-1.5 sm:grid-cols-2">
        <Effect arrow="↑" text={doc.raise} />
        <Effect arrow="↓" text={doc.lower} />
      </div>
    </div>
  )
}

function Effect({ arrow, text }) {
  return (
    <p className="rounded-lg bg-muted/40 px-2.5 py-1.5 text-[12px] leading-snug text-muted-foreground">
      <span className="mr-1 font-mono text-foreground/80">{arrow}</span>
      {text}
    </p>
  )
}
