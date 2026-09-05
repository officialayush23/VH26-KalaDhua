import { useEffect, useMemo, useState } from "react"

import { del, get, post } from "@/hooks/useLiveFeed"
import { bytes, cn, pct, usd } from "@/lib/utils"
import { Panel, Pill } from "./primitives"

/// Connecting an application to the cache.
///
/// The whole onboarding story is: mint a key, point a service at the URL, watch it appear.
/// There is no registration step and no schema to declare, because the key *is* the
/// application's identity - the engine attributes every request under it to the application
/// the key was issued to, and refuses to let a request body claim a different name. So the
/// moment a service makes its first call it shows up here, in the log, in the tuning panel
/// and in the cost breakdown, with nothing else done.
///
/// The secret is shown exactly once. Nothing stores it: the engine keeps a SHA-256 hash, and
/// so does the control plane, which means a copy of either is not a working credential.

const BASE = import.meta.env.VITE_AURA_URL || "http://localhost:8080"

export function OnboardingPanel({ frame }) {
  const [keys, setKeys] = useState([])
  const [enforced, setEnforced] = useState(false)
  const [name, setName] = useState("")
  const [minted, setMinted] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const seen = useMemo(
    () => new Set((frame?.applications ?? []).map((a) => a.application)),
    [frame]
  )

  const refresh = async () => {
    try {
      const data = await get("/v1/keys")
      setKeys(data?.keys ?? [])
      setEnforced(Boolean(data?.enforced))
      setError(null)
    } catch (err) {
      setError(String(err))
    }
  }

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const data = await get("/v1/keys")
        if (!alive) return
        setKeys(data?.keys ?? [])
        setEnforced(Boolean(data?.enforced))
      } catch (err) {
        if (alive) setError(String(err))
      }
    })()
    return () => {
      alive = false
    }
  }, [])

  const mint = async () => {
    const application = name.trim()
    if (!application) return
    setBusy(true)
    try {
      const res = await post("/v1/keys", { application })
      if (res?.error) {
        setError(res.error)
      } else {
        setMinted({ application, secret: res.secret, id: res.key?.id })
        setName("")
        await refresh()
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  const revoke = async (id) => {
    setBusy(true)
    try {
      await del(`/v1/keys/${encodeURIComponent(id)}`)
      await refresh()
    } catch (err) {
      setError(String(err.message ?? err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Connect an application"
        subtitle="Mint a key, point your service at the cache, and it appears here. Nothing else is registered: the key is the application's identity, so its objects, its costs and its tuning attach themselves."
        actions={
          <Pill tone={enforced ? "accent" : "warn"}>
            {enforced ? "keys required" : "open engine"}
          </Pill>
        }
      >
        {!enforced && (
          <p className="mb-4 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[13px] text-amber-300">
            This engine is running open: it answers every request whether or not a key is
            presented. Keys still work and still identify the caller, but set
            <code className="mx-1 rounded bg-black/20 px-1 font-mono">AURA_AUTH=enforced</code>
            before it has a public address.
          </p>
        )}

        <div className="flex flex-wrap items-end gap-3">
          <label className="flex-1 min-w-[220px]">
            <span className="text-[12.5px] font-medium text-muted-foreground">
              Application name
            </span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && mint()}
              placeholder="checkout-api"
              className="mt-1.5 h-10 w-full rounded-xl border border-border bg-background px-3 text-[14px] outline-none focus:border-primary/60"
            />
          </label>
          <button
            onClick={mint}
            disabled={busy || !name.trim()}
            className="h-10 rounded-xl bg-primary px-4 text-[13.5px] font-semibold text-primary-foreground disabled:opacity-50"
          >
            Mint key
          </button>
        </div>

        {error && (
          <p className="mt-3 rounded-lg border border-rose-400/40 bg-rose-400/10 px-3 py-2 text-[13px] text-rose-300">
            {error}
          </p>
        )}

        {minted && <Minted minted={minted} connected={seen.has(minted.application)} />}
      </Panel>

      <ConnectedServices />

      <Panel
        title="Issued keys"
        subtitle="Hashes and prefixes only. The secret existed once, at mint time, and is not recoverable from here or from the database."
      >
        {keys.length === 0 ? (
          <p className="text-[13.5px] text-muted-foreground">No keys yet.</p>
        ) : (
          <div className="divide-y divide-border/70">
            {keys.map((k) => (
              <div key={k.id} className="flex flex-wrap items-center gap-3 py-2.5">
                <span className="w-[150px] shrink-0 text-[13.5px] font-semibold">
                  {k.application}
                </span>
                <code className="rounded bg-muted/50 px-1.5 py-0.5 font-mono text-[12.5px]">
                  {k.hint}…
                </code>
                <span className="text-[12.5px] text-muted-foreground">{k.id}</span>
                {seen.has(k.application) ? (
                  <Pill tone="accent">connected</Pill>
                ) : (
                  <Pill>no traffic yet</Pill>
                )}
                {k.revoked ? (
                  <Pill tone="bad">revoked</Pill>
                ) : (
                  <button
                    onClick={() => revoke(k.id)}
                    disabled={busy}
                    className="ml-auto rounded-lg border border-border px-2.5 py-1 text-[12.5px] hover:bg-accent disabled:opacity-50"
                  >
                    Revoke
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  )
}

/// Who is actually using this cache, as opposed to who was issued a credential.
///
/// The engine records the moment it last saw each key, so this is an observation rather
/// than a configuration file: a service appears here when it makes its first authenticated
/// call and falls quiet on its own when it stops.
function ConnectedServices() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const d = await get("/v1/connections")
        if (alive) {
          setData(d)
          setError(null)
        }
      } catch (err) {
        // "Nobody is connected" and "the engine would not tell me" look identical on this
        // panel and have completely different fixes, so the second one has to say so.
        if (alive) setError(String(err.message ?? err))
      }
    }
    load()
    const id = setInterval(load, 4000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const keys = data?.keys ?? []
  const unkeyed = data?.without_a_key ?? []
  const live = keys.filter((k) => k.connected).length

  return (
    <Panel
      title="Connected services"
      subtitle="Observed, not declared. A service appears the moment it makes its first authenticated call, and goes quiet on its own."
      actions={<Pill tone={live > 0 ? "accent" : "default"}>{live} live now</Pill>}
    >
      {error && (
        <p className="mb-3 rounded-lg border border-rose-400/40 bg-rose-400/10 px-3 py-2 text-[13px] text-rose-300">
          {error}
        </p>
      )}
      {keys.length === 0 && unkeyed.length === 0 ? (
        <p className="text-[13.5px] text-muted-foreground">
          Nothing has called this engine yet. Mint a key above and point a service at it.
        </p>
      ) : (
        <div className="space-y-2.5">
          {keys.map((k) => (
            <div
              key={k.key_id}
              className={cn(
                "rounded-xl border px-4 py-3",
                k.connected ? "border-primary/45 bg-primary/[0.06]" : "border-border bg-background"
              )}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full",
                      k.connected ? "animate-pulse bg-primary" : "bg-muted-foreground/50"
                    )}
                  />
                  <span className="text-[14px] font-semibold">{k.application}</span>
                  <code className="rounded bg-muted/50 px-1.5 py-0.5 font-mono text-[11.5px] text-muted-foreground">
                    {k.hint}…
                  </code>
                  {k.revoked && <Pill tone="bad">revoked</Pill>}
                </div>
                <span className="text-[12.5px] text-muted-foreground">
                  {k.last_seen_ms_ago == null
                    ? "never used"
                    : `last call ${humanAgo(k.last_seen_ms_ago)}`}
                </span>
              </div>
              {k.traffic && (
                <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-muted-foreground">
                  <span>
                    requests{" "}
                    <span className="font-mono tabular-nums text-foreground">
                      {Number(k.traffic.requests ?? 0).toLocaleString()}
                    </span>
                  </span>
                  <span>
                    hit rate{" "}
                    <span className="font-mono tabular-nums text-foreground">
                      {pct(k.traffic.hit_rate ?? 0, 1)}
                    </span>
                  </span>
                  <span>
                    holding{" "}
                    <span className="font-mono tabular-nums text-foreground">
                      {bytes(k.traffic.resident_bytes ?? 0)}
                    </span>
                  </span>
                  <span>
                    spent rebuilding{" "}
                    <span className="font-mono tabular-nums text-foreground">
                      {usd(k.traffic.cost_usd ?? 0)}
                    </span>
                  </span>
                </div>
              )}
            </div>
          ))}

          {unkeyed.map((u) => (
            <div key={u.application} className="rounded-xl border border-dashed border-border px-4 py-3">
              <div className="flex items-center gap-2.5">
                <span className="h-2 w-2 rounded-full bg-amber-400/70" />
                <span className="text-[14px] font-semibold">{u.application}</span>
                <span className="text-[12.5px] text-muted-foreground">
                  calling without a key — fine on an open engine, refused once enforcement is on
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

function humanAgo(ms) {
  const s = Math.round(ms / 1000)
  if (s < 2) return "just now"
  if (s < 60) return `${s}s ago`
  const m = Math.round(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.round(m / 60)}h ago`
}

function Minted({ minted, connected }) {
  const [copied, setCopied] = useState(false)
  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard access is denied in some browsers over plain http; the value is on
      // screen and selectable, which is the fallback.
    }
  }

  const curl = `curl -X PUT ${BASE}/v1/cache/report:q42 \\
  -H "Authorization: Bearer ${minted.secret}" \\
  -H "content-type: application/json" \\
  -d '{"value":{"rows":[]},"context":{"application":"${minted.application}","object_type":"report","size_bytes":48000,"ttl_ms":300000},"measured":{"db_ms":412}}'`

  const python = `from common.aura_client import AuraClient

cache = AuraClient(
    base_url="${BASE}",
    api_key="${minted.secret}",
)

hit = await cache.get(key)
if hit is None:
    value, cost = await rebuild(key)   # your work, measured
    await cache.put(key, value, cost)  # AURA decides whether to keep it`

  return (
    <div className="mt-5 rounded-2xl border border-primary/45 bg-primary/[0.06] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-[14px] font-semibold">
            Key for {minted.application}
          </h3>
          <p className="mt-0.5 text-[12.5px] text-muted-foreground">
            Copy it now. This is the only time it is shown, and nothing stores it.
          </p>
        </div>
        <button
          onClick={() => copy(minted.secret)}
          className="rounded-lg border border-primary/50 bg-primary/10 px-3 py-1.5 text-[13px] font-medium"
        >
          {copied ? "Copied" : "Copy key"}
        </button>
      </div>

      <code className="mt-3 block overflow-x-auto rounded-lg border border-border bg-background px-3 py-2.5 font-mono text-[13px]">
        {minted.secret}
      </code>

      <div className="mt-4 space-y-3">
        <Snippet title="Try it with curl" body={curl} />
        <Snippet title="Or from the client the example apps use" body={python} />
      </div>

      <div
        className={cn(
          "mt-4 flex items-center gap-2 rounded-lg border px-3 py-2 text-[13px]",
          connected
            ? "border-primary/50 bg-primary/10"
            : "border-border bg-background text-muted-foreground"
        )}
      >
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            connected ? "bg-primary" : "animate-pulse bg-amber-400"
          )}
        />
        {connected
          ? `${minted.application} is connected and its traffic is in the log.`
          : `Waiting for the first call from ${minted.application}…`}
      </div>
    </div>
  )
}

function Snippet({ title, body }) {
  return (
    <div>
      <div className="text-[12.5px] font-medium text-muted-foreground">{title}</div>
      <pre className="mt-1.5 overflow-x-auto rounded-lg border border-border bg-background px-3 py-2.5 font-mono text-[12.5px] leading-relaxed">
        {body}
      </pre>
    </div>
  )
}
