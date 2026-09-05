import { useCallback, useEffect, useRef, useState } from "react"

import { authHeaders, token } from "@/lib/session"

const API = import.meta.env.VITE_AURA_URL || "http://localhost:8080"
const WS = API.replace(/^http/, "ws")
const HISTORY_LIMIT = 240
// Scrollback the client keeps. The engine only ever sends the tail of its own log, so this
// is what makes the console readable: the page accumulates, the socket stays small.
const LOG_LIMIT = 500

/// Subscribes to the engine's telemetry socket. Every frame is a complete snapshot, so a
/// dropped frame costs nothing and there is no state to reconcile on reconnect. If the
/// socket cannot be established the hook polls the same payload instead.
///
/// History is accumulated here rather than in each chart, so every chart on the page is
/// drawn from the same series and cannot disagree with its neighbour.
export function useLiveFeed() {
  const [frame, setFrame] = useState(null)
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState("connecting")
  const [log, setLog] = useState([])
  const socketRef = useRef(null)
  const retryRef = useRef(null)
  const pollRef = useRef(null)
  const attemptsRef = useRef(0)

  const record = useCallback((next) => {
    setFrame(next)

    // Audit entries arrive newest first and repeat across frames. `seq` is monotonic and
    // unique, so merging on it means a slow tab, a dropped frame or a reconnect all end up
    // with the same scrollback and never a duplicate line.
    const incoming = next?.audit
    if (Array.isArray(incoming) && incoming.length) {
      setLog((prev) => {
        const seen = new Set(prev.map((e) => e.seq))
        const fresh = incoming.filter((e) => e && !seen.has(e.seq))
        if (!fresh.length) return prev
        const out = [...fresh, ...prev].sort((a, b) => b.seq - a.seq)
        return out.length > LOG_LIMIT ? out.slice(0, LOG_LIMIT) : out
      })
    }

    setHistory((prev) => {
      const t = (next?.virtual_time_s ?? 0)
      const last = prev[prev.length - 1]
      if (last && Math.abs(last.t - t) < 0.2) return prev
      // Every field a chart on the Evidence tab needs, captured here rather than in each
      // chart. The engine sends complete snapshots and keeps no history of its own, so this
      // buffer is the only series that exists -- a value not recorded here can never be
      // plotted, no matter that it arrives in every frame.
      const base = next?.cost?.baselines ?? {}
      const point = {
        t,
        hitRate: next?.layers?.l2?.hit_rate ?? 0,
        // The applications' own in-process caches, as they report them. The engine cannot
        // measure this: a request served from a local copy never reaches it.
        hitRateL1: next?.tier1?.hit_rate ?? 0,
        l1Reporting: next?.tier1?.reporting ?? 0,
        admissionWindow: next?.layers?.l1?.hit_rate ?? 0,
        byteHitRate: next?.layers?.l2?.byte_hit_rate ?? 0,
        p95: next?.latency?.p95_ms ?? 0,
        p50: next?.latency?.p50_ms ?? 0,
        totalUsd: next?.cost?.total_usd ?? 0,
        savedUsd: next?.cost?.saved_vs_no_cache_usd ?? 0,
        rps: next?.sim?.rps ?? 0,
        baseRps: next?.sim?.base_rps ?? 0,
        pressure: next?.capacity?.pressure ?? 0,
        capacity: next?.capacity?.logical_bytes ?? 0,
        usedBytes: next?.capacity?.used_bytes ?? 0,
        decision: next?.capacity?.decision ?? "hold",
        // Cumulative counters. Charts difference consecutive points to get a rate, which is
        // the only way to see a burst: a monotonically rising total hides everything.
        admissions: next?.engine?.admissions ?? 0,
        rejections: next?.engine?.admissions_rejected ?? 0,
        evictions: next?.engine?.evictions ?? 0,
        requests: next?.engine?.requests ?? 0,
        mixture: next?.policy?.mixture ?? null,
        regime: next?.workload?.regime ?? null,
        baselineUsd: {
          lru: base?.lru?.total_usd ?? 0,
          lfu: base?.lfu?.total_usd ?? 0,
          gds: base?.gds?.total_usd ?? 0,
        },
      }
      const out = [...prev, point]
      return out.length > HISTORY_LIMIT ? out.slice(out.length - HISTORY_LIMIT) : out
    })
  }, [])

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const startPolling = useCallback(() => {
    if (pollRef.current) return
    const tick = async () => {
      try {
        const res = await fetch(`${API}/v1/stats`, { headers: authHeaders() })
        if (!res.ok) throw new Error(String(res.status))
        record(await res.json())
        // Polling is a working state, not a failure. Saying so is the difference between
        // a console that looks broken while showing correct numbers and one that tells the
        // truth about how it got them.
        setStatus("polling")
      } catch {
        setStatus("offline")
      }
    }
    // Fire immediately, then on the interval: waiting a full second before the first
    // request means a second of "offline" every time the socket drops.
    tick()
    pollRef.current = setInterval(tick, 1000)
  }, [record])

  useEffect(() => {
    let closed = false

    const connect = () => {
      if (closed) return
      let socket
      try {
        // A browser cannot set a header on a socket handshake, so the token goes in the
        // query string. It is the same token, over the same TLS connection.
        const t = token()
        socket = new WebSocket(`${WS}/v1/live${t ? `?token=${encodeURIComponent(t)}` : ""}`)
      } catch {
        startPolling()
        return
      }
      socketRef.current = socket

      socket.onopen = () => {
        attemptsRef.current = 0
        stopPolling()
        setStatus("live")
      }
      socket.onmessage = (event) => {
        try {
          record(JSON.parse(event.data))
        } catch {
          // A malformed frame is dropped; the next one is a full snapshot anyway.
        }
      }
      socket.onerror = () => startPolling()
      socket.onclose = () => {
        if (closed) return
        attemptsRef.current += 1
        // Do NOT declare "reconnecting" here. Polling has already been started and is
        // setting the status from what it actually observes; overwriting it on every
        // failed handshake is what pinned the header to "reconnecting" while correct data
        // was arriving underneath it every second.
        startPolling()

        // Exponential backoff with jitter, capped. A socket that is being refused for a
        // structural reason -- a proxy that will not upgrade, a plan that does not carry
        // WebSockets -- is not going to succeed on the next attempt either, and retrying
        // every two seconds forever only fills the console with noise. After the backoff
        // saturates we still try occasionally, because the reason may be transient.
        const n = Math.min(attemptsRef.current, 6)
        const base = Math.min(2000 * 2 ** (n - 1), 60_000)
        const delay = base + Math.random() * base * 0.3
        retryRef.current = setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      closed = true
      stopPolling()
      if (retryRef.current) clearTimeout(retryRef.current)
      if (socketRef.current) socketRef.current.close()
    }
  }, [startPolling, record])

  const send = useCallback((message) => {
    const socket = socketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message))
      return true
    }
    return false
  }, [])

  return { frame, history, status, send, log, api: API }
}

/// One request, with the status actually checked.
///
/// The version of this that did `return res.json()` was the single worst bug in the
/// console. Every gated route answers a refusal with `{ "error": ... }` and a 401, so a
/// helper that ignores the status hands that object back as if it were data: the keys panel
/// reads `data.keys ?? []` and says "no keys yet", the connections panel shows nobody
/// connected, saving a profile appears to succeed and changes nothing, and the benchmark
/// button finishes with no table. Four unrelated-looking features quietly broken by one
/// missing line, and every one of them lying about *why* -- an empty list is a much worse
/// answer than "you are not signed in".
async function request(path, init) {
  let res
  try {
    res = await fetch(`${API}${path}`, init)
  } catch {
    // A network-level failure is not a server response, and it has a different fix: the
    // engine's address is wrong, it is asleep, or this browser cannot reach it at all.
    throw new Error(`could not reach the engine at ${API}`)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const err = new Error(
      body?.error || `${init?.method ?? "GET"} ${path} failed (${res.status})`
    )
    err.status = res.status
    err.fix = body?.fix
    throw err
  }
  return body
}

const json = (method, body) => ({
  method,
  headers: { "content-type": "application/json", ...authHeaders() },
  body: JSON.stringify(body ?? {}),
})

export async function get(path) {
  return request(path, { headers: authHeaders() })
}

export async function del(path) {
  return request(path, { method: "DELETE", headers: authHeaders() })
}

export async function put(path, body) {
  return request(path, json("PUT", body))
}

export async function post(path, body) {
  return request(path, json("POST", body))
}
