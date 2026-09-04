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
      const point = {
        t,
        hitRate: next?.layers?.l2?.hit_rate ?? 0,
        byteHitRate: next?.layers?.l2?.byte_hit_rate ?? 0,
        p95: next?.latency?.p95_ms ?? 0,
        p50: next?.latency?.p50_ms ?? 0,
        totalUsd: next?.cost?.total_usd ?? 0,
        savedUsd: next?.cost?.saved_vs_no_cache_usd ?? 0,
        rps: next?.sim?.rps ?? 0,
        pressure: next?.capacity?.pressure ?? 0,
        capacity: next?.capacity?.logical_bytes ?? 0,
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
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/v1/stats`, { headers: authHeaders() })
        if (!res.ok) throw new Error(String(res.status))
        record(await res.json())
        setStatus("polling")
      } catch {
        setStatus("offline")
      }
    }, 1000)
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
        setStatus("reconnecting")
        startPolling()
        retryRef.current = setTimeout(connect, 2000)
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

export async function post(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body ?? {}),
  })
  return res.json()
}

export async function get(path) {
  const res = await fetch(`${API}${path}`, { headers: authHeaders() })
  return res.json()
}

export async function del(path) {
  const res = await fetch(`${API}${path}`, { method: "DELETE", headers: authHeaders() })
  return res.json()
}

export async function put(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "PUT",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body ?? {}),
  })
  return res.json()
}
