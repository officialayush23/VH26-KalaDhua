import { useCallback, useEffect, useRef, useState } from "react"

const API = import.meta.env.VITE_AURA_URL || "http://localhost:8080"
const WS = API.replace(/^http/, "ws")

/// Subscribes to the engine's telemetry socket. Every frame is a complete snapshot, so a
/// dropped frame costs nothing and there is no state to reconcile on reconnect. When the
/// socket cannot be established the hook falls back to polling the same payload.
export function useLiveFeed() {
  const [frame, setFrame] = useState(null)
  const [status, setStatus] = useState("connecting")
  const socketRef = useRef(null)
  const retryRef = useRef(null)
  const pollRef = useRef(null)

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
        const res = await fetch(`${API}/v1/stats`)
        if (!res.ok) throw new Error(String(res.status))
        setFrame(await res.json())
        setStatus("polling")
      } catch {
        setStatus("offline")
      }
    }, 1000)
  }, [])

  useEffect(() => {
    let closed = false

    const connect = () => {
      if (closed) return
      let socket
      try {
        socket = new WebSocket(`${WS}/v1/live`)
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
          setFrame(JSON.parse(event.data))
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
  }, [startPolling])

  const send = useCallback((message) => {
    const socket = socketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message))
      return true
    }
    return false
  }, [])

  return { frame, status, send, api: API }
}

export async function post(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
  return res.json()
}

export async function get(path) {
  const res = await fetch(`${API}${path}`)
  return res.json()
}
