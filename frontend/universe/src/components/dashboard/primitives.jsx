import { useEffect, useRef } from "react"
import gsap from "gsap"

import { cn } from "@/lib/utils"

/// Tweens the displayed number instead of snapping to it. A metric that jumps between
/// frames is unreadable at four frames a second; a metric that travels is legible.
export function Ticker({ value, format, className, duration = 0.6 }) {
  const ref = useRef(null)
  const current = useRef(0)

  useEffect(() => {
    const target = Number(value) || 0
    const node = ref.current
    if (!node) return
    const state = { v: current.current }
    const tween = gsap.to(state, {
      v: target,
      duration,
      ease: "power2.out",
      onUpdate: () => {
        current.current = state.v
        node.textContent = format ? format(state.v) : state.v.toFixed(2)
      },
    })
    return () => tween.kill()
  }, [value, format, duration])

  return <span ref={ref} className={className} />
}

export function Panel({ title, subtitle, actions, className, children }) {
  const ref = useRef(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const tween = gsap.fromTo(
      node,
      { opacity: 0, y: 14 },
      { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }
    )
    return () => tween.kill()
  }, [])

  return (
    <section
      ref={ref}
      className={cn(
        "rounded-xl border border-border bg-card/60 p-4 backdrop-blur",
        className
      )}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title && (
              <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, hint, tone = "default" }) {
  const tones = {
    default: "text-foreground",
    good: "text-emerald-500",
    warn: "text-amber-500",
    bad: "text-rose-500",
  }
  return (
    <div className="rounded-lg border border-border/70 bg-background/40 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className={cn("mt-1 font-mono text-xl tabular-nums", tones[tone])}>
        {value}
      </div>
      {hint && (
        <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
          {hint}
        </div>
      )}
    </div>
  )
}

/// Width-animated bar. The tween runs on the element rather than through React state so a
/// four-hertz feed does not schedule a render per bar per frame.
export function Bar({ fraction, className, tone }) {
  const ref = useRef(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const tween = gsap.to(node, {
      width: `${Math.max(0, Math.min(1, fraction || 0)) * 100}%`,
      duration: 0.5,
      ease: "power2.out",
    })
    return () => tween.kill()
  }, [fraction])

  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div ref={ref} className={cn("h-full w-0 rounded-full", tone)} />
    </div>
  )
}

export function Sparkline({ points, width = 240, height = 48, tone = "stroke-primary" }) {
  if (!points || points.length < 2) {
    return <div style={{ height }} className="text-xs text-muted-foreground">no data yet</div>
  }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys)
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1

  const d = points
    .map((p, i) => {
      const x = ((p.x - minX) / spanX) * width
      const y = height - ((p.y - minY) / spanY) * height
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={d} className={cn("fill-none", tone)} strokeWidth="2" />
    </svg>
  )
}
