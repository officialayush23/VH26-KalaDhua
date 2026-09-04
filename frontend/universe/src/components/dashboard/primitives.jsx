import { useEffect, useRef } from "react"
import gsap from "gsap"

import { cn } from "@/lib/utils"

/// Tweens the displayed number rather than snapping to it. At four frames a second a
/// metric that jumps is unreadable; one that travels can be followed.
export function Ticker({ value, format, className, duration = 0.55 }) {
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

export function Panel({ title, subtitle, actions, footer, className, children }) {
  const ref = useRef(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const tween = gsap.fromTo(
      node,
      { opacity: 0, y: 12 },
      { opacity: 1, y: 0, duration: 0.45, ease: "power2.out" }
    )
    return () => tween.kill()
  }, [])

  return (
    <section
      ref={ref}
      className={cn(
        "flex min-w-0 flex-col rounded-2xl border border-border bg-card shadow-sm",
        className
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-border/80 px-5 py-4">
          <div className="min-w-0">
            {title && (
              <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
            )}
            {subtitle && (
              <p className="mt-1 text-[13.5px] leading-snug text-muted-foreground">
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </header>
      )}
      <div className="flex-1 px-5 py-4">{children}</div>
      {footer && (
        <footer className="border-t border-border/80 px-5 py-3 text-[12px] text-muted-foreground">
          {footer}
        </footer>
      )}
    </section>
  )
}

const TONES = {
  default: "text-foreground",
  good: "text-emerald-400",
  warn: "text-amber-400",
  bad: "text-rose-400",
  accent: "text-primary",
}

/// A labelled number. `unit` and `explain` exist because a bare figure on a dashboard is
/// a puzzle: the reader should never have to guess what they are looking at.
export function Metric({ label, value, unit, explain, tone = "default", size = "md" }) {
  const sizes = {
    sm: "text-xl",
    md: "text-[27px]",
    lg: "text-[38px]",
  }
  return (
    <div className="flex flex-col rounded-xl border border-border bg-background/80 px-4 py-3">
      <div className="text-[12px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span
          className={cn(
            "font-mono font-medium leading-none tabular-nums",
            sizes[size],
            TONES[tone]
          )}
        >
          {value}
        </span>
        {unit && (
          <span className="text-[13.5px] font-medium text-muted-foreground">{unit}</span>
        )}
      </div>
      {explain && (
        <div className="mt-2 text-[12px] leading-snug text-muted-foreground">{explain}</div>
      )}
    </div>
  )
}

/// Width-animated bar. The tween targets the element directly so a four-hertz feed does
/// not schedule a React render per bar per frame.
export function Bar({ fraction, tone = "bg-primary", track = "bg-muted/60", height = "h-2" }) {
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
    <div className={cn("w-full overflow-hidden rounded-full", height, track)}>
      <div ref={ref} className={cn("h-full w-0 rounded-full", tone)} />
    </div>
  )
}

export function Gauge({ fraction, label, value, tone = "stroke-primary" }) {
  const ref = useRef(null)
  const r = 42
  const circumference = 2 * Math.PI * r

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const clamped = Math.max(0, Math.min(1, fraction || 0))
    const tween = gsap.to(node, {
      strokeDashoffset: circumference * (1 - clamped),
      duration: 0.6,
      ease: "power2.out",
    })
    return () => tween.kill()
  }, [fraction, circumference])

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 100" className="h-28 w-28 -rotate-90">
        <circle cx="50" cy="50" r={r} className="fill-none stroke-muted/50" strokeWidth="9" />
        <circle
          ref={ref}
          cx="50"
          cy="50"
          r={r}
          className={cn("fill-none", tone)}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
        />
      </svg>
      <div className="-mt-[4.6rem] mb-8 text-center">
        <div className="font-mono text-2xl font-medium tabular-nums">{value}</div>
        <div className="mt-0.5 text-[12px] uppercase tracking-wide text-muted-foreground">
          {label}
        </div>
      </div>
    </div>
  )
}

/// Line chart with labelled axes. An unlabelled sparkline says "something changed" and
/// nothing else, which is not worth the space it takes.
export function LineChart({
  points,
  height = 150,
  tone = "stroke-primary",
  fill = "fill-primary/10",
  yFormat = (v) => v.toFixed(2),
  xFormat = (v) => String(v),
  yLabel,
  xLabel,
}) {
  if (!points || points.length < 2) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center rounded-lg border border-dashed border-border text-[12px] text-muted-foreground"
      >
        collecting data
      </div>
    )
  }

  const width = 480
  const padL = 46
  const padB = 22
  const padT = 8
  const padR = 8
  const plotW = width - padL - padR
  const plotH = height - padT - padB

  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys) || 1
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1

  const px = (x) => padL + ((x - minX) / spanX) * plotW
  const py = (y) => padT + plotH - ((y - minY) / spanY) * plotH

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`).join(" ")
  const area = `${line} L${px(maxX).toFixed(1)},${py(minY).toFixed(1)} L${px(minX).toFixed(1)},${py(minY).toFixed(1)} Z`

  const yTicks = [minY, minY + spanY / 2, maxY]
  const xTicks = [minX, minX + spanX / 2, maxX]

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={padL}
              x2={width - padR}
              y1={py(t)}
              y2={py(t)}
              className="stroke-border/50"
              strokeWidth="1"
            />
            <text
              x={padL - 6}
              y={py(t) + 3.5}
              textAnchor="end"
              className="fill-muted-foreground text-[9px]"
            >
              {yFormat(t)}
            </text>
          </g>
        ))}
        <path d={area} className={cn("stroke-none", fill)} />
        <path d={line} className={cn("fill-none", tone)} strokeWidth="2" strokeLinejoin="round" />
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
      {(yLabel || xLabel) && (
        <div className="mt-1 flex justify-between text-[12px] text-muted-foreground">
          <span>{yLabel}</span>
          <span>{xLabel}</span>
        </div>
      )}
    </div>
  )
}

export function Legend({ items }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-1.5 text-[12px]">
          <span className={cn("h-2.5 w-2.5 shrink-0 rounded-sm", it.tone)} />
          <span className="text-muted-foreground">{it.label}</span>
          {it.value !== undefined && (
            <span className="font-mono tabular-nums">{it.value}</span>
          )}
        </div>
      ))}
    </div>
  )
}

export function Pill({ children, tone = "default" }) {
  const tones = {
    default: "border-border bg-muted/40 text-muted-foreground",
    good: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
    warn: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    bad: "border-rose-500/40 bg-rose-500/10 text-rose-400",
    accent: "border-primary/40 bg-primary/10 text-primary",
  }
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[12px] font-medium",
        tones[tone]
      )}
    >
      {children}
    </span>
  )
}
