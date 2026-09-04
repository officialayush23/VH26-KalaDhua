import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

export function usd(v, digits = 2) {
  if (v === undefined || v === null || Number.isNaN(v)) return "$0.00"
  return `$${Number(v).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`
}

export function pct(v, digits = 1) {
  if (v === undefined || v === null || Number.isNaN(v)) return "0.0%"
  return `${(Number(v) * 100).toFixed(digits)}%`
}

export function bytes(b) {
  const units = ["B", "KB", "MB", "GB", "TB"]
  let v = Number(b) || 0
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function ms(v) {
  const n = Number(v) || 0
  if (n >= 1000) return `${(n / 1000).toFixed(2)} s`
  return `${n.toFixed(n < 10 ? 2 : 0)} ms`
}
