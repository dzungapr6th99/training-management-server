import { useEffect, useState, useCallback } from 'react'

export async function api(path, opts) {
  const res = await fetch('/api' + path, opts)
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const e = new Error(body?.detail?.message || body?.detail || res.statusText)
    e.problems = body?.detail?.problems
    throw e
  }
  return body
}

export function useApi(path) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const load = useCallback(() => {
    setError(null)
    api(path).then(setData).catch(setError)
  }, [path])
  useEffect(() => { setData(null); load() }, [load])
  return { data, error, reload: load }
}

export const pct = (x, d = 0) => (x == null ? '—' : (x * 100).toFixed(d) + '%')
export const num = (x) => (x == null ? '—' : Number(x).toLocaleString('en-US'))
export function bytes(b) {
  if (b == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let v = b
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + u[i]
}
export const ci = (lo, hi) => (lo == null ? '—' : `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`)
