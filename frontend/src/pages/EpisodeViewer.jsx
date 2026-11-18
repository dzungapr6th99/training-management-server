import React from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, useApi } from '../api.js'
import { Badge, Load } from '../ui.jsx'

const COLORS = ['#2563eb', '#d97706', '#7c3aed', '#0f766e', '#be185d', '#65a30d', '#0891b2', '#dc2626']

/** Group the channels of a 16-D humanoid vector (left/right arm + gripper). Other robots: one group per 8 channels. */
function groupsFor(names) {
  if (!names || !names.length) return []
  const idx = (pred) => names.map((n, i) => (pred(n) ? i : -1)).filter((i) => i >= 0)
  const g = [
    ['Tay trái (khớp)', idx((n) => /left.*joint/i.test(n))],
    ['Tay phải (khớp)', idx((n) => /right.*joint/i.test(n))],
    ['Gripper', idx((n) => /gripper/i.test(n))],
  ].filter(([, v]) => v.length)
  if (g.length) return g
  const out = []
  for (let i = 0; i < names.length; i += 8) out.push([`Kênh ${i}–${Math.min(i + 7, names.length - 1)}`, names.slice(i, i + 8).map((_, k) => i + k)])
  return out
}

function Cam({ name, v, vref, onTime, onEnded, rate }) {
  // each camera is a window [from,to] inside one long mp4; seek there and stop at `to`
  return (
    <div style={{ minWidth: 0 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>{name}</div>
      <video
        ref={vref} src={v.url} muted playsInline preload="auto" style={{ width: '100%', background: '#000', borderRadius: 8, aspectRatio: '1/1' }}
        onLoadedMetadata={(e) => { e.currentTarget.currentTime = v.from }}
        onTimeUpdate={(e) => onTime(name, e.currentTarget.currentTime - v.from)}
        onEnded={onEnded}
      />
    </div>
  )
}

export default function EpisodeViewer({ dataset, episode }) {
  const det = useApi(`/datasets/${dataset}/episodes/${episode}`)
  const ser = useApi(`/datasets/${dataset}/episodes/${episode}/series`)
  const refs = React.useRef({})
  const [t, setT] = React.useState(0)
  const [playing, setPlaying] = React.useState(false)
  const [rate, setRate] = React.useState(1)
  const [group, setGroup] = React.useState(0)
  const [kind, setKind] = React.useState('action')

  React.useEffect(() => { setT(0); setPlaying(false) }, [dataset, episode])

  const cams = det.data ? Object.entries(det.data.videos).filter(([, v]) => v.available) : []
  const dur = det.data ? det.data.length / det.data.fps : 0
  const all = () => Object.values(refs.current).filter(Boolean)
  const seek = (s) => {
    const c = det.data.videos
    Object.entries(refs.current).forEach(([n, el]) => { if (el && c[n]) el.currentTime = c[n].from + s })
    setT(s)
  }
  const play = () => {
    if (t >= dur - 0.05) seek(0)
    all().forEach((el) => { el.playbackRate = rate; el.play().catch(() => {}) })
    setPlaying(true)
  }
  const pause = () => { all().forEach((el) => el.pause()); setPlaying(false) }
  const onTime = (name, s) => {
    if (cams[0] && name === cams[0][0]) {
      if (s >= dur - 0.02) { pause(); setT(dur); return }
      setT(Math.max(0, s))
    }
  }
  React.useEffect(() => { all().forEach((el) => { el.playbackRate = rate }) }, [rate])

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <Load state={det}>{(d) => {
        const names = d.feature_names[kind] || []
        const groups = groupsFor(names)
        const g = groups[Math.min(group, groups.length - 1)]
        const chart = ser.data ? ser.data.frames.map((f, i) => {
          const row = { time: +(f / d.fps).toFixed(2) }
          if (g) g[1].forEach((ch) => { row[names[ch] || `ch${ch}`] = ser.data[kind][i][ch] })
          return row
        }) : []
        return (
          <>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <b>Episode {d.episode}</b><Badge>{d.length} frame · {d.seconds}s · {d.fps} fps</Badge>
              <span>{d.task}</span>
            </div>
            {cams.length === 0 && (
              <div className="note" style={{ marginTop: 12 }}>
                Máy này không có file video của dataset (chỉ có metadata), nên không phát được. Chép thư mục <code>videos/</code> của dataset vào project để xem.
              </div>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(cams.length, 1)}, minmax(0,1fr))`, gap: 10, marginTop: 12 }}>
              {cams.map(([n, v]) => <Cam key={n} name={n} v={v} vref={(el) => { refs.current[n] = el }} onTime={onTime} onEnded={pause} rate={rate} />)}
            </div>
            {cams.length > 0 && <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 12 }}>
              <button className="btn sm" onClick={playing ? pause : play} style={{ width: 70 }}>{playing ? 'Dừng' : 'Phát'}</button>
              <input type="range" min={0} max={dur} step={0.04} value={t} onChange={(e) => { pause(); seek(Number(e.target.value)) }} style={{ flex: 1 }} />
              <span className="mono small" style={{ width: 90, textAlign: 'right' }}>{t.toFixed(1)} / {dur.toFixed(1)} s</span>
              <select value={rate} onChange={(e) => setRate(Number(e.target.value))}>
                {[0.5, 1, 2, 4].map((r) => <option key={r} value={r}>{r}×</option>)}
              </select>
            </div>}

            <div style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '16px 0 6px', flexWrap: 'wrap' }}>
              <b className="small">Tín hiệu</b>
              {['action', 'state'].map((k) => <button key={k} className={'btn sm' + (kind === k ? ' primary' : '')} onClick={() => { setKind(k); setGroup(0) }}>{k}</button>)}
              <span className="muted small">·</span>
              {groups.map(([label], i) => <button key={label} className={'btn sm' + (i === group ? ' primary' : '')} onClick={() => setGroup(i)}>{label}</button>)}
            </div>
            <div style={{ height: 240 }}>
              {ser.data ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chart} margin={{ left: 0, right: 12, top: 6, bottom: 0 }}>
                    <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                    <XAxis dataKey="time" type="number" domain={[0, dur]} tick={{ fill: 'var(--muted)', fontSize: 12 }} tickFormatter={(v) => v.toFixed(0) + 's'} />
                    <YAxis tick={{ fill: 'var(--muted)', fontSize: 12 }} width={46} />
                    <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--line)' }} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)} />
                    {g && g[1].map((ch, i) => <Line key={ch} type="monotone" dataKey={names[ch] || `ch${ch}`} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.6} isAnimationActive={false} />)}
                    <ReferenceLine x={+t.toFixed(2)} stroke="var(--ink)" strokeDasharray="4 3" />
                  </LineChart>
                </ResponsiveContainer>
              ) : ser.error ? <div className="note">Không có dữ liệu state/action của dataset này trên máy này (chỉ có metadata).</div> : <p className="muted">Đang tải tín hiệu…</p>}
            </div>
            <p className="muted small" style={{ margin: '4px 0 0' }}>
              Mỗi camera là một đoạn trong file mp4 dài của dataset (LeRobot v3): trình phát tua tới đoạn của episode này và dừng ở cuối.
              Đường đứt đứng trên biểu đồ là thời điểm đang xem.
            </p>
          </>
        )
      }}</Load>
    </div>
  )
}
