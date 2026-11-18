import React from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ci, pct, useApi } from '../api.js'
import { Badge, Load, Page, Table } from '../ui.jsx'

const KINDS = [
  ['eval', 'Eval chính thức'], ['validation', 'Validation (chọn exec-steps)'], ['demo', 'Demo'], ['expert', 'Expert'], ['smoke', 'Smoke/kiểm tra'],
]
const PALETTE = ['var(--c-act)', 'var(--c-smolvla)', 'var(--c-groot)', 'var(--c-other)', '#0f766e', '#be185d']

function CompareChart({ rows, level }) {
  const sel = rows.filter((r) => r.level === level)
  const models = [...new Set(sel.map((r) => r.model_id))]
  const splits = [...new Set(sel.map((r) => r.split))]
  const data = splits.map((s) => {
    const o = { split: s }
    sel.filter((r) => r.split === s).forEach((r) => {
      o[r.model_id] = r.rate
      o[r.model_id + '__err'] = [r.rate - r.lo, r.hi - r.rate]
    })
    return o
  })
  if (!data.length) return null
  return (
    <div className="panel" style={{ height: 340 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: 0, right: 12, top: 10, bottom: 0 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="split" tick={{ fill: 'var(--muted)', fontSize: 12 }} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} tick={{ fill: 'var(--muted)', fontSize: 12 }} width={44} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? pct(v, 1) : v)} contentStyle={{ background: 'var(--panel)', border: '1px solid var(--line)' }} />
          <Legend />
          {models.map((m, i) => (
            <Bar key={m} dataKey={m} fill={PALETTE[i % PALETTE.length]} isAnimationActive={false} radius={[3, 3, 0, 0]}>
              <ErrorBar dataKey={m + '__err'} width={5} strokeWidth={1.5} stroke="var(--ink)" />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function Compare({ task }) {
  const [level, setLevel] = React.useState(1)
  const st = useApi(`/eval/compare?task=${task}&kind=eval`)
  return (
    <Load state={st}>{(d) => {
      const levels = [...new Set(d.rows.map((r) => r.level).filter((x) => x != null))].sort()
      const lv = levels.includes(level) ? level : levels[0]
      const rows = d.rows.filter((r) => r.level === lv)
      const ref = d.reference_conditions[String(lv)]
      const bad = rows.filter((r) => !r.comparable)
      return (
        <>
          <div className="filters">
            {levels.map((l) => <button key={l} className={'btn sm' + (l === lv ? ' primary' : '')} onClick={() => setLevel(l)}>Level {l}</button>)}
          </div>
          {ref && (
            <div className={'note ' + (bad.length ? '' : 'ok')}>
              Điều kiện so sánh: seed test <b>{ref.seeds[0]}–{ref.seeds[1]}</b>, episode_seconds <b>{ref.episode_seconds}</b>, level {lv}.
              {bad.length ? ` ${bad.length} dòng dưới đây khác điều kiện nên KHÔNG so sánh được với phần còn lại.` : ' Mọi dòng dưới đây cùng điều kiện, so sánh được.'}
              {' '}Lưu ý: so sánh model chỉ hợp lệ khi cùng dataset; hiện tại tất cả model của task này train trên cùng <code>yellowarm_pallet_all</code>.
            </div>
          )}
          <CompareChart rows={d.rows} level={lv} />
          <p className="muted small">Cột = tỉ lệ thành công; thanh sai số = khoảng tin cậy Wilson 95% (đọc từ results/*.json).</p>
          <EvalTable rows={rows} showCompare />
        </>
      )
    }}</Load>
  )
}

function EvalTable({ rows, showCompare }) {
  return (
    <Table empty="Không có kết quả" rows={rows} cols={[
      { h: 'Tag', r: (r) => <code>{r.tag}</code> },
      { h: 'Model', r: (r) => (r.run_id ? <Link to={`/runs/${r.run_id}`}>{r.model_id}</Link> : <span className="muted">—</span>) },
      { h: 'Điều kiện', r: (r) => (r.level ? `L${r.level} · ${r.split}` : r.split || '—') },
      { h: 'exec', n: true, r: (r) => r.exec_steps ?? '—' },
      { h: 'Seed', n: true, r: (r) => (r.seed_first != null ? `${r.seed_first}–${r.seed_last}` : '—') },
      { h: 'ep. giây', n: true, r: (r) => r.episode_seconds ?? '—' },
      { h: 'Thành công', n: true, r: (r) => <b>{r.k}/{r.n} = {pct(r.rate)}</b> },
      { h: 'Wilson 95%', n: true, r: (r) => ci(r.lo, r.hi) },
      { h: 'Infer (ms)', n: true, r: (r) => (r.infer_ms != null ? r.infer_ms.toFixed(1) : '—') },
      { h: '', r: (r) => (
        <>
          {showCompare && (r.comparable ? <Badge kind="ok">so sánh được</Badge> : <Badge kind="warn">khác điều kiện</Badge>)}
          {!r.complete && <Badge kind="bad">chưa đủ</Badge>}
          {r.wilson_ok === 0 && <Badge kind="bad">Wilson lệch</Badge>}
        </>) },
    ]} />
  )
}

export default function Evals() {
  const [kind, setKind] = React.useState('eval')
  const [task, setTask] = React.useState('humanoid_pallet')
  const tasks = useApi('/tasks')
  const list = useApi(kind === 'eval' && task === 'humanoid_pallet' ? '/health' : `/evals?kind=${kind}&task=${task}`)
  return (
    <Page title="Eval" sub="Kết quả đọc từ results/*.json do eval_harness.py ghi. Con số không được chép tay hay tính lại.">
      <div className="filters">
        {KINDS.map(([k, label]) => <button key={k} className={'btn sm' + (kind === k ? ' primary' : '')} onClick={() => setKind(k)}>{label}</button>)}
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          {(tasks.data || []).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>
      {kind === 'eval' && task === 'humanoid_pallet'
        ? <Compare task={task} />
        : <Load state={list}>{(rows) => <EvalTable rows={Array.isArray(rows) ? rows : []} />}</Load>}
    </Page>
  )
}
