import React from 'react'
import { Link } from 'react-router-dom'
import { api, bytes, ci, num, pct, useApi } from '../api.js'
import { Badge, Load, Page, Table } from '../ui.jsx'

export default function Dashboard() {
  const st = useApi('/overview')
  const [busy, setBusy] = React.useState(false)
  const reindex = async () => {
    setBusy(true)
    try { await api('/reindex', { method: 'POST' }); st.reload() } finally { setBusy(false) }
  }
  return (
    <Page title="Tổng quan" sub="Trạng thái dự án đọc từ datasets/, outputs/, results/ và design/."
      actions={<button className="btn" onClick={reindex} disabled={busy}>{busy ? 'Đang quét…' : 'Quét lại dữ liệu'}</button>}>
      <Load state={st}>{(d) => (
        <>
          <div className="cards">
            {[['robot', 'Robots', '/robots'], ['task', 'Tasks', '/tasks'], ['dataset', 'Datasets', '/datasets'],
              ['train_run', 'Lần train', '/runs'], ['model', 'Models (checkpoint)', '/models'], ['eval_run', 'Kết quả eval', '/evals']]
              .map(([k, label, to]) => (
                <Link key={k} to={to} className="card stat" style={{ color: 'inherit', textDecoration: 'none' }}>
                  <b>{num(d.counts[k])}</b><span>{label}</span>
                </Link>
              ))}
            <div className="card stat"><b>{bytes(d.dataset_bytes)}</b><span>Dữ liệu LeRobot</span></div>
          </div>
          <p className="muted small">Quét lần cuối: {d.indexed_at} · gốc: <code>{d.root}</code></p>

          <h2>Eval gần nhất</h2>
          <Table rows={d.latest_evals} cols={[
            { h: 'Tag', r: (r) => <code>{r.tag}</code> },
            { h: 'Loại', r: (r) => <Badge>{r.kind}</Badge> },
            { h: 'Model', r: (r) => r.model_id || '—' },
            { h: 'Điều kiện', r: (r) => (r.level ? `L${r.level} · ${r.split}` : '—') },
            { h: 'Thành công', n: true, r: (r) => `${r.k}/${r.n} = ${pct(r.rate)}` },
            { h: 'Wilson 95%', n: true, r: (r) => ci(r.lo, r.hi) },
          ]} />

          <h2>Robot đang gán</h2>
          {d.assignments.length === 0
            ? <div className="panel muted">Chưa gán dataset/model nào cho robot. Vào trang <Link to="/robots">Robots</Link> để gán.</div>
            : <Table rows={d.assignments} cols={[
              { h: 'Robot', r: (r) => <Link to={`/robots/${r.robot_id}`}>{r.robot_id}</Link> },
              { h: 'Loại', r: (r) => r.slot },
              { h: 'Đang dùng', r: (r) => <code>{r.ref_id}</code> }]} />}
        </>
      )}</Load>
    </Page>
  )
}
