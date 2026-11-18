import React from 'react'
import { Link } from 'react-router-dom'
import { api, num, useApi } from '../api.js'
import { Badge, KV, Load, Page, Table } from '../ui.jsx'

const STAT = {
  queued: ['', 'chờ'], running: ['info', 'đang chạy'], succeeded: ['ok', 'xong'], failed: ['bad', 'lỗi'],
  cancelled: ['warn', 'đã huỷ'], interrupted: ['warn', 'bị ngắt'],
}
const JobStatus = ({ s }) => <Badge kind={(STAT[s] || [''])[0]}>{(STAT[s] || ['', s])[1]}</Badge>

function stamp() {
  const d = new Date(); const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`
}

function NewTrain({ onStarted }) {
  const ov = useApi('/overview')
  const canTrain = !ov.data || ov.data.can_train
  const ds = useApi('/datasets?include_old=false')
  const [f, setF] = React.useState({ dataset_id: '', model_type: 'act', steps: 100, batch_size: 8, run_name: '' })
  const [err, setErr] = React.useState(null)
  const [busy, setBusy] = React.useState(false)
  const rows = (ds.data || []).filter((d) => d.robot_id)
  React.useEffect(() => { if (!f.dataset_id && rows.length) setF((x) => ({ ...x, dataset_id: rows.find((d) => d.id === 'yellowarm_pallet_L1')?.id || rows[0].id })) }, [rows.length])
  const name = f.run_name || `studio_${f.model_type}_${f.dataset_id.replace('yellowarm_', '')}_${stamp()}`
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === 'number' ? Number(e.target.value) : e.target.value })
  const go = async () => {
    setErr(null); setBusy(true)
    try {
      const r = await api('/jobs/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...f, run_name: name }) })
      onStarted(r.id)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="panel">
      <b>Chạy train mới</b>
      <div className="filters" style={{ marginTop: 10 }}>
        <select value={f.dataset_id} onChange={set('dataset_id')}>
          {rows.map((d) => <option key={d.id} value={d.id}>{d.id} ({num(d.episodes)} ep · {d.state_dim}-D)</option>)}
        </select>
        <select value={f.model_type} onChange={set('model_type')}><option value="act">ACT (từ đầu)</option><option value="smolvla">SmolVLA (từ base)</option></select>
        <label className="small">steps <input type="number" value={f.steps} min={1} onChange={set('steps')} style={{ width: 90 }} /></label>
        <label className="small">batch <input type="number" value={f.batch_size} min={1} max={64} onChange={set('batch_size')} style={{ width: 64 }} /></label>
        <input value={f.run_name} placeholder={name} onChange={set('run_name')} style={{ minWidth: 300 }} />
        <button className="btn primary" disabled={busy || !f.dataset_id || !canTrain} onClick={go}>{busy ? 'Đang gửi…' : 'Chạy train'}</button>
      </div>
      <p className="muted small" style={{ margin: 0 }}>
        Job chạy bằng <code>env\python.exe -m lerobot.scripts.lerobot_train</code>, ghi vào <code>outputs/&lt;tên run&gt;</code> (không ghi đè run cũ).
        Mỗi lần chỉ một job GPU; GPU phải còn ≥ 4 GiB trống.
      </p>
      {!canTrain && <div className="note">Máy này chưa có môi trường train (thiếu <code>env\python.exe</code>), nên app chỉ để xem dữ liệu. Cài môi trường train trong thư mục project để chạy job.</div>}
      {err && <div className="note">{err}</div>}
    </div>
  )
}

function JobPanel({ id, onChanged }) {
  const st = useApi(`/jobs/${id}`)
  const [lines, setLines] = React.useState([])
  const live = st.data && (st.data.status === 'running' || st.data.status === 'queued')
  React.useEffect(() => {
    let stop = false
    const tick = async () => {
      try { const l = await api(`/jobs/${id}/log?tail=60`); if (!stop) setLines(l.lines) } catch { /* ignore */ }
      if (!stop) st.reload()
    }
    tick()
    const t = setInterval(tick, live ? 2500 : 60000)
    return () => { stop = true; clearInterval(t) }
  }, [id, live])
  const cancel = async () => { await api(`/jobs/${id}/cancel`, { method: 'POST' }); st.reload(); onChanged() }
  if (!st.data) return <p className="muted">Đang tải job #{id}…</p>
  const j = st.data
  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <b>Job #{j.id} · {j.name}</b> <JobStatus s={j.status} />
        <span style={{ flex: 1 }} />
        {(j.status === 'running' || j.status === 'queued') && <button className="btn sm" onClick={cancel}>Huỷ</button>}
      </div>
      <KV items={[
        ['Tạo / bắt đầu / kết thúc', `${j.created || '—'} / ${j.started || '—'} / ${j.ended || '—'}`],
        ['Exit code', j.exit_code],
        ['Kết quả', j.result && (j.result.error ? <span className="err">{j.result.error}</span>
          : <>{j.result.finished ? 'train xong' : 'chưa đủ step'} · checkpoint: {(j.result.checkpoints || []).join(', ') || 'không có'} · loss cuối {j.result.final_loss ?? '—'} · <Link to={`/runs/${j.params.run}`}>mở run</Link></>)],
        ['Log', j.log && <code>{j.log}</code>],
      ]} />
      <pre style={{ background: 'var(--bg)', border: '1px solid var(--line)', borderRadius: 8, padding: 10, maxHeight: 320, overflow: 'auto', fontSize: 12, margin: '10px 0 0', whiteSpace: 'pre-wrap' }}>
        {lines.length ? lines.join('\n') : '(chưa có log)'}
      </pre>
    </div>
  )
}

export default function Jobs() {
  const st = useApi('/jobs')
  const [sel, setSel] = React.useState(null)
  React.useEffect(() => {
    const busy = st.data && st.data.some((j) => j.status === 'running' || j.status === 'queued')
    const t = setInterval(st.reload, busy ? 3000 : 20000)
    return () => clearInterval(t)
  }, [st.data && st.data.length, st.data && st.data.map((j) => j.status).join()])
  return (
    <Page title="Jobs" sub="Chạy và theo dõi các job training. Hiện hỗ trợ job train (ACT, SmolVLA).">
      <NewTrain onStarted={(id) => { setSel(id); st.reload() }} />
      {sel && <JobPanel id={sel} onChanged={st.reload} />}
      <h2>Lịch sử job</h2>
      <Load state={st}>{(rows) => (
        <Table empty="Chưa có job nào" rows={rows} onRow={(r) => setSel(r.id)} cols={[
          { h: '#', n: true, r: (r) => r.id },
          { h: 'Run', r: (r) => <code>{r.name}</code> },
          { h: 'Loại', r: (r) => r.type },
          { h: 'Trạng thái', r: (r) => <JobStatus s={r.status} /> },
          { h: 'Dataset', r: (r) => <code>{r.params.dataset}</code> },
          { h: 'Model/Step', r: (r) => `${r.params.model} · ${num(r.params.steps)}` },
          { h: 'Bắt đầu', r: (r) => r.started || r.created },
          { h: 'Exit', n: true, r: (r) => r.exit_code ?? '—' },
        ]} />
      )}</Load>
    </Page>
  )
}
