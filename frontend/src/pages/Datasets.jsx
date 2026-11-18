import React from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { bytes, num, useApi } from '../api.js'
import { Badge, KV, Load, ModelBadge, Page, Status, Table } from '../ui.jsx'
import EpisodeViewer from './EpisodeViewer.jsx'

function Episodes({ dataset }) {
  const [q, setQ] = React.useState('')
  const [page, setPage] = React.useState(0)
  const [sel, setSel] = React.useState(null)
  const size = 15
  const st = useApi(`/datasets/${dataset}/episodes?offset=${page * size}&limit=${size}${q ? `&q=${encodeURIComponent(q)}` : ''}`)
  React.useEffect(() => { setPage(0) }, [q])
  return (
    <>
      <h2>Episodes {st.data && <span className="muted small">({num(st.data.total)})</span>}</h2>
      <div className="filters">
        <input placeholder="Lọc theo câu lệnh hoặc số episode (vd: blue cylinder, 301)" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 340 }} />
        <button className="btn sm" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button>
        <span className="small muted">trang {page + 1}{st.data ? ` / ${Math.max(1, Math.ceil(st.data.total / size))}` : ''}</span>
        <button className="btn sm" disabled={!st.data || (page + 1) * size >= st.data.total} onClick={() => setPage(page + 1)}>›</button>
      </div>
      <Load state={st}>{(d) => (
        <Table empty="Không có episode khớp" rows={d.episodes.map((e) => ({ ...e, id: e.episode }))} onRow={(r) => setSel(r.episode)} cols={[
          { h: 'Ep', n: true, r: (e) => (sel === e.episode ? <b>{e.episode} ▶</b> : e.episode) },
          { h: 'Câu lệnh', r: (e) => e.task },
          { h: 'Frame', n: true, r: (e) => e.length },
          { h: 'Giây', n: true, r: (e) => e.seconds },
        ]} />
      )}</Load>
      {sel != null && <EpisodeViewer dataset={dataset} episode={sel} />}
    </>
  )
}

export function Datasets() {
  const [robot, setRobot] = React.useState('')
  const [showOld, setShowOld] = React.useState(true)
  const robots = useApi('/robots')
  const st = useApi(`/datasets?include_old=${showOld}${robot ? `&robot=${robot}` : ''}`)
  const nav = useNavigate()
  return (
    <Page title="Datasets" sub="Bộ dữ liệu LeRobot trong datasets/lerobot (số episode/frame đọc từ meta/info.json).">
      <div className="filters">
        <select value={robot} onChange={(e) => setRobot(e.target.value)}>
          <option value="">Mọi robot</option>
          {(robots.data || []).map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
        </select>
        <label className="small"><input type="checkbox" checked={showOld} onChange={(e) => setShowOld(e.target.checked)} /> hiện cả bản cũ/đã bỏ</label>
      </div>
      <Load state={st}>{(rows) => (
        <Table rows={rows} onRow={(r) => nav(`/datasets/${r.id}`)} cols={[
          { h: 'Dataset', r: (d) => <code>{d.id}</code> },
          { h: 'Robot', r: (d) => d.robot_id || <span className="muted">?</span> },
          { h: 'Task', r: (d) => d.task_id || '—' },
          { h: 'Trạng thái', r: (d) => <Status s={d.status} /> },
          { h: 'Episode', n: true, r: (d) => num(d.episodes) },
          { h: 'Frame', n: true, r: (d) => num(d.frames) },
          { h: 'State/Action', n: true, r: (d) => `${d.state_dim}/${d.action_dim}` },
          { h: 'Camera', r: (d) => (d.cameras || []).join(', ') },
          { h: 'Dung lượng', n: true, r: (d) => bytes(d.size_bytes) },
        ]} />
      )}</Load>
    </Page>
  )
}

function Dist({ title, data }) {
  const entries = Object.entries(data).filter(([, v]) => typeof v === 'number')
  if (!entries.length) return null
  const max = Math.max(...entries.map(([, v]) => v))
  return (
    <div className="panel" style={{ minWidth: 0 }}>
      <b className="small">{title}</b>
      {entries.map(([k, v]) => (
        <div key={k} style={{ margin: '6px 0' }}>
          <div className="small" style={{ display: 'flex', justifyContent: 'space-between' }}><span>{k}</span><span className="muted">{v}</span></div>
          <div className="bar"><i style={{ width: `${(v / max) * 100}%` }} /></div>
        </div>
      ))}
    </div>
  )
}

export function DatasetDetail() {
  const { id } = useParams()
  const st = useApi(`/datasets/${id}`)
  return (
    <Load state={st}>{(d) => {
      const s = d.stats || {}
      const scalars = Object.entries(s).filter(([, v]) => typeof v !== 'object')
      const dists = Object.entries(s).filter(([, v]) => v && typeof v === 'object' && !Array.isArray(v))
      return (
        <Page crumb={<Link to="/datasets">Datasets</Link>} title={d.id} sub={d.note}>
          <div className="panel">
            <KV items={[
              ['Trạng thái', <Status s={d.status} />],
              ['Robot', d.robot_id ? <Link to={`/robots/${d.robot_id}`}>{d.robot_id}</Link> : null],
              ['Task', d.task_id], ['robot_type', <code>{d.robot_type}</code>],
              ['Episode / Frame', `${num(d.episodes)} / ${num(d.frames)} (${d.fps} fps)`],
              ['Số câu lệnh khác nhau', d.n_tasks],
              ['State / Action', `${d.state_dim}-D / ${d.action_dim}-D`],
              ['Camera', (d.cameras || []).join(', ')],
              ['Dung lượng', `${bytes(d.size_bytes)} (${d.video_files} file video)`],
              ['Đường dẫn', <code>{d.path}</code>],
              ['Tên chiều state', d.feature_names?.state && <span className="small mono">{d.feature_names.state.join(', ')}</span>],
            ]} />
          </div>
          <Episodes dataset={d.id} />
          {d.trained_runs.length > 0 && (
            <>
              <h2>Đã dùng để train</h2>
              <Table rows={d.trained_runs} cols={[
                { h: 'Run', r: (r) => <Link to={`/runs/${r.id}`}><code>{r.id}</code></Link> },
                { h: 'Loại', r: (r) => <ModelBadge t={r.model_type} /> },
                { h: 'Step', n: true, r: (r) => num(r.steps) }]} />
            </>
          )}
          {d.stats && (
            <>
              <h2>Thống kê độ phủ <span className="muted small">(results/dataset_stats.json)</span></h2>
              {scalars.length > 0 && <div className="panel"><KV items={scalars.map(([k, v]) => [k, String(v)])} /></div>}
              <div className="cards" style={{ marginTop: 12, gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
                {dists.map(([k, v]) => <Dist key={k} title={k} data={v} />)}
              </div>
              <p className="muted small">Các nhóm có giá trị lồng nhau (ví dụ độ dài episode) xem trực tiếp trong file JSON.</p>
            </>
          )}
          {!d.stats && <p className="muted small" style={{ marginTop: 16 }}>Dataset này chưa có thống kê độ phủ trong results/dataset_stats.json.</p>}
        </Page>
      )
    }}</Load>
  )
}
