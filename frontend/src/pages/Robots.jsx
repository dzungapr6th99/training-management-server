import React from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, num, useApi } from '../api.js'
import { Confirm } from '../forms.jsx'
import { Badge, KV, Load, ModelBadge, Page, Status, Table } from '../ui.jsx'
import RobotForm from './RobotForm.jsx'

/** URDF / USD are required, the USD with ROS is optional: show each as a chip. */
function AssetChips({ r }) {
  const a = r.assets
  return (
    <div className="chip-row">
      <Badge kind={a.urdf ? 'ok' : 'bad'}>URDF{a.urdf ? '' : ' thiếu'}</Badge>
      <Badge kind={a.usd ? 'ok' : 'bad'}>USD{a.usd ? '' : ' thiếu'}</Badge>
      <Badge kind={a.usd_ros ? 'ok' : ''}>USD+ROS{a.usd_ros ? '' : ' —'}</Badge>
    </div>
  )
}

export function Robots() {
  const [showHidden, setShowHidden] = React.useState(false)
  const st = useApi(`/robots?include_hidden=${showHidden}`)
  const nav = useNavigate()
  const [adding, setAdding] = React.useState(false)
  const restore = async (id) => { await api(`/robots/${id}/restore`, { method: 'POST' }); st.reload() }
  return (
    <Page title="Robots" sub="Mỗi robot cần URDF và USD cho Isaac Sim (USD có ROS 2 là tùy chọn), cùng các dataset và model tương thích."
      actions={<button className="btn primary" onClick={() => setAdding(true)}>+ Thêm robot</button>}>
      <div className="filters">
        <label className="small"><input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> hiện cả robot đã xóa</label>
      </div>
      <Load state={st}>{(rows) => (
        <Table rows={rows} onRow={(r) => nav(`/robots/${r.id}`)} cols={[
          { h: 'Robot', r: (r) => <><b>{r.name}</b> {r.hidden ? <Badge kind="warn">đã xóa</Badge> : null}{r.source === 'user' && <Badge kind="info">tự tạo</Badge>}<div className="muted small"><code>{r.id}</code></div></> },
          { h: 'Trạng thái', r: (r) => <Status s={r.status} /> },
          { h: 'State / Action', r: (r) => (r.state_dim ? `${r.state_dim} / ${r.action_dim}` : '—') },
          { h: 'Camera', r: (r) => (r.cameras.length ? r.cameras.join(', ') : '—') },
          { h: 'Mô hình Isaac Sim', r: (r) => <AssetChips r={r} /> },
          { h: 'Datasets', n: true, r: (r) => r.n_datasets },
          { h: 'Models', n: true, r: (r) => r.n_models },
          { h: 'Đang gán', r: (r) => (Object.keys(r.assignment).length
            ? Object.entries(r.assignment).map(([k, v]) => <div key={k} className="small"><code>{k}: {v}</code></div>)
            : <span className="muted">—</span>) },
          { h: '', r: (r) => (r.hidden ? <button className="btn sm" onClick={(e) => { e.stopPropagation(); restore(r.id) }}>Khôi phục</button> : null) },
        ]} />
      )}</Load>
      {adding && <RobotForm onClose={() => setAdding(false)} onSaved={(id) => nav(`/robots/${id}`)} />}
    </Page>
  )
}

function jointSummary(emb) {
  const jg = emb && emb.joint_groups
  if (!jg || typeof jg !== 'object') return null
  return Object.entries(jg)
    .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.length : typeof v === 'object' && v ? Object.keys(v).length : v}`)
    .join(' · ')
}

export function RobotDetail() {
  const { id } = useParams()
  const nav = useNavigate()
  const st = useApi(`/robots/${id}`)
  const [msg, setMsg] = React.useState(null)
  const [editing, setEditing] = React.useState(false)
  const [deleting, setDeleting] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [delErr, setDelErr] = React.useState(null)
  const assign = async (slot, ref_id) => {
    setMsg(null)
    try {
      await api(`/robots/${id}/assign`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ slot, ref_id }),
      })
      setMsg({ ok: true, text: ref_id ? `Đã gán ${slot} ${ref_id}` : `Đã bỏ gán ${slot}` })
      st.reload()
    } catch (e) {
      setMsg({ ok: false, text: e.message, problems: e.problems })
    }
  }
  const remove = async () => {
    setBusy(true); setDelErr(null)
    try { await api(`/robots/${id}`, { method: 'DELETE' }); nav('/robots') } catch (e) { setDelErr(e.message) } finally { setBusy(false) }
  }
  return (
    <Load state={st}>{({ robot: r, configured, datasets, models, tasks, assignment, history }) => (
      <Page crumb={<Link to="/robots">Robots</Link>} title={r.name} sub={r.description}
        actions={r.hidden ? <Badge kind="warn">đã xóa</Badge> : (
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={() => setEditing(true)}>Sửa</button>
            <button className="btn" style={{ color: 'var(--bad)' }} onClick={() => setDeleting(true)}>Xóa</button>
          </div>)}>
        <div className="grid2">
          <div className="panel">
            <KV items={[
              ['ID', <code>{r.id}</code>],
              ['Trạng thái', <Status s={r.status} />],
              ['Nguồn', r.source === 'user' ? 'tự tạo trong studio' : r.source],
              ['State / Action', r.state_dim ? `${r.state_dim}-D / ${r.action_dim}-D` : null],
              ['Camera', r.cameras.length ? r.cameras.join(', ') : null],
              ['Thư mục design', r.design_dir && <code>{r.design_dir}</code>],
              ['Khớp (embodiment)', jointSummary(r.embodiment)],
              ['Ghi chú', r.notes],
            ]} />
          </div>
          <div className="panel">
            <b>Ba bản mô hình cho Isaac Sim</b>
            <KV items={[
              ['URDF', r.assets.urdf ? <code>{r.assets.urdf}</code> : <span className="err">thiếu (bắt buộc)</span>],
              ['USD', r.assets.usd ? <code>{r.assets.usd}</code> : <span className="err">thiếu (bắt buộc)</span>],
              ['USD có ROS', r.assets.usd_ros ? <code>{r.assets.usd_ros}</code> : <span className="muted">không có (tùy chọn)</span>],
            ]} />
            {!r.assets_complete && <div className="note">Robot này chưa đủ URDF và USD. Bấm Sửa để khai báo.</div>}
          </div>
        </div>

        <h2>Đang dùng</h2>
        <div className="panel">
          <KV items={[
            ['Dataset', assignment.dataset
              ? <><code>{assignment.dataset.ref_id}</code> <button className="btn sm" onClick={() => assign('dataset', null)}>bỏ gán</button></>
              : <span className="muted">chưa gán</span>],
            ['Model', assignment.model
              ? <><Link to={`/model?id=${encodeURIComponent(assignment.model.ref_id)}`}><code>{assignment.model.ref_id}</code></Link> <button className="btn sm" onClick={() => assign('model', null)}>bỏ gán</button></>
              : <span className="muted">chưa gán</span>],
          ]} />
          <p className="muted small" style={{ marginBottom: 0 }}>
            Gán chỉ ghi vào database của studio, chưa triển khai gì lên robot. Server từ chối gán khi số chiều
            state/action hoặc camera không khớp.
          </p>
          {msg && (
            <div className={'note ' + (msg.ok ? 'ok' : '')}>
              {msg.text}
              {msg.problems && <ul style={{ margin: '4px 0 0' }}>{msg.problems.map((p) => <li key={p}>{p}</li>)}</ul>}
            </div>
          )}
        </div>

        <h2>Tasks ({tasks.length})</h2>
        {tasks.length
          ? <Table rows={tasks} cols={[
            { h: 'Task', r: (t) => <><b>{t.name}</b> <Status s={t.status} /></> },
            { h: 'Script sim', r: (t) => <code>{t.sim_script}</code> }]} />
          : <div className="panel muted">Chưa có task.</div>}

        <h2>Datasets ({datasets.length})</h2>
        <Table empty="Chưa có dataset cho robot này" rows={datasets} cols={[
          { h: 'Dataset', r: (d) => <Link to={`/datasets/${d.id}`}><code>{d.id}</code></Link> },
          { h: 'Episode', n: true, r: (d) => num(d.episodes) },
          { h: 'Frame', n: true, r: (d) => num(d.frames) },
          { h: 'Trạng thái', r: (d) => <Status s={d.status} /> },
          { h: 'Tương thích', r: (d) => (d.compat.length ? <span className="err small">{d.compat.join('; ')}</span> : <Badge kind="ok">khớp</Badge>) },
          { h: '', r: (d) => (
            <button className="btn sm" disabled={d.compat.length > 0 || assignment.dataset?.ref_id === d.id} onClick={() => assign('dataset', d.id)}>
              {assignment.dataset?.ref_id === d.id ? 'đang dùng' : 'Gán'}
            </button>) },
        ]} />

        <h2>Models ({models.length})</h2>
        <Table empty="Chưa có model cho robot này" rows={models} cols={[
          { h: 'Model', r: (m) => <Link to={`/model?id=${encodeURIComponent(m.id)}`}>{m.name ? <b>{m.name}</b> : <code>{m.id}</code>}{m.name && <div className="muted small"><code>{m.id}</code></div>}</Link> },
          { h: 'Loại', r: (m) => <ModelBadge t={m.model_type} /> },
          { h: 'Dataset', r: (m) => <code>{m.dataset_id}</code> },
          { h: 'Bản', r: (m) => <Status s={m.status} /> },
          { h: 'Tương thích', r: (m) => (m.compat.length ? <span className="err small">{m.compat.join('; ')}</span> : <Badge kind="ok">khớp</Badge>) },
          { h: '', r: (m) => (
            <button className="btn sm" disabled={m.compat.length > 0 || assignment.model?.ref_id === m.id} onClick={() => assign('model', m.id)}>
              {assignment.model?.ref_id === m.id ? 'đang dùng' : 'Chọn'}
            </button>) },
        ]} />

        {history.length > 0 && (
          <>
            <h2>Lịch sử gán</h2>
            <Table rows={history} cols={[
              { h: 'Lúc', r: (h) => h.at },
              { h: 'Loại', r: (h) => h.slot },
              { h: 'Giá trị', r: (h) => (h.ref_id ? <code>{h.ref_id}</code> : <span className="muted">bỏ gán</span>) }]} />
          </>
        )}

        {editing && <RobotForm robot={r} configured={configured} onClose={() => setEditing(false)} onSaved={() => { setEditing(false); st.reload() }} />}
        {deleting && (
          <Confirm title={`Xóa robot "${r.name}"?`} danger confirmText="Xóa robot" busy={busy} error={delErr} onConfirm={remove} onClose={() => setDeleting(false)}>
            <p style={{ marginTop: 0 }}>
              {r.source === 'user' ? 'Robot tự tạo sẽ bị gỡ khỏi studio.' : 'Robot có sẵn sẽ được ẩn khỏi studio (có thể khôi phục ở trang Robots bằng "hiện cả robot đã xóa").'}
              {' '}Việc gán dataset/model của robot này sẽ bị bỏ.
            </p>
            <p className="muted small" style={{ marginBottom: 0 }}>
              File URDF/USD/CAD trong <code>design/</code>, dataset và model <b>không bị xóa</b>. Hiện có {datasets.length} dataset và {models.length} model liên kết với robot này.
            </p>
          </Confirm>
        )}
      </Page>
    )}</Load>
  )
}
