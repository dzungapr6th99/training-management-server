import React from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, bytes, ci, pct, useApi } from '../api.js'
import { Confirm, Field, Modal, wide } from '../forms.jsx'
import { Badge, KV, Load, ModelBadge, Page, Status, Table } from '../ui.jsx'

const STATUSES = [['final', 'cuối'], ['intermediate', 'trung gian'], ['custom', 'tự đăng ký'], ['archived', 'lưu trữ']]
const post = (path, body, method = 'POST') => api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
const label = (m) => m.name || m.id

/** Register a model folder (a LeRobot pretrained_model directory) or edit an existing entry. */
function ModelForm({ model, onClose, onSaved }) {
  const editing = !!model
  const robots = useApi('/robots')
  const datasets = useApi('/datasets')
  const [f, setF] = React.useState({
    path: model?.path || '', name: model?.name || '', notes: model?.notes || '', status: model?.status || 'custom',
    robot_id: model?.robot_id || '', dataset_id: model?.dataset_id || '',
  })
  const [err, setErr] = React.useState(null)
  const [busy, setBusy] = React.useState(false)
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })
  const save = async () => {
    setErr(null); setBusy(true)
    const body = { name: f.name, notes: f.notes || null, status: f.status, robot_id: f.robot_id || null, dataset_id: f.dataset_id || null }
    try {
      if (editing) { await post(`/model-edit?id=${encodeURIComponent(model.id)}`, body, 'PUT'); onSaved(model.id) }
      else { const r = await post('/models', { ...body, path: f.path }); onSaved(r.id) }
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  return (
    <Modal title={editing ? `Sửa model: ${label(model)}` : 'Đăng ký model'} onClose={onClose}>
      {!editing && (
        <Field label="Thư mục model" required hint="Thư mục có config.json của LeRobot, tính từ training-management-server/ (thường là outputs/<run>/checkpoints/<step>/pretrained_model).">
          <input value={f.path} onChange={set('path')} placeholder="outputs/my_run/checkpoints/010000/pretrained_model" style={wide} />
        </Field>
      )}
      <Field label="Tên hiển thị" required><input value={f.name} onChange={set('name')} style={wide} /></Field>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Field label="Trạng thái">
          <select value={f.status} onChange={set('status')} style={wide}>{STATUSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
        </Field>
        <Field label="Robot">
          <select value={f.robot_id} onChange={set('robot_id')} style={wide}>
            <option value="">(theo dataset)</option>
            {(robots.data || []).map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
        </Field>
        <Field label="Dataset đã train">
          <select value={f.dataset_id} onChange={set('dataset_id')} style={wide}>
            <option value="">—</option>
            {(datasets.data || []).map((d) => <option key={d.id} value={d.id}>{d.id}</option>)}
          </select>
        </Field>
      </div>
      <Field label="Ghi chú"><input value={f.notes} onChange={set('notes')} style={wide} /></Field>
      {err && <div className="note" style={{ borderColor: 'var(--bad)', background: 'var(--bad-soft)' }}>{err}</div>}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button className="btn" onClick={onClose}>Hủy</button>
        <button className="btn primary" disabled={busy || !f.name || (!editing && !f.path)} onClick={save}>{busy ? 'Đang lưu…' : editing ? 'Lưu' : 'Đăng ký'}</button>
      </div>
    </Modal>
  )
}

export default function Models() {
  const [onlyFinal, setOnlyFinal] = React.useState(true)
  const [showHidden, setShowHidden] = React.useState(false)
  const st = useApi(`/models?only_final=${onlyFinal}&include_hidden=${showHidden}`)
  const nav = useNavigate()
  const [adding, setAdding] = React.useState(false)
  const restore = async (id) => { await api(`/model-restore?id=${encodeURIComponent(id)}`, { method: 'POST' }); st.reload() }
  return (
    <Page title="Models" sub="Mỗi checkpoint trong outputs/ và các model bạn đăng ký thêm. Mở một model để sửa, xóa hoặc chọn cho robot."
      actions={<button className="btn primary" onClick={() => setAdding(true)}>+ Đăng ký model</button>}>
      <div className="filters">
        <label className="small"><input type="checkbox" checked={onlyFinal} onChange={(e) => setOnlyFinal(e.target.checked)} /> chỉ checkpoint cuối của mỗi run</label>
        <label className="small"><input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> hiện cả model đã xóa</label>
      </div>
      <Load state={st}>{(rows) => (
        <Table rows={rows} onRow={(m) => nav(`/model?id=${encodeURIComponent(m.id)}`)} cols={[
          { h: 'Model', r: (m) => <><b>{label(m)}</b> {m.hidden ? <Badge kind="warn">đã xóa</Badge> : null}{m.source === 'user' && <Badge kind="info">tự đăng ký</Badge>}{m.name && <div className="muted small"><code>{m.id}</code></div>}</> },
          { h: 'Loại', r: (m) => <ModelBadge t={m.model_type} /> },
          { h: 'Bản', r: (m) => <Status s={m.status} /> },
          { h: 'Dataset', r: (m) => (m.dataset_id ? <code>{m.dataset_id}</code> : '—') },
          { h: 'Robot', r: (m) => (m.robot_id ? m.robot_id : <span className="muted">?</span>) },
          { h: 'State/Action', n: true, r: (m) => `${m.state_dim ?? '—'}/${m.action_dim ?? '—'}` },
          { h: 'Dung lượng', n: true, r: (m) => bytes(m.size_bytes) },
          { h: 'Eval', n: true, r: (m) => m.n_evals },
          { h: 'Đang dùng cho', r: (m) => (m.assigned_to.length ? m.assigned_to.map((r) => <Badge key={r} kind="info">{r}</Badge>) : <span className="muted">—</span>) },
          { h: '', r: (m) => (m.hidden ? <button className="btn sm" onClick={(e) => { e.stopPropagation(); restore(m.id) }}>Khôi phục</button> : null) },
        ]} />
      )}</Load>
      {adding && <ModelForm onClose={() => setAdding(false)} onSaved={(id) => nav(`/model?id=${encodeURIComponent(id)}`)} />}
    </Page>
  )
}

export function ModelDetail() {
  const [sp] = useSearchParams()
  const id = sp.get('id')
  const nav = useNavigate()
  const st = useApi(`/model-detail?id=${encodeURIComponent(id)}`)
  const [editing, setEditing] = React.useState(false)
  const [deleting, setDeleting] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [err, setErr] = React.useState(null)
  const [msg, setMsg] = React.useState(null)
  const choose = async (robotId) => {
    setMsg(null)
    try { await post(`/robots/${robotId}/assign`, { slot: 'model', ref_id: id }); setMsg({ ok: true, text: `Đã chọn model này cho ${robotId}` }); st.reload() }
    catch (e) { setMsg({ ok: false, text: e.message, problems: e.problems }) }
  }
  const unchoose = async (robotId) => { await post(`/robots/${robotId}/assign`, { slot: 'model', ref_id: null }); st.reload() }
  const remove = async () => {
    setBusy(true); setErr(null)
    try { await api(`/model-edit?id=${encodeURIComponent(id)}`, { method: 'DELETE' }); nav('/models') } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  return (
    <Load state={st}>{({ model: m, evals, run, robots, assigned_to }) => (
      <Page crumb={<Link to="/models">Models</Link>} title={label(m)} sub={m.notes}
        actions={m.hidden ? <Badge kind="warn">đã xóa</Badge> : (
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={() => setEditing(true)}>Sửa</button>
            <button className="btn" style={{ color: 'var(--bad)' }} onClick={() => setDeleting(true)}>Xóa</button>
          </div>)}>
        <div className="panel">
          <KV items={[
            ['ID', <code>{m.id}</code>],
            ['Loại', <ModelBadge t={m.model_type} />], ['Bản', <Status s={m.status} />],
            ['Nguồn', m.source === 'user' ? 'tự đăng ký' : m.source === 'disk+user' ? 'checkpoint trên đĩa (đã chỉnh trong studio)' : 'checkpoint trên đĩa'],
            ['Run', run ? <Link to={`/runs/${run.id}`}>{run.id}</Link> : null],
            ['Dataset', m.dataset_id ? <Link to={`/datasets/${m.dataset_id}`}><code>{m.dataset_id}</code></Link> : null],
            ['State / Action', `${m.state_dim ?? '—'}-D / ${m.action_dim ?? '—'}-D`],
            ['Camera', (m.cameras || []).join(', ')],
            ['Chunk / n_action_steps', `${m.chunk_size ?? '—'} / ${m.n_action_steps ?? '—'}`],
            ['Dung lượng', bytes(m.size_bytes)],
            ['Đường dẫn', <code>{m.path}</code>],
          ]} />
        </div>

        <h2>Chọn model này cho robot</h2>
        <p className="muted small" style={{ marginTop: -4 }}>Chỉ robot có state/action và camera khớp model mới chọn được. Chọn chỉ ghi vào studio, chưa triển khai lên robot.</p>
        {msg && <div className={'note ' + (msg.ok ? 'ok' : '')}>{msg.text}{msg.problems && <ul style={{ margin: '4px 0 0' }}>{msg.problems.map((p) => <li key={p}>{p}</li>)}</ul>}</div>}
        <Table rows={robots} cols={[
          { h: 'Robot', r: (r) => <Link to={`/robots/${r.id}`}>{r.name}</Link> },
          { h: 'Tương thích', r: (r) => (r.problems.length ? <span className="err small">{r.problems.join('; ')}</span> : <Badge kind="ok">khớp</Badge>) },
          { h: '', r: (r) => (assigned_to.includes(r.id)
            ? <><Badge kind="info">đang dùng</Badge> <button className="btn sm" onClick={() => unchoose(r.id)}>bỏ chọn</button></>
            : <button className="btn sm primary" disabled={r.problems.length > 0} onClick={() => choose(r.id)}>Chọn cho robot này</button>) },
        ]} />

        <h2>Kết quả eval ({evals.length})</h2>
        <Table empty="Chưa có eval cho model này" rows={evals.map((e) => ({ ...e, id: e.tag }))} cols={[
          { h: 'Tag', r: (e) => <code>{e.tag}</code> }, { h: 'Loại', r: (e) => <Badge>{e.kind}</Badge> },
          { h: 'Điều kiện', r: (e) => (e.level ? `L${e.level} · ${e.split}` : '—') },
          { h: 'exec', n: true, r: (e) => e.exec_steps ?? '—' },
          { h: 'Thành công', n: true, r: (e) => `${e.k}/${e.n} = ${pct(e.rate)}` },
          { h: 'Wilson 95%', n: true, r: (e) => ci(e.lo, e.hi) },
        ]} />

        {editing && <ModelForm model={m} onClose={() => setEditing(false)} onSaved={() => { setEditing(false); st.reload() }} />}
        {deleting && (
          <Confirm title={`Xóa model "${label(m)}"?`} danger confirmText="Xóa model" busy={busy} error={err} onConfirm={remove} onClose={() => setDeleting(false)}>
            <p style={{ marginTop: 0 }}>
              {m.source === 'user' ? 'Model tự đăng ký sẽ bị gỡ khỏi studio.' : 'Model sẽ được ẩn khỏi studio (khôi phục ở trang Models bằng "hiện cả model đã xóa").'}
              {assigned_to.length > 0 && ` Model đang được chọn cho ${assigned_to.join(', ')}; việc chọn đó sẽ bị bỏ.`}
            </p>
            <p className="muted small" style={{ marginBottom: 0 }}>Checkpoint trên đĩa <b>không bị xóa</b>.</p>
          </Confirm>
        )}
      </Page>
    )}</Load>
  )
}
