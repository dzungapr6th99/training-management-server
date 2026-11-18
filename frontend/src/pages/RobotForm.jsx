import React from 'react'
import { api, useApi } from '../api.js'
import { Field, Modal, wide } from '../forms.jsx'

const STATUSES = [['design', 'thiết kế'], ['active', 'đang dùng'], ['component', 'bộ phận'], ['deprecated', 'đã bỏ']]

/** Create (robot == null) or edit a robot. URDF and USD are required; the USD with the ROS 2 graph is optional. */
export default function RobotForm({ robot, configured, onClose, onSaved }) {
  const editing = !!robot
  const dirs = useApi('/design/dirs')
  const c = configured || {}
  const [f, setF] = React.useState({
    id: robot?.id || '', name: c.name || '', kind: c.kind || 'other', status: c.status || 'design', description: c.description || '',
    state_dim: c.state_dim ?? '', action_dim: c.action_dim ?? '', cameras: (c.cameras || []).join(', '),
    design_dir: c.design_dir ? c.design_dir.replace(/^design\//, '') : '', urdf: c.urdf || '', usd: c.usd || '', usd_ros: c.usd_ros || '',
    notes: c.notes || '',
  })
  const [err, setErr] = React.useState(null)
  const [busy, setBusy] = React.useState(false)
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })
  const dir = (dirs.data || []).find((d) => d.dir === f.design_dir)
  const num = (v) => (v === '' || v == null ? null : Number(v))

  const save = async () => {
    setErr(null); setBusy(true)
    const body = {
      name: f.name, kind: f.kind, status: f.status, description: f.description || null, notes: f.notes || null,
      state_dim: num(f.state_dim), action_dim: num(f.action_dim),
      cameras: f.cameras.split(',').map((s) => s.trim()).filter(Boolean),
      design_dir: f.design_dir || null, urdf: f.urdf || null, usd: f.usd || null, usd_ros: f.usd_ros || null,
    }
    try {
      if (editing) await api(`/robots/${robot.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      else await api('/robots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, id: f.id }) })
      onSaved(editing ? robot.id : f.id)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  const FilePick = ({ k, label, options, required, hint }) => (
    <Field label={label} required={required} hint={hint}>
      <input list={`dl-${k}`} value={f[k]} onChange={set(k)} placeholder={options.length ? 'chọn hoặc gõ đường dẫn tương đối' : 'đường dẫn tương đối trong thư mục design'} style={wide} />
      <datalist id={`dl-${k}`}>{options.map((o) => <option key={o} value={o} />)}</datalist>
    </Field>
  )

  return (
    <Modal title={editing ? `Sửa robot: ${robot.name}` : 'Thêm robot'} onClose={onClose} wide>
      {!editing && (
        <Field label="ID" required hint="Chữ thường, số, _ hoặc -; không đổi được sau khi tạo.">
          <input value={f.id} onChange={set('id')} placeholder="vd: humanoid_v2" style={wide} />
        </Field>
      )}
      <Field label="Tên" required><input value={f.name} onChange={set('name')} style={wide} /></Field>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <Field label="Loại"><input value={f.kind} onChange={set('kind')} style={wide} /></Field>
        <Field label="Trạng thái">
          <select value={f.status} onChange={set('status')} style={wide}>{STATUSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
        </Field>
        <Field label="Camera" hint="cách nhau bằng dấu phẩy"><input value={f.cameras} onChange={set('cameras')} placeholder="camera1, camera2" style={wide} /></Field>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Field label="State (số chiều)"><input type="number" min={1} value={f.state_dim} onChange={set('state_dim')} style={wide} /></Field>
        <Field label="Action (số chiều)"><input type="number" min={1} value={f.action_dim} onChange={set('action_dim')} style={wide} /></Field>
      </div>

      <div className="panel" style={{ background: 'var(--bg)', margin: '4px 0 12px' }}>
        <b className="small">Ba bản mô hình cho Isaac Sim</b>
        <p className="muted small" style={{ margin: '2px 0 10px' }}>URDF và USD là <b>bắt buộc</b>. USD có ROS 2 (đồ thị ROS bridge) tùy chọn, có thể để trống.</p>
        <Field label="Thư mục trong design/" required>
          <select value={f.design_dir} onChange={set('design_dir')} style={wide}>
            <option value="">— chọn —</option>
            {(dirs.data || []).map((d) => <option key={d.dir} value={d.dir}>{d.dir}</option>)}
          </select>
        </Field>
        <FilePick k="urdf" label="URDF" required options={dir?.urdf || []} />
        <FilePick k="usd" label="USD" required options={dir?.usd || []} />
        <FilePick k="usd_ros" label="USD có ROS (tùy chọn)" options={dir?.usd || []} hint="Ví dụ stage chứa ROS 2 bridge graph để Isaac Sim nối với robot-agent." />
      </div>

      <Field label="Mô tả"><textarea value={f.description} onChange={set('description')} rows={2} style={{ ...wide, font: 'inherit', padding: 8, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--panel)', color: 'var(--ink)' }} /></Field>
      <Field label="Ghi chú"><input value={f.notes} onChange={set('notes')} style={wide} /></Field>
      {err && <div className="note" style={{ borderColor: 'var(--bad)', background: 'var(--bad-soft)' }}>{err}</div>}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button className="btn" onClick={onClose}>Hủy</button>
        <button className="btn primary" disabled={busy} onClick={save}>{busy ? 'Đang lưu…' : editing ? 'Lưu' : 'Thêm robot'}</button>
      </div>
    </Modal>
  )
}
