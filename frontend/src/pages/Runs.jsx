import React from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ci, num, pct, useApi } from '../api.js'
import { Badge, KV, Load, ModelBadge, Page, Table } from '../ui.jsx'

const fin = (r) => (r.finished == null ? <Badge>không rõ</Badge> : r.finished ? <Badge kind="ok">xong</Badge> : <Badge kind="warn">chưa xong</Badge>)

export function Runs() {
  const st = useApi('/runs')
  const nav = useNavigate()
  return (
    <Page title="Training" sub="Các lần train trong outputs/. Số liệu step, thời gian, VRAM, loss lấy từ results/train_*.json (nếu có).">
      <Load state={st}>{(rows) => (
        <Table rows={rows} onRow={(r) => nav(`/runs/${r.id}`)} cols={[
          { h: 'Run', r: (r) => <code>{r.id}</code> },
          { h: 'Loại', r: (r) => <ModelBadge t={r.model_type} /> },
          { h: 'Dataset', r: (r) => <code>{r.dataset_id || '—'}</code> },
          { h: 'Trạng thái', r: fin },
          { h: 'Step', n: true, r: (r) => num(r.steps) },
          { h: 'Batch', n: true, r: (r) => r.batch_size ?? '—' },
          { h: 'Giờ', n: true, r: (r) => (r.wall_hours != null ? r.wall_hours.toFixed(2) : '—') },
          { h: 'VRAM đỉnh (GB)', n: true, r: (r) => r.peak_mem_gb ?? '—' },
          { h: 'Loss cuối', n: true, r: (r) => r.final_loss ?? '—' },
          { h: 'Checkpoint', n: true, r: (r) => r.checkpoints.length },
        ]} />
      )}</Load>
    </Page>
  )
}

export function RunDetail() {
  const { id } = useParams()
  const st = useApi(`/runs/${id}`)
  const loss = useApi(`/runs/${id}/loss`)
  return (
    <Load state={st}>{(r) => (
      <Page crumb={<Link to="/runs">Training</Link>} title={r.id} sub={r.command ? undefined : 'Không có file tóm tắt results/train_*.json cho lần train này.'}>
        <div className="panel">
          <KV items={[
            ['Loại model', <ModelBadge t={r.model_type} />],
            ['Dataset', r.dataset_id ? <Link to={`/datasets/${r.dataset_id}`}><code>{r.dataset_id}</code></Link> : null],
            ['Khởi tạo', r.init], ['Trạng thái', fin(r)],
            ['Step / Batch', `${num(r.steps)} / ${r.batch_size ?? '—'}`],
            ['Thời gian', r.wall_hours != null ? `${r.wall_hours} giờ` : null],
            ['VRAM đỉnh (log)', r.peak_mem_gb != null ? `${r.peak_mem_gb} GB` : null],
            ['Loss đầu → cuối (TB 5 log cuối)', r.loss_first != null || r.final_loss != null ? `${r.loss_first ?? '—'} → ${r.final_loss ?? '—'}` : null],
            ['Checkpoint', r.checkpoints.length ? r.checkpoints.join(', ') : 'không có trong outputs/'],
            ['Tóm tắt', r.source && <code>{r.source}</code>],
            ['Log', r.log && <code>{r.log}</code>],
          ]} />
        </div>

        <h2>Đường loss</h2>
        <Load state={loss}>{(l) => (l.points.length ? (
          <div className="panel" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={l.points.map(([s, v]) => ({ step: s, loss: v }))} margin={{ left: 0, right: 16, top: 6, bottom: 0 }}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                <XAxis dataKey="step" tick={{ fill: 'var(--muted)', fontSize: 12 }} tickFormatter={(v) => (v >= 1000 ? v / 1000 + 'K' : v)} />
                <YAxis scale="log" domain={['auto', 'auto']} tick={{ fill: 'var(--muted)', fontSize: 12 }} width={50} />
                <Tooltip formatter={(v) => v} labelFormatter={(v) => 'step ' + v} contentStyle={{ background: 'var(--panel)', border: '1px solid var(--line)' }} />
                <Line type="monotone" dataKey="loss" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <div className="panel muted">Không tìm thấy dòng loss trong logs/train_{r.id}*.log.</div>)}</Load>
        <p className="muted small">Trục loss dùng thang log. Dữ liệu đọc từ log train (gộp cả log resume).</p>

        <h2>Checkpoint ({r.models.length})</h2>
        <Table empty="Không có checkpoint" rows={r.models} cols={[
          { h: 'Model', r: (m) => <code>{m.id}</code> },
          { h: 'Bản', r: (m) => <Badge kind={m.status === 'final' ? 'ok' : ''}>{m.status === 'final' ? 'cuối' : 'trung gian'}</Badge> },
          { h: 'State/Action', n: true, r: (m) => `${m.state_dim}/${m.action_dim}` },
          { h: 'Chunk / n_action_steps', n: true, r: (m) => `${m.chunk_size ?? '—'} / ${m.n_action_steps ?? '—'}` },
        ]} />

        <h2>Kết quả eval của run này ({r.evals.length})</h2>
        <Table empty="Chưa có eval" rows={r.evals} cols={[
          { h: 'Tag', r: (e) => <code>{e.tag}</code> }, { h: 'Loại', r: (e) => <Badge>{e.kind}</Badge> },
          { h: 'Điều kiện', r: (e) => (e.level ? `L${e.level} · ${e.split}` : '—') },
          { h: 'exec', n: true, r: (e) => e.exec_steps ?? '—' },
          { h: 'Thành công', n: true, r: (e) => `${e.k}/${e.n} = ${pct(e.rate)}` },
          { h: 'Wilson 95%', n: true, r: (e) => ci(e.lo, e.hi) },
        ]} />
      </Page>
    )}</Load>
  )
}
