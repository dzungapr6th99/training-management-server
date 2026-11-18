import React from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api.js'
import { Load, Page, Status, Table } from '../ui.jsx'

export default function Tasks() {
  const st = useApi('/tasks')
  return (
    <Page title="Tasks" sub="Các tác vụ mô phỏng, robot thực hiện, level/split và khoảng seed.">
      <Load state={st}>{(rows) => (
        <Table rows={rows} cols={[
          { h: 'Task', r: (t) => <><b>{t.name}</b><div className="muted small"><code>{t.id}</code></div></> },
          { h: 'Trạng thái', r: (t) => <Status s={t.status} /> },
          { h: 'Robot', r: (t) => <Link to={`/robots/${t.robot_id}`}>{t.robot_id}</Link> },
          { h: 'Level', r: (t) => (t.levels.length ? t.levels.map((l) => 'L' + l).join(', ') : '—') },
          { h: 'Split', r: (t) => (t.splits.length
            ? <div className="chip-row">{t.splits.map((s) => <span className="chip" key={s}>{s}</span>)}</div> : '—') },
          { h: 'Seed', r: (t) => Object.entries(t.seeds).map(([k, v]) => <div className="small" key={k}><span className="muted">{k}:</span> {v}</div>) },
          { h: 'Datasets', n: true, r: (t) => t.n_datasets },
          { h: 'Eval', n: true, r: (t) => t.n_evals },
          { h: 'Mô tả', r: (t) => <span className="small">{t.description}</span> },
        ]} />
      )}</Load>
    </Page>
  )
}
