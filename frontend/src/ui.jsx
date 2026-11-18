import React from 'react'

export function Badge({ kind = '', children }) {
  return <span className={'badge ' + kind}>{children}</span>
}

const STATUS = {
  active: ['ok', 'đang dùng'], ok: ['ok', 'ok'], final: ['ok', 'cuối'], intermediate: ['', 'trung gian'],
  deprecated: ['warn', 'đã bỏ'], superseded: ['warn', 'bản cũ'], design: ['info', 'thiết kế'], component: ['', 'bộ phận'],
}
export function Status({ s }) {
  const [k, t] = STATUS[s] || ['', s]
  return <Badge kind={k}>{t}</Badge>
}
export const ModelBadge = ({ t }) => <Badge kind={['act', 'smolvla', 'groot'].includes(t) ? t : ''}>{t || '—'}</Badge>

export function Page({ title, sub, crumb, children, actions }) {
  return (
    <div>
      {crumb && <div className="crumb">{crumb}</div>}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <h1>{title}</h1>
          {sub && <p className="sub">{sub}</p>}
        </div>
        {actions}
      </div>
      {children}
    </div>
  )
}

export function Load({ state, children }) {
  if (state.error) return <p className="err">Lỗi: {String(state.error.message || state.error)}</p>
  if (!state.data) return <p className="muted">Đang tải…</p>
  return children(state.data)
}

export function Table({ cols, rows, empty = 'Không có dữ liệu', onRow }) {
  if (!rows.length) return <div className="panel empty">{empty}</div>
  return (
    <div className="tablewrap">
      <table>
        <thead><tr>{cols.map((c) => <th key={c.h} className={c.n ? 'n' : ''}>{c.h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id ?? r.tag ?? i} onClick={onRow ? () => onRow(r) : undefined} style={onRow ? { cursor: 'pointer' } : undefined}>
              {cols.map((c) => <td key={c.h} className={c.n ? 'n' : ''}>{c.r(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function KV({ items }) {
  return (
    <dl className="kv">
      {items.filter(Boolean).map(([k, v]) => (<React.Fragment key={k}><dt>{k}</dt><dd>{v ?? '—'}</dd></React.Fragment>))}
    </dl>
  )
}
