import React from 'react'

export function Modal({ title, onClose, children, wide }) {
  React.useEffect(() => {
    const k = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(10,14,20,.5)', display: 'grid', placeItems: 'start center', padding: '6vh 16px', overflowY: 'auto', zIndex: 50 }}>
      <div onClick={(e) => e.stopPropagation()} className="panel" style={{ width: wide ? 760 : 560, maxWidth: '100%', boxShadow: '0 20px 60px rgba(0,0,0,.35)' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
          <b style={{ fontSize: 16, flex: 1 }}>{title}</b>
          <button className="btn sm" onClick={onClose}>Đóng</button>
        </div>
        {children}
      </div>
    </div>
  )
}

export function Field({ label, hint, required, children }) {
  return (
    <label style={{ display: 'block', margin: '0 0 12px' }}>
      <div className="small" style={{ fontWeight: 600, marginBottom: 3 }}>
        {label}{required && <span style={{ color: 'var(--bad)' }}> *</span>}
      </div>
      {children}
      {hint && <div className="muted small" style={{ marginTop: 3 }}>{hint}</div>}
    </label>
  )
}

export const wide = { width: '100%' }

export function Confirm({ title, children, confirmText = 'Xác nhận', danger, busy, onConfirm, onClose, error }) {
  return (
    <Modal title={title} onClose={onClose}>
      <div style={{ marginBottom: 14 }}>{children}</div>
      {error && <div className="note" style={{ borderColor: 'var(--bad)', background: 'var(--bad-soft)' }}>{error}</div>}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button className="btn" onClick={onClose}>Hủy</button>
        <button className="btn primary" style={danger ? { background: 'var(--bad)', borderColor: 'var(--bad)' } : undefined} disabled={busy} onClick={onConfirm}>
          {busy ? 'Đang xử lý…' : confirmText}
        </button>
      </div>
    </Modal>
  )
}
