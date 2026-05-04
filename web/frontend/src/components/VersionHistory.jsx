import { useEffect, useState } from 'react'

function fmtTime(iso) {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return iso
  }
}

export default function VersionHistory({ onReverted }) {
  const [versions, setVersions] = useState([])
  const [open,     setOpen]     = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [reverting, setReverting] = useState(null)

  const fetchVersions = async () => {
    try {
      const res  = await fetch('/api/versions')
      const data = await res.json()
      setVersions(data.versions || [])
    } catch (_) {}
  }

  useEffect(() => {
    fetchVersions()
  }, [])

  // Re-fetch when panel opens
  const toggleOpen = () => {
    if (!open) fetchVersions()
    setOpen(o => !o)
  }

  const handleRevert = async (version) => {
    if (loading || reverting) return
    setReverting(version)
    try {
      const res = await fetch(`/api/revert/${version}`, { method: 'POST' })
      if (!res.ok) throw new Error((await res.json()).detail)
      await fetchVersions()
      if (onReverted) onReverted(version)
    } catch (err) {
      alert(`Revert failed: ${err.message}`)
    } finally {
      setReverting(null)
    }
  }

  if (versions.length === 0) return null

  return (
    <div className="version-panel">
      <button className="version-toggle" onClick={toggleOpen}>
        <span>Version History</span>
        <span className="version-count">{versions.length}</span>
        <span className="version-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="version-list">
          {versions.map((v, idx) => (
            <div key={v.version} className={`version-row ${idx === 0 ? 'version-current' : ''}`}>
              <div className="version-meta">
                <span className="version-tag">{v.version}</span>
                {idx === 0 && <span className="version-current-badge">current</span>}
                <span className="version-time">{fmtTime(v.timestamp)}</span>
              </div>
              <p className="version-desc">{v.description}</p>
              {idx !== 0 && (
                <button
                  className="btn-revert"
                  onClick={() => handleRevert(v.version)}
                  disabled={!!reverting}
                >
                  {reverting === v.version
                    ? <span className="spinner" />
                    : 'Revert'}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
