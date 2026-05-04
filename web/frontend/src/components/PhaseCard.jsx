import { useEffect, useRef, useState } from 'react'

const STATUS_LABEL = {
  idle:    'IDLE',
  running: 'RUNNING',
  done:    'DONE',
  error:   'ERROR',
}

export default function PhaseCard({ phaseNum, label, state, onRerun, pipelineRunning, canRerun }) {
  const { status, logs } = state
  const logRef   = useRef(null)
  const [open, setOpen] = useState(false)

  // Auto-open and scroll when running
  useEffect(() => {
    if (status === 'running') setOpen(true)
  }, [status])

  useEffect(() => {
    if (open && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs, open])

  const canShowLogs = status !== 'idle' && logs.length > 0

  return (
    <div className={`phase-card phase-${status}`}>
      <div className="phase-header">
        <div className="phase-title-row">
          <span className="phase-num">Phase {phaseNum}</span>
          <span className="phase-label">{label}</span>
          <span className={`status-badge status-${status}`}>
            {status === 'running' && <span className="spinner" />}
            {STATUS_LABEL[status] ?? status.toUpperCase()}
          </span>
        </div>
        <div className="phase-actions">
          {canShowLogs && (
            <button className="btn-toggle" onClick={() => setOpen(o => !o)}>
              {open ? 'Hide logs' : 'Show logs'}
            </button>
          )}
          {canRerun && status !== 'idle' && (
            <button
              className="btn-rerun"
              onClick={onRerun}
              disabled={pipelineRunning}
            >
              Re-run
            </button>
          )}
        </div>
      </div>

      {open && canShowLogs && (
        <div className="log-console" ref={logRef}>
          {logs.map((line, i) => (
            <div key={i} className={`log-line ${line.startsWith('[ERROR]') ? 'log-error' : ''}`}>
              {line}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
