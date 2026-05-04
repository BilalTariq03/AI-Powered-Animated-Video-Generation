import { useState } from 'react'

const EXAMPLES = [
  'Make scene 2 darker',
  'Change voice tone to deep for all characters',
  'Add hopeful background music to scene 3',
  'Remove subtitles',
  'Speed up scene 1 by 1.5x',
  'Change the character design to noir style',
  'Regenerate the script',
]

export default function EditPanel({ onEditDone, pipelineRunning }) {
  const [query,   setQuery]   = useState('')
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)

  const handleSubmit = async () => {
    if (!query.trim() || loading || pipelineRunning) return
    setLoading(true)
    setResult(null)

    try {
      const res  = await fetch('/api/edit', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ query, thread_id: 'session' }),
      })
      const data = await res.json()
      setResult(data)
      if (data.phase_to_rerun && onEditDone) {
        onEditDone(data.phase_to_rerun)
      }
    } catch (err) {
      setResult({ ok: false, message: String(err) })
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit()
  }

  return (
    <div className="edit-panel">
      <div className="edit-panel-header">
        <span className="section-title" style={{ marginBottom: 0 }}>Edit</span>
        <span className="edit-hint">Describe what you want to change in plain language</span>
      </div>

      {/* Example chips */}
      <div className="edit-examples">
        {EXAMPLES.map(ex => (
          <button
            key={ex}
            className="edit-chip"
            onClick={() => setQuery(ex)}
            disabled={loading}
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="edit-input-row">
        <input
          className="edit-input"
          placeholder='e.g. "Make scene 2 darker" or "Change voice tone to deep"'
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button
          className={`btn-edit-submit ${loading || pipelineRunning ? 'btn-disabled' : ''}`}
          onClick={handleSubmit}
          disabled={loading || pipelineRunning || !query.trim()}
        >
          {loading ? <span className="spinner" /> : 'Apply'}
        </button>
      </div>

      {result && (
        <div className={`edit-result ${result.ok ? 'edit-result-ok' : 'edit-result-err'}`}>
          {result.intent && (
            <div className="edit-intent-row">
              <span className="edit-intent-badge">{result.intent.intent}</span>
              <span className="edit-intent-scope">{result.intent.scope}</span>
              {result.phase_to_rerun && (
                <span className="edit-rerun-badge">→ Phase {result.phase_to_rerun} re-running</span>
              )}
            </div>
          )}
          <p className="edit-result-msg">{result.message || result.error || 'Done.'}</p>
        </div>
      )}
    </div>
  )
}
