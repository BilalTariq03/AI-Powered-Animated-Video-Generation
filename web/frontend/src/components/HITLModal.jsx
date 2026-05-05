import { useState } from 'react'

export default function HITLModal({ script, onRespond }) {
  const [rejecting,  setRejecting]  = useState(false)
  const [newPrompt,  setNewPrompt]  = useState('')
  const [submitting, setSubmitting] = useState(false)

  const approve = async () => {
    setSubmitting(true)
    await fetch('/api/hitl/respond', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action: 'approve' }),
    })
    onRespond('approve')
  }

  const reject = async () => {
    if (!newPrompt.trim()) return
    setSubmitting(true)
    await fetch('/api/hitl/respond', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action: 'regenerate', prompt: newPrompt.trim() }),
    })
    onRespond('regenerate')
  }

  return (
    <div className="hitl-overlay">
      <div className="hitl-modal">
        <h2 className="hitl-title">Review Generated Script</h2>
        <p className="hitl-subtitle">
          The AI has written a screenplay. Review it below and approve or request changes.
        </p>

        <div className="hitl-story-meta">
          <span className="hitl-tag">{script.genre}</span>
          <strong>{script.title}</strong>
          {script.synopsis && <p className="hitl-synopsis">{script.synopsis}</p>}
        </div>

        <div className="hitl-scenes">
          {script.scenes?.map(scene => (
            <div key={scene.scene_id} className="hitl-scene">
              <div className="hitl-scene-header">
                <span className="hitl-scene-num">Scene {scene.scene_id}</span>
                <span className="hitl-scene-loc">{scene.location}</span>
                <span className={`hitl-mood hitl-mood-${scene.mood}`}>{scene.mood}</span>
              </div>
              {scene.characters?.length > 0 && (
                <p className="hitl-chars">
                  {scene.characters.join(', ')}
                </p>
              )}
              <div className="hitl-dialogue">
                {scene.dialogue?.map((d, i) => (
                  <div key={i} className="hitl-line">
                    <span className="hitl-speaker">{d.speaker}:</span>
                    <span className="hitl-text">"{d.line}"</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {rejecting ? (
          <div className="hitl-reject-form">
            <label className="hitl-reject-label">What would you like to change?</label>
            <textarea
              className="hitl-reject-input"
              placeholder="e.g. Make it a comedy instead of noir. Add a robot sidekick."
              value={newPrompt}
              onChange={e => setNewPrompt(e.target.value)}
              rows={3}
              autoFocus
            />
            <div className="hitl-actions">
              <button
                className="btn-hitl-back"
                onClick={() => setRejecting(false)}
                disabled={submitting}
              >
                ← Back
              </button>
              <button
                className="btn-hitl-regenerate"
                onClick={reject}
                disabled={submitting || !newPrompt.trim()}
              >
                {submitting ? 'Sending...' : 'Regenerate Script'}
              </button>
            </div>
          </div>
        ) : (
          <div className="hitl-actions">
            <button
              className="btn-hitl-reject"
              onClick={() => setRejecting(true)}
              disabled={submitting}
            >
              Request Changes
            </button>
            <button
              className="btn-hitl-approve"
              onClick={approve}
              disabled={submitting}
            >
              {submitting ? 'Approving...' : '✓ Approve & Continue'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
