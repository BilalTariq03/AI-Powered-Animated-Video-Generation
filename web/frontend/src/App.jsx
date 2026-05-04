import { useState, useEffect, useRef, useCallback } from 'react'
import PhaseCard      from './components/PhaseCard'
import VideoPlayer    from './components/VideoPlayer'
import EditPanel      from './components/EditPanel'
import VersionHistory from './components/VersionHistory'

const PHASES = [
  { id: 1, label: 'Story & Script Generation' },
  { id: 2, label: 'Audio & Voice Generation'  },
  { id: 3, label: 'Video Rendering'            },
]

const blank      = () => ({ status: 'idle', logs: [] })
const blankPhases = () => ({ 1: blank(), 2: blank(), 3: blank() })

export default function App() {
  const [prompt,       setPrompt]       = useState('')
  const [phases,       setPhases]       = useState(blankPhases)
  const [hasVideo,     setHasVideo]     = useState(false)
  const [running,      setRunning]      = useState(false)
  const [jobId,        setJobId]        = useState(null)       // eslint-disable-line no-unused-vars
  const [versionKey,   setVersionKey]   = useState(0)
  // One-way gate: once true it never goes back to false (avoids EditPanel unmounting)
  const [editUnlocked, setEditUnlocked] = useState(false)
  const esRef = useRef(null)

  const handleEvent = useCallback((event) => {
    switch (event.type) {
      case 'phase_start':
        setRunning(true)
        setPhases(p => ({ ...p, [event.phase]: { status: 'running', logs: [] } }))
        break

      case 'log':
        setPhases(p => ({
          ...p,
          [event.phase]: {
            ...p[event.phase],
            logs: [...(p[event.phase]?.logs ?? []), event.message],
          },
        }))
        break

      case 'phase_end':
        setPhases(p => ({
          ...p,
          [event.phase]: { ...p[event.phase], status: event.status },
        }))
        if (event.status === 'done') setEditUnlocked(true)
        break

      case 'done':
        setRunning(false)
        setHasVideo(event.has_video)
        if (event.has_video) setEditUnlocked(true)
        break

      case 'snapshot':
        // New snapshot created — refresh version list
        setVersionKey(k => k + 1)
        break

      case 'reverted':
        setVersionKey(k => k + 1)
        break

      default:
        break
    }
  }, [])

  const connectSSE = useCallback(() => {
    if (esRef.current) esRef.current.close()
    const es = new EventSource('/api/stream')
    esRef.current = es
    es.onmessage = (e) => {
      try { handleEvent(JSON.parse(e.data)) } catch (_) {}
    }
  }, [handleEvent])

  // Restore state on page load
  useEffect(() => {
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        if (data.job.prompt) setPrompt(data.job.prompt)
        if (data.job.phases) {
          setPhases(p => {
            const next = { ...p }
            for (const [k, v] of Object.entries(data.job.phases))
              next[parseInt(k)] = v
            return next
          })
        }
        setHasVideo(data.has_video)
        if (data.has_video) setEditUnlocked(true)
        if (data.job.active) {
          setRunning(true)
          connectSSE()
        }
      })
      .catch(() => {})

    return () => esRef.current?.close()
  }, [connectSSE])

  const handleStart = async () => {
    if (!prompt.trim() || running) return
    setPhases(blankPhases())
    setHasVideo(false)
    setRunning(true)
    setEditUnlocked(false)
    const res  = await fetch('/api/start', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt }),
    })
    const data = await res.json()
    setJobId(data.job_id)
    connectSSE()
  }

  const handleRerun = async (phase) => {
    if (running) return
    setPhases(p => {
      const next = { ...p }
      for (let i = phase; i <= 3; i++) next[i] = blank()
      return next
    })
    setHasVideo(false)
    setRunning(true)
    await fetch('/api/rerun', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ phase }),
    })
    connectSSE()
  }

  // Called by EditPanel when an edit requires a phase rerun
  const handleEditDone = useCallback((phaseToRerun) => {
    if (!phaseToRerun) return
    setVersionKey(k => k + 1)
    handleRerun(phaseToRerun)
  }, [running]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleReverted = useCallback(() => {
    setVersionKey(k => k + 1)
    setHasVideo(false)
    setPhases(blankPhases())
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <h1>AI Story Pipeline</h1>
          <span className="header-sub">CS-4015 Agentic AI &mdash; NUCES</span>
        </div>
      </header>

      <main className="main">
        {/* Prompt */}
        <section className="prompt-card">
          <label className="prompt-label">Story Prompt</label>
          <textarea
            className="prompt-input"
            placeholder="e.g. A detective in 2087 discovers AI has been committing crimes."
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            rows={3}
            disabled={running}
          />
          <button
            className={`btn-generate ${running ? 'btn-disabled' : ''}`}
            onClick={handleStart}
            disabled={running || !prompt.trim()}
          >
            {running ? 'Generating...' : 'Generate Story'}
          </button>
        </section>

        {/* Phase cards */}
        <section className="phases">
          {PHASES.map((ph, idx) => {
            const prevDone = idx === 0 || phases[ph.id - 1]?.status === 'done'
            return (
              <PhaseCard
                key={ph.id}
                phaseNum={ph.id}
                label={ph.label}
                state={phases[ph.id]}
                onRerun={() => handleRerun(ph.id)}
                pipelineRunning={running}
                canRerun={!running && prevDone}
              />
            )
          })}
        </section>

        {/* Video output */}
        {hasVideo && (
          <section className="video-section">
            <h2 className="section-title">Final Video</h2>
            <VideoPlayer />
          </section>
        )}

        {/* Edit & Undo (Phase 5) — visible once any phase completes, never unmounts after */}
        {editUnlocked && (
          <>
            <EditPanel
              onEditDone={handleEditDone}
              pipelineRunning={running}
            />
            <VersionHistory
              key={versionKey}
              onReverted={handleReverted}
            />
          </>
        )}
      </main>
    </div>
  )
}
