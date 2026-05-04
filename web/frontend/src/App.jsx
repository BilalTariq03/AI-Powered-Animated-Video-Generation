import { useState, useEffect, useRef, useCallback } from 'react'
import PhaseCard from './components/PhaseCard'
import VideoPlayer from './components/VideoPlayer'

const PHASES = [
  { id: 1, label: 'Story & Script Generation' },
  { id: 2, label: 'Audio & Voice Generation'  },
  { id: 3, label: 'Video Rendering'            },
]

const blank = () => ({ status: 'idle', logs: [] })
const blankPhases = () => ({ 1: blank(), 2: blank(), 3: blank() })

export default function App() {
  const [prompt,   setPrompt]   = useState('')
  const [phases,   setPhases]   = useState(blankPhases)
  const [hasVideo, setHasVideo] = useState(false)
  const [running,  setRunning]  = useState(false)
  const [jobId,    setJobId]    = useState(null)
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
        break

      case 'done':
        setRunning(false)
        setHasVideo(event.has_video)
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
      </main>
    </div>
  )
}
