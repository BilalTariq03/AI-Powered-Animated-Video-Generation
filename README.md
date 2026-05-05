# The Writer's Room — Agentic AI Story-to-Video Pipeline

> CS-4015 Agentic AI | NUCES | Spring 2026

A four-phase agentic pipeline that turns a single story prompt into a complete animated short film — with screenplay, character voices, background music, and lip-synced video — using locally-running and free-tier AI tools.

---

## Demo Output

```
Prompt: "A detective in 2087 discovers AI has been committing crimes."
  → 5-scene screenplay with character profiles
  → Voiced dialogue for each character (Kokoro TTS)
  → Mood-matched background music per scene (procedural synthesis)
  → Animated video with Ken Burns, emotion tints, subtitles
  → final_video.mp4
```

---

## Architecture

```
[User Prompt]
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Phase 1 — Story Generation                      │
│  LangGraph  +  Groq LLaMA 3.3 70B               │
│  ScriptwriterAgent → HITLAgent → CharacterAgent  │
│  → scene_manifest.json                           │
│  → phase2_audio_handoff.json                     │
│  → phase3_video_handoff.json                     │
│  → data/outputs/images/*.png                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  Phase 2 — Audio Generation                      │
│  Kokoro TTS  +  Procedural BGM Synthesis         │
│  → data/outputs/audio/dialogue/*.mp3             │
│  → data/outputs/audio/bgm/*.mp3                  │
│  → data/outputs/audio/timing_manifest.json       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  Phase 3 — Video Generation                      │
│  MoviePy  +  HF FLUX.1-schnell  +  Wav2Lip      │
│  → data/outputs/video/scenes/*.mp4               │
│  → data/outputs/video/final_video.mp4            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
         [Edit Agent — optional loop]
         Free-text command → LangGraph classify
         → mutate JSON → re-run affected phase only
```

Each phase reads the previous phase's handoff JSON and writes its own outputs. Any phase can be re-run independently without restarting the full pipeline.

---

## Project Structure

```
project/
├── main.py                          # Phase 1 entry point
├── config.py                        # API keys, model names, directory paths
├── requirements.txt
│
├── agents/
│   ├── base.py                      # BaseAgent with MCP tool discovery
│   ├── orchestrator/
│   │   └── graph.py                 # LangGraph workflow (Phase 1)
│   ├── story_agent/
│   │   ├── agent.py                 # ScriptwriterAgent (5-step reasoning loop)
│   │   ├── character_agent.py       # CharacterDesignerAgent
│   │   ├── image_agent.py           # ImageSynthesizerAgent
│   │   └── tests/test_story_agent.py
│   ├── audio_agent/
│   │   ├── agent.py                 # AudioGenerationAgent
│   │   ├── run.py                   # Phase 2 entry point
│   │   └── tests/test_audio.py
│   ├── video_agent/
│   │   ├── agent.py                 # VideoGenerationAgent
│   │   ├── lip_sync.py              # Wav2Lip wrapper
│   │   ├── setup_wav2lip.py         # One-time Wav2Lip setup
│   │   ├── run.py                   # Phase 3 entry point
│   │   └── tests/test_video_agent.py
│   └── edit_agent/
│       ├── agent.py                 # EditAgent (LangGraph classify → execute)
│       ├── executor.py              # EditExecutor (9 intent handlers)
│       └── tests/test_edit_agent.py
│
├── mcp/
│   ├── tool_registry.py             # MCPRegistry — central tool registry
│   └── tools/
│       ├── audio_tools/
│       │   ├── tts_tool.py          # TTSEngine (ElevenLabs → Kokoro → Edge → gTTS)
│       │   └── bgm_tool.py          # BGMEngine (procedural additive synthesis)
│       └── ...
│
├── shared/
│   └── schemas/
│       └── schema.py                # Pydantic models (PhaseOneOutput, SceneModel, etc.)
│
└── data/
    └── outputs/                     # All generated files (git-ignored)
        ├── scene_manifest.json
        ├── character_db.json
        ├── phase2_audio_handoff.json
        ├── phase3_video_handoff.json
        ├── images/
        ├── audio/
        │   ├── dialogue/
        │   ├── bgm/
        │   └── timing_manifest.json
        └── video/
            ├── scenes/
            ├── lipsync/
            └── final_video.mp4
```

---

## Prerequisites

- Python 3.10–3.12
- [ffmpeg](https://ffmpeg.org/download.html) installed and on `PATH` (required by MoviePy and pydub)
- A [Groq API key](https://console.groq.com/) (free tier is sufficient)
- A [HuggingFace API key](https://huggingface.co/settings/tokens) (free tier, for image generation)

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd project
python -m venv venv
```

**2. Activate the venv**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> The `en_core_web_sm` spaCy model is included in `requirements.txt` as a direct wheel URL and installs automatically.

**4. Create your `.env` file**

```bash
cp .env.example .env   # or create manually
```

```env
GROQ_API_KEY=your_groq_key_here
HF_API_KEY=your_huggingface_key_here

# Optional — enables ElevenLabs premium TTS (falls back to Kokoro if not set)
ELEVENLABS_API_KEY=
```

**5. (Optional) Set up Wav2Lip for lip sync**

```bash
python agents/video_agent/setup_wav2lip.py
```

This clones the Wav2Lip repository and attempts to download the checkpoint automatically. If automatic download fails, follow the printed instructions to download `wav2lip_gan.pth` (~430 MB) manually and place it at:

```
agents/video_agent/Wav2Lip/checkpoints/wav2lip_gan.pth
```

Wav2Lip is fully optional — the pipeline runs without it and falls back to static portrait animation.

---

## Running the Pipeline

### Phase 1 — Story Generation

```bash
python main.py
```

You will be prompted to choose a mode:

- **Auto** — enter a story prompt and the agent generates the full screenplay
- **Manual** — paste your own script JSON

After generation, a human-in-the-loop step lets you approve or request changes (up to 3 regeneration attempts).

You can also skip the interactive prompt for scripted use:

```bash
python main.py --prompt "A detective in 2087 discovers AI has been committing crimes." --auto
```

**Outputs:** `data/outputs/scene_manifest.json`, `phase2_audio_handoff.json`, `phase3_video_handoff.json`, `images/*.png`

---

### Phase 2 — Audio Generation

```bash
python agents/audio_agent/run.py
```

Reads `phase2_audio_handoff.json`. Synthesises voiced dialogue for every line using Kokoro TTS (or ElevenLabs if configured), and generates mood-matched background music for each scene using procedural additive synthesis.

**Outputs:** `data/outputs/audio/dialogue/*.mp3`, `audio/bgm/*.mp3`, `audio/timing_manifest.json`

---

### Phase 3 — Video Generation

```bash
python agents/video_agent/run.py
```

Reads `phase3_video_handoff.json` and `timing_manifest.json`. Generates scene backgrounds via HuggingFace FLUX.1-schnell (Pollinations.ai as fallback), composes animated frames with Ken Burns zoom, character portraits, emotion tints, and subtitle overlays, mixes audio, and concatenates all scenes.

**Output:** `data/outputs/video/final_video.mp4`

---

### Edit Agent — Post-Production Edits

After completing Phase 3, you can request changes in plain English. The agent classifies your intent, mutates the relevant handoff JSON, and re-runs only the affected phase.

```python
from agents.edit_agent.agent import EditAgent
from agents.edit_agent.executor import EditExecutor

agent    = EditAgent()
executor = EditExecutor()

result = agent.classify("make scene 3 darker and more dramatic")
executor.apply(result)
# → re-runs Phase 3 only
```

**Supported edit commands:**

| Command example | Phase re-run |
|---|---|
| `"change detective's voice to deep"` | Phase 2 |
| `"make scene 2 darker"` | Phase 3 |
| `"change scene 4 mood to hopeful"` | Phase 3 |
| `"remove subtitles"` / `"add subtitles"` | Phase 3 |
| `"change AI Drone's appearance to chrome humanoid robot"` | Phase 3 |
| `"speed up scene 1"` / `"slow down scene 5"` | Phase 3 |
| `"regenerate script"` | Phase 1 |

---

## Features

### TTS — Four-Tier Fallback Chain

| Tier | Provider | Requires | Quality |
|---|---|---|---|
| 1 | ElevenLabs | `ELEVENLABS_API_KEY` | Best |
| 2 | **Kokoro 82M** (local neural) | Nothing | High |
| 3 | Edge TTS (Microsoft cloud) | Nothing | Medium |
| 4 | gTTS (Google) | Nothing | Basic |

Kokoro is the default. The model is downloaded once from HuggingFace and cached locally. Seven distinct voice styles are supported: `deep`, `authoritative`, `raspy`, `soft`, `high-pitched`, `whispery`, `neutral`.

### BGM — Procedural Synthesis

Background music is generated in-process with no external API. Each of eight moods has a unique harmonic parameter set (base frequency, partials, waveform, LFO rate, pulse BPM):

`dramatic` · `tense` · `mysterious` · `hopeful` · `melancholic` · `comedic` · `romantic` · `fantasy`

### Lip Sync (Wav2Lip)

When the checkpoint is present, Wav2Lip generates per-segment mouth animations for any character whose portrait contains a detectable human face. Non-human/abstract characters automatically skip to static portrait fallback with no errors.

### Image Generation

Scene backgrounds and character portraits are generated via HuggingFace FLUX.1-schnell. If the API is unavailable, Pollinations.ai (free, no key) is tried next. A final fallback renders a mood-matched gradient so the pipeline always completes.

---

## Running Tests

```bash
# All tests
venv\Scripts\python -m pytest agents/ -v

# Individual modules
venv\Scripts\python -m pytest agents/story_agent/tests/ -v
venv\Scripts\python -m pytest agents/audio_agent/tests/ -v   # ~3 min (runs Kokoro)
venv\Scripts\python -m pytest agents/video_agent/tests/ -v
venv\Scripts\python -m pytest agents/edit_agent/tests/ -v
```

188 tests total across all four phases. Audio tests take ~3 minutes because they run real Kokoro synthesis.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLaMA 3.3 70B |
| `HF_API_KEY` | Recommended | HuggingFace key for FLUX.1-schnell image generation |
| `ELEVENLABS_API_KEY` | No | ElevenLabs key for premium TTS (Kokoro used otherwise) |

---

## Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `groq` | LLM API (LLaMA 3.3 70B) |
| `langgraph` + `langchain` | Agent orchestration & workflow graphs |
| `pydantic` | Schema validation for all pipeline data |
| `chromadb` | Vector store for script/character memory |
| `kokoro` | Local neural TTS |
| `pydub` + `scipy` + `numpy` | Audio processing and BGM synthesis |
| `moviepy` | Video composition and encoding |
| `Pillow` + `opencv-python` | Image processing and face detection |
| `python-dotenv` | `.env` loading |

---

## Known Issues

| Issue | Status |
|---|---|
| `OSError: [WinError 6]` at end of Phase 3 | Harmless MoviePy Windows bug — video is already written correctly |
| Wav2Lip auto-download fails | All original mirror URLs are dead; manual download required (setup script prints instructions) |
| Phase 1 takes 3–6 min | Bottlenecked by 4–5 sequential Groq LLM calls; no known workaround |
| Kokoro `dropout` UserWarning | PyTorch internal warning; harmless |

---

## License

Academic project — CS-4015 Agentic AI, NUCES, Spring 2026.
