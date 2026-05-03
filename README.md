# 🎬 THE WRITER'S ROOM
## Autonomous Story & Image Generation Layer
### CS-4015 Agentic AI — Phase 1 | NUCES

---

## 📁 Project Structure

```
writers_room/
├── main.py                   # Entry point — run this
├── config.py                 # API keys & paths
├── tools.py                  # MCP tool registrations (all tools live here)
├── mcp_registry.py           # MCP dynamic tool discovery engine
├── requirements.txt
├── .env.example              # Copy to .env and fill in keys
│
├── agents/
│   ├── base.py               # BaseAgent (Groq LLM + MCP discovery)
│   ├── scriptwriter.py       # Generates screenplay from prompt
│   ├── validator.py          # Validates manually uploaded scripts
│   ├── hitl.py               # Human-in-the-Loop review checkpoint
│   ├── character_designer.py # Extracts character profiles
│   └── image_synthesizer.py  # Generates character images (HuggingFace)
│
├── memory/
│   └── vector_store.py       # ChromaDB persistent shared memory
│
├── workflow/
│   └── graph.py              # LangGraph StateGraph (full pipeline)
│
└── output/                   # Generated on first run
    ├── scene_manifest.json   # Structured screenplay
    ├── character_db.json     # Character identity store
    └── images/               # AI-generated character portraits
```

---

## ⚡ Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env with your keys
```

**Getting your free keys:**
| Key | Where to get it | Cost |
|-----|----------------|------|
| `GROQ_API_KEY` | https://console.groq.com | Free |
| `HF_API_KEY` | https://huggingface.co/settings/tokens | Free |

> `HF_API_KEY` is optional — without it, placeholder images are created.

### 3. Run the system
```bash
python main.py
```

---

## 🔄 How It Works

```
User Input (prompt or script)
        │
        ▼
[mode_selector_node]
  ├─ auto  → [scriptwriter_node]  ← Groq LLM generates screenplay
  └─ manual → [validator_node]   ← Validates & standardizes script
        │
        ▼
[hitl_node]  ← YOU review & approve/reject the script
        │
        ▼
[character_node]  ← Extracts character profiles
        │
        ▼
[image_node]  ← Generates portraits via HuggingFace SD
        │
        ▼
[memory_commit_node]  ← Saves everything to ChromaDB
        │
        ▼
   OUTPUT FILES
```

---

## 🏗️ Architecture

### Multi-Agent System (Supervisor-Worker)
- **Orchestrator**: LangGraph StateGraph (implicit supervisor via routing)
- **Workers**: Scriptwriter, Validator, HITL, Character Designer, Image Synthesizer

### MCP Tool Discovery
All tools are registered in `tools.py` and discovered dynamically at runtime.
**No agent hardcodes any API call** — they all go through the MCP registry:
```python
tools = mcp.discover(tags=["script"])   # discover
result = mcp.invoke("generate_script_segment", {...})  # invoke
```

### Shared Memory (ChromaDB)
All agents share a persistent vector store for:
- Script history
- Character metadata
- Image references
- Recovery from failure

---

## 📦 Outputs

| File | Description |
|------|-------------|
| `output/scene_manifest.json` | Full structured screenplay |
| `output/character_db.json` | All character profiles |
| `output/images/*.png` | Character reference images |

---

## 📊 Evaluation Coverage

| Criterion | Implementation |
|-----------|---------------|
| Agent Definition (20 pts) | 5 agents with clear roles in `agents/` |
| Script Generation (15 pts) | Groq LLM via MCP `generate_script_segment` |
| MCP Integration (15 pts) | `mcp_registry.py` + `tools.py` — zero hardcoding |
| LangGraph Workflow (10 pts) | Full StateGraph in `workflow/graph.py` |
| Human-in-the-Loop (10 pts) | `agents/hitl.py` with approve/reject/edit |
| Output Completeness (5 pts) | JSON + images in `output/` |
