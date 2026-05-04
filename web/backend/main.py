"""
web/backend/main.py
────────────────────
FastAPI backend for Phase 4 web interface.

Start:
    uvicorn web.backend.main:app --reload --port 8000
"""

import asyncio
import json
import os
import subprocess as _subprocess
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from config import FINAL_VIDEO  # noqa: E402

app = FastAPI(title="AI Story Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────

PHASE_LABELS = {
    1: "Story & Script",
    2: "Audio & Voice",
    3: "Video Rendering",
}

PHASE_CMDS = {
    1: lambda prompt: [sys.executable, "-u", "main.py", "--auto", "--prompt", prompt],
    2: lambda _:      [sys.executable, "-u", "agents/audio_agent/run.py"],
    3: lambda _:      [sys.executable, "-u", "agents/video_agent/run.py"],
}


def _blank_phase():
    return {"status": "idle", "logs": []}


def _new_job(prompt=""):
    return {
        "id":     str(uuid.uuid4())[:8],
        "prompt": prompt,
        "active": False,
        "phases": {i: _blank_phase() for i in [1, 2, 3]},
    }


# Mutable global state (single-user app)
_job: dict = _new_job()
_events: list[dict] = []
_pipeline_done: bool = False
_active_task: asyncio.Task | None = None


# ── Event helpers ─────────────────────────────────────────────────────────────

def _emit(event: dict) -> None:
    _events.append(event)


# ── Phase runner ──────────────────────────────────────────────────────────────

async def _run_phase(phase: int, prompt: str) -> bool:
    _job["phases"][phase] = {"status": "running", "logs": []}
    _emit({"type": "phase_start", "phase": phase, "label": PHASE_LABELS[phase]})

    cmd = PHASE_CMDS[phase](prompt)
    env = {**os.environ, "HITL_AUTO_APPROVE": "1", "PYTHONUNBUFFERED": "1"}

    def _log(line: str):
        _job["phases"][phase]["logs"].append(line)
        _emit({"type": "log", "phase": phase, "message": line})

    proc_ref: list[_subprocess.Popen | None] = [None]

    def _run_sync() -> int:
        proc = _subprocess.Popen(
            cmd,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            env=env,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        proc_ref[0] = proc
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                _log(stripped)
        proc.stdout.close()
        proc.wait()
        return proc.returncode

    try:
        returncode = await asyncio.to_thread(_run_sync)
        success = returncode == 0
        if not success:
            _log(f"[ERROR] Process exited with code {returncode}")
    except asyncio.CancelledError:
        if proc_ref[0]:
            proc_ref[0].terminate()
        raise
    except Exception as exc:
        import traceback
        _log(f"[ERROR] {type(exc).__name__}: {exc}")
        for tb_line in traceback.format_exc().splitlines():
            if tb_line.strip():
                _log(f"  {tb_line}")
        success = False

    status = "done" if success else "error"
    _job["phases"][phase]["status"] = status
    _emit({"type": "phase_end", "phase": phase, "status": status})
    return success


async def _pipeline(from_phase: int, prompt: str) -> None:
    global _pipeline_done
    _job["active"] = True
    _pipeline_done = False

    try:
        for phase in range(from_phase, 4):
            ok = await _run_phase(phase, prompt)
            if not ok:
                break
    except asyncio.CancelledError:
        pass
    finally:
        _job["active"] = False
        _pipeline_done = True
        _emit({"type": "done", "has_video": os.path.exists(FINAL_VIDEO)})


# ── Request models ────────────────────────────────────────────────────────────

class StartReq(BaseModel):
    prompt: str


class RerunReq(BaseModel):
    phase: int


# ── Routes ────────────────────────────────────────────────────────────────────

async def _cancel_active():
    global _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
        try:
            await _active_task
        except asyncio.CancelledError:
            pass


@app.post("/api/start")
async def start(req: StartReq):
    global _job, _events, _pipeline_done, _active_task
    await _cancel_active()
    _job = _new_job(req.prompt)
    _events.clear()
    _pipeline_done = False
    _active_task = asyncio.create_task(_pipeline(1, req.prompt))
    return {"job_id": _job["id"]}


@app.post("/api/rerun")
async def rerun(req: RerunReq):
    global _events, _pipeline_done, _active_task
    if req.phase not in (1, 2, 3):
        raise HTTPException(400, "phase must be 1, 2, or 3")
    await _cancel_active()

    # Reset requested phase and all after it
    for p in range(req.phase, 4):
        _job["phases"][p] = _blank_phase()

    # Keep only events that belong to earlier phases
    kept = [e for e in _events if e.get("phase", 0) < req.phase or "phase" not in e]
    _events.clear()
    _events.extend(kept)

    _pipeline_done = False
    _active_task = asyncio.create_task(_pipeline(req.phase, _job["prompt"]))
    return {"ok": True}


@app.get("/api/stream")
async def stream_sse():
    async def gen() -> AsyncIterator[str]:
        cursor = 0
        idle_ticks = 0
        while True:
            # Drain all pending events from cursor
            while cursor < len(_events):
                yield f"data: {json.dumps(_events[cursor])}\n\n"
                cursor += 1
                idle_ticks = 0

            if _pipeline_done and cursor >= len(_events):
                break

            await asyncio.sleep(0.15)
            idle_ticks += 1
            # Send SSE comment keepalive every ~20 s to prevent proxy timeouts
            if idle_ticks % 133 == 0:
                yield ": keepalive\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status")
async def status():
    return {"job": _job, "has_video": os.path.exists(FINAL_VIDEO)}


@app.get("/api/outputs/video")
async def get_video():
    if not os.path.exists(FINAL_VIDEO):
        raise HTTPException(404, "Video not ready yet")
    return FileResponse(FINAL_VIDEO, media_type="video/mp4", filename="final_video.mp4")


@app.get("/api/health")
async def health():
    return {"ok": True}


# Serve all generated outputs (images, audio previews, etc.)
_out_dir = os.path.join(PROJECT_ROOT, "data", "outputs")
os.makedirs(_out_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=_out_dir), name="outputs")
