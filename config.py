import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
HF_API_KEY         = os.getenv("HF_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ── Model Settings ────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant"
# llama-3.3-70b-versatile
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# ── Directory Layout ──────────────────────────────────────────────────────────
OUTPUT_DIR = "data/outputs"
IMAGES_DIR = f"{OUTPUT_DIR}/images"

# Phase 1 outputs
MANIFEST    = f"{OUTPUT_DIR}/scene_manifest.json"
CHAR_DB     = f"{OUTPUT_DIR}/character_db.json"

# Phase handoff files
PHASE2_HANDOFF = f"{OUTPUT_DIR}/phase2_audio_handoff.json"
PHASE3_HANDOFF = f"{OUTPUT_DIR}/phase3_video_handoff.json"

# Phase 2 audio outputs
AUDIO_DIR       = f"{OUTPUT_DIR}/audio"
DIALOGUE_DIR    = f"{AUDIO_DIR}/dialogue"
BGM_DIR         = f"{AUDIO_DIR}/bgm"
TIMING_MANIFEST = f"{AUDIO_DIR}/timing_manifest.json"

# Phase 3 video outputs
VIDEO_DIR    = f"{OUTPUT_DIR}/video"
SCENES_DIR   = f"{VIDEO_DIR}/scenes"
LIPSYNC_DIR  = f"{VIDEO_DIR}/lipsync"
FINAL_VIDEO  = f"{VIDEO_DIR}/final_video.mp4"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH = ".chroma_db"
COLLECTION  = "writers_room_memory"
