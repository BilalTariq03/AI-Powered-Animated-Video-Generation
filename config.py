import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY   = os.getenv("HF_API_KEY", "")      # HuggingFace (free image gen)

# ── Model Settings ────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"          # Best free Groq model
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# ── Paths ─────────────────────────────────────────────────
OUTPUT_DIR  = "output"
IMAGES_DIR  = f"{OUTPUT_DIR}/images"
MANIFEST    = f"{OUTPUT_DIR}/scene_manifest.json"
CHAR_DB     = f"{OUTPUT_DIR}/character_db.json"

# ── ChromaDB ──────────────────────────────────────────────
CHROMA_PATH = ".chroma_db"
COLLECTION  = "writers_room_memory"
