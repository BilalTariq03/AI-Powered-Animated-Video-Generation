"""
tools.py
────────
Registers all MCP tools into the global registry.
Import this ONCE at startup (in main.py) before any agents run.

This is the ONLY place tool implementations live.
Agents discover and invoke tools via mcp_registry — never directly.
"""

from mcp_registry import mcp, MCPToolSchema
from memory.vector_store import memory


# ── Tool: generate_script_segment ────────────────────────────────────────────

def _generate_script_segment(prompt: str, num_scenes: int = 5) -> str:
    """
    Calls the Groq LLM to generate a screenplay segment.
    Returns raw LLM output (JSON string).
    """
    from groq import Groq
    from config import GROQ_API_KEY, GROQ_MODEL

    client = Groq(api_key=GROQ_API_KEY)
    system = """You are a professional screenplay writer. 
Generate a structured multi-scene screenplay in valid JSON ONLY.
Schema: {"title":"...","genre":"...","scenes":[{"scene_id":1,"location":"...","time_of_day":"DAY","characters":[],"action":"...","dialogue":[{"speaker":"...","line":"...","visual_cue":"..."}],"visual_notes":"..."}]}
Output ONLY JSON. No markdown, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Write a {num_scenes}-scene screenplay for: {prompt}"}
        ],
        temperature=0.8,
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()


mcp.register(
    MCPToolSchema(
        name="generate_script_segment",
        description="Generates a multi-scene screenplay from a story prompt using an LLM.",
        input_schema={
            "prompt": {"type": "string", "description": "Story premise or prompt"},
            "num_scenes": {"type": "integer", "description": "Number of scenes to generate", "default": 5}
        },
        tags=["script", "generation"]
    ),
    handler=_generate_script_segment
)


# ── Tool: commit_memory ───────────────────────────────────────────────────────

def _commit_memory(key: str, data, metadata: dict = None) -> str:
    """Stores data in the persistent vector memory store."""
    doc_id = memory.store(key, data, metadata or {})
    return f"Stored at key='{key}' with id={doc_id}"


mcp.register(
    MCPToolSchema(
        name="commit_memory",
        description="Persists agent output (scripts, characters, images) to vector memory.",
        input_schema={
            "key": {"type": "string", "description": "Unique storage key"},
            "data": {"type": "any", "description": "Data to store"},
            "metadata": {"type": "object", "description": "Optional metadata tags"}
        },
        tags=["memory", "storage"]
    ),
    handler=_commit_memory
)


# ── Tool: query_stock_footage ─────────────────────────────────────────────────

def _query_stock_footage(character_name: str, style: str = "cinematic") -> dict:
    """
    Searches memory for existing visual references for a character.
    Returns reference metadata if found.
    """
    results = memory.query(f"character {character_name} {style}", n_results=3)
    if results:
        return {"found": True, "references": results}
    return {"found": False, "references": [], "note": "No prior visual references found."}


mcp.register(
    MCPToolSchema(
        name="query_stock_footage",
        description="Searches memory for existing visual/stock references for a character.",
        input_schema={
            "character_name": {"type": "string"},
            "style": {"type": "string", "default": "cinematic"}
        },
        tags=["character", "memory", "image"]
    ),
    handler=_query_stock_footage
)


# ── Tool: generate_image ──────────────────────────────────────────────────────

def _generate_image(character_name: str, prompt: str, style: str = "cinematic realism") -> str | None:
    """
    Calls HuggingFace Inference API to generate a character portrait.
    Returns file path of saved image, or None if unavailable.
    """
    import io, os, time
    import requests
    from PIL import Image
    from config import HF_API_KEY, HF_IMAGE_MODEL, IMAGES_DIR

    if not HF_API_KEY:
        return None  # Caller will create placeholder

    full_prompt = f"{prompt}, {style}, highly detailed, professional portrait"
    api_url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": full_prompt}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code == 503:
            try:
                wait_time = response.json().get("estimated_time", 30)
            except Exception:
                wait_time = 30
            time.sleep(wait_time)
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"[MCP:generate_image] HF error {response.status_code}: {response.text[:500]}")
            return None

        os.makedirs(IMAGES_DIR, exist_ok=True)
        img = Image.open(io.BytesIO(response.content))
        safe = character_name.lower().replace(" ", "_")
        path = os.path.join(IMAGES_DIR, f"{safe}.png")
        img.save(path)
        return path
    except Exception as e:
        print(f"[MCP:generate_image] Error: {e}")
        return None


mcp.register(
    MCPToolSchema(
        name="generate_image",
        description="Generates a character portrait image via Stable Diffusion (HuggingFace API).",
        input_schema={
            "character_name": {"type": "string"},
            "prompt": {"type": "string", "description": "Stable Diffusion prompt"},
            "style": {"type": "string", "default": "cinematic realism"}
        },
        tags=["image", "generation"]
    ),
    handler=_generate_image
)

print(f"\n[MCP] Registry initialized with {len(mcp.list_tools())} tools: {mcp.list_tools()}")
