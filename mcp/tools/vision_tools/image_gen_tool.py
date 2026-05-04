"""
mcp/tools/vision_tools/image_gen_tool.py
─────────────────────────────────────────
MCP tool: generate_image
Primary:  Pollinations.ai  (free, no key needed)
Fallback: HuggingFace Inference API (requires HF_API_KEY)
"""

from mcp.tool_registry import mcp, MCPToolSchema


def _pollinations(prompt: str, width: int, height: int) -> bytes | None:
    import urllib.parse, requests
    encoded = urllib.parse.quote(prompt)
    url     = (f"https://image.pollinations.ai/prompt/{encoded}"
               f"?width={width}&height={height}&model=flux&nologo=true")
    try:
        resp = requests.get(url, timeout=90)
        if resp.status_code == 200 and resp.content:
            return resp.content
        print(f"[MCP:generate_image] Pollinations returned {resp.status_code}")
    except Exception as e:
        print(f"[MCP:generate_image] Pollinations error: {e}")
    return None


def _hf(prompt: str) -> bytes | None:
    import time, requests
    from config import HF_API_KEY, HF_IMAGE_MODEL
    if not HF_API_KEY:
        return None
    url     = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
        if resp.status_code == 503:
            time.sleep(min(resp.json().get("estimated_time", 30) if resp.content else 30, 60))
            resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
        if resp.status_code == 200 and resp.content:
            return resp.content
        print(f"[MCP:generate_image] HF error {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"[MCP:generate_image] HF error: {e}")
    return None


def _generate_image(character_name: str, prompt: str, style: str = "cinematic realism") -> str | None:
    import io, os
    from PIL import Image
    from config import IMAGES_DIR

    full_prompt = f"{prompt}, {style}, highly detailed, professional portrait, face visible"

    raw = _pollinations(full_prompt, 512, 512) or _hf(full_prompt)
    if not raw:
        return None

    os.makedirs(IMAGES_DIR, exist_ok=True)
    img  = Image.open(io.BytesIO(raw)).convert("RGB")
    safe = character_name.lower().replace(" ", "_")
    path = os.path.join(IMAGES_DIR, f"{safe}.png")
    img.save(path)
    print(f"[MCP:generate_image] Saved {path}")
    return path


mcp.register(
    MCPToolSchema(
        name="generate_image",
        description="Generates a character portrait via Pollinations.ai or HuggingFace.",
        input_schema={
            "character_name": {"type": "string"},
            "prompt":         {"type": "string", "description": "Image generation prompt"},
            "style":          {"type": "string", "default": "cinematic realism"}
        },
        tags=["image", "generation"]
    ),
    handler=_generate_image
)
