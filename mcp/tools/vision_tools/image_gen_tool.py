"""
mcp/tools/vision_tools/image_gen_tool.py
─────────────────────────────────────────
MCP tool: generate_image
Calls HuggingFace Inference API (Stable Diffusion) to produce character portraits.
"""

from mcp.tool_registry import mcp, MCPToolSchema


def _generate_image(character_name: str, prompt: str, style: str = "cinematic realism") -> str | None:
    import io, os, time
    import requests
    from PIL import Image
    from config import HF_API_KEY, HF_IMAGE_MODEL, IMAGES_DIR

    if not HF_API_KEY:
        return None

    full_prompt = f"{prompt}, {style}, highly detailed, professional portrait"
    api_url     = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
    headers     = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload     = {"inputs": full_prompt}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code == 503:
            wait_time = response.json().get("estimated_time", 30) if response.content else 30
            time.sleep(wait_time)
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"[MCP:generate_image] HF error {response.status_code}")
            return None
        os.makedirs(IMAGES_DIR, exist_ok=True)
        img  = Image.open(io.BytesIO(response.content))
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
        description="Generates a character portrait via Stable Diffusion (HuggingFace).",
        input_schema={
            "character_name": {"type": "string"},
            "prompt":         {"type": "string", "description": "Stable Diffusion prompt"},
            "style":          {"type": "string", "default": "cinematic realism"}
        },
        tags=["image", "generation"]
    ),
    handler=_generate_image
)
