"""
agents/image_synthesizer.py
────────────────────────────
Image Synthesizer Agent
  Role: Generate character portrait images using Stable Diffusion.
  Backend: HuggingFace Inference API (FREE — just needs HF_API_KEY).
  MCP Tools: generate_image (discovered at runtime)
"""

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image

from agents.base import BaseAgent
from config import HF_API_KEY, HF_IMAGE_MODEL, IMAGES_DIR
from memory.vector_store import memory

from mcp_registry import mcp

class ImageSynthesizerAgent(BaseAgent):
    name = "ImageSynthesizerAgent"
    tool_tags = ["image", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Generating character reference images...")
        characters = state.get("characters", [])

        if not characters:
            return {**state, "images": [], "status": "images_skipped"}

        os.makedirs(IMAGES_DIR, exist_ok=True)
        image_records = []

        for char in characters:
            name = char.get("name", "unknown")
            prompt = char.get("image_prompt", f"Portrait of {name}, cinematic, high detail")

            print(f"[{self.name}] Generating image for: {name}")

            # ── Invoke MCP tool ───────────────────────────────────────────────
            result = self.invoke_tool("generate_image", {
                "character_name": name,
                "prompt": prompt,
                "style": char.get("reference_style", "cinematic realism")
            })

            image_path = None

            if result.success and result.data:
                image_path = result.data  # MCP tool returns file path
            else:
                # Direct HuggingFace API call
                image_path = self._generate_via_hf(name, prompt)

            if image_path:
                image_records.append({
                    "character": name,
                    "path": image_path,
                    "prompt": prompt
                })
                # Commit image reference to memory
                self.invoke_tool("commit_memory", {
                    "key": f"image:{name.lower().replace(' ', '_')}",
                    "data": {"character": name, "path": image_path},
                    "metadata": {"type": "image"}
                })
                memory.store_image_ref(name, image_path)
                print(f"[{self.name}] ✓ Image saved: {image_path}")
            else:
                # Create placeholder if generation failed
                placeholder = self._create_placeholder(name)
                image_records.append({
                    "character": name,
                    "path": placeholder,
                    "prompt": prompt,
                    "note": "placeholder — add HF_API_KEY for real images"
                })
                print(f"[{self.name}] ⚠ Placeholder created for {name} (no HF_API_KEY set).")

        print(f"[{self.name}] ✓ Processed {len(image_records)} character images.")
        return {**state, "images": image_records, "status": "images_ready"}

    def _generate_via_hf(self, char_name: str, prompt: str) -> str | None:
        """Call HuggingFace Inference API for Stable Diffusion image generation."""
        if not HF_API_KEY:
            return None

        api_url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt}

        for attempt in range(3):
            try:
                print(f"[{self.name}] Calling HF API (attempt {attempt+1}/3): {api_url}")
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                print(f"[{self.name}] HF response status: {response.status_code}")

                if response.status_code == 503:
                    # Model loading — wait and retry
                    try:
                        wait_time = response.json().get("estimated_time", 20)
                    except Exception:
                        wait_time = 20
                    print(f"[{self.name}] Model loading, waiting {wait_time:.0f}s...")
                    time.sleep(min(wait_time, 60))
                    continue

                if response.status_code != 200:
                    print(f"[{self.name}] HF API error {response.status_code}: {response.text[:500]}")
                    time.sleep(5)
                    continue  # retry instead of returning None immediately

                # Save image
                image = Image.open(io.BytesIO(response.content))
                safe_name = char_name.lower().replace(" ", "_")
                path = os.path.join(IMAGES_DIR, f"{safe_name}.png")
                image.save(path)
                print(f"[{self.name}] Image saved: {path}")
                return path

            except Exception as e:
                print(f"[{self.name}] Attempt {attempt+1} failed: {e}")
                time.sleep(5)

        print(f"[{self.name}] All 3 attempts failed for {char_name}.")
        return None

    def _create_placeholder(self, char_name: str) -> str:
        """Create a labeled placeholder image using Pillow."""
        try:
            from PIL import ImageDraw, ImageFont

            img = Image.new("RGB", (512, 512), color=(40, 40, 60))
            draw = ImageDraw.Draw(img)

            # Draw border
            draw.rectangle([10, 10, 501, 501], outline=(100, 150, 200), width=3)

            # Character icon (simple silhouette placeholder text)
            draw.text((256, 200), "👤", fill=(180, 180, 200), anchor="mm")
            draw.text((256, 300), char_name, fill=(200, 200, 220), anchor="mm")
            draw.text((256, 340), "[ Add HF_API_KEY for AI image ]",
                      fill=(120, 120, 140), anchor="mm")

            safe_name = char_name.lower().replace(" ", "_")
            path = os.path.join(IMAGES_DIR, f"{safe_name}_placeholder.png")
            img.save(path)
            return path
        except Exception as e:
            print(f"[{self.name}] Placeholder creation failed: {e}")
            return ""
