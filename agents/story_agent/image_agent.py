"""
agents/story_agent/image_agent.py
───────────────────────────────────
Image Synthesizer Agent — generates character portrait images.
"""

import io
import os
import time
from typing import Any, Dict

import requests
from PIL import Image

from agents.base import BaseAgent
from config import HF_API_KEY, HF_IMAGE_MODEL, IMAGES_DIR
from shared.utils.vector_store import memory


class ImageSynthesizerAgent(BaseAgent):
    name      = "ImageSynthesizerAgent"
    tool_tags = ["image", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Generating character reference images...")
        characters = state.get("characters", [])
        if not characters:
            return {**state, "images": [], "status": "images_skipped"}

        os.makedirs(IMAGES_DIR, exist_ok=True)
        image_records = []

        for char in characters:
            name   = char.get("name", "unknown")
            prompt = char.get("image_prompt", f"Portrait of {name}, cinematic, high detail")
            print(f"[{self.name}] Generating image for: {name}")

            result     = self.invoke_tool("generate_image", {
                "character_name": name, "prompt": prompt,
                "style": char.get("reference_style", "cinematic realism")
            })
            image_path = result.data if result.success and result.data else self._generate_via_hf(name, prompt)

            if image_path:
                image_records.append({"character": name, "path": image_path, "prompt": prompt})
                self.invoke_tool("commit_memory", {
                    "key":      f"image:{name.lower().replace(' ','_')}",
                    "data":     {"character": name, "path": image_path},
                    "metadata": {"type": "image"}
                })
                memory.store_image_ref(name, image_path)
                print(f"[{self.name}] Image saved: {image_path}")
            else:
                placeholder = self._create_placeholder(name)
                image_records.append({"character": name, "path": placeholder, "prompt": prompt,
                                       "note": "placeholder - add HF_API_KEY for real images"})
                print(f"[{self.name}] Placeholder created for {name}.")

        print(f"[{self.name}] Processed {len(image_records)} character images.")
        return {**state, "images": image_records, "status": "images_ready"}

    def _generate_via_hf(self, char_name: str, prompt: str) -> str | None:
        if not HF_API_KEY:
            return None
        api_url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt}
        for attempt in range(3):
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 503:
                    wait = resp.json().get("estimated_time", 20) if resp.content else 20
                    print(f"[{self.name}] Model loading, waiting {wait}s...")
                    time.sleep(min(wait, 60))
                    continue
                if resp.status_code != 200:
                    time.sleep(5); continue
                img  = Image.open(io.BytesIO(resp.content))
                safe = char_name.lower().replace(" ", "_")
                path = os.path.join(IMAGES_DIR, f"{safe}.png")
                img.save(path)
                return path
            except Exception as e:
                print(f"[{self.name}] Attempt {attempt+1} failed: {e}")
                time.sleep(5)
        return None

    def _create_placeholder(self, char_name: str) -> str:
        try:
            from PIL import ImageDraw
            img  = Image.new("RGB", (512, 512), color=(40, 40, 60))
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 501, 501], outline=(100, 150, 200), width=3)
            draw.text((256, 300), char_name,                        fill=(200, 200, 220), anchor="mm")
            draw.text((256, 340), "[ Add HF_API_KEY for AI image ]", fill=(120, 120, 140), anchor="mm")
            safe = char_name.lower().replace(" ", "_")
            path = os.path.join(IMAGES_DIR, f"{safe}_placeholder.png")
            img.save(path)
            return path
        except Exception as e:
            print(f"[{self.name}] Placeholder creation failed: {e}")
            return ""
