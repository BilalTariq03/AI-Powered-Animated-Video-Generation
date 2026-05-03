"""
agents/character_designer.py
─────────────────────────────
Character Designer Agent
  Role: Extract and formalize character identities from the script.
  Output: character_db.json with name, traits, appearance, style.
  MCP Tools: commit_memory, query_stock_footage
"""

from typing import Any, Dict, List

from agents.base import BaseAgent
from memory.vector_store import memory

from mcp_registry import mcp

CHARACTER_SYSTEM = """
You are a character design director. Extract all characters from the given screenplay
and return ONLY a JSON array of character objects.

Schema:
[
  {
    "name": "Full Character Name",
    "role": "protagonist | antagonist | supporting | minor",
    "age_range": "e.g. 20s, 30s, teen, elderly",
    "gender": "male | female | non-binary | unknown",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "appearance": {
      "build": "slim | athletic | stocky | average",
      "hair": "color and style description",
      "eyes": "color",
      "clothing_style": "description of typical clothing",
      "distinguishing_features": "any unique features"
    },
    "reference_style": "Art style reference e.g. 'cinematic realism', 'anime', 'noir'",
    "image_prompt": "Detailed Stable Diffusion prompt for this character portrait",
    "scenes_appeared": [1, 2, 3]
  }
]

Rules:
- Include EVERY named character who has dialogue or action.
- image_prompt must be detailed (40+ words), portrait-focused, photorealistic style.
- Maintain identity CONSISTENCY — same character must look the same across all scenes.
- Output ONLY the JSON array. No markdown, no extra text.
"""


class CharacterDesignerAgent(BaseAgent):
    name = "CharacterDesignerAgent"
    tool_tags = ["memory", "character"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        print(f"[{self.name}] [Reasoning 1/3] Reading script and identifying all characters...")
        script = state.get("script", {})

        if not script:
            return {**state, "characters": [], "status": "error", "error": "No script found for character extraction."}

        # ── Reasoning 1: LLM extraction ───────────────────────────────────────
        script_text = self._script_to_text(script)
        raw = self.chat(CHARACTER_SYSTEM, f"Extract characters from this screenplay:\n\n{script_text}")
        characters = self.parse_json(raw)

        if not isinstance(characters, list):
            print(f"[{self.name}] JSON parse failed, retrying...")
            raw = self.chat(CHARACTER_SYSTEM, f"Fix and return ONLY JSON array:\n{raw}", temperature=0.2)
            characters = self.parse_json(raw)

        if not isinstance(characters, list):
            return {**state, "characters": [], "status": "error", "error": "Character extraction failed."}

        print(f"[{self.name}] [Reasoning 2/3] Checking memory for existing visual references (MCP)...")
        # ── Reasoning 2: Query stock footage references via MCP ───────────────
        for char in characters:
            result = self.invoke_tool("query_stock_footage", {
                "character_name": char["name"],
                "style": char.get("reference_style", "cinematic")
            })
            if result.success:
                char["stock_reference"] = result.data

        print(f"[{self.name}] [Reasoning 3/3] Committing character identities to shared memory...")
        # ── Reasoning 3: Commit to memory ────────────────────────────────────
        self.invoke_tool("commit_memory", {
            "key": "characters:all",
            "data": characters,
            "metadata": {"type": "character_list", "count": str(len(characters))}
        })
        memory.store_characters(characters)

        print(f"[{self.name}] ✓ Extracted {len(characters)} characters: {[c['name'] for c in characters]}")
        return {**state, "characters": characters, "status": "characters_ready"}

    def _script_to_text(self, script: Dict) -> str:
        """Convert structured script dict to readable text for LLM."""
        lines = [f"TITLE: {script.get('title', '')}", f"GENRE: {script.get('genre', '')}", ""]
        for scene in script.get("scenes", []):
            lines.append(f"SCENE {scene['scene_id']}: {scene.get('location', '')} - {scene.get('time_of_day', '')}")
            lines.append(f"ACTION: {scene.get('action', '')}")
            for dlg in scene.get("dialogue", []):
                lines.append(f"{dlg['speaker']}: {dlg['line']}")
            lines.append("")
        return "\n".join(lines)
