"""
agents/character_designer.py
─────────────────────────────
Character Designer Agent
  Role: Extract and formalize character identities from the script.
  Output: character_db.json + populates characters[] in PhaseOneOutput.
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
    "voice_profile": {
      "voice_style": "deep | soft | raspy | high-pitched | neutral | authoritative | whispery",
      "speaking_speed": "slow | normal | fast",
      "pitch": "low | medium | high",
      "emotion_range": ["neutral", "angry", "sad"],
      "tts_description": "One sentence describing how this character sounds."
    },
    "reference_style": "cinematic realism | anime | noir",
    "image_prompt": "Detailed Stable Diffusion prompt for this character portrait (40+ words).",
    "scenes_appeared": [1, 2, 3]
  }
]

Rules:
- Include EVERY named character who has dialogue or action.
- voice_style must be ONE of: deep, soft, raspy, high-pitched, neutral, authoritative, whispery.
- speaking_speed must be ONE of: slow, normal, fast.
- pitch must be ONE of: low, medium, high.
- emotion_range: list emotions this character expresses (neutral, angry, sad, excited, fearful, surprised, disgusted, happy).
- tts_description: a single sentence describing the voice for a TTS engine.
- image_prompt must be 40+ words, portrait-focused, photorealistic.
- Output ONLY the JSON array. No markdown, no extra text.
"""


class CharacterDesignerAgent(BaseAgent):
    name = "CharacterDesignerAgent"
    tool_tags = ["memory", "character"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        script = state.get("script", {})

        if not script:
            return {**state, "characters": [], "status": "error",
                    "error": "No script found for character extraction."}

        # ── Step 1: Extract characters via LLM ───────────────────────────────
        print(f"[{self.name}] [Reasoning 1/4] Extracting characters from script...")
        script_text = self._script_to_text(script)
        raw = self.chat(CHARACTER_SYSTEM, f"Extract characters from this screenplay:\n\n{script_text}")
        characters = self.parse_json(raw)

        if not isinstance(characters, list):
            print(f"[{self.name}] JSON parse failed, retrying...")
            raw = self.chat(CHARACTER_SYSTEM, f"Fix and return ONLY JSON array:\n{raw}", temperature=0.2)
            characters = self.parse_json(raw)

        if not isinstance(characters, list):
            return {**state, "characters": [], "status": "error",
                    "error": "Character extraction failed."}

        # ── Step 2: Fill missing voice_profile defaults ───────────────────────
        print(f"[{self.name}] [Reasoning 2/4] Filling voice profile defaults...")
        characters = [self._fill_voice_defaults(c) for c in characters]

        # ── Step 3: Query stock footage references via MCP ────────────────────
        print(f"[{self.name}] [Reasoning 3/4] Querying stock references (MCP)...")
        for char in characters:
            result = self.invoke_tool("query_stock_footage", {
                "character_name": char["name"],
                "style": char.get("reference_style", "cinematic")
            })
            if result.success:
                char["stock_reference"] = result.data

        # ── Step 4: Pydantic validation of full PhaseOneOutput ────────────────
        print(f"[{self.name}] [Reasoning 4/4] Running Pydantic validation of PhaseOneOutput...")
        from schema import validate_phase1_output

        story_block = script.get("story", {})
        scenes_block = script.get("scenes", [])

        # Remove any scene character references that aren't in the extracted roster
        known_names = {c["name"] for c in characters}
        for scene in scenes_block:
            original = scene.get("characters", [])
            filtered = [n for n in original if n in known_names]
            if not filtered:
                filtered = [characters[0]["name"]] if characters else original
            scene["characters"] = filtered

        validated, errors = validate_phase1_output({
            "story":      story_block,
            "scenes":     scenes_block,
            "characters": characters
        })

        if errors:
            print(f"[{self.name}] ⚠ Pydantic validation errors: {errors}")
            # Auto-fix common issues and retry once
            characters = self._fix_validation_errors(characters, errors)
            validated, errors = validate_phase1_output({
                "story":      story_block,
                "scenes":     scenes_block,
                "characters": characters
            })
            if errors:
                print(f"[{self.name}] ⚠ Remaining validation errors (non-blocking): {errors}")
            else:
                print(f"[{self.name}] ✓ Pydantic validation passed after auto-fix.")
        else:
            print(f"[{self.name}] ✓ Pydantic validation passed.")

        # ── Commit to memory ──────────────────────────────────────────────────
        self.invoke_tool("commit_memory", {
            "key": "characters:all",
            "data": characters,
            "metadata": {"type": "character_list", "count": str(len(characters))}
        })
        memory.store_characters(characters)

        print(f"[{self.name}] ✓ Extracted {len(characters)} characters: {[c['name'] for c in characters]}")
        return {**state, "characters": characters, "status": "characters_ready"}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fill_voice_defaults(self, char: dict) -> dict:
        valid_voice_styles = {"deep", "soft", "raspy", "high-pitched", "neutral", "authoritative", "whispery"}
        valid_speeds       = {"slow", "normal", "fast"}
        valid_pitches      = {"low", "medium", "high"}
        valid_emotions     = {"neutral", "angry", "sad", "excited", "fearful", "surprised", "disgusted", "happy"}

        vp = char.get("voice_profile")
        if not isinstance(vp, dict):
            char["voice_profile"] = {}
            vp = char["voice_profile"]

        if vp.get("voice_style") not in valid_voice_styles:
            vp["voice_style"] = "neutral"
        if vp.get("speaking_speed") not in valid_speeds:
            vp["speaking_speed"] = "normal"
        if vp.get("pitch") not in valid_pitches:
            vp["pitch"] = "medium"

        # Filter emotion_range to only valid values
        raw_emotions = vp.get("emotion_range", [])
        if isinstance(raw_emotions, list):
            vp["emotion_range"] = [e for e in raw_emotions if e in valid_emotions] or ["neutral"]
        else:
            vp["emotion_range"] = ["neutral"]

        if not vp.get("tts_description") or len(vp.get("tts_description", "")) < 10:
            vp["tts_description"] = (
                f"A {vp['voice_style']} voice speaking at {vp['speaking_speed']} speed "
                f"with {vp['pitch']} pitch."
            )
        return char

    def _fix_validation_errors(self, characters: list, errors: list) -> list:
        """Auto-fix common Pydantic validation errors."""
        for char in characters:
            # Ensure image_prompt is long enough
            if len(char.get("image_prompt", "").split()) < 30:
                char["image_prompt"] = (
                    f"Portrait of {char['name']}, {char.get('role','character')}, "
                    f"{char.get('age_range','adult')}, {char.get('gender','person')}, "
                    f"cinematic realism, photorealistic, professional photography, "
                    f"studio lighting, high detail, 4K render, sharp focus."
                )
            # Ensure scenes_appeared is a list of ints
            sa = char.get("scenes_appeared", [1])
            if isinstance(sa, list):
                char["scenes_appeared"] = [int(s) for s in sa if str(s).isdigit()] or [1]
            else:
                char["scenes_appeared"] = [1]

            # Ensure role is valid
            valid_roles = {"protagonist", "antagonist", "supporting", "minor"}
            if char.get("role") not in valid_roles:
                char["role"] = "supporting"

            # Ensure personality_traits is non-empty list
            if not isinstance(char.get("personality_traits"), list) or not char["personality_traits"]:
                char["personality_traits"] = ["determined"]

            # Ensure appearance has all fields
            app = char.setdefault("appearance", {})
            app.setdefault("build", "average")
            app.setdefault("hair", "unknown")
            app.setdefault("eyes", "unknown")
            app.setdefault("clothing_style", "casual")
            app.setdefault("distinguishing_features", "")

        return characters

    def _script_to_text(self, script: Dict) -> str:
        story = script.get("story", {})
        lines = [
            f"TITLE: {story.get('title', '')}",
            f"GENRE: {story.get('genre', '')}",
            f"SYNOPSIS: {story.get('synopsis', '')}",
            ""
        ]
        for scene in script.get("scenes", []):
            lines.append(f"SCENE {scene['scene_id']}: {scene.get('location', '')} - {scene.get('time_of_day', '')}")
            lines.append(f"MOOD: {scene.get('mood', '')} | TONE: {scene.get('tone', '')}")
            lines.append(f"ACTION: {scene.get('action', '')}")
            for dlg in scene.get("dialogue", []):
                lines.append(f"{dlg['speaker']} [{dlg.get('emotion', 'neutral')}]: {dlg['line']}")
            lines.append("")
        return "\n".join(lines)