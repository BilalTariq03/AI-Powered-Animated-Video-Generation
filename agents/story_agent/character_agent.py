"""
agents/story_agent/character_agent.py
───────────────────────────────────────
Character Designer Agent — extracts character identities from the screenplay.
"""

from typing import Any, Dict

from agents.base import BaseAgent
from shared.utils.vector_store import memory

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
      "hair": "color and style",
      "eyes": "color",
      "clothing_style": "description",
      "distinguishing_features": "any unique features"
    },
    "voice_profile": {
      "voice_style": "deep | soft | raspy | high-pitched | neutral | authoritative | whispery",
      "speaking_speed": "slow | normal | fast",
      "pitch": "low | medium | high",
      "emotion_range": ["neutral", "angry"],
      "tts_description": "One sentence describing how this character sounds."
    },
    "reference_style": "cinematic realism | anime | noir",
    "image_prompt": "Detailed 40+ word portrait prompt.",
    "scenes_appeared": [1, 2, 3]
  }
]

Rules:
- Include EVERY named character with dialogue or action.
- voice_style must be one of: deep, soft, raspy, high-pitched, neutral, authoritative, whispery.
- Output ONLY the JSON array.
"""


class CharacterDesignerAgent(BaseAgent):
    name      = "CharacterDesignerAgent"
    tool_tags = ["memory", "character"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        script = state.get("script", {})
        if not script:
            return {**state, "characters": [], "status": "error",
                    "error": "No script found for character extraction."}

        print(f"[{self.name}] [Reasoning 1/4] Extracting characters from script...")
        raw        = self.chat(CHARACTER_SYSTEM, f"Extract characters:\n\n{self._script_to_text(script)}")
        characters = self.parse_json(raw)
        if not isinstance(characters, list):
            raw        = self.chat(CHARACTER_SYSTEM, f"Fix and return ONLY JSON array:\n{raw}", temperature=0.2)
            characters = self.parse_json(raw)
        if not isinstance(characters, list):
            return {**state, "characters": [], "status": "error", "error": "Character extraction failed."}

        print(f"[{self.name}] [Reasoning 2/4] Filling voice profile defaults...")
        characters = [self._fill_voice_defaults(c) for c in characters]

        print(f"[{self.name}] [Reasoning 3/4] Querying stock references (MCP)...")
        for char in characters:
            result = self.invoke_tool("query_stock_footage", {
                "character_name": char["name"],
                "style":          char.get("reference_style", "cinematic")
            })
            if result.success:
                char["stock_reference"] = result.data

        print(f"[{self.name}] [Reasoning 4/4] Running Pydantic validation of PhaseOneOutput...")
        from shared.schemas.schema import validate_phase1_output

        story_block  = script.get("story", {})
        scenes_block = script.get("scenes", [])

        # Strip unknown character refs from scenes before validation
        known_names = {c["name"] for c in characters}
        for scene in scenes_block:
            filtered = [n for n in scene.get("characters", []) if n in known_names]
            scene["characters"] = filtered or ([characters[0]["name"]] if characters else scene.get("characters", []))

        _, errors = validate_phase1_output({"story": story_block, "scenes": scenes_block, "characters": characters})
        if errors:
            print(f"[{self.name}] Pydantic errors: {errors}")
            characters = self._fix_validation_errors(characters, errors)
            _, errors  = validate_phase1_output({"story": story_block, "scenes": scenes_block, "characters": characters})
            if errors:
                print(f"[{self.name}] Remaining (non-blocking): {errors}")
            else:
                print(f"[{self.name}] Pydantic passed after auto-fix.")
        else:
            print(f"[{self.name}] Pydantic validation passed.")

        self.invoke_tool("commit_memory", {
            "key":      "characters:all",
            "data":     characters,
            "metadata": {"type": "character_list", "count": str(len(characters))}
        })
        memory.store_characters(characters)

        print(f"[{self.name}] Extracted {len(characters)} characters: {[c['name'] for c in characters]}")
        return {**state, "characters": characters, "status": "characters_ready"}

    def _fill_voice_defaults(self, char: dict) -> dict:
        valid_styles   = {"deep","soft","raspy","high-pitched","neutral","authoritative","whispery"}
        valid_speeds   = {"slow","normal","fast"}
        valid_pitches  = {"low","medium","high"}
        valid_emotions = {"neutral","angry","sad","excited","fearful","surprised","disgusted","happy"}

        vp = char.get("voice_profile")
        if not isinstance(vp, dict):
            char["voice_profile"] = {}
            vp = char["voice_profile"]

        if vp.get("voice_style")    not in valid_styles:  vp["voice_style"]    = "neutral"
        if vp.get("speaking_speed") not in valid_speeds:  vp["speaking_speed"] = "normal"
        if vp.get("pitch")          not in valid_pitches: vp["pitch"]          = "medium"

        raw_emo = vp.get("emotion_range", [])
        vp["emotion_range"] = ([e for e in raw_emo if e in valid_emotions] if isinstance(raw_emo, list) else []) or ["neutral"]

        if not vp.get("tts_description") or len(vp.get("tts_description", "")) < 10:
            vp["tts_description"] = (
                f"A {vp['voice_style']} voice speaking at {vp['speaking_speed']} speed "
                f"with {vp['pitch']} pitch."
            )
        return char

    def _fix_validation_errors(self, characters: list, errors: list) -> list:
        for char in characters:
            if len(char.get("image_prompt", "").split()) < 30:
                char["image_prompt"] = (
                    f"Portrait of {char['name']}, {char.get('role','character')}, "
                    f"{char.get('age_range','adult')}, {char.get('gender','person')}, "
                    "cinematic realism, photorealistic, professional photography, "
                    "studio lighting, high detail, 4K render, sharp focus."
                )
            sa = char.get("scenes_appeared", [1])
            char["scenes_appeared"] = ([int(s) for s in sa if str(s).isdigit()] if isinstance(sa, list) else []) or [1]

            if char.get("role") not in {"protagonist","antagonist","supporting","minor"}:
                char["role"] = "supporting"
            if not isinstance(char.get("personality_traits"), list) or not char["personality_traits"]:
                char["personality_traits"] = ["determined"]

            app = char.setdefault("appearance", {})
            app.setdefault("build", "average"); app.setdefault("hair", "unknown")
            app.setdefault("eyes", "unknown");  app.setdefault("clothing_style", "casual")
            app.setdefault("distinguishing_features", "")
        return characters

    def _script_to_text(self, script: Dict) -> str:
        story = script.get("story", {})
        lines = [
            f"TITLE: {story.get('title','')}",
            f"GENRE: {story.get('genre','')}",
            f"SYNOPSIS: {story.get('synopsis','')}",
            ""
        ]
        for scene in script.get("scenes", []):
            lines.append(f"SCENE {scene['scene_id']}: {scene.get('location','')} - {scene.get('time_of_day','')}")
            lines.append(f"ACTION: {scene.get('action','')}")
            for dlg in scene.get("dialogue", []):
                lines.append(f"{dlg['speaker']} [{dlg.get('emotion','neutral')}]: {dlg['line']}")
            lines.append("")
        return "\n".join(lines)
