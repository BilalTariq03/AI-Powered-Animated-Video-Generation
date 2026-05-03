"""
agents/scriptwriter.py
──────────────────────
Scriptwriter Agent
  Role: Transform a user prompt into a structured multi-scene screenplay.
  MCP Tools Used: generate_script_segment, commit_memory

"""

from typing import Any, Dict

from agents.base import BaseAgent
from memory.vector_store import memory
from mcp_registry import mcp

SYSTEM_PROMPT = """
You are a professional screenplay writer. Given a story prompt, generate a 
structured multi-scene screenplay in valid JSON format ONLY.

The JSON must follow this exact schema:
{
  "story": {
    "title": "Story Title",
    "genre": "Genre",
    "synopsis": "2-3 sentence summary of the full story.",
    "themes": ["theme1", "theme2"],
    "arc": "intro → conflict → climax → resolution",
    "estimated_duration_seconds": 180
  },
  "scenes": [
    {
      "scene_id": 1,
      "location": "Setting description",
      "time_of_day": "DAY | NIGHT | DAWN | DUSK",
      "duration_seconds": 30,
      "mood": "tense | mysterious | hopeful | dramatic | comedic | romantic | melancholic",
      "tone": "dark | light | neutral | suspenseful | uplifting",
      "characters": ["CharacterName1", "CharacterName2"],
      "action": "Description of what happens in this scene.",
      "dialogue": [
        {
          "speaker": "CharacterName",
          "line": "What the character says.",
          "emotion": "neutral | angry | sad | excited | fearful | surprised | disgusted | happy",
          "visual_cue": "Camera/lighting direction e.g. Close-up, warm light"
        }
      ],
      "visual_notes": "Overall visual direction for this scene.",
      "image_generation_prompt": "Detailed Stable Diffusion prompt for this scene's background."
    }
  ]
}

Rules:
- Minimum 4 scenes, maximum 8 scenes.
- Each scene must have at least 2 dialogue lines.
- duration_seconds must be between 20 and 60 per scene.
- estimated_duration_seconds = sum of all scene durations.
- mood must be ONE of: tense, mysterious, hopeful, dramatic, comedic, romantic, melancholic.
- tone must be ONE of: dark, light, neutral, suspenseful, uplifting.
- emotion on each dialogue line must be ONE of: neutral, angry, sad, excited, fearful, surprised, disgusted, happy.
- image_generation_prompt must be 30+ words describing the background/environment only.
- Keep character names consistent across all scenes.
- Output ONLY the JSON. No extra text or markdown.
"""

STORY_SYSTEM = """
You are a story analyst. Given a story prompt, return ONLY a JSON object with keys:
"genre", "tone", "themes" (list of strings), "protagonist_archetype", "conflict_type".
Output ONLY JSON, no markdown.
"""


class ScriptwriterAgent(BaseAgent):
    name = "ScriptwriterAgent"
    tool_tags = ["script", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        prompt = state.get("user_prompt", "")

        # ── Step 1: INTERPRET ─────────────────────────────────────────────────
        print(f"[{self.name}] [Reasoning 1/5] Interpreting prompt...")
        interpretation = self.chat(STORY_SYSTEM, f"Analyze this story prompt: {prompt}", temperature=0.5)
        story_meta = self.parse_json(interpretation) or {}
        print(f"[{self.name}] Interpreted: genre={story_meta.get('genre','?')}, tone={story_meta.get('tone','?')}")

        # ── Step 2: DECOMPOSE ─────────────────────────────────────────────────
        print(f"[{self.name}] [Reasoning 2/5] Decomposing into scene outline...")
        outline_prompt = (
            f"Story prompt: {prompt}\n"
            f"Genre: {story_meta.get('genre','')}, Tone: {story_meta.get('tone','')}\n"
            f"Themes: {story_meta.get('themes','')}\n\n"
            "Return ONLY a JSON array of 5 scene outlines, each with: "
            "\"scene_id\", \"location\", \"time_of_day\", \"mood\", \"tone\", "
            "\"duration_seconds\", \"purpose\"."
        )
        outline_raw = self.chat("You are a screenplay story architect.", outline_prompt, temperature=0.6)
        scene_outline = self.parse_json(outline_raw) or []
        print(f"[{self.name}] Planned {len(scene_outline)} scenes.")

        # ── Step 3: GENERATE via MCP ──────────────────────────────────────────
        print(f"[{self.name}] [Reasoning 3/5] Generating full screenplay via MCP tool...")
        enriched_prompt = (
            f"{prompt}\n\nGenre: {story_meta.get('genre','')}, Tone: {story_meta.get('tone','')}\n"
            f"Scene outline: {scene_outline}"
        )
        result = self.invoke_tool("generate_script_segment", {
            "prompt": enriched_prompt,
            "num_scenes": 5
        })
        raw_text = result.data if result.success and result.data else None

        if not raw_text:
            print(f"[{self.name}] MCP tool fallback — calling LLM directly.")
            raw_text = self.chat(SYSTEM_PROMPT, f"Write a screenplay for: {enriched_prompt}")

        # ── Parse JSON ────────────────────────────────────────────────────────
        parsed = self.parse_json(raw_text)
        if not parsed:
            print(f"[{self.name}] JSON parse failed — asking LLM to fix it.")
            raw_text = self.chat(SYSTEM_PROMPT, f"Fix this JSON, return ONLY valid JSON:\n{raw_text}", temperature=0.2)
            parsed = self.parse_json(raw_text)

        if not parsed:
            return {**state, "status": "error", "error": "Scriptwriter failed to produce valid JSON."}

        # ── Step 4: ENRICH — fill any missing fields ──────────────────────────
        print(f"[{self.name}] [Reasoning 4/5] Enriching and validating fields...")
        parsed = self._enrich(parsed, story_meta, prompt)

        # ── Step 5: PYDANTIC VALIDATION ───────────────────────────────────────
        print(f"[{self.name}] [Reasoning 5/5] Running Pydantic schema validation...")
        from schema import validate_phase1_output

        # Build a partial PhaseOneOutput (characters will be added by CharacterDesignerAgent)
        # We validate just story + scenes here with a dummy character list
        validation_payload = {
            "story":      parsed.get("story", {}),
            "scenes":     parsed.get("scenes", []),
            "characters": [
                {
                    "name": name,
                    "role": "supporting",
                    "age_range": "unknown",
                    "gender": "unknown",
                    "personality_traits": ["unknown"],
                    "appearance": {
                        "build": "average", "hair": "unknown",
                        "eyes": "unknown", "clothing_style": "unknown",
                        "distinguishing_features": ""
                    },
                    "voice_profile": {
                        "voice_style": "neutral", "speaking_speed": "normal",
                        "pitch": "medium", "emotion_range": ["neutral"],
                        "tts_description": "A neutral voice."
                    },
                    "reference_style": "cinematic realism",
                    "image_prompt": f"Portrait of {name}, cinematic realism, photorealistic, detailed, professional photography, studio lighting, high quality render.",
                    "scenes_appeared": [1]
                }
                for name in self._all_character_names(parsed)
            ]
        }

        validated, errors = validate_phase1_output(validation_payload)
        if errors:
            print(f"[{self.name}] ⚠ Pydantic validation warnings (non-blocking): {errors}")
        else:
            print(f"[{self.name}] ✓ Pydantic validation passed.")

        # ── Commit to memory ──────────────────────────────────────────────────
        self.invoke_tool("commit_memory", {
            "key": "script:latest",
            "data": parsed,
            "metadata": {"type": "script", "title": parsed.get("story", {}).get("title", "untitled")}
        })

        print(f"[{self.name}] ✓ Complete. "
              f"Story: '{parsed.get('story', {}).get('title')}' | "
              f"{len(parsed.get('scenes', []))} scenes.")
        return {**state, "script": parsed, "status": "script_ready"}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _all_character_names(self, parsed: dict) -> list:
        names = set()
        for scene in parsed.get("scenes", []):
            for name in scene.get("characters", []):
                names.add(name)
        return list(names)

    def _enrich(self, parsed: dict, story_meta: dict, original_prompt: str) -> dict:
        """Fill any missing fields so Pydantic validation has a chance to pass."""
        valid_moods    = {"tense", "mysterious", "hopeful", "dramatic", "comedic", "romantic", "melancholic"}
        valid_tones    = {"dark", "light", "neutral", "suspenseful", "uplifting"}
        valid_emotions = {"neutral", "angry", "sad", "excited", "fearful", "surprised", "disgusted", "happy"}
        valid_times    = {"DAY", "NIGHT", "DAWN", "DUSK"}

        # ── Ensure top-level story object ─────────────────────────────────────
        # Handle case where LLM returned old format {title, genre, scenes}
        if "story" not in parsed and "title" in parsed:
            total_dur = sum(s.get("duration_seconds", 30) for s in parsed.get("scenes", []))
            parsed["story"] = {
                "title":   parsed.pop("title", "Untitled"),
                "genre":   parsed.pop("genre", story_meta.get("genre", "Drama")),
                "synopsis": "",
                "themes":  story_meta.get("themes", ["redemption"]),
                "arc":     "intro → conflict → climax → resolution",
                "estimated_duration_seconds": total_dur or 150
            }

        story = parsed.setdefault("story", {})
        story.setdefault("title", "Untitled")
        story.setdefault("genre", story_meta.get("genre", "Drama"))
        story.setdefault("themes", story_meta.get("themes", ["redemption"]))
        story.setdefault("arc", "intro → conflict → climax → resolution")

        if not story.get("synopsis") or len(story.get("synopsis", "")) < 20:
            synopsis = self.chat(
                "You are a story analyst. Given a story prompt, write a 2-3 sentence synopsis.",
                f"Story prompt: {original_prompt}"
            )
            story["synopsis"] = synopsis.strip()

        # Compute estimated_duration_seconds from scenes
        scenes = parsed.get("scenes", [])
        total_dur = sum(s.get("duration_seconds", 30) for s in scenes)
        story["estimated_duration_seconds"] = total_dur or 150

        # ── Enrich each scene ─────────────────────────────────────────────────
        for scene in scenes:
            scene.setdefault("duration_seconds", 30)
            tod = scene.get("time_of_day", "").upper()
            scene["time_of_day"] = tod if tod in valid_times else "DAY"

            if scene.get("mood") not in valid_moods:
                scene["mood"] = "dramatic"
            if scene.get("tone") not in valid_tones:
                scene["tone"] = "neutral"

            if not scene.get("image_generation_prompt"):
                vis = self.chat(
                    "You are a Stable Diffusion prompt engineer. Write a 30+ word image generation "
                    "prompt for the background/environment of this scene. No characters. Return only the prompt.",
                    f"Location: {scene.get('location', '')}. Visual notes: {scene.get('visual_notes', '')}."
                )
                scene["image_generation_prompt"] = vis.strip()

            if not scene.get("visual_notes"):
                scene["visual_notes"] = scene.get("image_generation_prompt", "Cinematic composition.")

            dialogue = scene.setdefault("dialogue", [])
            for dlg in dialogue:
                if dlg.get("emotion") not in valid_emotions:
                    dlg["emotion"] = "neutral"
                dlg.setdefault("visual_cue", "Medium shot.")
            if len(dialogue) < 2:
                speaker = dialogue[0]["speaker"] if dialogue else (scene.get("characters") or ["Character"])[0]
                dialogue.append({
                    "speaker": speaker,
                    "line": "...",
                    "emotion": "neutral",
                    "visual_cue": "Wide shot."
                })

        return parsed