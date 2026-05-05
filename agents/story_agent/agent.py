"""
agents/story_agent/agent.py
────────────────────────────
Scriptwriter Agent — transforms a user prompt into a structured screenplay.
"""

from typing import Any, Dict

from agents.base import BaseAgent
from shared.utils.vector_store import memory

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
    "arc": "intro -> conflict -> climax -> resolution",
    "estimated_duration_seconds": 180
  },
  "scenes": [
    {
      "scene_id": 1,
      "location": "Setting description",
      "time_of_day": "DAY | NIGHT | DAWN | DUSK",
      "duration_seconds": 30,
      "mood": "tense | mysterious | hopeful | dramatic | comedic | romantic | melancholic | fantasy",
      "tone": "dark | light | neutral | suspenseful | uplifting",
      "characters": ["CharacterName1"],
      "action": "Description of what happens.",
      "dialogue": [
        {
          "speaker": "CharacterName",
          "line": "What the character says.",
          "emotion": "neutral | angry | sad | excited | fearful | surprised | disgusted | happy",
          "visual_cue": "Camera direction"
        }
      ],
      "visual_notes": "Overall visual direction.",
      "image_generation_prompt": "Detailed 30+ word Stable Diffusion prompt for background."
    }
  ]
}

Rules:
- Minimum 4 scenes, maximum 8 scenes.
- Each scene must have at least 2 dialogue lines from AT LEAST 2 DIFFERENT speakers.
- Never have one character speak all the lines in a scene. Every scene is a conversation.
- Each line must be specific to its scene context — NO generic filler lines like
  "What does this mean?", "Understood.", "I see.", "Indeed.", "Interesting."
- Each character's lines must reflect their unique personality and the specific events of that scene.
- duration_seconds must be between 20 and 60.
- time_of_day must be UPPERCASE: DAY, NIGHT, DAWN, or DUSK.
- Output ONLY the JSON. No extra text or markdown.
"""

STORY_SYSTEM = """
You are a story analyst. Given a story prompt, return ONLY a JSON object with keys:
"genre", "tone", "themes" (list), "protagonist_archetype", "conflict_type",
"suggested_scenes" (integer 4-8, based on story complexity — simple one-act story = 4,
epic multi-thread story = 8, typical drama = 5).
Output ONLY JSON, no markdown.
"""


class ScriptwriterAgent(BaseAgent):
    name      = "ScriptwriterAgent"
    tool_tags = ["script", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        prompt = state.get("user_prompt", "")

        print(f"[{self.name}] [Reasoning 1/5] Interpreting prompt...")
        story_meta = self.parse_json(
            self.chat(STORY_SYSTEM, f"Analyze: {prompt}", temperature=0.5)
        ) or {}
        print(f"[{self.name}] Interpreted: genre={story_meta.get('genre','?')}, tone={story_meta.get('tone','?')}")

        print(f"[{self.name}] [Reasoning 2/5] Decomposing into scene outline...")
        n_scenes = max(4, min(8, int(story_meta.get("suggested_scenes", 5))))
        mood_arc = self._build_mood_arc(n_scenes)
        outline_prompt = (
            f"Story prompt: {prompt}\nGenre: {story_meta.get('genre','')}, "
            f"Tone: {story_meta.get('tone','')}\nThemes: {story_meta.get('themes','')}\n\n"
            f"Return ONLY a JSON array of exactly {n_scenes} scene outlines. "
            "Each object must have: "
            '"scene_id", "location", "time_of_day", "mood", "tone", "duration_seconds", "purpose", "characters" (list of 2+ character names). '
            "Every scene MUST include at least 2 different characters so dialogue is a conversation, not a monologue. "
            "Vary the character combinations across scenes. "
            f"You MUST assign these moods in order: {mood_arc}. "
            "Do not repeat or reorder them."
        )
        scene_outline = self.parse_json(
            self.chat("You are a screenplay story architect.", outline_prompt, temperature=0.6)
        ) or []
        print(f"[{self.name}] Planned {len(scene_outline)} scenes.")

        print(f"[{self.name}] [Reasoning 3/5] Generating full screenplay via MCP tool...")
        enriched_prompt = (
            f"{prompt}\n\nGenre: {story_meta.get('genre','')}, Tone: {story_meta.get('tone','')}\n"
            f"Scene outline: {scene_outline}"
        )
        result   = self.invoke_tool("generate_script_segment", {"prompt": enriched_prompt, "num_scenes": n_scenes})
        raw_text = result.data if result.success and result.data else None

        if not raw_text:
            print(f"[{self.name}] MCP fallback - calling LLM directly.")
            raw_text = self.chat(SYSTEM_PROMPT, f"Write a screenplay for: {enriched_prompt}")

        parsed = self.parse_json(raw_text)
        if not parsed:
            raw_text = self.chat(SYSTEM_PROMPT, f"Fix this JSON, return ONLY valid JSON:\n{raw_text}", temperature=0.2)
            parsed   = self.parse_json(raw_text)

        if not parsed:
            return {**state, "status": "error", "error": "Scriptwriter failed to produce valid JSON."}

        print(f"[{self.name}] [Reasoning 4/5] Enriching and validating fields...")
        parsed = self._enrich(parsed, story_meta, prompt)

        print(f"[{self.name}] [Reasoning 5/5] Running Pydantic schema validation...")
        from shared.schemas.schema import validate_phase1_output

        validation_payload = {
            "story":  parsed.get("story", {}),
            "scenes": parsed.get("scenes", []),
            "characters": [
                {
                    "name": name, "role": "supporting", "age_range": "unknown",
                    "gender": "unknown", "personality_traits": ["unknown"],
                    "appearance": {"build": "average", "hair": "unknown", "eyes": "unknown",
                                   "clothing_style": "unknown", "distinguishing_features": ""},
                    "voice_profile": {"voice_style": "neutral", "speaking_speed": "normal",
                                      "pitch": "medium", "emotion_range": ["neutral"],
                                      "tts_description": "A neutral voice."},
                    "reference_style": "cinematic realism",
                    "image_prompt": (f"Portrait of {name}, cinematic realism, photorealistic, "
                                     "detailed, professional photography, studio lighting, high quality."),
                    "scenes_appeared": [1]
                }
                for name in self._all_character_names(parsed)
            ]
        }
        _, errors = validate_phase1_output(validation_payload)
        if errors:
            print(f"[{self.name}] Pydantic warnings (non-blocking): {errors}")
        else:
            print(f"[{self.name}] Pydantic validation passed.")

        self.invoke_tool("commit_memory", {
            "key":      "script:latest",
            "data":     parsed,
            "metadata": {"type": "script", "title": parsed.get("story", {}).get("title", "untitled")}
        })

        print(f"[{self.name}] Complete. Story: '{parsed.get('story',{}).get('title')}' | "
              f"{len(parsed.get('scenes',[]))} scenes.")
        return {**state, "script": parsed, "status": "script_ready"}

    @staticmethod
    def _build_mood_arc(n: int) -> list:
        """Return a mood arc of length n (4–8) with natural dramatic progression."""
        arcs = {
            4: ["mysterious", "tense", "dramatic", "melancholic"],
            5: ["mysterious", "tense", "dramatic", "tense", "melancholic"],
            6: ["mysterious", "tense", "hopeful", "dramatic", "tense", "melancholic"],
            7: ["mysterious", "tense", "hopeful", "dramatic", "dramatic", "tense", "melancholic"],
            8: ["mysterious", "tense", "hopeful", "dramatic", "dramatic", "romantic", "tense", "melancholic"],
        }
        return arcs.get(n, arcs[5])

    def _all_character_names(self, parsed: dict) -> list:
        names = set()
        for scene in parsed.get("scenes", []):
            for entry in scene.get("characters", []):
                # LLM sometimes returns {"name": "..."} instead of plain string
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("character") or str(entry)
                else:
                    name = str(entry)
                if name:
                    names.add(name)
        return list(names)

    def _enrich(self, parsed: dict, story_meta: dict, original_prompt: str) -> dict:
        valid_moods    = {"tense","mysterious","hopeful","dramatic","comedic","romantic","melancholic","fantasy"}
        valid_tones    = {"dark","light","neutral","suspenseful","uplifting"}
        valid_emotions = {"neutral","angry","sad","excited","fearful","surprised","disgusted","happy"}
        valid_times    = {"DAY","NIGHT","DAWN","DUSK"}

        if "story" not in parsed and "title" in parsed:
            total_dur = sum(s.get("duration_seconds", 30) for s in parsed.get("scenes", []))
            parsed["story"] = {
                "title":   parsed.pop("title", "Untitled"),
                "genre":   parsed.pop("genre", story_meta.get("genre", "Drama")),
                "synopsis": "",
                "themes":  story_meta.get("themes", ["redemption"]),
                "arc":     "intro -> conflict -> climax -> resolution",
                "estimated_duration_seconds": total_dur or 150
            }

        story = parsed.setdefault("story", {})
        story.setdefault("title",  "Untitled")
        story.setdefault("genre",  story_meta.get("genre", "Drama"))
        story.setdefault("themes", story_meta.get("themes", ["redemption"]))
        story.setdefault("arc",    "intro -> conflict -> climax -> resolution")

        if not story.get("synopsis") or len(story.get("synopsis", "")) < 20:
            story["synopsis"] = self.chat(
                "Write a 2-3 sentence story synopsis.", f"Story prompt: {original_prompt}"
            ).strip()

        scenes    = parsed.get("scenes", [])
        total_dur = sum(s.get("duration_seconds", 30) for s in scenes)
        story["estimated_duration_seconds"] = total_dur or 150

        for scene_idx, scene in enumerate(scenes):
            # Coerce scene_id to int (LLM sometimes returns "1" instead of 1)
            try:
                scene["scene_id"] = int(str(scene.get("scene_id", scene_idx + 1)).strip())
            except (ValueError, TypeError):
                scene["scene_id"] = scene_idx + 1

            # Normalize characters to plain strings — LLM sometimes returns dicts
            def _to_name(c) -> str:
                if isinstance(c, dict):
                    return str(c.get("name") or c.get("character") or next(iter(c.values()), "Character"))
                return str(c) if c else "Character"
            scene["characters"] = [_to_name(c) for c in scene.get("characters", []) if c]

            # Normalize dialogue lines (LLM can produce empty strings)
            for dlg in scene.get("dialogue", []):
                if not dlg.get("line", "").strip():
                    dlg["line"] = "Understood."

            # Clamp duration to Pydantic limit
            scene["duration_seconds"] = min(int(scene.get("duration_seconds", 30)), 120)

            scene.setdefault("duration_seconds", 30)
            tod = scene.get("time_of_day", "").upper()
            scene["time_of_day"] = tod if tod in valid_times else "DAY"

            # Preserve outline-assigned mood; only fall back if truly missing/invalid
            fallback_arc = self._build_mood_arc(len(scenes))
            if scene.get("mood") not in valid_moods:
                scene["mood"] = fallback_arc[scene_idx % len(fallback_arc)]
            if scene.get("tone") not in valid_tones:
                scene["tone"] = "neutral"

            if not scene.get("image_generation_prompt"):
                scene["image_generation_prompt"] = self.chat(
                    "Write a 30+ word Stable Diffusion background prompt. No characters. Return only the prompt.",
                    f"Location: {scene.get('location','')}. Visual notes: {scene.get('visual_notes','')}."
                ).strip()

            scene.setdefault("visual_notes", scene.get("image_generation_prompt", "Cinematic composition."))

            dialogue = scene.setdefault("dialogue", [])
            for dlg in dialogue:
                if dlg.get("emotion") not in valid_emotions:
                    dlg["emotion"] = "neutral"
                dlg.setdefault("visual_cue", "Medium shot.")

            if len(dialogue) < 2:
                raw_chars = scene.get("characters") or ["Character"]
                chars = [
                    (c.get("name") or c.get("character") or "Character") if isinstance(c, dict) else str(c)
                    for c in raw_chars
                ]
                first_speaker = dialogue[0]["speaker"] if dialogue else chars[0]
                first_line    = dialogue[0]["line"]    if dialogue else ""
                # Pick a different character for the response line
                responder = next((c for c in chars if c != first_speaker), chars[0])
                # Generate a contextual response via LLM instead of a static fallback
                ctx = (
                    f"Scene: {scene.get('location','')} | Mood: {scene.get('mood','')} | "
                    f"Action: {scene.get('action','')[:120]}\n"
                    f"{first_speaker} just said: \"{first_line}\"\n"
                    f"Write ONE short, specific response line for {responder}. "
                    "Return ONLY the dialogue line, no speaker name, no quotes."
                )
                try:
                    response_line = self.chat(
                        "You are a screenplay writer. Write a single character's dialogue line.",
                        ctx, temperature=0.8
                    ).strip().strip('"').strip("'")
                except Exception:
                    response_line = "We have to keep moving."
                dialogue.append({
                    "speaker":    responder,
                    "line":       response_line or "We have to keep moving.",
                    "emotion":    "neutral",
                    "visual_cue": "Wide shot.",
                })

        return parsed
