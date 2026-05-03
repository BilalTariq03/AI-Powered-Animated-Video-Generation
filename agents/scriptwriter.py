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
  "title": "Story Title",
  "genre": "Genre",
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
      "image_generation_prompt": "Detailed Stable Diffusion prompt for this scene's background/setting image."
    }
  ]
}

Rules:
- Minimum 4 scenes, maximum 8 scenes.
- Each scene must have at least 2 dialogue lines.
- duration_seconds must be realistic: 20-60 seconds per scene.
- mood must be ONE of: tense, mysterious, hopeful, dramatic, comedic, romantic, melancholic.
- tone must be ONE of: dark, light, neutral, suspenseful, uplifting.
- emotion on each dialogue line must be ONE of: neutral, angry, sad, excited, fearful, surprised, disgusted, happy.
- image_generation_prompt must be 30+ words, describe the scene background/setting only (no characters).
- Keep character names consistent across scenes.
- Include vivid visual_cue and visual_notes for the image synthesizer.
- Output ONLY the JSON. No extra text or markdown.
"""


class ScriptwriterAgent(BaseAgent):
    name = "ScriptwriterAgent"
    tool_tags = ["script", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Starting reasoning loop...")
        prompt = state.get("user_prompt", "")

        # ── Reasoning Loop ────────────────────────────────────────────────────
        # Step 1: INTERPRET — understand the prompt's genre, tone, themes
        print(f"[{self.name}] [Reasoning 1/4] Interpreting prompt...")
        interpretation = self.chat(
            "You are a story analyst. Given a story prompt, return ONLY a JSON object with keys: "
            "\"genre\", \"tone\", \"themes\" (list), \"protagonist_archetype\", \"conflict_type\".",
            f"Analyze this story prompt: {prompt}",
            temperature=0.5
        )
        story_meta = self.parse_json(interpretation) or {}
        print(f"[{self.name}] Interpreted: genre={story_meta.get('genre','?')}, tone={story_meta.get('tone','?')}")

        # Step 2: DECOMPOSE — plan the scene structure
        print(f"[{self.name}] [Reasoning 2/4] Decomposing into scene outline...")
        outline_prompt = (
            f"Story prompt: {prompt}\n"
            f"Genre: {story_meta.get('genre','')}, Tone: {story_meta.get('tone','')}\n"
            f"Themes: {story_meta.get('themes','')}\n\n"
            "Return ONLY a JSON array of 5 scene outlines, each with: "
            "\"scene_id\", \"location\", \"time_of_day\", \"mood\", \"tone\", \"duration_seconds\", "
            "\"purpose\" (what this scene achieves narratively)."
        )
        outline_raw = self.chat("You are a screenplay story architect.", outline_prompt, temperature=0.6)
        scene_outline = self.parse_json(outline_raw) or []
        print(f"[{self.name}] Planned {len(scene_outline)} scenes.")

        # Step 3: GENERATE — invoke MCP tool to produce full screenplay
        print(f"[{self.name}] [Reasoning 3/4] Generating full screenplay via MCP tool...")
        tools = mcp.discover(tags=["script"])
        tool_names = [t.name for t in tools]
        print(f"[{self.name}] Available MCP tools: {tool_names}")

        enriched_prompt = (
            f"{prompt}\n\nGenre: {story_meta.get('genre','')}, Tone: {story_meta.get('tone','')}\n"
            f"Scene outline: {scene_outline}"
        )
        result = self.invoke_tool("generate_script_segment", {
            "prompt": enriched_prompt,
            "num_scenes": 5
        })

        if result.success and result.data:
            raw_text = result.data
        else:
            print(f"[{self.name}] MCP tool fallback — calling LLM directly.")
            raw_text = self.chat(SYSTEM_PROMPT, f"Write a screenplay for: {enriched_prompt}")

        # ── Parse JSON ────────────────────────────────────────────────────────
        script = self.parse_json(raw_text)
        if not script:
            print(f"[{self.name}] JSON parse failed — asking LLM to fix it.")
            fix_prompt = f"Fix this JSON and return ONLY valid JSON:\n{raw_text}"
            raw_text = self.chat(SYSTEM_PROMPT, fix_prompt, temperature=0.2)
            script = self.parse_json(raw_text)

        if not script:
            return {**state, "status": "error", "error": "Scriptwriter failed to produce valid JSON."}

        # Step 4: ENRICH — fill missing fields and attach visual context
        print(f"[{self.name}] [Reasoning 4/4] Enriching scenes with missing fields...")
        valid_moods = {"tense", "mysterious", "hopeful", "dramatic", "comedic", "romantic", "melancholic"}
        valid_tones = {"dark", "light", "neutral", "suspenseful", "uplifting"}
        valid_emotions = {"neutral", "angry", "sad", "excited", "fearful", "surprised", "disgusted", "happy"}

        for scene in script.get("scenes", []):
            # Fill duration_seconds if missing
            if not scene.get("duration_seconds"):
                scene["duration_seconds"] = 30

            # Fill mood if missing or invalid
            if scene.get("mood") not in valid_moods:
                mood_raw = self.chat(
                    "You are a film director. Given a scene description, return ONLY one word "
                    f"for the scene's mood from this list: {', '.join(valid_moods)}.",
                    f"Scene action: {scene.get('action', '')}"
                )
                scene["mood"] = mood_raw.strip().lower() if mood_raw.strip().lower() in valid_moods else "neutral" if "neutral" in valid_moods else "dramatic"

            # Fill tone if missing or invalid
            if scene.get("tone") not in valid_tones:
                scene["tone"] = "neutral"

            # Fill image_generation_prompt if missing
            if not scene.get("image_generation_prompt"):
                vis = self.chat(
                    "You are a Stable Diffusion prompt engineer. Given a scene setting, write a "
                    "detailed image generation prompt for the background/environment only (no characters). "
                    "Return ONLY the prompt text, 30+ words.",
                    f"Scene location: {scene.get('location', '')}. Visual notes: {scene.get('visual_notes', '')}."
                )
                scene["image_generation_prompt"] = vis.strip()

            # Fill visual_notes if missing
            if not scene.get("visual_notes"):
                vis = self.chat(
                    "You are a cinematographer. Given a scene description, return ONLY a one-sentence visual direction.",
                    f"Scene: {scene.get('action', '')}"
                )
                scene["visual_notes"] = vis.strip()

            # Fill emotion on each dialogue line if missing
            for dlg in scene.get("dialogue", []):
                if dlg.get("emotion") not in valid_emotions:
                    dlg["emotion"] = "neutral"

        # ── Step 5: Commit to memory via MCP ─────────────────────────────────
        self.invoke_tool("commit_memory", {
            "key": "script:latest",
            "data": script,
            "metadata": {"type": "script", "title": script.get("title", "untitled")}
        })

        print(f"[{self.name}] ✓ Reasoning loop complete. Script: '{script.get('title')}' with {len(script.get('scenes', []))} scenes.")
        return {**state, "script": script, "status": "script_ready"}
