"""
agents/edit_agent/executor.py
──────────────────────────────
Applies a classified EditIntent to the pipeline's JSON state files.

Each handler modifies only the relevant handoff / manifest JSON and
returns {"ok": bool, "message": str, "phase_to_rerun": int | None}.
The caller is responsible for triggering the phase re-run.
"""

import io
import json
import os
import time

import requests
from PIL import Image

from config import (
    PHASE2_HANDOFF, PHASE3_HANDOFF, TIMING_MANIFEST,
    IMAGES_DIR, HF_API_KEY, HF_IMAGE_MODEL,
    AUDIO_DIR, BGM_DIR, DIALOGUE_DIR, SCENES_DIR,
)

EDIT_SETTINGS = os.path.join("data", "outputs", "edit_settings.json")

_VALID_VOICE_STYLES = {
    "deep", "authoritative", "raspy", "soft", "high-pitched", "whispery", "neutral"
}

_VALID_MOODS = {
    "dramatic", "tense", "mysterious", "hopeful", "melancholic", "comedic", "romantic", "fantasy"
}


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_scope(scope: str):
    """Return (kind, value): ('character', 'Name'), ('scene', N), or ('all', None)."""
    s = (scope or "all").strip()
    if s.lower().startswith("character:"):
        return "character", s[10:].strip()
    if s.lower().startswith("scene:"):
        try:
            return "scene", int(s[6:].strip())
        except ValueError:
            return "scene", None
    return "all", None


def _missing(path: str, name: str) -> dict:
    return {"ok": False, "message": f"{name} not found — run the pipeline first.", "phase_to_rerun": None}


class EditExecutor:

    def execute(self, intent: dict) -> dict:
        action = intent.get("intent", "")
        scope  = intent.get("scope", "all")
        params = intent.get("parameters") or {}

        handlers = {
            "change_voice_tone":       self._change_voice_tone,
            "make_scene_darker":       self._make_scene_visual,
            "make_scene_lighter":      self._make_scene_visual,
            "change_scene_mood":       self._make_scene_visual,
            "change_scene_background": self._make_scene_visual,
            "add_background_music":    self._change_bgm,
            "remove_subtitle":         self._toggle_subtitle,
            "add_subtitle":            self._toggle_subtitle,
            "change_character_design": self._change_character_design,
            "speed_up_scene":          self._adjust_speed,
            "slow_down_scene":         self._adjust_speed,
            "regenerate_script":       self._regenerate_script,
            "add_scene":               self._add_scene,
            "delete_scene":            self._delete_scene,
        }

        handler = handlers.get(action)
        if not handler:
            return {"ok": False, "message": f"Unknown action: {action!r}", "phase_to_rerun": None}

        try:
            return handler(action, scope, params)
        except Exception as e:
            return {"ok": False, "message": str(e), "phase_to_rerun": None}

    # ── Voice tone ─────────────────────────────────────────────────────────────

    def _change_voice_tone(self, action, scope, params):
        if not os.path.exists(PHASE2_HANDOFF):
            return _missing(PHASE2_HANDOFF, "Phase 2 handoff")

        kind, value = _parse_scope(scope)
        tone = (params.get("tone") or params.get("voice_style") or
                params.get("style") or "neutral").lower()
        if tone not in _VALID_VOICE_STYLES:
            tone = "neutral"

        handoff = _read(PHASE2_HANDOFF)
        changed = 0
        for vc in handoff.get("voice_configs", []):
            match = (kind == "all" or
                     (kind == "character" and value and
                      value.lower() in vc.get("character_name", "").lower()))
            if match:
                vc["voice_style"] = tone
                changed += 1

        if not changed:
            return {"ok": False, "message": f"No character matched scope {scope!r}.", "phase_to_rerun": None}

        _write(PHASE2_HANDOFF, handoff)
        return {
            "ok": True,
            "message": f"Voice tone set to '{tone}' for {changed} character(s). Re-running Phase 2.",
            "phase_to_rerun": 2,
        }

    # ── Scene visuals ──────────────────────────────────────────────────────────

    _VISUAL_MODIFIERS = {
        "make_scene_darker":  "dark, shadowy, low-key lighting, deep shadows, night atmosphere",
        "make_scene_lighter": "bright, high-key lighting, golden sunlight, airy, warm",
    }

    def _make_scene_visual(self, action, scope, params):
        if not os.path.exists(PHASE3_HANDOFF):
            return _missing(PHASE3_HANDOFF, "Phase 3 handoff")

        kind, value = _parse_scope(scope)

        modifier = (
            self._VISUAL_MODIFIERS.get(action)
            or params.get("description")
            or params.get("mood")
            or "cinematic atmospheric"
        )

        handoff = _read(PHASE3_HANDOFF)
        changed = []
        for scene in handoff.get("scenes", []):
            sid = scene["scene_id"]
            match = kind == "all" or (kind == "scene" and value == sid)
            if match:
                existing = scene.get("image_generation_prompt", "")
                scene["image_generation_prompt"] = (
                    f"{existing}, {modifier}" if existing else modifier
                )
                # Delete cached bg so Phase 3 regenerates it
                bg_path = os.path.join(IMAGES_DIR, f"scene{sid}_bg.png")
                if os.path.exists(bg_path):
                    os.remove(bg_path)
                changed.append(sid)

        if not changed:
            return {"ok": False, "message": f"No scene matched scope {scope!r}.", "phase_to_rerun": None}

        _write(PHASE3_HANDOFF, handoff)
        return {
            "ok": True,
            "message": f"Visual prompt updated for scene(s) {changed}. Background will regenerate.",
            "phase_to_rerun": 3,
        }

    # ── Background music ───────────────────────────────────────────────────────

    def _change_bgm(self, action, scope, params):
        if not os.path.exists(PHASE2_HANDOFF):
            return _missing(PHASE2_HANDOFF, "Phase 2 handoff")

        kind, value = _parse_scope(scope)
        mood = (params.get("mood") or params.get("genre") or "dramatic").lower()
        if mood not in _VALID_MOODS:
            mood = "dramatic"

        handoff = _read(PHASE2_HANDOFF)
        changed = []
        for mm in handoff.get("music_moods", []):
            sid   = mm["scene_id"]
            match = kind == "all" or (kind == "scene" and value == sid)
            if match:
                mm["mood"] = mood
                changed.append(sid)

        if not changed:
            return {"ok": False, "message": f"No scene matched scope {scope!r}.", "phase_to_rerun": None}

        _write(PHASE2_HANDOFF, handoff)
        return {
            "ok": True,
            "message": f"BGM mood set to '{mood}' for scene(s) {changed}. Re-running Phase 2.",
            "phase_to_rerun": 2,
        }

    # ── Subtitle toggle ────────────────────────────────────────────────────────

    def _toggle_subtitle(self, action, scope, params):
        show     = (action == "add_subtitle")
        settings = _read(EDIT_SETTINGS) if os.path.exists(EDIT_SETTINGS) else {}
        settings["show_subtitles"] = show
        _write(EDIT_SETTINGS, settings)
        word = "enabled" if show else "disabled"
        return {
            "ok": True,
            "message": f"Subtitles {word}. Re-rendering video.",
            "phase_to_rerun": 3,
        }

    # ── Character design ───────────────────────────────────────────────────────

    def _change_character_design(self, action, scope, params):
        if not os.path.exists(PHASE3_HANDOFF):
            return _missing(PHASE3_HANDOFF, "Phase 3 handoff")

        kind, char_name = _parse_scope(scope)
        description     = params.get("description") or params.get("style") or ""

        handoff = _read(PHASE3_HANDOFF)
        changed = []
        for cv in handoff.get("character_visuals", []):
            match = (kind == "all" or
                     (kind == "character" and char_name and
                      char_name.lower() in cv.get("name", "").lower()))
            if match:
                # Merge the new description into the existing prompt so we keep
                # character-specific detail (e.g. "detective, 2087") and only
                # override the parts the user asked to change.
                existing = cv.get("image_prompt", f"portrait of {cv['name']}")
                if description:
                    cv["image_prompt"] = (
                        f"{existing}, {description}, "
                        "cinematic realism, highly detailed, photorealistic"
                    )

                # Delete stale cached image
                safe = cv["name"].lower().replace(" ", "_")
                for img in (f"{safe}.png", f"{safe}.jpg"):
                    p = os.path.join(IMAGES_DIR, img)
                    if os.path.exists(p):
                        os.remove(p)

                # Regenerate immediately — Phase 3 cannot do this itself
                new_path = self._regenerate_character_image(cv["name"], cv["image_prompt"])
                if new_path:
                    print(f"[EditExecutor] New portrait saved: {new_path}")
                else:
                    print(f"[EditExecutor] Warning: image regeneration failed for {cv['name']}. "
                          "Re-run Phase 1 or place image manually.")

                changed.append(cv["name"])

        if not changed:
            return {"ok": False, "message": f"No character matched scope {scope!r}.", "phase_to_rerun": None}

        _write(PHASE3_HANDOFF, handoff)
        return {
            "ok": True,
            "message": f"Portrait regenerated for {changed}. Re-rendering video.",
            "phase_to_rerun": 3,
        }

    def _regenerate_character_image(self, char_name: str, prompt: str) -> str | None:
        """Generate a character portrait PNG, trying HF then Pollinations."""
        os.makedirs(IMAGES_DIR, exist_ok=True)
        safe = char_name.lower().replace(" ", "_")
        out_path = os.path.join(IMAGES_DIR, f"{safe}.png")

        # ── HuggingFace FLUX.1-schnell ────────────────────────────────────────
        if HF_API_KEY:
            api_url = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            for attempt in range(3):
                try:
                    resp = requests.post(api_url, headers=headers,
                                         json={"inputs": prompt}, timeout=120)
                    if resp.status_code == 503:
                        wait = 20
                        try:
                            wait = resp.json().get("estimated_time", 20)
                        except Exception:
                            pass
                        time.sleep(min(wait, 60))
                        continue
                    if resp.status_code == 200:
                        Image.open(io.BytesIO(resp.content)).save(out_path)
                        return out_path
                except Exception as e:
                    print(f"[EditExecutor] HF attempt {attempt+1} failed: {e}")
                    time.sleep(5)

        # ── Pollinations.ai fallback (free, no key) ───────────────────────────
        try:
            import urllib.parse
            encoded = urllib.parse.quote(prompt)
            url     = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true"
            resp    = requests.get(url, timeout=60)
            if resp.status_code == 200:
                Image.open(io.BytesIO(resp.content)).save(out_path)
                return out_path
        except Exception as e:
            print(f"[EditExecutor] Pollinations fallback failed: {e}")

        return None

    # ── Scene speed ────────────────────────────────────────────────────────────

    def _adjust_speed(self, action, scope, params):
        if not os.path.exists(TIMING_MANIFEST):
            return _missing(TIMING_MANIFEST, "Timing manifest")

        kind, value = _parse_scope(scope)
        factor = float(params.get("factor", 1.5))
        scale  = (1.0 / factor) if action == "speed_up_scene" else factor

        manifest = _read(TIMING_MANIFEST)
        changed  = []
        for scene in manifest.get("scenes", []):
            sid   = scene["scene_id"]
            match = kind == "all" or (kind == "scene" and value == sid)
            if match:
                for seg in scene.get("dialogue_segments", []):
                    seg["start_ms"]    = int(seg["start_ms"]    * scale)
                    seg["end_ms"]      = int(seg["end_ms"]      * scale)
                    seg["duration_ms"] = int(seg["duration_ms"] * scale)
                scene["total_duration_ms"] = int(scene.get("total_duration_ms", 0) * scale)
                changed.append(sid)

        if not changed:
            return {"ok": False, "message": f"No scene matched scope {scope!r}.", "phase_to_rerun": None}

        manifest["flat_segments"] = [
            seg for sc in manifest["scenes"] for seg in sc.get("dialogue_segments", [])
        ]
        _write(TIMING_MANIFEST, manifest)
        verb = "sped up" if action == "speed_up_scene" else "slowed down"
        return {
            "ok": True,
            "message": f"Scene(s) {changed} {verb} by {factor}×. Re-rendering video.",
            "phase_to_rerun": 3,
        }

    # ── Add / delete scene ─────────────────────────────────────────────────────

    def _delete_scene(self, action, scope, params):
        for path in (PHASE2_HANDOFF, PHASE3_HANDOFF, TIMING_MANIFEST):
            if not os.path.exists(path):
                return _missing(path, os.path.basename(path))

        _, scene_num = _parse_scope(scope)
        if scene_num is None:
            return {"ok": False, "message": "Specify which scene to delete, e.g. 'delete scene 3'.", "phase_to_rerun": None}

        # ── Remove from phase3 handoff ────────────────────────────────────────
        p3 = _read(PHASE3_HANDOFF)
        before = len(p3.get("scenes", []))
        p3["scenes"] = [s for s in p3.get("scenes", []) if s["scene_id"] != scene_num]
        if len(p3["scenes"]) == before:
            return {"ok": False, "message": f"Scene {scene_num} not found.", "phase_to_rerun": None}
        _write(PHASE3_HANDOFF, p3)

        # ── Remove from phase2 handoff ────────────────────────────────────────
        p2 = _read(PHASE2_HANDOFF)
        removed_segments = [s["segment_id"] for s in p2.get("segments", []) if s["scene_id"] == scene_num]
        p2["segments"]    = [s for s in p2.get("segments", [])    if s["scene_id"] != scene_num]
        p2["music_moods"] = [m for m in p2.get("music_moods", []) if m["scene_id"] != scene_num]
        _write(PHASE2_HANDOFF, p2)

        # ── Remove from timing manifest ───────────────────────────────────────
        tm = _read(TIMING_MANIFEST)
        tm["scenes"] = [s for s in tm.get("scenes", []) if s["scene_id"] != scene_num]
        tm["flat_segments"] = [s for s in tm.get("flat_segments", []) if s["scene_id"] != scene_num]
        tm["total_scenes"]  = len(tm["scenes"])
        _write(TIMING_MANIFEST, tm)

        # ── Delete cached files ───────────────────────────────────────────────
        for f in [
            os.path.join(IMAGES_DIR,  f"scene{scene_num}_bg.png"),
            os.path.join(BGM_DIR,     f"scene{scene_num}_bgm.mp3"),
            os.path.join(SCENES_DIR,  f"scene{scene_num}.mp4"),
        ]:
            if os.path.exists(f):
                os.remove(f)
        for seg_id in removed_segments:
            dlg = os.path.join(DIALOGUE_DIR, f"{seg_id}.mp3")
            if os.path.exists(dlg):
                os.remove(dlg)

        return {
            "ok": True,
            "message": f"Scene {scene_num} deleted ({len(removed_segments)} audio segments removed). Re-rendering video.",
            "phase_to_rerun": 3,
        }

    def _add_scene(self, action, scope, params):
        for path in (PHASE2_HANDOFF, PHASE3_HANDOFF):
            if not os.path.exists(path):
                return _missing(path, os.path.basename(path))

        description = params.get("description") or params.get("style") or "a new dramatic scene"
        mood        = params.get("mood", "dramatic")
        if mood not in _VALID_MOODS:
            mood = "dramatic"

        # Determine insertion position
        s = (scope or "end").strip().lower()
        if s.startswith("after:"):
            try:
                insert_after = int(s[6:])
            except ValueError:
                insert_after = None
        else:
            insert_after = None  # append at end

        p3 = _read(PHASE3_HANDOFF)
        p2 = _read(PHASE2_HANDOFF)

        existing_ids = sorted(s["scene_id"] for s in p3.get("scenes", []))
        if not existing_ids:
            return {"ok": False, "message": "No existing scenes found — run Phase 1 first.", "phase_to_rerun": None}

        # ── Generate scene content via LLM ────────────────────────────────────
        print(f"[EditExecutor] Generating new scene: {description!r}")
        scene_data = self._generate_scene(
            description, mood, p3.get("title", ""), p3.get("genre", ""), existing_ids
        )
        if not scene_data:
            return {"ok": False, "message": "LLM failed to generate scene content.", "phase_to_rerun": None}

        # ── Assign new scene_id and insert into sorted list ───────────────────
        new_id = (insert_after + 1) if insert_after and insert_after in existing_ids else (max(existing_ids) + 1)
        scene_data["scene_id"] = new_id

        # If new_id collides or mid-insert, renumber all scenes after the insertion point
        all_scenes_p3 = p3.get("scenes", [])
        if insert_after and insert_after in existing_ids:
            # Shift every scene whose id >= new_id up by 1
            for sc in all_scenes_p3:
                if sc["scene_id"] >= new_id:
                    sc["scene_id"] += 1
            for sc in p2.get("segments", []):
                if sc["scene_id"] >= new_id:
                    sc["scene_id"] += 1
            for mm in p2.get("music_moods", []):
                if mm["scene_id"] >= new_id:
                    mm["scene_id"] += 1
            if os.path.exists(TIMING_MANIFEST):
                tm = _read(TIMING_MANIFEST)
                for sc in tm.get("scenes", []):
                    if sc["scene_id"] >= new_id:
                        sc["scene_id"] += 1
                for seg in tm.get("flat_segments", []):
                    if seg["scene_id"] >= new_id:
                        seg["scene_id"] += 1
                _write(TIMING_MANIFEST, tm)

        # ── Add new scene to phase3 handoff ───────────────────────────────────
        p3_scene = {
            "scene_id":                new_id,
            "location":                scene_data.get("location", "Unknown location"),
            "time_of_day":             scene_data.get("time_of_day", "DAY"),
            "duration_seconds":        scene_data.get("duration_seconds", 30),
            "mood":                    mood,
            "tone":                    scene_data.get("tone", "neutral"),
            "image_generation_prompt": scene_data.get("image_generation_prompt", description),
            "visual_notes":            scene_data.get("visual_notes", description),
            "characters_in_scene":     scene_data.get("characters", []),
            "camera_style":            "cinematic wide shot",
            "transition_to_next":      "fade",
        }
        all_scenes_p3.append(p3_scene)
        all_scenes_p3.sort(key=lambda x: x["scene_id"])
        p3["scenes"] = all_scenes_p3
        _write(PHASE3_HANDOFF, p3)

        # ── Add new segments and music mood to phase2 handoff ─────────────────
        dialogue = scene_data.get("dialogue", [])
        for i, dlg in enumerate(dialogue):
            p2["segments"].append({
                "segment_id": f"scene{new_id}_line{i+1}",
                "scene_id":   new_id,
                "speaker":    dlg.get("speaker", "Character"),
                "line":       dlg.get("line", ""),
                "emotion":    dlg.get("emotion", "neutral"),
                "mood":       mood,
            })
        p2["music_moods"].append({"scene_id": new_id, "mood": mood, "duration_seconds": 30})
        p2["music_moods"].sort(key=lambda x: x["scene_id"])
        _write(PHASE2_HANDOFF, p2)

        return {
            "ok": True,
            "message": (
                f"Scene {new_id} added"
                + (f" after scene {insert_after}" if insert_after else " at the end")
                + f" with {len(dialogue)} dialogue lines. Re-running Phase 2 then Phase 3."
            ),
            "phase_to_rerun": 2,
        }

    def _generate_scene(self, description: str, mood: str, title: str, genre: str,
                        existing_ids: list) -> dict | None:
        """Call Groq to generate a minimal scene JSON for the given description."""
        try:
            from langchain_groq import ChatGroq
            from config import GROQ_API_KEY, GROQ_MODEL
            llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.7)

            system = (
                "You are a screenplay writer. Return ONLY a JSON object with these keys: "
                "location, time_of_day (DAY/NIGHT/DAWN/DUSK), duration_seconds (20-60), "
                "tone (dark/light/neutral/suspenseful/uplifting), "
                "characters (list of character name strings already in the story), "
                "image_generation_prompt (30+ word Stable Diffusion background prompt, no characters), "
                "visual_notes, "
                "dialogue (list of {speaker, line, emotion, visual_cue}). "
                "Minimum 2 dialogue lines. Output ONLY JSON."
            )
            user = (
                f"Story: {title!r} ({genre}). "
                f"Add a new scene — mood: {mood}. Description: {description}"
            )
            raw = llm.invoke([
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]).content

            # Extract JSON from response
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                import json as _json
                return _json.loads(match.group())
        except Exception as e:
            print(f"[EditExecutor] Scene generation failed: {e}")
        return None

    # ── Full regeneration ──────────────────────────────────────────────────────

    def _regenerate_script(self, action, scope, params):
        return {
            "ok": True,
            "message": "Triggering full pipeline regeneration from Phase 1.",
            "phase_to_rerun": 1,
        }
