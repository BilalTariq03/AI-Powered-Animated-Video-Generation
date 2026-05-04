"""
agents/edit_agent/executor.py
──────────────────────────────
Applies a classified EditIntent to the pipeline's JSON state files.

Each handler modifies only the relevant handoff / manifest JSON and
returns {"ok": bool, "message": str, "phase_to_rerun": int | None}.
The caller is responsible for triggering the phase re-run.
"""

import json
import os

from config import (
    PHASE2_HANDOFF, PHASE3_HANDOFF,
    TIMING_MANIFEST, IMAGES_DIR,
)

EDIT_SETTINGS = os.path.join("data", "outputs", "edit_settings.json")

_VALID_VOICE_STYLES = {
    "deep", "authoritative", "raspy", "soft", "high-pitched", "whispery", "neutral"
}

_VALID_MOODS = {
    "dramatic", "tense", "mysterious", "hopeful", "melancholic", "comedic", "romantic"
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
                if description:
                    cv["image_prompt"] = (
                        f"{description}, cinematic realism, highly detailed, photorealistic"
                    )
                safe = cv["name"].lower().replace(" ", "_")
                for img in (f"{safe}.png", f"{safe}.jpg"):
                    p = os.path.join(IMAGES_DIR, img)
                    if os.path.exists(p):
                        os.remove(p)
                changed.append(cv["name"])

        if not changed:
            return {"ok": False, "message": f"No character matched scope {scope!r}.", "phase_to_rerun": None}

        _write(PHASE3_HANDOFF, handoff)
        return {
            "ok": True,
            "message": f"Design prompt updated for {changed}. Portrait will regenerate in Phase 3.",
            "phase_to_rerun": 3,
        }

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

    # ── Full regeneration ──────────────────────────────────────────────────────

    def _regenerate_script(self, action, scope, params):
        return {
            "ok": True,
            "message": "Triggering full pipeline regeneration from Phase 1.",
            "phase_to_rerun": 1,
        }
