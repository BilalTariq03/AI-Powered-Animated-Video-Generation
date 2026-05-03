"""
main.py

Run:
    python main.py
"""

import glob
import json
import os
import sys

import tools  # noqa: F401

from config import (
    OUTPUT_DIR, IMAGES_DIR,
    MANIFEST, CHAR_DB,
    PHASE2_HANDOFF, PHASE3_HANDOFF
)
from workflow.graph import workflow, WritersRoomState


# ─────────────────────────────────────────────────────────────────────────────
# Handoff builders
# ─────────────────────────────────────────────────────────────────────────────

def build_phase2_handoff(script: dict, characters: list) -> dict:
    """Build phase2_audio_handoff.json consumed by Phase 2 (Audio Generation)."""
    voice_configs = []
    for char in characters:
        vp = char.get("voice_profile", {})
        voice_configs.append({
            "character_name":  char["name"],
            "voice_style":     vp.get("voice_style", "neutral"),
            "speaking_speed":  vp.get("speaking_speed", "normal"),
            "pitch":           vp.get("pitch", "medium"),
            "tts_description": vp.get("tts_description", ""),
            "emotion_range":   vp.get("emotion_range", ["neutral"])
        })

    segments = []
    for scene in script.get("scenes", []):
        scene_id = scene["scene_id"]
        mood     = scene.get("mood", "neutral")
        for i, dlg in enumerate(scene.get("dialogue", [])):
            segments.append({
                "segment_id": f"scene{scene_id}_line{i+1}",
                "scene_id":   scene_id,
                "speaker":    dlg["speaker"],
                "line":       dlg["line"],
                "emotion":    dlg.get("emotion", "neutral"),
                "mood":       mood,
            })

    music_moods = [
        {
            "scene_id":         s["scene_id"],
            "mood":             s.get("mood", "neutral"),
            "tone":             s.get("tone", "neutral"),
            "duration_seconds": s.get("duration_seconds", 30)
        }
        for s in script.get("scenes", [])
    ]

    return {
        "title":         script.get("story", {}).get("title", ""),
        "voice_configs": voice_configs,
        "segments":      segments,
        "music_moods":   music_moods
    }


def build_phase3_handoff(script: dict, characters: list) -> dict:
    """Build phase3_video_handoff.json consumed by Phase 3 (Video Composition)."""
    scenes_visual = []
    for scene in script.get("scenes", []):
        scenes_visual.append({
            "scene_id":                scene["scene_id"],
            "location":                scene.get("location", ""),
            "time_of_day":             scene.get("time_of_day", "DAY"),
            "duration_seconds":        scene.get("duration_seconds", 30),
            "mood":                    scene.get("mood", "neutral"),
            "tone":                    scene.get("tone", "neutral"),
            "image_generation_prompt": scene.get("image_generation_prompt", scene.get("visual_notes", "")),
            "visual_notes":            scene.get("visual_notes", ""),
            "characters_in_scene":     scene.get("characters", []),
            "camera_style":            "cinematic wide shot",
            "transition_to_next":      "fade"
        })

    character_visuals = [
        {
            "name":            char["name"],
            "image_prompt":    char.get("image_prompt", ""),
            "reference_style": char.get("reference_style", "cinematic realism"),
            "scenes_appeared": char.get("scenes_appeared", [])
        }
        for char in characters
    ]

    return {
        "title":             script.get("story", {}).get("title", ""),
        "genre":             script.get("story", {}).get("genre", ""),
        "scenes":            scenes_visual,
        "character_visuals": character_visuals
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(state: WritersRoomState):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    script     = state.get("script", {})
    characters = state.get("characters", [])

    # ── scene_manifest.json — unified { story, scenes[], characters[] } ───────
    unified = {
        "story":      script.get("story", {}),
        "scenes":     script.get("scenes", []),
        "characters": characters
    }
    with open(MANIFEST, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"\n✅ Saved: {MANIFEST}  (unified PhaseOneOutput)")

    # ── character_db.json — standalone character store ────────────────────────
    char_db = {"total": len(characters), "characters": characters}
    with open(CHAR_DB, "w") as f:
        json.dump(char_db, f, indent=2)
    print(f"✅ Saved: {CHAR_DB}")

    # ── phase2_audio_handoff.json ─────────────────────────────────────────────
    phase2 = build_phase2_handoff(script, characters)
    with open(PHASE2_HANDOFF, "w") as f:
        json.dump(phase2, f, indent=2)
    print(f"✅ Saved: {PHASE2_HANDOFF}  "
          f"({len(phase2['segments'])} audio segments, {len(phase2['voice_configs'])} voice configs)")

    # ── phase3_video_handoff.json ─────────────────────────────────────────────
    phase3 = build_phase3_handoff(script, characters)
    with open(PHASE3_HANDOFF, "w") as f:
        json.dump(phase3, f, indent=2)
    print(f"✅ Saved: {PHASE3_HANDOFF}  "
          f"({len(phase3['scenes'])} scenes, {len(phase3['character_visuals'])} characters)")

    # ── Images summary ────────────────────────────────────────────────────────
    images = state.get("images", [])
    if images:
        print(f"✅ Images: {len(images)} saved in {IMAGES_DIR}/")
        for img in images:
            print(f"   • {img['character']} → {img.get('path', 'N/A')}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       THE WRITER'S ROOM — Autonomous Story Generator     ║
║       CS-4015 Agentic AI | NUCES | Phase 1               ║
╚══════════════════════════════════════════════════════════╝
""")


def get_user_input() -> WritersRoomState:
    print("Select mode:")
    print("  1. Auto — Generate script from a prompt (recommended)")
    print("  2. Manual — Upload your own script\n")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            break
        print("Please enter 1 or 2.")

    if choice == "1":
        print("\n📝 Auto Mode: Enter your story idea below.")
        print("   Example: 'A detective in 2087 discovers AI has been committing crimes.'")
        prompt = input("\nYour story prompt: ").strip()
        if not prompt:
            prompt = "A young astronaut discovers an alien signal that could save — or destroy — humanity."
            print(f"Using default prompt: {prompt}")

        return WritersRoomState(
            input_mode="auto", user_prompt=prompt, raw_script="",
            script={}, validated=False, validation_errors=[],
            hitl_approved=False, characters=[], images=[],
            status="init", error=None, iteration=0
        )
    else:
        print("\n📂 Manual Mode: Paste your script below.")
        print("   (Type END on a new line when done)\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)

        return WritersRoomState(
            input_mode="manual", user_prompt="", raw_script="\n".join(lines),
            script={}, validated=False, validation_errors=[],
            hitl_approved=False, characters=[], images=[],
            status="init", error=None, iteration=0
        )


def main():
    print_banner()

    from config import GROQ_API_KEY
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not set.")
        print("   Create a .env file with: GROQ_API_KEY=your_key_here")
        sys.exit(1)

    for f in glob.glob(os.path.join(IMAGES_DIR, "*.png")):
        os.remove(f)

    initial_state = get_user_input()
    print(f"\n🚀 Starting LangGraph workflow in '{initial_state['input_mode']}' mode...")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state.get("status") == "complete":
        print("🎉 PIPELINE COMPLETE!")
        save_outputs(final_state)
    elif final_state.get("error"):
        print(f"❌ Pipeline failed: {final_state['error']}")
    elif final_state.get("status") == "validation_failed":
        for e in final_state.get("validation_errors", []):
            print(f"   • {e}")
    else:
        print(f"⚠ Pipeline ended with status: {final_state.get('status')}")
        if final_state.get("script"):
            save_outputs(final_state)

    print("\nDone. Check the output/ folder for your files.")


if __name__ == "__main__":
    main()