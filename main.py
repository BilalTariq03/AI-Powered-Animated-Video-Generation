"""
main.py — The Writer's Room: Autonomous Story & Image Generation
═══════════════════════════════════════════════════════════════════
CS-4015 Agentic AI — Course Project Phase 1
NUCES

Run:
    python main.py

Environment variables needed (.env file):
    GROQ_API_KEY=your_groq_key_here
    HF_API_KEY=your_huggingface_key_here   (optional, for real images)
"""

import json
import os
import sys

# ── Bootstrap: register all MCP tools before agents start ────────────────────
import tools  # noqa: F401  — side-effect: registers all tools into mcp registry

from config import OUTPUT_DIR, IMAGES_DIR, MANIFEST, CHAR_DB
from workflow.graph import workflow, WritersRoomState


def save_outputs(state: WritersRoomState):
    """Write final JSON outputs to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # scene_manifest.json
    script = state.get("script", {})
    with open(MANIFEST, "w") as f:
        json.dump(script, f, indent=2)
    print(f"\n✅ Saved: {MANIFEST}")

    # character_db.json
    characters = state.get("characters", [])
    char_db = {
        "total": len(characters),
        "characters": characters
    }
    with open(CHAR_DB, "w") as f:
        json.dump(char_db, f, indent=2)
    print(f"✅ Saved: {CHAR_DB}")

    # images summary
    images = state.get("images", [])
    if images:
        print(f"✅ Images: {len(images)} saved in {IMAGES_DIR}/")
        for img in images:
            print(f"   • {img['character']} → {img.get('path', 'N/A')}")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       THE WRITER'S ROOM — Autonomous Story Generator     ║
║       CS-4015 Agentic AI | NUCES | Phase 1               ║
╚══════════════════════════════════════════════════════════╝
""")


def get_user_input() -> WritersRoomState:
    """Collect input from user and build initial state."""
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
            input_mode="auto",
            user_prompt=prompt,
            raw_script="",
            script={},
            validated=False,
            validation_errors=[],
            hitl_approved=False,
            characters=[],
            images=[],
            status="init",
            error=None,
            iteration=0
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
        raw_script = "\n".join(lines)

        return WritersRoomState(
            input_mode="manual",
            user_prompt="",
            raw_script=raw_script,
            script={},
            validated=False,
            validation_errors=[],
            hitl_approved=False,
            characters=[],
            images=[],
            status="init",
            error=None,
            iteration=0
        )


def main():
    print_banner()

    # Check API key
    from config import GROQ_API_KEY
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not set.")
        print("   Create a .env file with: GROQ_API_KEY=your_key_here")
        print("   Get your free key at: https://console.groq.com")
        sys.exit(1)

    # Get initial state from user
    initial_state = get_user_input()

    print(f"\n🚀 Starting LangGraph workflow in '{initial_state['input_mode']}' mode...")
    print("=" * 60)

    # ── Run LangGraph workflow ─────────────────────────────────────────────────
    final_state = workflow.invoke(initial_state)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if final_state.get("status") == "complete":
        print("🎉 PIPELINE COMPLETE!")
        save_outputs(final_state)
    elif final_state.get("error"):
        print(f"❌ Pipeline failed: {final_state['error']}")
    elif final_state.get("status") == "validation_failed":
        errors = final_state.get("validation_errors", [])
        print(f"❌ Script validation failed:")
        for e in errors:
            print(f"   • {e}")
    else:
        print(f"⚠ Pipeline ended with status: {final_state.get('status')}")
        # Save whatever we have
        if final_state.get("script"):
            save_outputs(final_state)

    print("\nDone. Check the output/ folder for your files.")


if __name__ == "__main__":
    main()
