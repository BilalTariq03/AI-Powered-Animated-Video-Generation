"""
agents/audio_agent/run.py
──────────────────────────
Entry point for Phase 2 - Audio Generation.

Usage:
    python agents/audio_agent/run.py
    python agents/audio_agent/run.py --handoff data/outputs/phase2_audio_handoff.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import ELEVENLABS_API_KEY, PHASE2_HANDOFF
from agents.audio_agent.agent import AudioGenerationAgent

BANNER = """
+----------------------------------------------------------+
|       AUDIO GENERATION - Phase 2                         |
|       CS-4015 Agentic AI | NUCES                         |
+----------------------------------------------------------+
"""


def main():
    print(BANNER)

    handoff_path = PHASE2_HANDOFF
    if len(sys.argv) == 3 and sys.argv[1] == "--handoff":
        handoff_path = sys.argv[2]

    if not os.path.exists(handoff_path):
        print(f"[ERROR] Handoff file not found: {handoff_path}")
        print("   Run Phase 1 first:  python main.py")
        sys.exit(1)

    if ELEVENLABS_API_KEY:
        print("[OK] ELEVENLABS_API_KEY found - using cloud TTS.")
    else:
        print("[i]  No ELEVENLABS_API_KEY set - using gTTS (free).")

    agent    = AudioGenerationAgent(ELEVENLABS_API_KEY)
    manifest = agent.run(handoff_path)

    segs   = len(manifest.get("flat_segments", []))
    scenes = manifest.get("total_scenes", 0)
    print(f"\n[DONE] Phase 2 complete - {segs} audio segments across {scenes} scenes.")
    print("   Check data/outputs/audio/ for generated files.")


if __name__ == "__main__":
    main()
