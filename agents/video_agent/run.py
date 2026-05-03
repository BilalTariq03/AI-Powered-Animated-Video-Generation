"""
agents/video_agent/run.py
──────────────────────────
Phase 3 entry point.

Run:
    python agents/video_agent/run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mcp  # noqa: F401 — triggers MCP tool registrations

from agents.video_agent.agent import VideoGenerationAgent
from config import PHASE3_HANDOFF, TIMING_MANIFEST

print("""
+----------------------------------------------------------+
|       VIDEO GENERATION - Phase 3                         |
|       CS-4015 Agentic AI | NUCES                         |
+----------------------------------------------------------+
""")

if not os.path.exists(PHASE3_HANDOFF):
    print(f"[ERROR] Phase 3 handoff not found: {PHASE3_HANDOFF}")
    print("        Run Phase 1 first: python main.py")
    sys.exit(1)

if not os.path.exists(TIMING_MANIFEST):
    print(f"[ERROR] Timing manifest not found: {TIMING_MANIFEST}")
    print("        Run Phase 2 first: python agents/audio_agent/run.py")
    sys.exit(1)

agent      = VideoGenerationAgent()
final_path = agent.run(PHASE3_HANDOFF, TIMING_MANIFEST)

print(f"\n[DONE] Phase 3 complete.")
print(f"   Final video -> {final_path}")
print(f"   Scene clips -> data/outputs/video/scenes/")
