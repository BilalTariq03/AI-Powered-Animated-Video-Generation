"""
agents/story_agent/hitl.py
───────────────────────────
Human-in-the-Loop checkpoint — pauses for human script approval.
"""

import json
from typing import Any, Dict

from agents.base import BaseAgent


class HITLAgent(BaseAgent):
    name      = "HITLAgent"
    tool_tags = []

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"[{self.name}] HUMAN REVIEW CHECKPOINT")
        print(f"{'='*60}")

        script = state.get("script", {})
        if not script:
            return {**state, "hitl_approved": False, "status": "error", "error": "No script to review."}

        story = script.get("story", script)
        print(f"\nSCRIPT: {story.get('title', script.get('title', 'Untitled'))}")
        print(f"  Genre : {story.get('genre', script.get('genre', 'Unknown'))}")
        print(f"  Scenes: {len(script.get('scenes', []))}")
        print()

        for scene in script.get("scenes", []):
            print(f"  Scene {scene['scene_id']}: {scene.get('location','')} [{scene.get('time_of_day','')}]")
            print(f"    Characters: {', '.join(scene.get('characters', []))}")
            print(f"    Action    : {scene.get('action','')[:80]}...")
            for dlg in scene.get("dialogue", [])[:2]:
                print(f"    {dlg['speaker']}: \"{dlg['line'][:60]}...\"")
            print()

        print(f"{'='*60}")

        while True:
            choice = input("\nApprove this script? [y = yes / n = no / e = edit prompt]: ").strip().lower()
            if choice == "y":
                print(f"[{self.name}] Script approved.")
                return {**state, "hitl_approved": True, "status": "approved"}
            elif choice == "n":
                new_prompt = input("Enter a revised story prompt: ").strip()
                return {**state, "hitl_approved": False,
                        "user_prompt": new_prompt or state.get("user_prompt", ""),
                        "status": "regenerate"}
            elif choice == "e":
                print("\nCurrent script JSON (truncated):")
                print(json.dumps(script, indent=2)[:2000], "...")
                new_prompt = input("\nEnter revised prompt (or Enter to keep): ").strip()
                if new_prompt:
                    return {**state, "hitl_approved": False, "user_prompt": new_prompt, "status": "regenerate"}
            else:
                print("Please enter 'y', 'n', or 'e'.")
