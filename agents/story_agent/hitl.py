"""
agents/story_agent/hitl.py
───────────────────────────
Human-in-the-Loop checkpoint — pauses for human script approval.
"""

import json
import os
import time
from typing import Any, Dict

from agents.base import BaseAgent

HITL_RESPONSE_FILE = os.path.join("data", "outputs", "hitl_response.json")
HITL_TIMEOUT_S     = 300   # 5 minutes before auto-approve


class HITLAgent(BaseAgent):
    name      = "HITLAgent"
    tool_tags = []

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        script = state.get("script", {})
        if not script:
            return {**state, "hitl_approved": False, "status": "error", "error": "No script to review."}

        # ── Legacy auto-approve (CI / old web flag) ───────────────────────────
        if os.environ.get("HITL_AUTO_APPROVE") == "1":
            print(f"[{self.name}] Auto-approve mode: script accepted.")
            return {**state, "hitl_approved": True, "status": "approved"}

        # ── Web mode: sentinel line + response-file polling ───────────────────
        if os.environ.get("HITL_WEB_MODE") == "1":
            return self._web_hitl(state, script)

        # ── Terminal interactive mode ─────────────────────────────────────────
        return self._terminal_hitl(state, script)

    # ── Web HITL ──────────────────────────────────────────────────────────────

    def _web_hitl(self, state: dict, script: dict) -> dict:
        # Clean up any stale response file from a previous run
        if os.path.exists(HITL_RESPONSE_FILE):
            os.remove(HITL_RESPONSE_FILE)

        # Build a compact summary to send to the frontend
        story  = script.get("story", {})
        scenes = script.get("scenes", [])
        summary = {
            "title":    story.get("title", "Untitled"),
            "genre":    story.get("genre", ""),
            "synopsis": story.get("synopsis", ""),
            "scenes": [
                {
                    "scene_id":   s["scene_id"],
                    "location":   s.get("location", ""),
                    "mood":       s.get("mood", ""),
                    "characters": s.get("characters", []),
                    "dialogue":   [
                        {"speaker": d["speaker"], "line": d["line"][:80]}
                        for d in s.get("dialogue", [])[:2]
                    ],
                }
                for s in scenes
            ],
        }

        # Emit sentinel — backend detects this line and forwards to frontend
        print(f"[HITL_WAITING] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        print(f"[{self.name}] Waiting for web approval (timeout {HITL_TIMEOUT_S}s)...", flush=True)

        # Poll for backend-written response file
        deadline = time.time() + HITL_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(0.5)
            if not os.path.exists(HITL_RESPONSE_FILE):
                continue
            try:
                with open(HITL_RESPONSE_FILE, encoding="utf-8") as f:
                    resp = json.load(f)
                os.remove(HITL_RESPONSE_FILE)
            except Exception:
                continue

            action = resp.get("action", "approve")
            if action == "approve":
                print(f"[{self.name}] Web approval received.", flush=True)
                return {**state, "hitl_approved": True, "status": "approved"}
            else:
                new_prompt = resp.get("prompt", state.get("user_prompt", ""))
                print(f"[{self.name}] Web rejection received — regenerating.", flush=True)
                return {**state, "hitl_approved": False,
                        "user_prompt": new_prompt, "status": "regenerate"}

        # Timed out — auto-approve so the pipeline doesn't hang forever
        print(f"[{self.name}] Timeout reached — auto-approving.", flush=True)
        return {**state, "hitl_approved": True, "status": "approved"}

    # ── Terminal HITL ─────────────────────────────────────────────────────────

    def _terminal_hitl(self, state: dict, script: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"[{self.name}] HUMAN REVIEW CHECKPOINT")
        print(f"{'='*60}")

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
