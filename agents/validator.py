"""
agents/validator.py
───────────────────
Script Validator Agent
  Role: Validate manually uploaded scripts for correct structure.
  Checks: scene headings, dialogue labels, action descriptions.
"""

import json
import re
from typing import Any, Dict, List

from agents.base import BaseAgent
from memory.vector_store import memory


VALIDATION_SYSTEM = """
You are a script validation expert. Analyze the given script and return ONLY a JSON object:
{
  "valid": true | false,
  "errors": ["list of structural errors if any"],
  "suggestions": ["list of improvement suggestions"],
  "standardized": { <standardized script JSON in scene_manifest format> }
}

Standardized format must follow:
{
  "title": "...",
  "genre": "...",
  "scenes": [
    {
      "scene_id": 1,
      "location": "...",
      "time_of_day": "DAY|NIGHT|DAWN|DUSK",
      "characters": [],
      "action": "...",
      "dialogue": [{"speaker": "...", "line": "...", "visual_cue": "..."}],
      "visual_notes": "..."
    }
  ]
}

If the script is unrecoverable, set valid=false and leave standardized as null.
Output ONLY JSON, no markdown.
"""


class ValidatorAgent(BaseAgent):
    name = "ValidatorAgent"
    tool_tags = ["validation", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Validating uploaded script...")
        raw_script = state.get("raw_script", "")

        if not raw_script.strip():
            return {**state, "validated": False, "validation_errors": ["No script content provided."]}

        # ── Structural checks (rule-based) ────────────────────────────────────
        errors = self._structural_checks(raw_script)

        # ── LLM-based deep validation + standardization ───────────────────────
        llm_response = self.chat(
            VALIDATION_SYSTEM,
            f"Validate and standardize this script:\n\n{raw_script}"
        )
        parsed = self.parse_json(llm_response)

        if not parsed:
            return {**state, "validated": False, "validation_errors": errors + ["LLM validation failed."]}

        # Hard failures: structural rule errors OR LLM explicitly says invalid
        structural_errors = errors  # from rule-based checks only
        llm_valid = parsed.get("valid", False)
        llm_errors = parsed.get("errors", [])
        suggestions = parsed.get("suggestions", [])
        standardized = parsed.get("standardized")

        # Only fail if structural checks found problems OR LLM says invalid
        is_valid = len(structural_errors) == 0 and llm_valid and standardized

        if is_valid:
            if llm_errors:
                print(f"[{self.name}] ⚠ Minor suggestions (not blocking): {llm_errors}")
            if suggestions:
                print(f"[{self.name}] 💡 Suggestions: {suggestions}")
            self.invoke_tool("commit_memory", {
                "key": "script:latest",
                "data": standardized,
                "metadata": {"type": "script", "source": "manual"}
            })
            print(f"[{self.name}] ✓ Script validated and standardized.")
            return {
                **state,
                "script": standardized,
                "validated": True,
                "validation_errors": [],
                "status": "script_ready"
            }
        else:
            blocking_errors = structural_errors + (llm_errors if not llm_valid else [])
            print(f"[{self.name}] ✗ Validation failed: {blocking_errors}")
            return {
                **state,
                "validated": False,
                "validation_errors": blocking_errors,
                "status": "validation_failed"
            }

    def _structural_checks(self, script: str) -> List[str]:
        """Rule-based structural checks before LLM validation."""
        errors = []
        lines = script.strip().split("\n")

        has_scene_heading = any(
            re.match(r'^(INT\.|EXT\.|SCENE|Scene)', line.strip())
            for line in lines
        )
        if not has_scene_heading:
            errors.append("No scene headings found (expected INT./EXT. or SCENE N).")

        has_dialogue = any(":" in line or re.match(r'^[A-Z]{2,}$', line.strip()) for line in lines)
        if not has_dialogue:
            errors.append("No dialogue labels detected.")

        if len(lines) < 10:
            errors.append("Script is too short (minimum 10 lines expected).")

        return errors
