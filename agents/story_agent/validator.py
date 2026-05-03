"""
agents/story_agent/validator.py
─────────────────────────────────
Validates manually uploaded scripts.
"""

import re
from typing import Any, Dict, List

from agents.base import BaseAgent
from shared.utils.vector_store import memory

VALIDATION_SYSTEM = """
You are a script validation expert. Analyze the given script and return ONLY a JSON object:
{
  "valid": true | false,
  "errors": ["list of structural errors if any"],
  "suggestions": ["list of improvement suggestions"],
  "standardized": { <standardized script JSON in scene_manifest format> }
}

If the script is unrecoverable, set valid=false and leave standardized as null.
Output ONLY JSON, no markdown.
"""


class ValidatorAgent(BaseAgent):
    name      = "ValidatorAgent"
    tool_tags = ["validation", "memory"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[{self.name}] Validating uploaded script...")
        raw_script = state.get("raw_script", "")
        if not raw_script.strip():
            return {**state, "validated": False, "validation_errors": ["No script content provided."]}

        errors = self._structural_checks(raw_script)
        parsed = self.parse_json(self.chat(VALIDATION_SYSTEM, f"Validate:\n\n{raw_script}"))

        if not parsed:
            return {**state, "validated": False, "validation_errors": errors + ["LLM validation failed."]}

        llm_valid    = parsed.get("valid", False)
        llm_errors   = parsed.get("errors", [])
        standardized = parsed.get("standardized")
        is_valid     = len(errors) == 0 and llm_valid and standardized

        if is_valid:
            self.invoke_tool("commit_memory", {
                "key": "script:latest", "data": standardized, "metadata": {"type": "script", "source": "manual"}
            })
            print(f"[{self.name}] Script validated and standardized.")
            return {**state, "script": standardized, "validated": True,
                    "validation_errors": [], "status": "script_ready"}
        else:
            blocking = errors + (llm_errors if not llm_valid else [])
            print(f"[{self.name}] Validation failed: {blocking}")
            return {**state, "validated": False, "validation_errors": blocking, "status": "validation_failed"}

    def _structural_checks(self, script: str) -> List[str]:
        errors = []
        lines  = script.strip().split("\n")
        if not any(re.match(r'^(INT\.|EXT\.|SCENE|Scene)', l.strip()) for l in lines):
            errors.append("No scene headings found.")
        if not any(":" in l or re.match(r'^[A-Z]{2,}$', l.strip()) for l in lines):
            errors.append("No dialogue labels detected.")
        if len(lines) < 10:
            errors.append("Script too short (minimum 10 lines).")
        return errors
