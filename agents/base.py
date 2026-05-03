"""
agents/base.py
──────────────
Base class for all agents.
Each agent discovers its tools from MCP registry at init time — no hardcoding.
"""

import json
from typing import Any, Dict, List, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from mcp_registry import mcp, MCPToolResult


class BaseAgent:
    """
    All agents inherit from this.
    Provides:
      - LLM access via Groq
      - MCP tool discovery + invocation
      - JSON parsing utilities
    """

    name: str = "BaseAgent"
    tool_tags: List[str] = []       # tags used to discover relevant MCP tools

    def __init__(self):
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.tools = self._discover_tools()

    def _discover_tools(self) -> Dict[str, Any]:
        """Dynamically discover tools from MCP registry at runtime."""
        discovered = mcp.discover(tags=self.tool_tags)
        tool_map = {t.name: t for t in discovered}
        if tool_map:
            print(f"[{self.name}] Discovered {len(tool_map)} MCP tools: {list(tool_map.keys())}")
        return tool_map

    def invoke_tool(self, tool_name: str, inputs: Dict) -> MCPToolResult:
        """Invoke an MCP tool by name."""
        if tool_name not in self.tools:
            # Re-discover in case registry was updated
            self.tools = self._discover_tools()
        return mcp.invoke(tool_name, inputs)

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        """Call Groq LLM and return response text."""
        response = self.llm.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()

    def parse_json(self, text: str) -> Optional[Any]:
        """Extract and parse JSON from LLM output (handles markdown fences)."""
        # Strip markdown code fences
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                try:
                    return json.loads(part.strip())
                except Exception:
                    continue

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            # Try to find JSON object/array within text
            import re
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None
