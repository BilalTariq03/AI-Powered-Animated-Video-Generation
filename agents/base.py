"""
agents/base.py
───────────────
Base class for all agents. Provides LLM access, MCP discovery, JSON parsing.
"""

import json
import re
from typing import Any, Dict, List, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from mcp.tool_registry import mcp, MCPToolResult


class BaseAgent:
    name: str       = "BaseAgent"
    tool_tags: List[str] = []

    def __init__(self):
        self.llm   = Groq(api_key=GROQ_API_KEY)
        self.tools = self._discover_tools()

    def _discover_tools(self) -> Dict[str, Any]:
        discovered = mcp.discover(tags=self.tool_tags)
        tool_map   = {t.name: t for t in discovered}
        if tool_map:
            print(f"[{self.name}] Discovered {len(tool_map)} MCP tools: {list(tool_map.keys())}")
        return tool_map

    def invoke_tool(self, tool_name: str, inputs: Dict) -> MCPToolResult:
        if tool_name not in self.tools:
            self.tools = self._discover_tools()
        return mcp.invoke(tool_name, inputs)

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
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
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                try:
                    return json.loads(part.strip())
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None
