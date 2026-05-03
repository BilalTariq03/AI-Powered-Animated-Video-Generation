"""
mcp_registry.py
───────────────
Simulates an MCP (Model Context Protocol) tool registry server.
Agents NEVER call tools directly — they always query this registry
at runtime to discover available tools and invoke them via schema.

This satisfies the assignment constraint:
  "All tools must be discovered dynamically via MCP – no hardcoded APIs."
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel


# ── Tool Schema (MCP-style) ───────────────────────────────────────────────────

class MCPToolSchema(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    tags: List[str] = []


class MCPToolResult(BaseModel):
    tool: str
    success: bool
    data: Any
    error: Optional[str] = None
    timestamp: float = 0.0

    def model_post_init(self, __context):
        self.timestamp = time.time()


# ── MCP Registry ──────────────────────────────────────────────────────────────

class MCPRegistry:
    """
    Central registry for all MCP tools.
    Agents call `discover(tags)` to find relevant tools,
    then `invoke(tool_name, inputs)` to execute them.
    """

    def __init__(self):
        self._tools: Dict[str, MCPToolSchema] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(self, schema: MCPToolSchema, handler: Callable):
        """Register a tool with its schema and handler function."""
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler
        print(f"[MCP] Registered tool: {schema.name}")

    def discover(self, tags: Optional[List[str]] = None) -> List[MCPToolSchema]:
        """
        Agents call this to discover available tools at runtime.
        Optionally filter by tags (e.g., 'script', 'image', 'memory').
        """
        if not tags:
            return list(self._tools.values())
        return [t for t in self._tools.values() if any(tag in t.tags for tag in tags)]

    def invoke(self, tool_name: str, inputs: Dict[str, Any]) -> MCPToolResult:
        """Execute a discovered tool by name with given inputs."""
        if tool_name not in self._handlers:
            return MCPToolResult(
                tool=tool_name,
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found in MCP registry."
            )
        try:
            result = self._handlers[tool_name](**inputs)
            return MCPToolResult(tool=tool_name, success=True, data=result)
        except Exception as e:
            return MCPToolResult(tool=tool_name, success=False, data=None, error=str(e))

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schema(self, tool_name: str) -> Optional[MCPToolSchema]:
        return self._tools.get(tool_name)


# ── Global Registry Singleton ─────────────────────────────────────────────────

mcp = MCPRegistry()
