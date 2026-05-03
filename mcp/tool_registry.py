"""
mcp/tool_registry.py
─────────────────────
MCP tool registry — agents discover and invoke tools through here.
No agent ever imports a tool implementation directly.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


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


class MCPRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPToolSchema] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(self, schema: MCPToolSchema, handler: Callable):
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler
        print(f"[MCP] Registered tool: {schema.name}")

    def discover(self, tags: Optional[List[str]] = None) -> List[MCPToolSchema]:
        if not tags:
            return list(self._tools.values())
        return [t for t in self._tools.values() if any(tag in t.tags for tag in tags)]

    def invoke(self, tool_name: str, inputs: Dict[str, Any]) -> MCPToolResult:
        if tool_name not in self._handlers:
            return MCPToolResult(
                tool=tool_name, success=False, data=None,
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


mcp = MCPRegistry()
