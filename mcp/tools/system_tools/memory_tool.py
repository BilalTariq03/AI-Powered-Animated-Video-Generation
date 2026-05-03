"""
mcp/tools/system_tools/memory_tool.py
───────────────────────────────────────
MCP tools: commit_memory, query_stock_footage
Wrap the shared VectorMemory store for agent use.
"""

from mcp.tool_registry import mcp, MCPToolSchema
from shared.utils.vector_store import memory


def _commit_memory(key: str, data, metadata: dict = None) -> str:
    doc_id = memory.store(key, data, metadata or {})
    return f"Stored at key='{key}' with id={doc_id}"


def _query_stock_footage(character_name: str, style: str = "cinematic") -> dict:
    results = memory.query(f"character {character_name} {style}", n_results=3)
    if results:
        return {"found": True, "references": results}
    return {"found": False, "references": [], "note": "No prior visual references found."}


mcp.register(
    MCPToolSchema(
        name="commit_memory",
        description="Persists agent output to vector memory.",
        input_schema={
            "key":      {"type": "string", "description": "Unique storage key"},
            "data":     {"type": "any",    "description": "Data to store"},
            "metadata": {"type": "object", "description": "Optional metadata tags"}
        },
        tags=["memory", "storage"]
    ),
    handler=_commit_memory
)

mcp.register(
    MCPToolSchema(
        name="query_stock_footage",
        description="Searches memory for existing visual references for a character.",
        input_schema={
            "character_name": {"type": "string"},
            "style":          {"type": "string", "default": "cinematic"}
        },
        tags=["character", "memory", "image"]
    ),
    handler=_query_stock_footage
)
