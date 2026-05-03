"""
mcp/tools/llm_tools/text_generator.py
───────────────────────────────────────
MCP tool: generate_script_segment
Uses Groq LLM to produce a structured screenplay JSON from a prompt.
"""

from mcp.tool_registry import mcp, MCPToolSchema


def _generate_script_segment(prompt: str, num_scenes: int = 5) -> str:
    from groq import Groq
    from config import GROQ_API_KEY, GROQ_MODEL

    client = Groq(api_key=GROQ_API_KEY)
    system = (
        "You are a professional screenplay writer. "
        "Generate a structured multi-scene screenplay in valid JSON ONLY. "
        'Schema: {"title":"...","genre":"...","scenes":[{"scene_id":1,"location":"...",'
        '"time_of_day":"DAY","characters":[],"action":"...","dialogue":[{"speaker":"...",'
        '"line":"...","visual_cue":"..."}],"visual_notes":"..."}]} '
        "Output ONLY JSON. No markdown, no explanation."
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Write a {num_scenes}-scene screenplay for: {prompt}"}
        ],
        temperature=0.8,
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()


mcp.register(
    MCPToolSchema(
        name="generate_script_segment",
        description="Generates a multi-scene screenplay from a story prompt using an LLM.",
        input_schema={
            "prompt":     {"type": "string",  "description": "Story premise"},
            "num_scenes": {"type": "integer", "description": "Number of scenes", "default": 5}
        },
        tags=["script", "generation"]
    ),
    handler=_generate_script_segment
)
