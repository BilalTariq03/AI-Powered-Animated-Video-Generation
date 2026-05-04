"""
agents/edit_agent/agent.py
───────────────────────────
LangGraph-based edit intent classification agent.

Graph: classify → execute → END

The MemorySaver checkpointer keeps conversation context across
multiple edits within the same browser session (thread_id).
"""

from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ── Pydantic model for structured LLM output ──────────────────────────────────

class EditIntent(BaseModel):
    intent: str = Field(
        description=(
            "Exact intent name — one of: change_voice_tone, make_scene_darker, "
            "make_scene_lighter, change_scene_mood, add_background_music, "
            "remove_subtitle, add_subtitle, change_character_design, "
            "speed_up_scene, slow_down_scene, regenerate_script, change_scene_background"
        )
    )
    target: Literal["audio", "video_frame", "video", "script"] = Field(
        description="Pipeline component being edited"
    )
    scope: str = Field(
        description=(
            "What is being targeted: 'character:Name', 'scene:N' (N=scene number), or 'all'"
        )
    )
    parameters: dict = Field(
        default_factory=dict,
        description=(
            "Action-specific values extracted from the query. "
            "Examples: {tone: 'deep'}, {mood: 'hopeful'}, {factor: 1.5}, "
            "{description: 'anime style warrior'}"
        ),
    )
    reasoning: str = Field(description="One-sentence explanation of the classification")


# ── LangGraph state ────────────────────────────────────────────────────────────

class EditState(TypedDict):
    query:  str
    intent: Optional[dict]
    result: Optional[dict]
    error:  Optional[str]


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """You classify free-text video editing requests for an AI story pipeline.
Output a structured JSON with intent, target, scope, parameters, and reasoning.

INTENT NAMES (use exactly):
  change_voice_tone       — change a character's voice style/tone
  make_scene_darker       — darken a scene visually
  make_scene_lighter      — brighten a scene visually
  change_scene_mood       — change the overall aesthetic/atmosphere of a scene
  add_background_music    — add or change BGM for a scene
  remove_subtitle         — remove subtitle captions from the video
  add_subtitle            — add subtitle captions to the video
  change_character_design — regenerate a character portrait / change appearance
  speed_up_scene          — make a scene play faster
  slow_down_scene         — make a scene play slower
  regenerate_script       — regenerate the entire story from scratch
  change_scene_background — replace the background image for a scene

TARGET CATEGORIES:
  "audio"       — voice or background music
  "video_frame" — character images or scene backgrounds
  "video"       — subtitle, speed, or compositing
  "script"      — story / script regeneration

SCOPE FORMAT (pick the most specific one):
  "character:Name"  — specific character (e.g. "character:Detective Kim")
  "scene:N"         — specific scene number (e.g. "scene:2")
  "all"             — applies to everything

PARAMETER KEYS TO EXTRACT:
  tone / voice_style : deep | soft | authoritative | raspy | whispery | high-pitched | neutral
  mood               : dramatic | tense | mysterious | hopeful | melancholic | comedic | romantic
  factor             : numeric speed multiplier (e.g. 2.0)
  description        : free-text image description for visual changes
  style              : art style descriptor (e.g. "anime", "noir", "watercolor")"""


# ── LLM singleton ─────────────────────────────────────────────────────────────

_structured_llm = None


def _get_llm():
    global _structured_llm
    if _structured_llm is None:
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY, GROQ_MODEL
        base = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
        _structured_llm = base.with_structured_output(EditIntent)
    return _structured_llm


# ── Graph nodes ────────────────────────────────────────────────────────────────

def _classify_node(state: EditState) -> EditState:
    try:
        llm: EditIntent = _get_llm().invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": state["query"]},
        ])
        print(
            f"[EditAgent] intent={llm.intent!r} target={llm.target} "
            f"scope={llm.scope!r} params={llm.parameters}"
        )
        return {**state, "intent": llm.model_dump(), "error": None}
    except Exception as exc:
        return {**state, "intent": None, "error": f"Classification failed: {exc}"}


def _execute_node(state: EditState) -> EditState:
    if state.get("error") or not state.get("intent"):
        return state
    from agents.edit_agent.executor import EditExecutor
    try:
        result = EditExecutor().execute(state["intent"])
        return {**state, "result": result}
    except Exception as exc:
        return {**state, "result": None, "error": f"Execution failed: {exc}"}


# ── Build and compile graph ────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(EditState)
    g.add_node("classify", _classify_node)
    g.add_node("execute",  _execute_node)
    g.add_edge(START,      "classify")
    g.add_edge("classify", "execute")
    g.add_edge("execute",  END)
    return g.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_edit(query: str, thread_id: str = "default") -> dict:
    """
    Classify and execute a free-text edit query.

    Returns the final EditState dict which includes:
      - intent:  classified EditIntent (dict)
      - result:  executor result with ok, message, phase_to_rerun
      - error:   string if something went wrong, else None
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"query": query, "intent": None, "result": None, "error": None},
        config,
    )
