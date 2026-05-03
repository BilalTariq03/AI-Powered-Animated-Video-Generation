"""
workflow/graph.py
─────────────────
LangGraph Stateful Workflow for The Writer's Room.

Nodes (LangGraph):
  mode_selector_node  → routes to validator or scriptwriter
  validator_node      → validates manual script
  scriptwriter_node   → generates script from prompt
  hitl_node           → human review checkpoint
  character_node      → extracts character identities
  image_node          → generates character images
  memory_commit_node  → final output persistence

State flows through all nodes via WritersRoomState (TypedDict).
"""

from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from agents.scriptwriter import ScriptwriterAgent
from agents.validator import ValidatorAgent
from agents.hitl import HITLAgent
from agents.character_designer import CharacterDesignerAgent
from agents.image_synthesizer import ImageSynthesizerAgent
from memory.vector_store import memory


# ── Shared State Schema ───────────────────────────────────────────────────────

class WritersRoomState(TypedDict):
    input_mode: str               # "manual" | "auto"
    user_prompt: str              # prompt for auto mode
    raw_script: str               # raw text for manual mode
    script: Dict                  # structured screenplay dict
    validated: bool               # was manual script validated?
    validation_errors: List[str]  # errors from validator
    hitl_approved: bool           # approved by human?
    characters: List[Dict]        # extracted character profiles
    images: List[Dict]            # generated image metadata
    status: str                   # current pipeline status
    error: Optional[str]          # error message if any
    iteration: int                # regeneration counter


# ── Agent Instances ───────────────────────────────────────────────────────────

scriptwriter  = ScriptwriterAgent()
validator     = ValidatorAgent()
hitl          = HITLAgent()
char_designer = CharacterDesignerAgent()
img_synth     = ImageSynthesizerAgent()


# ── Node Functions ────────────────────────────────────────────────────────────

def mode_selector_node(state: WritersRoomState) -> WritersRoomState:
    """Decides whether to run validator or scriptwriter based on input_mode."""
    print(f"\n[mode_selector_node] Mode: {state['input_mode']}")
    if state["input_mode"] == "manual" and state.get("raw_script"):
        return {**state, "status": "validate"}
    else:
        return {**state, "status": "generate"}


def validator_node(state: WritersRoomState) -> WritersRoomState:
    """Validates manually uploaded script."""
    return validator.run(state)


def scriptwriter_node(state: WritersRoomState) -> WritersRoomState:
    """Generates screenplay autonomously from prompt."""
    return scriptwriter.run(state)


def hitl_node(state: WritersRoomState) -> WritersRoomState:
    """Human review checkpoint."""
    return hitl.run(state)


def character_node(state: WritersRoomState) -> WritersRoomState:
    """Extracts character identities from approved script."""
    return char_designer.run(state)


def image_node(state: WritersRoomState) -> WritersRoomState:
    """Generates character images."""
    return img_synth.run(state)


def memory_commit_node(state: WritersRoomState) -> WritersRoomState:
    """Final node: persist all outputs to memory and mark complete."""
    print(f"\n[memory_commit_node] Committing final outputs...")

    memory.store("output:final_state", {
        "title": state.get("script", {}).get("title"),
        "num_scenes": len(state.get("script", {}).get("scenes", [])),
        "num_characters": len(state.get("characters", [])),
        "num_images": len(state.get("images", [])),
        "status": "complete"
    }, {"type": "final_output"})

    print(f"[memory_commit_node] ✓ All outputs committed to memory.")
    return {**state, "status": "complete"}


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_mode(state: WritersRoomState) -> str:
    if state["status"] == "validate":
        return "validator_node"
    return "scriptwriter_node"


def route_after_script_ready(state: WritersRoomState) -> str:
    """After script is ready (from either agent), go to HITL."""
    if state["status"] in ("script_ready", "approved"):
        return "hitl_node"
    if state["status"] == "validation_failed":
        return END   # Can't proceed without valid script
    return "hitl_node"


def route_after_hitl(state: WritersRoomState) -> str:
    """After human review, either continue or regenerate."""
    if state.get("hitl_approved"):
        return "character_node"
    if state.get("status") == "regenerate":
        iteration = state.get("iteration", 0) + 1
        if iteration >= 3:
            print("[route_after_hitl] Max regeneration attempts reached.")
            return END
        state["iteration"] = iteration
        return "scriptwriter_node"
    return END


# ── Build LangGraph StateGraph ────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    graph = StateGraph(WritersRoomState)

    # Add all nodes
    graph.add_node("mode_selector_node", mode_selector_node)
    graph.add_node("validator_node",     validator_node)
    graph.add_node("scriptwriter_node",  scriptwriter_node)
    graph.add_node("hitl_node",          hitl_node)
    graph.add_node("character_node",     character_node)
    graph.add_node("image_node",         image_node)
    graph.add_node("memory_commit_node", memory_commit_node)

    # Entry point
    graph.set_entry_point("mode_selector_node")

    # Edges
    graph.add_conditional_edges("mode_selector_node", route_after_mode, {
        "validator_node":    "validator_node",
        "scriptwriter_node": "scriptwriter_node"
    })

    graph.add_conditional_edges("validator_node", route_after_script_ready, {
        "hitl_node": "hitl_node",
        END: END
    })

    graph.add_conditional_edges("scriptwriter_node", route_after_script_ready, {
        "hitl_node": "hitl_node",
        END: END
    })

    graph.add_conditional_edges("hitl_node", route_after_hitl, {
        "character_node":    "character_node",
        "scriptwriter_node": "scriptwriter_node",
        END: END
    })

    graph.add_edge("character_node",     "image_node")
    graph.add_edge("image_node",         "memory_commit_node")
    graph.add_edge("memory_commit_node", END)

    return graph.compile()


# Compiled workflow (importable)
workflow = build_workflow()
