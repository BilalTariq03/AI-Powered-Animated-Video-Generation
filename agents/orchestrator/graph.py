"""
agents/orchestrator/graph.py
──────────────────────────────
LangGraph workflow for Phase 1 pipeline.
"""

from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agents.story_agent.agent           import ScriptwriterAgent
from agents.story_agent.validator       import ValidatorAgent
from agents.story_agent.hitl            import HITLAgent
from agents.story_agent.character_agent import CharacterDesignerAgent
from agents.story_agent.image_agent     import ImageSynthesizerAgent
from shared.utils.vector_store          import memory


class WritersRoomState(TypedDict):
    input_mode:        str
    user_prompt:       str
    raw_script:        str
    script:            Dict
    validated:         bool
    validation_errors: List[str]
    hitl_approved:     bool
    characters:        List[Dict]
    images:            List[Dict]
    status:            str
    error:             Optional[str]
    iteration:         int


scriptwriter  = ScriptwriterAgent()
validator     = ValidatorAgent()
hitl          = HITLAgent()
char_designer = CharacterDesignerAgent()
img_synth     = ImageSynthesizerAgent()


def mode_selector_node(state: WritersRoomState) -> WritersRoomState:
    print(f"\n[mode_selector_node] Mode: {state['input_mode']}")
    if state["input_mode"] == "manual" and state.get("raw_script"):
        return {**state, "status": "validate"}
    return {**state, "status": "generate"}


def validator_node(state: WritersRoomState) -> WritersRoomState:
    return validator.run(state)


def scriptwriter_node(state: WritersRoomState) -> WritersRoomState:
    return scriptwriter.run(state)


def hitl_node(state: WritersRoomState) -> WritersRoomState:
    return hitl.run(state)


def character_node(state: WritersRoomState) -> WritersRoomState:
    return char_designer.run(state)


def image_node(state: WritersRoomState) -> WritersRoomState:
    return img_synth.run(state)


def memory_commit_node(state: WritersRoomState) -> WritersRoomState:
    print(f"\n[memory_commit_node] Committing final outputs...")
    memory.store("output:final_state", {
        "title":          state.get("script", {}).get("story", {}).get("title"),
        "num_scenes":     len(state.get("script", {}).get("scenes", [])),
        "num_characters": len(state.get("characters", [])),
        "num_images":     len(state.get("images", [])),
        "status":         "complete"
    }, {"type": "final_output"})
    print(f"[memory_commit_node] All outputs committed to memory.")
    return {**state, "status": "complete"}


def route_after_mode(state: WritersRoomState) -> str:
    return "validator_node" if state["status"] == "validate" else "scriptwriter_node"


def route_after_script_ready(state: WritersRoomState) -> str:
    if state["status"] == "validation_failed":
        return END
    return "hitl_node"


def route_after_hitl(state: WritersRoomState) -> str:
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


def build_workflow() -> StateGraph:
    graph = StateGraph(WritersRoomState)
    graph.add_node("mode_selector_node", mode_selector_node)
    graph.add_node("validator_node",     validator_node)
    graph.add_node("scriptwriter_node",  scriptwriter_node)
    graph.add_node("hitl_node",          hitl_node)
    graph.add_node("character_node",     character_node)
    graph.add_node("image_node",         image_node)
    graph.add_node("memory_commit_node", memory_commit_node)

    graph.set_entry_point("mode_selector_node")
    graph.add_conditional_edges("mode_selector_node", route_after_mode,
                                {"validator_node": "validator_node", "scriptwriter_node": "scriptwriter_node"})
    graph.add_conditional_edges("validator_node",    route_after_script_ready,
                                {"hitl_node": "hitl_node", END: END})
    graph.add_conditional_edges("scriptwriter_node", route_after_script_ready,
                                {"hitl_node": "hitl_node", END: END})
    graph.add_conditional_edges("hitl_node",         route_after_hitl,
                                {"character_node": "character_node", "scriptwriter_node": "scriptwriter_node", END: END})
    graph.add_edge("character_node",     "image_node")
    graph.add_edge("image_node",         "memory_commit_node")
    graph.add_edge("memory_commit_node", END)
    return graph.compile()


workflow = build_workflow()
