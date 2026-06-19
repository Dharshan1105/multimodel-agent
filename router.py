"""
Router and LangGraph workflow orchestration.

This module is the entry point of the application. It defines the LangGraph
workflow that routes each user message to the correct module based on the
currently selected mode.

The router reads the mode from the ApplicationState and directs the message to:
- text_node for "text" mode (text conversation)
- audio_node for "audio" mode (audio response)
- vision_node for "vision" mode (image understanding)
- imagegen_node for "imagine" mode (image generation)

All nodes use the same ApplicationState object, which is passed between them by
LangGraph. This allows shared state (like conversation history) to be maintained
across module calls.
"""

from langgraph.graph import StateGraph
from state import ApplicationState
from nodes.text_node import text_node
from nodes.audio_node import audio_node
from nodes.vision_node import vision_node
from nodes.imagegen_node import imagegen_node


def route_by_mode(state: ApplicationState) -> str:
    """
    Determine which node to route to based on the current mode.
    
    This function reads the mode field from the state and returns the name of
    the node that should handle this request. LangGraph uses the returned string
    to determine which node to execute next.
    
    Args:
        state: The ApplicationState object containing the current mode
    
    Returns:
        A string with the node name:
        - "text" for text conversation mode
        - "audio" for audio response mode
        - "vision" for image understanding mode
        - "imagine" for image generation mode
    
    Raises:
        ValueError: If mode is not recognized
    """
    valid_modes = {"text", "audio", "vision", "imagine"}
    
    if state.mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {state.mode}. "
            f"Valid modes are: {', '.join(valid_modes)}"
        )
    
    return state.mode


def build_graph() -> StateGraph:
    """
    Build and return the LangGraph workflow.
    
    This function creates a state graph with four nodes (one for each module)
    and configures conditional routing based on the current mode. The graph
    is the complete workflow that LangGraph executes.
    
    Args:
        None
    
    Returns:
        A compiled LangGraph StateGraph ready to execute
    """
    
    # Create a new state graph with ApplicationState as the state schema
    graph = StateGraph(ApplicationState)
    
    # Add nodes for each module
    # Each node is a function that takes state → state
    graph.add_node("text", text_node)
    graph.add_node("audio", audio_node)
    graph.add_node("vision", vision_node)
    graph.add_node("imagine", imagegen_node)
    
    # Add an entry point that routes based on mode
    # The entry point sends every request to the router
    graph.add_conditional_edges(
        source="__start__",  # Start from the beginning
        path=route_by_mode,  # Use route_by_mode to decide which node
        path_map={
            "text": "text",
            "audio": "audio",
            "vision": "vision",
            "imagine": "imagine"
        }
    )
    
    # Define end points for each node
    # Each node returns to END when done
    graph.add_edge("text", "__end__")
    graph.add_edge("audio", "__end__")
    graph.add_edge("vision", "__end__")
    graph.add_edge("imagine", "__end__")
    
    # Compile the graph into a runnable workflow
    return graph.compile()


def create_workflow():
    """
    Create and return a compiled LangGraph workflow.
    
    This is a convenience function that builds the graph and returns it
    ready to use. Call this once at application startup, then reuse the
    workflow for multiple requests.
    
    Args:
        None
    
    Returns:
        A compiled LangGraph workflow ready to invoke
    """
    return build_graph()
