"""
Vision node for image understanding in the LangGraph workflow.

This node accepts an image file path (and optionally a specific question) and
uses LLaVA to return a natural language description of what is in the image.

The node stores each analyzed image in its own separate memory store, allowing
users to ask follow-up questions about an image without having to upload it again.
For example, after analyzing an image, the user can ask "What color is the box in
the top left?" and the node will retrieve the most recent image and use it as
context for answering the question.
"""

from state import ApplicationState
from memory.image_interaction_memory import ImageInteractionMemory
from models.llava_client import LLaVAClient
import state


def vision_node(state: ApplicationState) -> ApplicationState:
    """
    Analyze an image and return a text description.
    
    This is a LangGraph node that:
    1. Parses the user input to extract image path and optional question
    2. If no image path is provided, uses the most recent image from memory (for follow-ups)
    3. Calls LLaVA to generate a description of the image
    4. Stores the interaction (image path, description, timestamp) in memory
    5. Prints the description to the user
    6. Returns updated state
    
    The user can invoke this in two ways:
    - Provide an image path: "Describe docker_architecture.png"
    - Ask a follow-up: "What color is the logo?" (uses most recent image)
    
    Args:
        state: The ApplicationState object containing:
               - user_input: The message typed by the user (image path + optional question)
               - image_interactions: List of previous image analyses
    
    Returns:
        The updated ApplicationState with:
        - image_interactions: Now contains the new image analysis
    """
    
    # Initialize image interaction memory from existing history in state
    memory = ImageInteractionMemory(initial_interactions=state.image_interactions)
    
    # Parse the user input to extract image path and question
    image_path, question = _parse_vision_input(state.user_input)
    
    # If no image path provided, this is a follow-up question
    # Use the most recent image from memory
    if image_path is None:
        image_path = memory.get_most_recent_image_path()
        if image_path is None:
            print("Error: No image uploaded yet. Please provide an image path first.")
            return state
    
    # Initialize the LLaVA client (connects to local Ollama)
    llava = LLaVAClient()
    
    # Generate a description of the image
    # Pass the question if provided, otherwise LLaVA will generate a general description
    try:
        description = llava.describe_image(image_path, question)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return state
    except ValueError as e:
        print(f"Error: {e}")
        return state
    
    # Store the interaction in memory (image path, description, timestamp)
    memory.add_interaction(image_path, description)
    
    # Update the state with the new image interactions
    state.image_interactions = memory.get_all_interactions()
    
    # Print the description to the user
    print(f"Assistant: {description}")
    
    return state


def _parse_vision_input(user_input: str) -> tuple[str | None, str | None]:
    """
    Internal helper: Parse user input to extract image path and optional question.
    
    The user can provide input in these formats:
    1. Just an image path: "docker_architecture.png"
    2. Image path with a question: "docker_architecture.png What components are shown?"
    3. Just a question (follow-up): "What color is the logo?"
    
    This function tries to detect if the first word looks like a file path (contains
    a dot and extension), and if so, treats it as the image path. Everything after
    is treated as the question.
    
    Args:
        user_input: The raw user input string
    
    Returns:
        A tuple of (image_path, question) where either can be None:
        - ("docker_architecture.png", "What components?") - both provided
        - ("docker_architecture.png", None) - just image path
        - (None, "What color is the logo?") - just question (follow-up)
        - (None, None) - empty input
    """
    user_input = user_input.strip()
    
    if not user_input:
        return None, None
    
    # Split the input into parts
    parts = user_input.split(maxsplit=1)
    
    if not parts:
        return None, None
    
    first_part = parts[0]
    
    # Check if the first part looks like a file path
    # (contains a dot followed by an extension like .png, .jpg, etc.)
    if "." in first_part and len(first_part) > 2:
        image_path = first_part
        question = parts[1] if len(parts) > 1 else None
        return image_path, question
    else:
        # First part is not a file path, so it's all a question (follow-up)
        question = user_input
        return None, question
