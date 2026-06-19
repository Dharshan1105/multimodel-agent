"""
Text conversation node for the LangGraph workflow.

This node handles plain text conversations. It receives a user message,
passes the full conversation history to Gemma3 for context, returns a text
response, and appends both the user message and response to the shared
conversation history.

This node is used when the user selects "text" mode. The conversation history
is shared with the audio node, so facts mentioned here are remembered in
audio responses.
"""

from state import ApplicationState
from memory.conversation_memory import ConversationMemory
from models.gemma_client import GemmaClient


def text_node(state: ApplicationState) -> ApplicationState:
    """
    Process a user message and generate a text response.
    
    This is a LangGraph node that:
    1. Creates/loads the shared conversation memory
    2. Adds the user's input to conversation history
    3. Sends the full history to Gemma3 for context-aware response
    4. Adds the assistant's response to conversation history
    5. Updates the state and returns it
    
    Args:
        state: The ApplicationState object containing:
               - user_input: The message typed by the user
               - conversation_history: List of previous messages (may be empty on first turn)
    
    Returns:
        The updated ApplicationState with:
        - conversation_history: Now contains the user message and assistant response
        - user_input: Remains unchanged
    """
    
    # Initialize conversation memory from existing history in state
    memory = ConversationMemory(initial_history=state.conversation_history)
    
    # Add the user's input to the conversation history
    memory.add_user_message(state.user_input)
    
    # Initialize the Gemma3 client (connects to local Ollama)
    gemma = GemmaClient()
    
    # Get the full conversation history as dicts for the LLM API
    messages = memory.get_history_as_dict_list()
    
    # Generate a response from Gemma3 using the full conversation context
    assistant_response = gemma.generate_response(messages)
    
    # Add the assistant's response to the conversation history
    memory.add_assistant_message(assistant_response)
    
    # Update the state with the new conversation history
    state.conversation_history = memory.get_history()
    
    # Print the response to the user
    print(f"Assistant: {assistant_response}")
    
    return state
