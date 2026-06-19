"""
Audio response node for the LangGraph workflow.

This node does everything the text node does, but instead of printing a response,
it speaks the response out loud using text-to-speech and waits for playback to
finish before returning control to the user.

The audio node uses the same shared conversation history as the text node, so
facts mentioned during a text turn are remembered during an audio turn.

During development, this uses pyttsx3 for TTS. Once the audio flow is confirmed
working, the TTS engine can be swapped to Kokoro (only tts_client.py changes).
"""

from state import ApplicationState
from memory.conversation_memory import ConversationMemory
from models.gemma_client import GemmaClient
from models.tts_client import TTSClient


def audio_node(state: ApplicationState) -> ApplicationState:
    """
    Process a user message, generate a text response, and speak it out loud.
    
    This is a LangGraph node that:
    1. Creates/loads the shared conversation memory
    2. Adds the user's input to conversation history
    3. Sends the full history to Gemma3 for context-aware response
    4. Adds the assistant's response to conversation history
    5. Passes the response text to the TTS engine
    6. Plays the audio on a background thread
    7. Waits until audio playback finishes before returning
    
    The entire flow follows this sequence:
    - User types message → Gemma3 generates text → TTS converts to audio →
    - Audio plays on background thread → Program waits for playback to finish →
    - Ready for next input
    
    Args:
        state: The ApplicationState object containing:
               - user_input: The message typed by the user
               - conversation_history: List of previous messages (shared with text node)
    
    Returns:
        The updated ApplicationState with:
        - conversation_history: Now contains the user message and assistant response
        - user_input: Remains unchanged
    """
    
    # Initialize conversation memory from existing history in state
    # This is the same shared history used by the text node
    memory = ConversationMemory(initial_history=state.conversation_history)
    
    # Add the user's input to the conversation history
    memory.add_user_message(state.user_input)
    
    # Initialize the Gemma3 client (connects to local Ollama)
    gemma = GemmaClient()
    
    # Get the full conversation history as dicts for the LLM API
    messages = memory.get_history_as_dict_list()
    
    # Generate a text response from Gemma3 using the full conversation context
    assistant_response = gemma.generate_response(messages)
    
    # Add the assistant's response to the conversation history
    memory.add_assistant_message(assistant_response)
    
    # Initialize the TTS client (pyttsx3 during dev, Kokoro later)
    tts = TTSClient()
    
    # Print the response so the user sees it while audio plays
    print(f"Assistant: (Speaking...)")
    
    # Pass the response text to the TTS engine and play it
    # This method blocks until playback finishes, so the user sees the audio
    # fully complete before the next input prompt appears
    tts.speak(assistant_response)
    
    # Update the state with the new conversation history
    state.conversation_history = memory.get_history()
    
    return state
