"""
Conversation memory manager for text and audio modules.

This module manages the shared conversation history that is used by both the
text conversation module and the audio response module. It provides methods to
add messages, retrieve the full history, and format messages for LLM context.

The conversation history persists for the entire session. When the program
exits, the memory is cleared (no persistence to disk).
"""

from state import ConversationMessage


class ConversationMemory:
    """
    Manager for the shared conversation history between text and audio modules.
    
    This class provides an interface to store and retrieve messages in a
    conversation. Both text and audio modules read from and write to the
    same history, so facts mentioned in one mode are remembered in the other.
    """
    
    def __init__(self, initial_history: list[ConversationMessage] | None = None):
        """
        Initialize the conversation memory.
        
        Args:
            initial_history: Optional list of ConversationMessage objects to
                           start with. If None, starts with empty history.
        
        Returns:
            None (this is a constructor)
        """
        self.history = initial_history if initial_history is not None else []
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content: The text of the user's message
        
        Returns:
            None
        """
        message = ConversationMessage(role="user", content=content)
        self.history.append(message)
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant response to the conversation history.
        
        Args:
            content: The text of the assistant's response
        
        Returns:
            None
        """
        message = ConversationMessage(role="assistant", content=content)
        self.history.append(message)
    
    def get_history(self) -> list[ConversationMessage]:
        """
        Get the complete conversation history.
        
        Args:
            None
        
        Returns:
            A list of ConversationMessage objects in chronological order
        """
        return self.history
    
    def get_history_as_dict_list(self) -> list[dict]:
        """
        Get conversation history formatted as dictionaries for LLM APIs.
        
        This format is what most LLM APIs (like those called through LangChain)
        expect for the messages parameter.
        
        Args:
            None
        
        Returns:
            A list of dicts with 'role' and 'content' keys, in chronological order
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history
        ]
    
    def get_last_n_messages(self, n: int) -> list[ConversationMessage]:
        """
        Get the last N messages from the conversation history.
        
        Useful for providing recent context to the LLM while keeping
        the context window smaller.
        
        Args:
            n: Number of messages to retrieve (if n > history length, returns entire history)
        
        Returns:
            A list of the last N ConversationMessage objects
        """
        return self.history[-n:] if n > 0 else []
    
    def clear(self) -> None:
        """
        Clear all conversation history.
        
        Use this to reset the conversation when starting a new session
        or when explicitly requested by the user.
        
        Args:
            None
        
        Returns:
            None
        """
        self.history = []
    
    def get_message_count(self) -> int:
        """
        Get the total number of messages in the conversation history.
        
        Args:
            None
        
        Returns:
            The number of messages (both user and assistant) in the history
        """
        return len(self.history)
