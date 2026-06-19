"""
Main entry point for the Multimodal Agentic AI Assistant.

This is the file the user runs to start the assistant. It handles:
1. Prompting the user to select a mode (text, audio, vision, or imagine)
2. Creating the LangGraph workflow
3. Continuously reading user input and processing it through the workflow
4. Maintaining state across multiple turns (memory persists in the session)

The user can start the assistant with:
    python main.py

Then follow the prompts to select a mode and start conversing.
"""

from state import ApplicationState
from router import create_workflow


def display_welcome_message():
    """
    Display welcome message and mode selection options.
    
    Args:
        None
    
    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("MULTIMODAL AGENTIC AI ASSISTANT")
    print("=" * 60)
    print("\nSelect a mode to begin:\n")
    print("  text    - Text conversation with memory")
    print("  audio   - Speak responses aloud automatically")
    print("  vision  - Describe images you provide")
    print("  imagine - Generate images from text prompts")
    print("\nType 'exit' to quit the assistant.")
    print("=" * 60 + "\n")


def select_mode() -> str:
    """
    Prompt the user to select a mode.
    
    Args:
        None
    
    Returns:
        The selected mode as a string ("text", "audio", "vision", or "imagine")
    """
    valid_modes = {"text", "audio", "vision", "imagine"}
    
    while True:
        mode = input("Enter mode (text/audio/vision/imagine): ").strip().lower()
        
        if mode in valid_modes:
            return mode
        else:
            print(f"Invalid mode. Please choose from: {', '.join(valid_modes)}")


def main():
    """
    Main loop for the Multimodal Agentic AI Assistant.
    
    This function:
    1. Displays welcome message
    2. Prompts user to select a mode
    3. Creates the LangGraph workflow
    4. Enters a loop that:
       - Reads user input
       - Creates ApplicationState with the input
       - Invokes the workflow
       - Displays any output
       - Repeats
    
    The session persists until the user types 'exit'.
    Memory (conversation history, image interactions, image generations)
    is maintained across all turns in the same session.
    
    Args:
        None
    
    Returns:
        None
    """
    
    # Display welcome and mode selection
    display_welcome_message()
    mode = select_mode()
    
    print(f"\nYou selected: {mode} mode")
    print("Type 'exit' to quit or 'switch' to change modes.\n")
    print("-" * 60 + "\n")
    
    # Create the workflow (this initializes all nodes)
    workflow = create_workflow()
    
    # Initialize application state with selected mode
    # Memory persists in this state object across all turns
    state = ApplicationState(mode=mode)
    
    # Main conversation loop
    while True:
        try:
            # Read user input
            user_input = input("You: ").strip()
            
            # Handle special commands
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "switch":
                mode = select_mode()
                state.mode = mode
                print(f"Switched to {mode} mode.\n")
                continue
            
            if not user_input:
                # Empty input, skip
                continue
            
            # Set the current user input in state
            state.user_input = user_input
            
            # Invoke the workflow
            # The workflow routes to the appropriate node based on state.mode
            # and returns the updated state
            result = workflow.invoke(state)
            state.conversation_history = result.get("conversation_history", state.conversation_history)
            state.image_interactions = result.get("image_interactions", state.image_interactions)
            state.image_generations = result.get("image_generations", state.image_generations)
            
            print()  # Blank line for readability
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nAssistant interrupted by user. Goodbye!")
            break
        except Exception as e:
            # Catch unexpected errors and display them
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
