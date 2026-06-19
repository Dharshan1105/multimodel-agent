"""
Image generation node for the LangGraph workflow.

This node accepts a text prompt describing an image and uses Stable Diffusion
to generate a new image. The generated image is saved to the outputs directory
with an auto-incrementing filename (generated_001.png, generated_002.png, etc.).

The node stores each generation in its own separate memory store, allowing users
to reference or modify previously generated images later in the session. For
example, after generating an image, the user can say "Generate a similar image
but at night" and the module can retrieve the previous generation to use as
context for the new request.
"""

from state import ApplicationState
from memory.image_generation_memory import ImageGenerationMemory
from models.stable_diffusion_client import StableDiffusionClient


def imagegen_node(state: ApplicationState) -> ApplicationState:
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    This is a LangGraph node that:
    1. Takes the user input as the image prompt
    2. Initializes the Stable Diffusion client
    3. Generates an image from the prompt
    4. Saves the image to the outputs directory with auto-incrementing filename
    5. Stores the generation (prompt, file_path, metadata) in memory
    6. Prints the result to the user
    7. Returns updated state
    
    Args:
        state: The ApplicationState object containing:
               - user_input: The text prompt describing the image to generate
               - image_generations: List of previous image generations
    
    Returns:
        The updated ApplicationState with:
        - image_generations: Now contains the new image generation record
    """
    
    # Initialize image generation memory from existing history in state
    memory = ImageGenerationMemory(initial_generations=state.image_generations)
    
    # Use the user input as the image generation prompt
    prompt = state.user_input.strip()
    
    if not prompt:
        print("Error: Please provide a text description of the image to generate.")
        return state
    
    # Initialize the Stable Diffusion client (connects locally, no API calls)
    # The first time this runs, it will download the model (several GB)
    # Subsequent runs will use the cached model
    try:
        sd = StableDiffusionClient()
    except Exception as e:
        print(f"Error initializing Stable Diffusion: {e}")
        return state
    
    # Generate the image from the prompt
    # Returns tuple of (file_path, metadata)
    try:
        file_path, metadata = sd.generate_image(prompt)
    except Exception as e:
        print(f"Error generating image: {e}")
        return state
    
    # Store the generation in memory (prompt, file_path, metadata)
    # This allows the user to reference or modify it later
    memory.add_generation(prompt, file_path, metadata)
    
    # Update the state with the new image generations
    state.image_generations = memory.get_all_generations()
    
    # Print the result to the user so they know where to find the image
    print(f"Assistant: Image saved to {file_path}")
    
    return state
