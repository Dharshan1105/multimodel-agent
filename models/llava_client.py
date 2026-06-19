"""
LLaVA vision client for image understanding and description.

This module uses the ollama Python library directly instead of LangChain
because LangChain's OllamaLLM does not handle image inputs to LLaVA correctly.
The ollama library sends images in the format LLaVA actually expects.
"""

import ollama
from pathlib import Path


class LLaVAClient:
    """
    Client for LLaVA vision model running locally through Ollama.

    Uses the ollama Python library directly to send images and prompts
    to LLaVA and receive full text descriptions in return.
    """

    def __init__(self, model_name: str = "llava"):
        """
        Initialize the LLaVA client.

        Args:
            model_name: Name of the Ollama model to use (default: "llava")

        Returns:
            None
        """
        self.model_name = model_name
        self.supported_formats = {".png", ".jpg", ".jpeg"}

    def describe_image(self, image_path: str, question: str = None) -> str:
        """
        Analyze an image and return a text description.

        Args:
            image_path: File path to the image file (PNG, JPG, or JPEG)
            question: Optional specific question to ask about the image.
                     If None, asks for a general description.

        Returns:
            A string containing LLaVA's text description of the image

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image format is not supported
        """
        # Validate the file exists
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate the format
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(self.supported_formats)}"
            )

        # Set the prompt
        if question:
            prompt = question
        else:
            prompt = "Please describe what you see in this image in detail."

        # Call LLaVA using the ollama library directly
        # This is the correct way to send images to LLaVA through Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(path)]
                }
            ]
        )

        return response["message"]["content"]