"""
Stable Diffusion client for image generation.

Uses lazy loading to avoid GPU out of memory errors. The model is only
loaded into GPU memory when the user actually requests image generation,
and unloaded immediately after to free memory for other models.
"""

import os
import torch
from pathlib import Path
from datetime import datetime
import requests

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class StableDiffusionClient:
    """
    Client for Stable Diffusion image generation running locally.

    The pipeline is not loaded on init. It is loaded only when generate_image
    is called and unloaded immediately after to free GPU memory.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "outputs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the client without loading the model yet.

        The model is intentionally not loaded here. It will be loaded
        only when generate_image is called. This keeps GPU memory free
        for Gemma3 and LLaVA which are used more frequently.

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory where generated images will be saved
            device: Device to run on (cuda or cpu)

        Returns:
            None
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Pipeline is None until generate_image is called
        # This is intentional to avoid loading into GPU memory on startup
        self.pipeline = None

    def _load_pipeline(self) -> None:
        """
        Load Stable Diffusion into GPU memory.

        Called internally just before generation starts. Separated from
        __init__ so the model only occupies GPU memory when actually needed.

        Args:
            None

        Returns:
            None
        """
        from diffusers import StableDiffusionPipeline

        print("Loading Stable Diffusion into memory...")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()

        print("Stable Diffusion loaded.")

    def _unload_pipeline(self) -> None:
        """
        Unload Stable Diffusion from GPU memory after generation.

        Frees the GPU memory so Gemma3 and LLaVA can use it again.
        The model will be reloaded next time generate_image is called.

        Args:
            None

        Returns:
            None
        """
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Stable Diffusion unloaded. GPU memory freed.")
        
    def _unload_ollama_models(self) -> None:
        """
        Tell Ollama to unload all models from GPU memory.

        Ollama keeps models loaded in GPU memory for fast responses.
        We need to evict them before loading Stable Diffusion otherwise
        there is not enough GPU memory for both at the same time.

        Args:
            None

        Returns:
            None
        """
        try:
            # Setting keep_alive to 0 tells Ollama to unload the model immediately
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3", "keep_alive": 0}
            )
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llava", "keep_alive": 0}
            )
            print("Ollama models unloaded from GPU.")

            # Give Ollama a moment to fully release the memory
            import time
            time.sleep(2)

        except Exception as e:
            print(f"Could not unload Ollama models: {e}")

    def _reload_ollama_models(self) -> None:
        """
        Warm up Ollama models back into memory after image generation.

        This is optional but makes the next text or audio response faster
        since the model does not need to reload from scratch.

        Args:
            None

        Returns:
            None
        """
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3",
                    "prompt": "hello",
                    "keep_alive": "5m"
                }
            )
            print("Ollama models reloaded.")
        except Exception as e:
            print(f"Could not reload Ollama models: {e}")

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = None,
        width: int = 512,
        height: int = 512
    ) -> tuple[str, dict]:
        """
        Generate an image from a text prompt.

        Loads the model, generates the image, saves it, then unloads
        the model immediately to free GPU memory.

        Args:
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default 30)
            guidance_scale: How strongly to follow the prompt (default 7.5)
            seed: Random seed for reproducibility (default None)
            width: Image width in pixels (default 512)
            height: Image height in pixels (default 512)

        Returns:
            A tuple of file_path (str) and metadata (dict)
        """
        try:
    # Step 1 - Free Ollama GPU memory before loading Stable Diffusion
            self._unload_ollama_models()

            # Step 2 - Load Stable Diffusion
            self._load_pipeline()

            # Set seed
            if seed is not None:
                torch.manual_seed(seed)
            else:
                seed = int(torch.randint(0, 2**32, (1,)).item())

            # Generate the image
            with torch.no_grad():
                image = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]

            # Save the image
            file_path = self._get_next_filename()
            image.save(file_path)

            metadata = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "model": "stable-diffusion-v1-5",
                "device": self.device,
                "timestamp": datetime.now().isoformat()
            }

            return str(file_path), metadata

        finally:
            # Step 3 - Unload Stable Diffusion
            self._unload_pipeline()

            # Step 4 - Reload Ollama so text and audio mode work again
            self._reload_ollama_models()

    def _get_next_filename(self) -> Path:
        """
        Generate the next auto incrementing filename.

        Args:
            None

        Returns:
            A Path object for the next filename (e.g. outputs/generated_001.png)
        """
        existing_files = list(self.output_dir.glob("generated_*.png"))

        numbers = []
        for file in existing_files:
            try:
                num_str = file.stem.replace("generated_", "")
                numbers.append(int(num_str))
            except (ValueError, AttributeError):
                pass

        next_number = max(numbers) + 1 if numbers else 1
        filename = self.output_dir / f"generated_{next_number:03d}.png"

        return filename