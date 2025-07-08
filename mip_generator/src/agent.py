from typing import List, Dict
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class SimulatedAgent:
    """
    A simulated OS agent that uses a VLM to interpret screen content.
    """

    def __init__(self, model: AutoModelForCausalLM, processor: AutoProcessor):
        """
        Initializes the agent with a loaded VLM and processor.

        Args:
            model: The Vision-Language Model.
            processor: The processor for the VLM.
        """
        self.model = model
        self.processor = processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, image: Image.Image, prompt: str) -> str:
        """
        Simulates the agent's workflow: processing an image and prompt to get a response.

        Args:
            image: The input image (e.g., a screenshot).
            prompt: The text prompt for the agent.

        Returns:
            The generated response from the VLM.
        """
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        
        inputs = self.processor(text=messages, images=[image], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # The output is a list containing one item, so we extract it
        return generated_texts[0].strip()
