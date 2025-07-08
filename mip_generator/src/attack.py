

import torch
from tqdm import tqdm

class PGDAttack:
    """
    A simple PGD attack for Vision-Language Models.
    """
    def __init__(self, model, processor, eps: float = 8/255, alpha: float = 1/255, steps: int = 100):
        """
        Args:
            model: The VLM model.
            processor: The processor for the VLM.
            eps (float): Maximum perturbation.
            alpha (float): Step size for each iteration.
            steps (int): Number of PGD steps.
        """
        self.model = model
        self.processor = processor
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device

    def run(self, image, target_text: str):
        """
        Runs the PGD attack.

        Args:
            image (Image.Image): The input image.
            target_text (str): The target malicious text prompt.

        Returns:
            torch.Tensor: The adversarial image tensor.
        """
        # Prepare the image and target text
        prompt = f"USER: <image>\n{target_text} ASSISTANT:"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        image_tensor = inputs['pixel_values'].clone().detach()
        
        # The target is the tokenized text, we want the model to generate this
        target_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        adv_image_tensor = image_tensor.clone().detach()
        adv_image_tensor.requires_grad = True

        print("Running PGD attack...")
        for _ in tqdm(range(self.steps)):
            # Forward pass
            outputs = self.model(pixel_values=adv_image_tensor, labels=target_ids)
            loss = outputs.loss

            # Backward pass to get gradients
            loss.backward()

            # Get the gradient
            grad = adv_image_tensor.grad

            # PGD step: update the image
            adv_image_tensor.data = adv_image_tensor.data + self.alpha * grad.sign()
            
            # Project back into epsilon ball
            perturbation = torch.clamp(adv_image_tensor.data - image_tensor.data, -self.eps, self.eps)
            adv_image_tensor.data = image_tensor.data + perturbation
            
            # Clamp to valid image range
            adv_image_tensor.data = torch.clamp(adv_image_tensor.data, 0, 1)

            # Zero the gradient for the next iteration
            adv_image_tensor.grad.zero_()

        print("Attack finished.")
        return adv_image_tensor
