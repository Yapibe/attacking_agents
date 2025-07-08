
from PIL import Image
import torch

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image.Image: The loaded image.
    """
    return Image.open(image_path).convert("RGB")

def save_image(tensor: torch.Tensor, output_path: str):
    """
    Saves a tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save. Assumes values are in [0, 1].
        output_path (str): The path to save the image to.
    """
    # Clamp tensor to be sure it is in [0,1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert tensor to PIL Image
    image = Image.fromarray((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
    
    # Save the image
    image.save(output_path)
    print(f"Saved adversarial image to {output_path}")

