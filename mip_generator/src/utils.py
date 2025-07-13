
import logging
from PIL import Image
import torch

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image.Image: The loaded image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        logger.info("Image loaded and converted to RGB successfully.")
        return image
    except FileNotFoundError:
        logger.error(f"Image file not found at path: {image_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the image: {e}")
        raise

def save_image(tensor: torch.Tensor, output_path: str):
    """
    Saves a tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save. Assumes values are in [0, 1].
        output_path (str): The path to save the image to.
    """
    logger.info(f"Saving image tensor to path: {output_path}")
    try:
        # Clamp tensor to be sure it is in [0,1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert tensor to PIL Image
        # Squeeze batch dim, permute C,H,W to H,W,C, move to CPU, convert to numpy, scale to 0-255
        image_numpy = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        image = Image.fromarray(image_numpy)
        
        # Save the image
        image.save(output_path)
        logger.info(f"Image saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the image: {e}")
        raise

