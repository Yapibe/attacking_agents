
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
    Handles both 4D (B, C, H, W) and 5D (B, N, C, H, W) tensors from different models.

    Args:
        tensor (torch.Tensor): The image tensor to save. Assumes values are in [0, 1].
        output_path (str): The path to save the image to.
    """
    logger.info(f"Saving image tensor to path: {output_path}")
    logger.info(f"Original tensor shape: {tensor.shape}")

    try:
        # --- Handle 5D tensors from Llava-NeXT ---
        if tensor.dim() == 5:
            logger.info("Detected a 5D tensor. Selecting the first image [0, 0] for saving.")
            # Select the first image from the batch and the first from the num_images dimension
            tensor_to_save = tensor[0, 0, :, :, :]
        # --- Handle standard 4D tensors ---
        elif tensor.dim() == 4:
            logger.info("Detected a 4D tensor. Selecting the first image [0] for saving.")
            tensor_to_save = tensor.squeeze(0)
        else:
            tensor_to_save = tensor

        logger.info(f"Shape of tensor being saved: {tensor_to_save.shape}")

        # Clamp tensor to be sure it is in [0,1] range
        tensor_to_save = torch.clamp(tensor_to_save, 0, 1)
        
        # Convert tensor to PIL Image
        # Permute C,H,W to H,W,C, move to CPU, convert to numpy, scale to 0-255
        image_numpy = (tensor_to_save.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        image = Image.fromarray(image_numpy)
        
        # Save the image
        image.save(output_path)
        logger.info(f"Image saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the image: {e}")
        raise

