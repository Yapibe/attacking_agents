import torch
import logging
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Loads a Vision-Language Model and its processor from Hugging Face.

    Args:
        model_id (str): The identifier of the model to load.

    Returns:
        (LlavaForConditionalGeneration, AutoProcessor): A tuple containing the loaded model and processor.
    """
    logger.info(f"Starting model loading for model_id: {model_id}")
    
    # --- Load Model ---
    logger.info("Loading model from pretrained...")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # --- Load Processor ---
    logger.info("Loading processor from pretrained...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Processor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    # --- Move to GPU ---
    if torch.cuda.is_available():
        logger.info("CUDA is available. Moving model to GPU...")
        try:
            model.to("cuda")
            logger.info("Model successfully moved to GPU.")
        except Exception as e:
            logger.error(f"Failed to move model to GPU: {e}")
            raise
    else:
        logger.warning("CUDA not available. Model will run on CPU.")

    return model, processor
