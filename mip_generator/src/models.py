import torch
import logging
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Loads a Vision-Language Model and its processor from Hugging Face.

    Args:
        model_id (str): The identifier of the model to load.

    Returns:
        (LlavaNextForConditionalGeneration, AutoProcessor): A tuple containing the loaded model and processor.
    """
    logger.info(f"Starting model loading for model_id: {model_id}")
    
    # --- Load Model ---
    # Use LlavaNextForConditionalGeneration for LLaVA v1.6+ models
    logger.info("Loading model from pretrained using LlavaNextForConditionalGeneration...")
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
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
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        logger.info("Processor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    # --- Set Chat Template ---
    # The Vicuna-based model needs a chat template to be explicitly set.
    # This defines how the conversation is formatted.
    vicuna_template = (
        "{% for message in messages %}"
        "  {% if message['role'] == 'system' %}"
        "    {{ message['content'] + '\n' }}"
        "  {% elif message['role'] == 'user' %}"
        "    {{ 'USER: ' + message['content'] + '\n' }}"
        "  {% elif message['role'] == 'assistant' %}"
        "    {{ 'ASSISTANT: ' + message['content'] + eos_token + '\n' }}"
        "  {% endif %}"
        "{% endfor %}"
    )
    processor.tokenizer.chat_template = vicuna_template
    logger.info("Vicuna chat template has been set on the tokenizer.")

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
