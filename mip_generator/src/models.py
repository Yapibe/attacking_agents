import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Loads a Vision-Language Model and its processor from Hugging Face.

    Args:
        model_id (str): The identifier of the model to load.

    Returns:
        (LlavaForConditionalGeneration, AutoProcessor): A tuple containing the loaded model and processor.
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")

    return model, processor