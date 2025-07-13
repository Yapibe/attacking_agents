import os
import torch
import wandb
import logging
from PIL import Image
from src.models import load_model
from src.utils import load_image, save_image
from src.PGDAttacks import VLMWhiteBoxPGDAttack

# --- Configuration ---
# Use the more powerful 13B parameter model
MODEL_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"
INPUT_IMAGE_PATH = "data/backgrounds/example_desktop.jpg"
# Change the target to a joke
TARGET_TEXT = "Why did the car get a ticket? Because it was parked in a 'no-joking' zone!"
OUTPUT_DIR = "outputs/generated_mips"
OUTPUT_IMAGE_NAME = "joke_attack_desktop.png"

# Attack parameters
config = {
    "model_id": MODEL_ID,
    "eps": 16 / 255,
    "alpha": 1 / 255,
    "steps": 250, # Increased steps for a potentially harder task
    "target_text": TARGET_TEXT,
}

def setup_logging(log_dir="logs"):
    """Sets up logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "mip_generation.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # Suppress overly verbose logs from libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main():
    """
    Main function to generate a malicious image patch.
    """
    setup_logging()
    logging.info("--- Starting MIP Generation ---")

    # --- 0. Initialize W&B ---
    try:
        run = wandb.init(project="mip-generator-attack", config=config)
        logging.info(f"W&B run initialized successfully. Run name: {run.name}")
    except Exception as e:
        logging.error(f"Failed to initialize W&B: {e}")
        # Decide if you want to exit or continue without W&B
        run = None

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_NAME)
    logging.info(f"Output directory set to: {output_path}")

    # --- 1. Load Model and Processor ---
    logging.info(f"Loading model and processor for: {config['model_id']}")
    model, processor = load_model(config['model_id'])
    logging.info("Model and processor loaded successfully.")
    device = next(model.parameters()).device
    logging.info(f"Model loaded on device: {device}")

    # --- 2. Load Image ---
    logging.info(f"Loading input image from: {INPUT_IMAGE_PATH}")
    image = load_image(INPUT_IMAGE_PATH)
    logging.info(f"Image loaded successfully. Size: {image.size}")
    
    # Log original image to W&B
    if run:
        run.log({"original_image": wandb.Image(image)})

    # --- 3. Prepare Inputs for the Attack ---
    logging.info("Preparing inputs for the attack...")
    prompt = f"USER: <image>\n{config['target_text']} ASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # The Llava-NeXT model requires the original image sizes.
    image_sizes = inputs.get('image_sizes')
    labels = input_ids.clone()
    logging.info("Inputs prepared.")

    # --- 4. Initialize and Run Attack ---
    logging.info("Initializing PGD attack...")
    attack = VLMWhiteBoxPGDAttack(
        model, 
        processor, 
        eps=config['eps'], 
        n=config['steps'], 
        alpha=config['alpha'], 
        early_stop=True,
        wandb_run=run # Pass the run object
    )
    
    logging.info("Starting PGD attack execution...")
    adversarial_image_tensor = attack.execute(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_sizes=image_sizes
    )
    logging.info("PGD attack finished.")

    # --- 5. Save the Adversarial Image ---
    logging.info(f"Saving adversarial image to: {output_path}")
    save_image(adversarial_image_tensor, output_path)
    logging.info("Adversarial image saved successfully.")
    
    # Log the final adversarial image
    if run:
        try:
            adv_image_pil = Image.open(output_path)
            run.log({"adversarial_image": wandb.Image(adv_image_pil)})
        except Exception as e:
            logging.error(f"Could not log adversarial image to W&B: {e}")

    # --- 6. Verification (Optional) ---
    logging.info("Verifying the attack effectiveness...")
    # Load the saved adversarial image for a true verification
    try:
        adv_image_pil_for_verify = Image.open(output_path)
        logging.info("Successfully loaded saved adversarial image for verification.")
    except Exception as e:
        logging.error(f"Could not load the saved image for verification: {e}")
        # Skip verification if the image can't be loaded
        adv_image_pil_for_verify = None

    if adv_image_pil_for_verify:
        verify_prompt = f"USER: <image>\nWhat is written in the image? ASSISTANT:"
        # Process the PIL image, not the tensor
        verify_inputs = processor(text=verify_prompt, images=adv_image_pil_for_verify, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**verify_inputs, max_new_tokens=100)
        
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        verification_result = generated_text.split("ASSISTANT:")[-1].strip()
        
        logging.info(f"Verification - Model's output: '{verification_result}'")
        if run:
            run.log({"verification_output": verification_result})
    
    # Finish the W&B run
    if run:
        run.finish()
        logging.info("W&B run finished.")
    
    logging.info("--- MIP Generation Complete ---")


if __name__ == "__main__":
    # The script needs to be run from the root of the `mip_generator` directory
    # Example: python -m scripts.generate_mip
    # Adjusting path to be relative to the project root for execution
    if os.path.basename(os.getcwd()) == 'attacking_agents':
        os.chdir('mip_generator')

    main()

