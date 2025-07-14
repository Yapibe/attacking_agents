import torch
import torch.nn as nn
import logging
from tqdm import tqdm

# Get a logger for this module
logger = logging.getLogger(__name__)


class VLMWhiteBoxPGDAttack:
    """
    PGD L_inf white-box attack for Vision-Language Models.
    Targeted attack: tries to force the model to generate the desired output sequence for an image.
    """

    def __init__(self, model, processor, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True, wandb_run=None):
        """
        Parameters:
        - model: VLM agent model with image + text input, outputs logits or token sequences
        - processor: The processor object from Hugging Face, which includes the tokenizer.
        - eps: maximum perturbation
        - n: number of PGD steps
        - alpha: step size
        - rand_init: whether to randomly initialize image
        - early_stop: stop if target sequence is already achieved
        - wandb_run: optional wandb run object for logging
        """
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer # Explicitly get the tokenizer
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.wandb_run = wandb_run
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')  # token-level loss

        logger.info("VLMWhiteBoxPGDAttack initialized.")
        logger.info(f"  - Epsilon (eps): {self.eps:.4f}")
        logger.info(f"  - Alpha (step size): {self.alpha:.4f}")
        logger.info(f"  - Steps (n): {self.n}")
        logger.info(f"  - Random Init: {self.rand_init}")
        logger.info(f"  - Early Stopping: {self.early_stop}")

    def execute(self, pixel_values, input_ids, attention_mask, labels, image_sizes):
        """
        Performs a targeted PGD attack on an input image.

        Args:
            pixel_values (Tensor): Input image tensor, shape (B, C, H, W), values in [0,1]
            input_ids (Tensor): Input token ids.
            attention_mask (Tensor): Attention mask for the input tokens.
            labels (Tensor): Target output tokens (B, L)
            image_sizes (Tensor): Original sizes of the images.

        Returns:
            Adversarial image tensor
        """
        logger.info("Starting PGD attack execution.")
        x_adv = pixel_values.clone().detach()

        if self.rand_init:
            logger.info("Applying random initialization to the image tensor.")
            x_adv += torch.empty_like(pixel_values).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv.requires_grad = True

        # Wrap the loop with tqdm for a progress bar
        for i in tqdm(range(self.n), desc="PGD Attack Steps"):
            outputs = self.model(
                pixel_values=x_adv, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels,
                image_sizes=image_sizes
            )
            loss = outputs.loss

            if i % 10 == 0:
                # This logging is still useful for file logs
                logger.info(f"Step [{i}/{self.n}] - Loss: {loss.item():.4f}")

            if self.wandb_run:
                self.wandb_run.log({"step": i, "loss": loss.item()})
            
            # We want to maximize the probability of the target tokens, so we minimize the negative loss
            # The model already returns the loss for the labels, so we can just use it.

            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv - self.alpha * torch.sign(grad) # Use - because we want to minimize the loss
            x_adv = torch.min(torch.max(x_adv, pixel_values - self.eps), pixel_values + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()
            x_adv.requires_grad = True

            if self.early_stop and (i % 10 == 0 or i == self.n - 1): # Check every 10 steps
                with torch.no_grad():
                    # For early stopping, we generate text and see if it matches the target
                    # We need to decode the labels to get the target text
                    target_text_full = self.tokenizer.decode(labels[0], skip_special_tokens=True)
                    # The actual target is what comes after "ASSISTANT:"
                    target_text = target_text_full.split("ASSISTANT:")[1].strip()

                    # Generate from the adversarial image
                    gen_ids = self.model.generate(
                        pixel_values=x_adv, 
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=50, 
                        image_sizes=image_sizes
                    )
                    gen_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    
                    # The generated text includes the prompt, so we check if our target is in the newly generated part
                    gen_assistant_part = gen_text.split("ASSISTANT:")[-1].strip()

                    if target_text in gen_assistant_part:
                        logger.info(f"Early stopping at step {i}. Target achieved.")
                        if self.wandb_run:
                            self.wandb_run.log({"early_stop_step": i})
                        break
        
        logger.info("PGD attack finished.")
        return x_adv
