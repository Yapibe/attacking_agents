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

    def __init__(self, model, processor, eps=8 / 255., n=50, alpha=1e-2,
                 rand_init=True, early_stop=True, wandb_run=None):
        """
        Parameters:
        - model: VLM agent model with image + text input, outputs logits or token sequences
        - processor: The processor object from Hugging Face, which includes the tokenizer.
        - eps: maximum perturbation
        - n: number of PGD steps
        - alpha: learning rate for Adam optimizer
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
        logger.info(f"  - Alpha (learning rate): {self.alpha:.4f}")
        logger.info(f"  - Steps (n): {self.n}")
        logger.info(f"  - Random Init: {self.rand_init}")
        logger.info(f"  - Early Stopping: {self.early_stop}")

    def execute(self, pixel_values, input_ids_for_loss, attention_mask_for_loss,
                input_ids_for_gen, attention_mask_for_gen, labels, image_sizes, target_text):
        """
        Performs a targeted PGD attack on an input image.

        Args:
            pixel_values (Tensor): Input image tensor.
            input_ids_for_loss (Tensor): Token IDs for loss calculation (includes target).
            attention_mask_for_loss (Tensor): Attention mask for loss calculation.
            input_ids_for_gen (Tensor): Token IDs for generation check (prompt only).
            attention_mask_for_gen (Tensor): Attention mask for generation check.
            labels (Tensor): Target output tokens with masking.
            image_sizes (Tensor): Original sizes of the images.
            target_text (str): The target string for early stopping.

        Returns:
            Adversarial image tensor
        """
        logger.info("Starting PGD attack execution.")
        x_adv = pixel_values.clone().detach()

        if self.rand_init:
            logger.info("Applying random initialization to the image tensor.")
            x_adv += torch.empty_like(pixel_values).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = torch.round(x_adv * 255) / 255

        x_adv.requires_grad = True

        optimizer = torch.optim.Adam([x_adv], lr=self.alpha, betas=(0.9, 0.9))


        target_ids = labels[0, labels[0] != -100]

        # Wrap the loop with tqdm for a progress bar
        for i in tqdm(range(self.n), desc="PGD Attack Steps"):
            optimizer.zero_grad()
            outputs = self.model(
                pixel_values=x_adv,
                input_ids=input_ids_for_loss,
                attention_mask=attention_mask_for_loss,
                labels=labels,
                image_sizes=image_sizes
            )
            loss = outputs.loss

            if i % 10 == 0:
                logger.info(f"Step [{i}/{self.n}] - Loss: {loss.item():.4f}")

            if self.wandb_run:
                self.wandb_run.log({"step": i, "loss": loss.item()})

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta = torch.clamp(x_adv - pixel_values, -self.eps, self.eps)
                x_adv.data = pixel_values + delta
                x_adv.data = torch.clamp(x_adv.data, 0, 1)
                x_adv.data = torch.round(x_adv.data * 255) / 255
            x_adv.requires_grad = True

            # Efficient Early Stopping Check
            # Efficient Early Stopping Check
            if self.early_stop and (i % 10 == 0 or i == self.n - 1):
                with torch.no_grad():
                    # Reuse the logits from the forward pass we already did for the loss
                    logits = outputs.logits

                    # The model internally shifts logits and labels, so the logit at position j
                    # corresponds to the label at position j. We just need to find where our
                    # target labels are.
                    
                    # Find the indices of the target tokens (where labels are not -100)
                    target_token_indices = (labels != -100).nonzero(as_tuple=True)
                    
                    # Get the logits that correspond to the positions of our target tokens
                    relevant_logits = logits[target_token_indices]
                    
                    # Get the token IDs of our target tokens
                    target_token_ids = labels[target_token_indices]

                    # Calculate the probabilities of all tokens at the relevant positions
                    probs = torch.softmax(relevant_logits, dim=-1)
                    
                    # Get the specific probabilities of the correct target tokens
                    correct_token_probs = probs.gather(1, target_token_ids.unsqueeze(1)).squeeze()

                    # Check if all target tokens have a probability > 99%
                    if torch.all(correct_token_probs > 0.99):
                        logger.info(f"Early stopping at step {i}. All target token probabilities > 99%.")
                        if self.wandb_run:
                            self.wandb_run.log({"early_stop_step": i})
                        break
        
        logger.info("PGD attack finished.")
        return x_adv
