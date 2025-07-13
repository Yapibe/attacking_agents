import torch
import torch.nn as nn
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


class VLMWhiteBoxPGDAttack:
    """
    PGD L_inf white-box attack for Vision-Language Models.
    Targeted attack: tries to force the model to generate the desired output sequence for an image.
    """

    def __init__(self, model, tokenizer, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True, wandb_run=None):
        """
        Parameters:
        - model: VLM agent model with image + text input, outputs logits or token sequences
        - tokenizer: Tokenizer used by the VLM (e.g., T5Tokenizer, CLIPTokenizer)
        - eps: maximum perturbation
        - n: number of PGD steps
        - alpha: step size
        - rand_init: whether to randomly initialize image
        - early_stop: stop if target sequence is already achieved
        - wandb_run: optional wandb run object for logging
        """
        self.model = model
        self.tokenizer = tokenizer
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

    def execute(self, pixel_values, input_ids, attention_mask, labels):
        """
        Performs a targeted PGD attack on an input image.

        Args:
            pixel_values (Tensor): Input image tensor, shape (B, C, H, W), values in [0,1]
            input_ids (Tensor): Input token ids.
            attention_mask (Tensor): Attention mask for the input tokens.
            labels (Tensor): Target output tokens (B, L)

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

        for i in range(self.n):
            # VLM models might need prompt text and image
            outputs = self.model(pixel_values=x_adv, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if i % 10 == 0:
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
                    gen_ids = self.model.generate(pixel_values=x_adv, max_new_tokens=50)
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

class VLMBlackBoxPGDAttack:
    """
    Black-box NES-based PGD L_inf attack for Vision-Language Models.
    Targeted: pushes image towards a model output that matches a target token sequence.
    """

    def __init__(self, model, tokenizer, eps=8 / 255., n=50, alpha=1 / 255.,
                 momentum=0., k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Args:
            model: vision-language model (e.g., BLIP2, Flamingo, LLaVA)
            tokenizer: tokenizer matching the model (e.g., LLaMATokenizer)
            eps: max perturbation
            n: attack steps
            alpha: step size
            momentum: momentum factor in [0,1)
            k: # of NES samples
            sigma: noise std for NES
            rand_init: whether to randomly perturb the input at start
            early_stop: whether to stop if attack succeeds
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def estimate_gradient(self, x_adv, target_token_ids, prompt):
        """
        NES gradient estimate by querying the model with positive and negative perturbations.
        """
        B = x_adv.shape[0]
        grads = torch.zeros_like(x_adv)
        queries = 2 * self.k

        for i in range(self.k):
            noise = torch.randn_like(x_adv)
            x_pos = torch.clamp(x_adv + self.sigma * noise, 0, 1)
            x_neg = torch.clamp(x_adv - self.sigma * noise, 0, 1)

            with torch.no_grad():
                out_pos = self.model(image=x_pos, prompt=prompt, labels=target_token_ids)
                out_neg = self.model(image=x_neg, prompt=prompt, labels=target_token_ids)

            logits_pos = out_pos.logits
            logits_neg = out_neg.logits

            B, L, V = logits_pos.shape
            loss_pos = self.loss_func(logits_pos.view(B*L, V), target_token_ids.view(-1))
            loss_neg = self.loss_func(logits_neg.view(B*L, V), target_token_ids.view(-1))

            loss = loss_pos - loss_neg
            loss = -loss  # targeted: minimize distance to target
            grads += loss * noise

        grads /= (queries * self.sigma)
        return grads, queries

    def execute(self, x, target_token_ids, prompt=None):
        """
        Runs the black-box attack on input images `x` toward desired sequence `target_token_ids`.
        """
        torch.cuda.empty_cache()

        x_adv = x.clone().detach()
        if self.rand_init:
            x_adv += torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        velocity = torch.zeros_like(x)
        query_counts = torch.zeros(x.shape[0], dtype=torch.int, device=x.device)

        for i in range(self.n):
            grads, queries = self.estimate_gradient(x_adv, target_token_ids, prompt)
            query_counts += queries

            velocity = self.momentum * velocity + grads
            x_adv = x_adv + self.alpha * torch.sign(velocity)
            x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()

            if self.early_stop:
                with torch.no_grad():
                    gen_ids = self.model.generate(image=x_adv, prompt=prompt)
                    if (gen_ids == target_token_ids).all():
                        break

        return x_adv, query_counts


class VLMWhiteBoxPGDEnsembleAttack:
    """
    White-box PGD L_inf attack against an ensemble of vision-language models,
    targeting a desired sequence output from the image.
    """

    def __init__(self, models, tokenizer, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Args:
            models: list of VLM models
            tokenizer: tokenizer for the models
            eps: maximum perturbation
            n: number of PGD iterations
            alpha: step size
            rand_init: whether to initialize randomly within [x-eps, x+eps]
            early_stop: whether to stop when all models generate the target
        """
        self.models = models
        self.tokenizer = tokenizer
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def execute(self, image, target_token_ids, prompt=None):
        """
        Args:
            image (Tensor): batch of images (B, C, H, W)
            target_token_ids (Tensor): tokenized target sequences (B, L)
            prompt (str or list): prompt for context

        Returns:
            x_adv: adversarial images
        """
        x_adv = image.clone().detach()
        if self.rand_init:
            x_adv += torch.empty_like(image).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv.requires_grad = True

        for i in range(self.n):
            total_loss = 0.0
            for model in self.models:
                output = model(image=x_adv, prompt=prompt, labels=target_token_ids)
                logits = output.logits  # shape: (B, L, V)
                B, L, V = logits.shape
                loss = self.loss_func(logits.view(B * L, V), target_token_ids.view(-1))
                total_loss += loss

            total_loss /= len(self.models)
            loss = -total_loss  # targeted attack: maximize likelihood of target

            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv + self.alpha * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, image - self.eps), image + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()
            x_adv.requires_grad = True

            if self.early_stop:
                with torch.no_grad():
                    all_match = True
                    for model in self.models:
                        gen_ids = model.generate(image=x_adv, prompt=prompt)
                        if not (gen_ids == target_token_ids).all():
                            all_match = False
                            break
                    if all_match:
                        break

        return x_adv
