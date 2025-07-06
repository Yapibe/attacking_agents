import torch
import torch.nn as nn


class VLMWhiteBoxPGDAttack:
    """
    PGD L_inf white-box attack for Vision-Language Models.
    Targeted attack: tries to force the model to generate the desired output sequence for an image.
    """

    def __init__(self, model, tokenizer, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: VLM agent model with image + text input, outputs logits or token sequences
        - tokenizer: Tokenizer used by the VLM (e.g., T5Tokenizer, CLIPTokenizer)
        - eps: maximum perturbation
        - n: number of PGD steps
        - alpha: step size
        - rand_init: whether to randomly initialize image
        - early_stop: stop if target sequence is already achieved
        """
        self.model = model
        self.tokenizer = tokenizer
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')  # token-level loss

    def execute(self, image, target_token_ids, prompt=None):
        """
        Performs a targeted PGD attack on an input image.

        Args:
            image (Tensor): Input image tensor, shape (B, C, H, W), values in [0,1]
            target_token_ids (Tensor): Target output tokens (B, L)
            prompt (str or list of str): Optional prompt for generation context

        Returns:
            Adversarial image tensor
        """
        x_adv = image.clone().detach()

        if self.rand_init:
            x_adv += torch.empty_like(image).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv.requires_grad = True

        for i in range(self.n):
            # VLM models might need prompt text and image
            outputs = self.model(image=x_adv, prompt=prompt, labels=target_token_ids)
            logits = outputs.logits  # shape: (B, L, vocab_size)

            # Reshape for token-level CE loss
            B, L, V = logits.shape
            loss = self.loss_func(logits.view(B * L, V), target_token_ids.view(-1))
            loss = -loss  # Minimize the loss => maximize likelihood of target

            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv + self.alpha * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, image - self.eps), image + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()
            x_adv.requires_grad = True

            if self.early_stop:
                with torch.no_grad():
                    gen_ids = self.model.generate(image=x_adv, prompt=prompt)
                    if (gen_ids == target_token_ids).all():
                        break

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