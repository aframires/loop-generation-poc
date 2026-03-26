from typing import Any, Callable, Optional
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LRScheduler

from loop_generation_poc.models.components.transformer import Transformer

class LinearWarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        max_steps: int, 
        eta_min: float = 1e-6, 
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps 
                for base_lr in self.base_lrs
            ]
        
        # Cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

class ConditionalDiffusion(pl.LightningModule):
    """
    PyTorch Lightning Module for training a Conditional Diffusion model.
    Supports both 'v_prediction' and 'rectified_flow' objectives.
    """
    def __init__(
        self,
        latent_dim: int = 64,
        text_dim: int = 768,
        hidden_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        diffusion_objective: str = "v_prediction",
        optimizer: Optional[Callable[..., torch.optim.Optimizer]] = None,
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        loss_fn: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Only save architecture arguments so inference works without Hydra
        self.save_hyperparameters(
            "latent_dim", "text_dim", "hidden_dim", "depth", "num_heads", "diffusion_objective"
        )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = Transformer(
                latent_dim=self.hparams.latent_dim,
                text_dim=self.hparams.text_dim,
                hidden_dim=self.hparams.hidden_dim,
                depth=self.hparams.depth,
                num_heads=self.hparams.num_heads,
            )

    @classmethod
    def load_for_inference(cls, ckpt_path: str, map_location: str = "cpu"):
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(map_location)
        return model

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass, mainly used for inference.
        x: [Batch, SeqLen, LatentDim]
        t: [Batch]
        context: [Batch, ContextSeqLen, ContextDim]
        """
        return self.backbone(x, t, context)

    def get_alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the cosine schedule coefficients for continuous time t in [0, 1].
        Returns alpha_t and sigma_t.
        """
        # [Batch]
        theta = t * (torch.pi / 2.0)
        alpha = torch.cos(theta)
        sigma = torch.sin(theta)
        return alpha, sigma

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Computes the diffusion loss based on the selected objective.
        """
        # [Batch, SeqLen, LatentDim]
        data = batch["latents"]
        # [Batch, ContextSeqLen, ContextDim]
        context = batch["text_embeds"]
        
        B, S, D = data.shape

        # Sample continuous timesteps t ~ Uniform(0, 1)
        t = torch.rand((B,), device=self.device)

        if self.hparams.diffusion_objective in ["v_prediction", "v-prediction"]:
            # Get schedule coefficients and reshape for broadcasting
            alpha, sigma = self.get_alpha_sigma(t)
            alpha = alpha.view(-1, 1, 1)
            sigma = sigma.view(-1, 1, 1)

            # Sample noise epsilon ~ N(0, I)
            eps = torch.randn_like(data)

            # Corrupt the data (q sample)
            x_t = alpha * data + sigma * eps

            # Compute the V-prediction target
            v_target = alpha * eps - sigma * data

        elif self.hparams.diffusion_objective in ["rectified_flow", "rectified-flow"]:
            # Interpolate between noise (x_0) and data (x_1)
            t_expand = t.view(-1, 1, 1)
            noise = torch.randn_like(data)
            
            x_t = t_expand * data + (1.0 - t_expand) * noise
            
            # Target velocity is the path derivative (x_1 - x_0)
            v_target = data - noise
            
        else:
            raise ValueError(f"Unknown diffusion_objective: {self.hparams.diffusion_objective}")

        # Model prediction
        v_pred = self.forward(x_t, t, context)

        # Compute dynamically configured Loss
        loss = self.loss_fn(v_pred, v_target)

        # Log to wandb
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step (exact same math as training, just logged differently).
        """
        data = batch["latents"]
        context = batch["text_embeds"]
        B = data.shape[0]

        t = torch.rand((B,), device=self.device)

        if self.hparams.diffusion_objective in ["v_prediction", "v-prediction"]:
            alpha, sigma = self.get_alpha_sigma(t)
            alpha = alpha.view(-1, 1, 1)
            sigma = sigma.view(-1, 1, 1)
            eps = torch.randn_like(data)
            x_t = alpha * data + sigma * eps
            v_target = alpha * eps - sigma * data

        elif self.hparams.diffusion_objective in ["rectified_flow", "rectified-flow"]:
            t_expand = t.view(-1, 1, 1)
            noise = torch.randn_like(data)
            x_t = t_expand * data + (1.0 - t_expand) * noise
            v_target = data - noise
            
        else:
            raise ValueError(f"Unknown diffusion_objective: {self.hparams.diffusion_objective}")

        v_pred = self.forward(x_t, t, context)
        loss = self.loss_fn(v_pred, v_target)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Any:
        """
        Instantiates the optimizer and optional learning rate scheduler from Hydra configs.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not configured. Cannot train.")

        optimizer = self.optimizer(params=self.backbone.parameters())
        
        # If no scheduler is provided in the config, just return the optimizer
        if self.scheduler is None:
            return {"optimizer": optimizer}
            
        # Instantiate the partially defined scheduler with the optimizer
        scheduler = self.scheduler(optimizer=optimizer)
        
        # Lightning requires this specific dictionary format for schedulers
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the LR after every batch (standard for Transformers/Cosine)
                "frequency": 1,
            },
        }