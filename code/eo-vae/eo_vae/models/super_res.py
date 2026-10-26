"""Lightning Module to train Prior Diffusion model."""
from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float,
    final_lr: float,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        lr_scale = (base_lr - final_lr) * cosine_decay + final_lr
        return lr_scale / base_lr

    return LambdaLR(optimizer, lr_lambda)


class DiffusionSuperRes(LightningModule):
    """Lightning Module to train a Prior Diffusion model with Azula modules."""

    def __init__(
        self,
        denoiser: nn.Module,
        sampler: partial[nn.Module],
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        base_lr: float = 1e-4,
        final_lr: float | None = None,
        warmup_epochs: int | None = None,
        decay_end_epoch: int | None = None,
    ) -> None:
        """Initialize the DiffusionModule.

        Args:
            denoiser: the denoising model
            sampler: the sampling model (e.g., DDIMSampler)
            optimizer: the optimizer to use
            lr_scheduler: the learning rate scheduler to use
            base_lr: base learning rate
            final_lr: final learning rate after decay
            warmup_epochs: number of warmup epochs
            decay_end_epoch: epoch to end decay
        """
        super().__init__()
        self.denoiser = denoiser
        self.sampler = sampler(denoiser=denoiser)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = decay_end_epoch

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the denoiser.

        Args:
            x: target input tensor from which to gather shape
            t: time steps
            kwargs: additional keyword arguments for the denoiser

        Returns:
            denoised tensor at t - 1 step
        """
        x_t = torch.randn_like(x)
        return self.denoiser(x_t, t, **kwargs)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for the diffusion model.

        Args:
            batch: a batch of data
            batch_idx: index of the batch

        Returns:
            the training loss
        """
        batch_size = batch['image_hr'].shape[0]
        # sample random ode continous time interval [0, 1]
        t = torch.rand(batch_size, device=self.device)
        # azula computes loss internally, expecting x and t, with everything else being kwargs
        # that go to underlying backbone of denoiser
        train_loss = self.denoiser.loss(
            x=batch['image_hr'], t=t, cond=batch['image_lr']
        )
        self.log(
            'train_loss', train_loss, prog_bar=True, on_step=True, batch_size=batch_size
        )
        return train_loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for the diffusion model.

        Uses standard sampler without guidance to evaluate denoiser quality.

        Args:
            batch: a batch of data
            batch_idx: index of the batch

        Returns:
            the validation mse
        """
        batch_size = batch['image_hr'].shape[0]

        # again any other args than x1_shape go to sampler kwargs
        # which will be passed to denoiser backbone forward
        x0 = self.sample(
            x1_shape=batch['image_hr'].shape,
            cond=batch['image_lr'],  # if we have conditioning
        )

        val_mse = F.mse_loss(x0, batch['image_hr'])
        self.log(
            'val_mse', val_mse, prog_bar=True, on_step=False, batch_size=batch_size
        )

        return val_mse

    def sample(self, x1_shape: tuple[int, ...], **sampler_kwargs) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            x1_shape: shape of the initial noise tensor
            sampler_kwargs: additional keyword arguments for the sampler, such as
                conditioning information

        Returns:
            fully denoised tensor at time step 0
        """
        x1 = self.sampler.init(x1_shape, device=self.device)
        return self.sampler(x1, **sampler_kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        if self.base_lr is not None:
            optimizer = self.optimizer(self.denoiser.parameters(), lr=self.base_lr)
        else:
            optimizer = self.optimizer(self.denoiser.parameters())

        if (
            all([self.final_lr, self.warmup_epochs, self.decay_end_epoch])
            and self.base_lr is not None
        ):
            steps_per_epoch = 152  # Estimate
            num_warmup = self.warmup_epochs * steps_per_epoch
            num_total = self.decay_end_epoch * steps_per_epoch

            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=num_total,
                base_lr=self.base_lr,
                final_lr=self.final_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'},
            }

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': 'val_loss'},
            }
        else:
            return {'optimizer': optimizer}
