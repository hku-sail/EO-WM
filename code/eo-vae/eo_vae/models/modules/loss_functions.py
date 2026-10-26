"""Loss module for EO-Autoencoder training using DOFA-based losses."""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
# from focal_frequency_loss import FocalFrequencyLoss as FFL
from .ffl import FocalFrequencyLoss as FFL

from .loss_utils import hinge_d_loss, vanilla_d_loss, vanilla_g_loss


class EOGenerativeLoss(nn.Module):
    """Combined loss for RAE training.
    Routes multispectral data and wavelengths (wvs) to DOFA-based LPIPS and Discriminator.
    """

    def __init__(
        self,
        # Modules (injected)
        discriminator: nn.Module,
        lpips: nn.Module,
        # Loss weights
        perceptual_weight: float = 1.0,
        disc_weight: float = 0.75,
        # Training schedule
        gan_start_step: int = 0,
        disc_update_start_step: int = 0,
        # Optimization
        max_d_weight: float = 1e4,
        disc_loss_type: str = 'hinge',
        # Focal loss
        focal_loss_weight: float = 0.0,
        focal_loss_alpha: float = 0.0,
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.gan_start_step = gan_start_step
        self.disc_update_start_step = disc_update_start_step
        self.max_d_weight = max_d_weight

        self.discriminator = discriminator
        self.lpips_loss = lpips

        # Optional Focal Frequency Loss
        self.focal_loss_weight = focal_loss_weight
        if self.focal_loss_weight > 0.0:
            self.ffl = FFL(loss_weight=focal_loss_weight, alpha=focal_loss_alpha)

        # Discriminator Loss Function
        if disc_loss_type == 'hinge':
            self.disc_loss_fn = hinge_d_loss
        else:
            self.disc_loss_fn = vanilla_d_loss
        self.gen_loss_fn = vanilla_g_loss

    def calculate_adaptive_weight(
        self,
        recon_loss: torch.Tensor,
        gan_loss: torch.Tensor,
        last_layer: torch.nn.Parameter,
    ) -> torch.Tensor:
        """Balances Gradient norms between Reconstruction and GAN loss."""
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
        return torch.clamp(d_weight, 0.0, self.max_d_weight).detach()

    def robust_normalize(self, x: torch.Tensor, clip_val: float = 3.0) -> torch.Tensor:
        """Linearly maps Z-scored data from roughly [-clip_val, clip_val] to [-1, 1].
        Preserves linearity (texture details) better than Tanh for the Discriminator.
        """
        # Clamp outliers to prevent gradient explosions in the discriminator
        x_clamped = torch.clamp(x, -clip_val, clip_val)
        return x_clamped / clip_val

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        wvs: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.nn.Parameter | None = None,
        split: str = 'train',
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Schedule checks
        use_gan = global_step >= self.gan_start_step and self.disc_weight > 0.0
        train_disc = (
            global_step >= self.disc_update_start_step and self.disc_weight > 0.0
        )

        # -------------------------------------------------------------------
        # Generator Step (optimizer_idx == 0)
        # -------------------------------------------------------------------
        if optimizer_idx == 0:
            if self.discriminator is not None:
                self.discriminator.eval()  # Important: Freeze disc during gen step

            # 1. Pixel-wise Loss (L1)
            rec_loss = F.l1_loss(reconstructions, inputs)

            # 2. Focal Frequency Loss (Optional)
            if self.focal_loss_weight > 0.0:
                rec_loss = rec_loss + self.ffl(reconstructions, inputs)

            # 3. DOFA-LPIPS Loss (Perceptual)
            if self.perceptual_weight > 0.0:
                # Pass wvs so DOFA can process the bands correctly
                p_loss = self.lpips_loss(inputs, reconstructions, wvs)
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0)

            # 4. GAN Loss (Generator side)
            if use_gan:
                reconstructions = self.robust_normalize(reconstructions)
                logits_fake, _ = self.discriminator(reconstructions, None, wvs)
                g_loss = self.gen_loss_fn(logits_fake)

                # Adaptive Weighting
                if last_layer is not None:
                    d_weight = self.calculate_adaptive_weight(
                        rec_loss, g_loss, last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)

                total_loss = rec_loss + d_weight * self.disc_weight * g_loss
            else:
                g_loss = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)
                total_loss = rec_loss

            log = {
                f'{split}/loss_total': total_loss.detach(),
                f'{split}/loss_rec': rec_loss.detach(),
                f'{split}/loss_lpips': p_loss.detach(),
                f'{split}/loss_gan': g_loss.detach(),
                f'{split}/d_weight': d_weight,
            }
            return total_loss, log

        # -------------------------------------------------------------------
        # Discriminator Step (optimizer_idx == 1)
        # -------------------------------------------------------------------
        elif optimizer_idx == 1:
            if not train_disc:
                return torch.tensor(0.0, requires_grad=True), {}

            self.discriminator.train()

            # Compute logits on Real and Fake (detached)
            logits_fake, logits_real = self.discriminator(
                reconstructions.detach(), inputs, wvs
            )

            d_loss = self.disc_loss_fn(logits_real, logits_fake)

            log = {
                f'{split}/loss_disc': d_loss.detach(),
                f'{split}/logits_real': logits_real.mean().detach(),
                f'{split}/logits_fake': logits_fake.mean().detach(),
            }
            return d_loss, log

        else:
            raise ValueError(f'Unknown optimizer_idx {optimizer_idx}')
