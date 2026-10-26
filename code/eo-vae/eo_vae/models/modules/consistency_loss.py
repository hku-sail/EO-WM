import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

# from focal_frequency_loss import FocalFrequencyLoss as FFL
from .ffl import FocalFrequencyLoss as FFL


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1-like loss)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + self.eps**2))


class SSIMLoss(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        # MS-SSIM calculates similarity at multiple resolutions.
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            sigma=1.5,
            kernel_size=5,
            betas=(0.0448, 0.2856, 0.3001, 0.2363),  # 4 scales
            normalize="relu",
        )

    def forward(self, pred, target):
        # MS-SSIM returns 1.0 for perfect match. Loss = 1 - score.
        return 1.0 - self.msssim(pred, target)


import torch.nn as nn


class DynamicPatchGAN(nn.Module):
    def __init__(self, input_conv_generator, ndf=128, n_layers=3):
        """input_conv_generator: Hypernetwork layer that generates
        weights for variable channels.
        """
        super().__init__()
        self.dynamic_input = input_conv_generator  # Wavelength-conditioned input convolution.

        # Standard PatchGAN backbone
        layers = []
        curr_dim = ndf

        # Initial block (ndf)
        # Spectral normalization stabilizes hinge-loss training.
        layers.append(
            nn.Sequential(
                spectral_norm(nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, True),
            )
        )

        # Intermediate blocks
        for i in range(1, n_layers):
            prev_dim = curr_dim
            curr_dim = min(ndf * (2**i), 512)
            layers.append(
                nn.Sequential(
                    # Spectral normalization followed by instance normalization.
                    spectral_norm(
                        nn.Conv2d(
                            prev_dim,
                            curr_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        )
                    ),
                    nn.InstanceNorm2d(curr_dim),
                    nn.LeakyReLU(0.2, True),
                )
            )

        # Final 1-channel prediction map (the "Patch")
        # Spectrally normalized output layer.
        layers.append(
            spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1))
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, wvs):
        # x: [B, C, H, W], wvs: [B, C]
        x = self.dynamic_input(x, wvs)  # Map variable C -> ndf
        return self.model(x)


class EOPatchLoss(nn.Module):
    def __init__(
        self, discriminator, disc_start=10000, disc_weight=0.5, ssim_weight=0.2
    ):
        super().__init__()
        self.discriminator = discriminator
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.ssim_weight = ssim_weight
        self.ssim = SSIMLoss()

    def forward(
        self,
        inputs,
        wvs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split='train',
    ):
        reconstructions = torch.clamp(reconstructions, -2.5, 5.0)
        if not torch.isfinite(inputs).all():
            raise RuntimeError("inputs has NaN/Inf")
        if not torch.isfinite(reconstructions).all():
            raise RuntimeError("reconstructions has NaN/Inf")

        # 1. GENERATOR BRANCH
        if optimizer_idx == 0:
            self.discriminator.eval()  # Freeze disc during gen step
            # Pixel-wise Accuracy (Radiometric Fidelity)
            rec_loss = F.l1_loss(reconstructions, inputs)

            # Structural Consistency (Works for any C-channel modality)
            # SSIM helps reconstruct fine-scale details without ImageNet bias
            ssim_loss = self.ssim(reconstructions, inputs)

            # Adversarial Loss (Texture & Sharpness)
            g_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
            logits_fake_mean = torch.tensor(
                0.0, device=inputs.device, dtype=inputs.dtype
            )
            weight = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

            if global_step >= self.disc_start:
                logits_fake = self.discriminator(reconstructions, wvs)
                g_loss = -torch.mean(logits_fake)
                logits_fake_mean = logits_fake.mean()  # Log this!

                # Adaptive weighting
                if last_layer is not None:
                    weight = self.calculate_adaptive_weight(
                        rec_loss, g_loss, last_layer
                    )
                    g_loss = g_loss * weight

            total_loss = (
                rec_loss + (self.disc_weight * g_loss) + (self.ssim_weight * ssim_loss)
            )

            return total_loss, {
                f'{split}/loss_rec': rec_loss,
                f'{split}/loss_g': g_loss,
                f'{split}/disc_weight': weight,
                f'{split}/loss_msssim': ssim_loss,
                f'{split}/logits_fake_g': logits_fake_mean,
            }

        # 2. DISCRIMINATOR BRANCH
        if optimizer_idx == 1:
            self.discriminator.train()
            logits_real = self.discriminator(inputs.detach(), wvs)
            logits_fake = self.discriminator(reconstructions.detach(), wvs)

            d_loss = 0.5 * (
                torch.mean(F.relu(1.0 - logits_real))
                + torch.mean(F.relu(1.0 + logits_fake))
            )
            return d_loss, {
                f'{split}/loss_disc': d_loss,
                f'{split}/logits_real': logits_real.mean(),
                f'{split}/logits_fake_d': logits_fake.mean(),
            }

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        # Adaptive weight logic to balance GAN vs Reconstruction
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(d_weight, 0.0, 2.0).detach()


class SAMLoss(nn.Module):
    """Optimizes Spectral Angle without using acos().
    Minimizing (1 - cos(theta)) is numerically stable.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_rec, x_true):
        # Dot product along channel dimension
        dot = torch.sum(x_rec * x_true, dim=1)

        # Norms
        norm_rec = torch.norm(x_rec, dim=1)
        norm_true = torch.norm(x_true, dim=1)

        # Cosine Similarity
        # eps prevents division by zero
        cos_sim = dot / (norm_rec * norm_true + self.eps)

        # Loss = 1 - CosineSimilarity
        # Range: [0, 2] (0 = aligned, 1 = orthogonal, 2 = opposite)
        return (1.0 - cos_sim).mean()


class BerHuLoss(nn.Module):
    """The 'BerHu' Loss.
    - Small errors (|x| <= c): Penalized linearly (L1). Good for texture.
    - Large errors (|x| > c): Penalized quadratically (L2). Good for hotspots.
    """

    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(pred - target)

        # Mask for small vs large errors
        mask = diff <= self.threshold

        # L1 part
        l1_loss = diff[mask]

        # L2 part (Rewritten to connect smoothly to L1)
        # (diff^2 + c^2) / (2c)
        l2_loss = (diff[~mask] ** 2 + self.threshold**2) / (2 * self.threshold)

        if l1_loss.numel() == 0:
            return l2_loss.mean()
        if l2_loss.numel() == 0:
            return l1_loss.mean()

        return (l1_loss.sum() + l2_loss.sum()) / diff.numel()


class GradientDifferenceLoss(nn.Module):
    """Penalizes differences in the gradient of the image.
    Popular in Medical Imaging and Video Frame Interpolation.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Calculate gradients in X and Y direction
        # shape: [B, C, H, W]

        # Vertical Gradients (dy)
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Horizontal Gradients (dx)
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        # We want the MAGNITUDE of the gradients to match (Texture preservation)
        # AND the values to match (Edge alignment)
        loss_y = torch.pow(torch.abs(pred_dy - target_dy), self.alpha)
        loss_x = torch.pow(torch.abs(pred_dx - target_dx), self.alpha)

        return loss_x.mean() + loss_y.mean()


class SpatialGradientLoss(nn.Module):
    """Penalizes differences in gradients to preserve edges and reduce blur."""

    def __init__(self):
        super().__init__()
        # Simple Sobel kernels for gradients
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x_rec, x_true):
        b, c, h, w = x_rec.shape

        # Flatten channels for group conv
        x_rec_flat = x_rec.reshape(-1, 1, h, w)
        x_true_flat = x_true.reshape(-1, 1, h, w)

        grad_x_rec = F.conv2d(x_rec_flat, self.kernel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_flat, self.kernel_y, padding=1)

        grad_x_true = F.conv2d(x_true_flat, self.kernel_x, padding=1)
        grad_y_true = F.conv2d(x_true_flat, self.kernel_y, padding=1)

        loss = F.l1_loss(grad_x_rec, grad_x_true) + F.l1_loss(grad_y_rec, grad_y_true)
        return loss


class DOFASemanticLoss(nn.Module):
    """Computes semantic feature loss using DOFA features."""

    def __init__(self, dofa_net: nn.Module):
        super().__init__()
        self.dofa_net = dofa_net
        # Freeze DOFA
        for p in self.dofa_net.parameters():
            p.requires_grad = False

    def forward(
        self, inputs: torch.Tensor, reconstructions: torch.Tensor, wvs: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            f_in = self.dofa_net.forward_features(inputs, wvs)
        # Allow gradients to flow to reconstructions
        f_rec = self.dofa_net.forward_features(reconstructions, wvs)

        l_feat = 0.0
        for fi, fr in zip(f_in, f_rec):
            l_feat += (1.0 - F.cosine_similarity(fi, fr, dim=1)).mean()

        return l_feat


class EOConsistencyLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,  # Reconstruction loss weight (L1 or Charbonnier)
        rec_loss_type: str = 'l1',  # 'l1' or 'char' for reconstruction loss
        spectral_weight: float = 0.0,  # SAM for spectral accuracy
        spatial_weight: float = 0.0,  # Gradient for edges
        freq_weight: float = 0.0,  # FFT for textures
        feature_weight: float = 0.0,  # Optional DOFA features
        msssim_weight: float = 0.0,  # MS-SSIM loss
        temporal_coherence_weight: float = 0.0,  # Latent smoothness across time
        spectral_start_step: int = 0,
        spatial_start_step: int = 0,
        freq_start_step: int = 0,
        feature_start_step: int = 0,
        msssim_start_step: int = 0,
        patch_factor: int = 2,
        ffl_alpha: float = 1.0,
        dofa_net: nn.Module = None,
    ):
        """Initializes the EO Consistency Loss with multiple components.

        Args:
            pixel_weight: Weight for reconstruction loss (L1 or Charbonnier).
            rec_loss_type: Type of reconstruction loss ('l1' or 'char').
            spectral_weight: Weight for spectral angle mapper loss, useful for multispectral data.
            spatial_weight: Weight for spatial gradient loss to preserve edges.
            freq_weight: Weight for focal frequency loss to capture textures.
            feature_weight: Weight for semantic feature loss using a pretrained DOFA network.
            msssim_weight: Weight for MS-SSIM loss.
            spectral_start_step: Global step to start applying spectral loss.
            spatial_start_step: Global step to start applying spatial loss.
            freq_start_step: Global step to start applying frequency loss.
            feature_start_step: Global step to start applying feature loss.
            msssim_start_step: Global step to start applying MS-SSIM loss.
            dofa_net: Pretrained DOFA network for feature extraction (if feature_weight > 0).

        """
        super().__init__()

        self.rec_loss_type = rec_loss_type
        self.starts = {
            'spectral': spectral_start_step,
            'spatial': spatial_start_step,
            'freq': freq_start_step,
            'feature': feature_start_step,
            'msssim': msssim_start_step,
        }
        self.weights = {
            'pixel': pixel_weight,
            'spectral': spectral_weight,
            'spatial': spatial_weight,
            'freq': freq_weight,
            'feature': feature_weight,
            'msssim': msssim_weight,
            'temporal_coherence': temporal_coherence_weight,
        }

        self.sam_loss = SAMLoss()
        self.grad_loss = GradientDifferenceLoss()
        self.fft_loss = FFL(
            loss_weight=1.0,
            alpha=ffl_alpha,
            patch_factor=patch_factor,
            ave_spectrum=False,
            batch_matrix=True,
            log_matrix=True,
        )
        self.char_loss = CharbonnierLoss()
        self.msssim_loss = SSIMLoss()
        self.feature_loss = DOFASemanticLoss(dofa_net) if dofa_net is not None else None

    def forward(
        self,
        inputs: torch.Tensor,
        wvs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int = 0,
        split: str = 'train',
        **kwargs,
    ):
        # clamp reconstructions to valid range

        logs = {}
        # Initialize as tensor to ensure device consistency
        total_loss = torch.tensor(0.0, device=inputs.device)

        # 1. Reconstruction Loss (Always Active, L1 or Charbonnier)
        if self.weights['pixel'] > 0:
            if self.rec_loss_type == 'l1':
                l_rec = F.l1_loss(reconstructions, inputs)
            elif self.rec_loss_type == 'char':
                l_rec = self.char_loss(reconstructions, inputs)
            else:
                raise ValueError("rec_loss_type must be 'l1' or 'char'")
            total_loss = total_loss + self.weights['pixel'] * l_rec
            logs[f'{split}/loss_rec'] = l_rec.detach()

        # 2. Spectral Loss (Scheduled)
        if self.weights['spectral'] > 0:
            if global_step >= self.starts['spectral']:
                l_sam = self.sam_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['spectral'] * l_sam
                logs[f'{split}/loss_spectral'] = l_sam.detach()

        # 3. Spatial Loss (Scheduled)
        if self.weights['spatial'] > 0:
            if global_step >= self.starts['spatial']:
                l_spat = self.grad_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['spatial'] * l_spat
                logs[f'{split}/loss_spatial'] = l_spat.detach()

        if self.weights['freq'] > 0 and global_step >= self.starts['freq']:
            # 1. Calculate the raw loss once
            raw_focal_freq_loss = self.fft_loss(reconstructions, inputs)

            # 2. Calculate the warmup factor (0.0 to 1.0)
            # Using max(0, ...) ensures we never have negative weight
            # Warmup duration: 1000 steps.
            warmup_steps = 1000
            start_step = self.starts['freq']  # 6000

            warmup_factor = min(
                1.0, max(0.0, (global_step - start_step) / warmup_steps)
            )

            # 3. Apply the warmup multiplier and base frequency weight.
            # The configured frequency weight sets the final scale.
            current_ffl_weight = self.weights['freq'] * warmup_factor

            weighted_ffl_loss = raw_focal_freq_loss * current_ffl_weight

            # 4. Update total loss and logs
            total_loss = total_loss + weighted_ffl_loss

            # Log the raw loss so you can see if the model is actually learning textures,
            # and log the weight to verify the warmup in WandB/Tensorboard.
            logs[f'{split}/loss_freq_raw'] = raw_focal_freq_loss.detach()
            logs[f'{split}/ffl_weight'] = torch.tensor(current_ffl_weight)

        # 5. MS-SSIM Loss (Scheduled)
        if self.weights['msssim'] > 0:
            if global_step >= self.starts['msssim']:
                l_msssim = self.msssim_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['msssim'] * l_msssim
                logs[f'{split}/loss_msssim'] = l_msssim.detach()

        # 6. Semantic Feature Loss (Scheduled + Requires Net)
        if self.weights['feature'] > 0:
            if global_step >= self.starts['feature']:
                l_feat = self.feature_loss(inputs, reconstructions, wvs)
                total_loss += self.weights['feature'] * l_feat
                logs[f'{split}/loss_feature'] = l_feat.detach()

        # 7. Temporal Coherence Loss (optional; expects z_sequence [B,T,...]
        # passed from the training step). Penalizes L1 between consecutive
        # frames' latents to encourage smooth temporal structure. Dormant when
        # temporal_coherence_weight == 0 (the default), so non-temporal configs
        # are unaffected.
        z_sequence = kwargs.get('z_sequence', None)
        if self.weights['temporal_coherence'] > 0 and z_sequence is not None:
            if z_sequence.dim() >= 2 and z_sequence.shape[1] > 1:
                l_tc = F.l1_loss(
                    z_sequence[:, 1:], z_sequence[:, :-1]
                )
                total_loss = total_loss + self.weights['temporal_coherence'] * l_tc
                logs[f'{split}/loss_temporal_coherence'] = l_tc.detach()

        logs[f'{split}/loss_total'] = total_loss.detach()
        return total_loss, logs
