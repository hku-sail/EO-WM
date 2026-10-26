import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (
    1,
    7,
    1,
)
if IS_HIGH_VERSION:
    import torch.fft


import torch


class FocalFrequencyLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False,
        batch_matrix=False,
    ):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # Force FP32 inside the loss to prevent overflow in ComplexHalf (FFT)
        # Even if the trainer uses AMP, FFT coefficients grow too large for FP16.
        x = x.float()

        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        patch_h, patch_w = h // patch_factor, w // patch_factor

        # Optimized patch extraction
        y = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        y = y.permute(0, 2, 3, 1, 4, 5).reshape(
            x.size(0), -1, x.size(1), patch_h, patch_w
        )

        # FFT calculation
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)

        # Safety check: Remove any INF/NaN before distance calculation
        return torch.nan_to_num(freq, nan=0.0, posinf=1e6, neginf=-1e6)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # Distance in frequency domain
            matrix_tmp = (recon_freq - real_freq) ** 2
            # Add eps to sqrt to avoid infinite gradients at distance 0
            matrix_tmp = (
                torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1] + 1e-8) ** self.alpha
            )

            if self.log_matrix:
                matrix_tmp = torch.log1p(matrix_tmp)

            # --- NUMERICAL STABILITY FIX ---
            # Avoid dividing by INF or 0
            if self.batch_matrix:
                max_val = matrix_tmp.max()
            else:
                max_val = matrix_tmp.flatten(2).max(-1).values[:, :, :, None, None]

            # If max_val is INF or NaN, fallback to 1.0 to avoid Nan weight matrix
            max_val = torch.where(
                torch.isfinite(max_val) & (max_val > 0),
                max_val,
                torch.ones_like(max_val),
            )
            weight_matrix = (matrix_tmp / max_val).clamp(0.0, 1.0).detach()

        # Frequency distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        # Predictions are expected to be clamped to the valid input range.
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
