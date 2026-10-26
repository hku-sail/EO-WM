from __future__ import annotations
import os

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],  # R, G, B
    'S2L2A': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        1.610,
        2.190,
        0.945,
    ],  # 12 bands
    'S2L1C': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        0.945,
        1.375,
        1.610,
        2.190,
    ],  # 13 bands
}

RGB_INDICES = {
    'S2RGB': [0, 1, 2],  # Already RGB
    'S2L2A': [3, 2, 1],  # R: 0.665 (B04), G: 0.560 (B03), B: 0.490 (B02)
    'S2L1C': [3, 2, 1],  # Same as S2L2A
}

# Legacy normalization stats (from original TerraMesh)
NORM_STATS_LEGACY = {
    'S2L2A': {
        'mean': [
            1375.648,
            1489.600,
            1709.087,
            1831.752,
            2186.075,
            2794.358,
            3008.528,
            3096.780,
            3155.180,
            3169.651,
            2415.761,
            1838.622,
        ],
        'std': [
            2101.107,
            2138.673,
            2033.628,
            2118.186,
            2061.646,
            1869.234,
            1801.386,
            1841.173,
            1734.404,
            1751.174,
            1375.131,
            1284.165,
        ],
    },
    'S1RTC': {'mean': [-10.793, -17.198], 'std': [4.278, 4.346]},
    'S2L1C': {
        'mean': [
            2475.625,
            2260.839,
            2143.561,
            2230.225,
            2445.427,
            2992.950,
            3257.843,
            3171.695,
            3440.958,
            1567.433,
            561.076,
            2562.809,
            1924.178,
        ],
        'std': [
            1761.905,
            1804.267,
            1661.263,
            1932.020,
            1918.007,
            1812.421,
            1795.179,
            1734.280,
            1780.039,
            1082.531,
            512.077,
            1350.580,
            1177.511,
        ],
    },
    'S2RGB': {'mean': [110.349, 99.507, 75.843], 'std': [69.905, 53.708, 53.378]},
    'DEM': {'mean': [651.663], 'std': [928.168]},
}

# Custom normalization stats (with 10k clipping and time-aware harmonization)
NORM_STATS_CUSTOM = {
    'S2L2A': {
        'mean': [
            1718.9949,
            1825.5669,
            2043.5834,
            2175.4543,
            2522.9522,
            3114.2216,
            3323.3469,
            3417.3660,
            3470.9655,
            3489.4869,
            2725.9735,
            2152.0551,
        ],
        'std': [
            2126.3409,
            2140.1035,
            2044.6618,
            2125.3351,
            2065.3251,
            1874.4652,
            1808.0426,
            1839.0210,
            1737.9521,
            1738.5136,
            1456.5919,
            1365.1743,
        ],
    },
    'S2L1C': {
        'mean': [
            2424.2556,
            2207.7019,
            2098.2302,
            2167.1584,
            2382.3115,
            2938.8499,
            3204.8447,
            3126.6599,
            3389.0706,
            1580.1287,
            572.5726,
            2552.1208,
            1917.9390,
        ],
        'std': [
            1700.3824,
            1731.5450,
            1610.9904,
            1833.5536,
            1808.5067,
            1694.4427,
            1678.2327,
            1625.7446,
            1659.3112,
            1093.5255,
            515.6395,
            1300.8892,
            1151.6169,
        ],
    },
}


class ImageLogger(Callback):
    def __init__(self, max_images=8, save_dir='images'):
        super().__init__()
        self.max_images = max_images
        self.save_dir = save_dir

    @rank_zero_only
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        """Called automatically by Lightning when the validation loop starts.
        We strictly intercept the very first batch (batch_idx == 0).
        """
        if batch_idx == 0:
            self.log_local(
                self.save_dir,
                batch,
                pl_module,
                global_step=trainer.global_step,
                split='val',
            )

    def log_local(self, save_dir, batch, pl_module, global_step, split='val'):
        root = os.path.join(save_dir, 'image_log', split)
        os.makedirs(root, exist_ok=True)

        # Extract batch data
        images = batch['image']
        wvs = batch['wvs']
        modality = batch.get('modality', 'S2RGB')
        t_index = batch.get('t_index', None)
        is_temporal = images.dim() == 5

        # Determine normalization scheme from datamodule
        dm = pl_module.trainer.datamodule
        norm_scheme = getattr(dm, 'norm_scheme', 'legacy')

        # 1. Forward Pass (keep temporal dims; flatten to per-frame images
        # below for visualization so the rest of this method stays 4D).
        with torch.no_grad():
            reconstruction = pl_module(images, wvs, t_index=t_index)
            if isinstance(reconstruction, tuple):
                reconstruction = reconstruction[0]

        # Flatten [B,T,C,H,W] -> [B*T,C,H,W] for 4D-compatible visualization.
        if is_temporal:
            images = images.reshape(-1, *images.shape[2:])
            reconstruction = reconstruction.reshape(-1, *reconstruction.shape[2:])

        # 2. Slice batch
        N = min(images.shape[0], self.max_images)
        inputs = images[:N]
        recons = reconstruction[:N]

        # 3. Un-normalize to Physical Units
        inputs_phys = self._denormalize(inputs, modality, norm_scheme, pl_module.device)
        recons_phys = self._denormalize(recons, modality, norm_scheme, pl_module.device)

        # 4. Helper: Physical -> Visual [0, 1]
        def to_vis(x, rgb_indices):
            # Select RGB channels
            x = x[:, rgb_indices, :, :]
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1)

            vis_batch = []
            for i in range(b):
                img = x_flat[i]
                # Robust scaling (2% - 98%) per image
                low = torch.quantile(img, 0.02)
                high = torch.quantile(img, 0.98)
                img_norm = (x[i] - low) / (high - low + 1e-5)
                vis_batch.append(torch.clamp(img_norm, 0, 1))

            return torch.stack(vis_batch)

        rgb_indices = RGB_INDICES.get(modality, [0, 1, 2])
        inputs_vis = to_vis(inputs_phys, rgb_indices)
        recons_vis = to_vis(recons_phys, rgb_indices)

        # Normalize Error Map (use mean diff across selected channels)
        diff = torch.abs(
            inputs_phys[:, rgb_indices, :, :] - recons_phys[:, rgb_indices, :, :]
        ).mean(dim=1, keepdim=True)
        diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
        diff_vis = diff_vis.repeat(1, 3, 1, 1)

        # 5. Grid Construction
        rows = []
        for i in range(N):
            rows.append(torch.cat((inputs_vis[i], recons_vis[i], diff_vis[i]), dim=2))

        grid = torch.cat(rows, dim=1)

        # 6. Plot & Save
        filename = f'global_step_{global_step:04}_{modality}_{norm_scheme}.png'
        path = os.path.join(root, filename)

        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, N * 3))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(
            f'Global Step: {global_step} | Modality: {modality} | Scheme: {norm_scheme}\n'
            f'Input (Phys) | Recon (Phys) | Error (Phys)'
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _denormalize(self, x, modality, norm_scheme, device):
        """Denormalize images back to physical units.

        Args:
            x: Normalized tensor (B, C, H, W)
            modality: Modality name (e.g., 'S2L2A', 'S2L1C', 'S2RGB')
            norm_scheme: 'legacy' or 'custom'
            device: Device to move stats to

        Returns:
            Denormalized tensor in physical units
        """
        # Select appropriate stats based on scheme
        if norm_scheme == 'custom' and modality in NORM_STATS_CUSTOM:
            stats = NORM_STATS_CUSTOM[modality]
        elif modality in NORM_STATS_LEGACY:
            stats = NORM_STATS_LEGACY[modality]
        else:
            # No denormalization possible, return as is
            return x

        # Move stats to device
        mean = torch.tensor(stats['mean'], device=device).view(1, -1, 1, 1)
        std = torch.tensor(stats['std'], device=device).view(1, -1, 1, 1)

        # Denormalize: x_phys = x_norm * std + mean
        x_phys = x * std + mean

        # For custom scheme with S2L2A/S2L1C, the data was clipped to [0, 10000]
        # before normalization, so we should clip the denormalized values too
        if norm_scheme == 'custom' and modality in ['S2L2A', 'S2L1C']:
            x_phys = torch.clamp(x_phys, min=0.0, max=10000.0)

        return x_phys
