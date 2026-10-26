from __future__ import annotations
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class SuperResImageLogger(Callback):
    def __init__(self, max_images=8, save_dir='images', log_every_n_epochs=10):
        super().__init__()
        self.max_images = max_images
        self.save_dir = save_dir
        self.log_every_n_epochs = log_every_n_epochs

    @rank_zero_only
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        """Called automatically by Lightning when the validation loop starts.
        We strictly intercept the very first batch (batch_idx == 0).
        """
        if batch_idx == 0 and trainer.current_epoch % self.log_every_n_epochs == 0:
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

        # 1. Get Data
        # DiffusionSuperRes expects 'image_lr' (cond) and 'image_hr' (target)
        img_lr = batch['image_lr']
        img_hr = batch['image_hr']

        # 2. Slice batch
        N = min(img_hr.shape[0], self.max_images)
        img_lr = img_lr[:N]
        img_hr = img_hr[:N]

        # 3. Sample Prediction
        with torch.no_grad():
            # Call the sample method from DiffusionSuperRes
            # This uses the 'image_lr' as conditioning to generate 'image_hr'
            reconstruction = pl_module.sample(x1_shape=img_hr.shape, cond=img_lr)

        # 4. Helper: Value -> Visualization [0, 1]
        # Since we don't have explicit stats, we use robust min-max normalization per image.
        # This handles both latents and images reasonably well for visual check.
        def to_vis(x):
            # Select first 3 channels for RGB visualization
            if x.shape[1] > 3:
                x = x[:, :3, :, :]
            elif x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1)

            vis_batch = []
            for i in range(b):
                img = x_flat[i]
                # Robust scaling (2% - 98%) per image to ignore outliers
                low = torch.quantile(img, 0.02)
                high = torch.quantile(img, 0.98)
                denom = high - low
                if denom == 0:
                    denom = 1e-5

                img_norm = (x[i] - low) / denom
                vis_batch.append(torch.clamp(img_norm, 0, 1))

            return torch.stack(vis_batch)

        # 5. Prepare Grids
        # Upsample LR to match HR size for simple concatenation
        img_lr_up = F.interpolate(
            img_lr,
            size=img_hr.shape[-2:],
            mode='nearest',  # Use nearest to clearly see the low-res inputs
        )

        vis_lr = to_vis(img_lr_up)
        vis_pred = to_vis(reconstruction)
        vis_hr = to_vis(img_hr)

        # 6. Grid Construction: LR | Prediction | HR
        rows = []
        for i in range(N):
            # Concatenate along width: (C, H, W_lr + W_pred + W_hr)
            row = torch.cat((vis_lr[i], vis_pred[i], vis_hr[i]), dim=2)
            rows.append(row)

        # Concatenate along height for the full batch grid
        grid = torch.cat(rows, dim=1)

        # 7. Plot & Save
        filename = f'global_step_{global_step:04}.png'
        path = os.path.join(root, filename)

        # (C, H, W) -> (H, W, C) for matplotlib
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, N * 3))
        plt.imshow(grid_np)
        plt.axis('off')

        plt.title(
            f'Step: {global_step} | Model: DiffusionSuperRes\nLeft: LR Input | Mid: Prediction | Right: HR Target'
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
