import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback


class ImageLogger(Callback):
    def __init__(self, rgb_indices=None, num_images: int = 4):
        """Args:
        rgb_indices: List of channel indices to use for RGB plotting.
            Defaults to [0, 1, 2] if not provided.
        num_images: Number of images from the batch to log.
        """
        self.rgb_indices = rgb_indices if rgb_indices is not None else [0, 1, 2]
        self.num_images = num_images

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_images(trainer, pl_module, mode='validation')

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_images(trainer, pl_module, mode='training')

    def log_images(self, trainer, pl_module, mode='validation', batch=None):
        """Performs a forward pass on a batch and logs input and reconstructions.
        If batch is None, the first batch from the respective dataloader is used.
        """
        # Get the batch if not provided.
        if batch is None:
            if mode == 'validation':
                loader = trainer.datamodule.val_dataloader()
            else:
                loader = trainer.datamodule.train_dataloader()
            batch = next(iter(loader))

        # Assume the batch contains key "image" with tensor shape (B, C, H, W)
        input_images = batch['image'].to(pl_module.device)
        wvs = batch['wvs'][0, :].to(pl_module.device)

        # Run forward pass on the model.
        with torch.no_grad():
            outputs = pl_module(input_images, wvs)
            if isinstance(outputs, (list, tuple)):
                reconstructions = outputs[0]
            else:
                reconstructions = outputs

        # Extract only the RGB channels (using provided indices)
        input_rgb = input_images[:, self.rgb_indices, :, :]
        recon_rgb = reconstructions[:, self.rgb_indices, :, :]

        fig = self.plot_images(input_rgb, recon_rgb, self.num_images)
        # Log figure to wandb through the experiment logger.
        log_key = f'{mode}/reconstructions'
        trainer.logger.experiment.log(
            {log_key: wandb.Image(fig)}, step=trainer.global_step
        )
        plt.close(fig)

    @staticmethod
    def plot_images(inputs: torch.Tensor, recons: torch.Tensor, n: int) -> plt.Figure:
        """Plots input and reconstruction images side-by-side in a grid.

        Args:
            inputs: Tensor of input images (B, C, H, W)
            recons: Tensor of reconstructed images (B, C, H, W)
            n: Number of images in the batch to plot

        Returns:
            A matplotlib Figure.
        """
        count = min(n, inputs.shape[0])
        # Create subplots: each image gets two columns (input & reconstruction)
        fig, axes = plt.subplots(count, 2, figsize=(8, 4 * count))
        if count == 1:
            axes = [axes]
        for i in range(count):
            inp_img = inputs[i].detach().cpu().numpy().transpose(1, 2, 0)
            rec_img = recons[i].detach().cpu().numpy().transpose(1, 2, 0)
            # Clip and normalize the image if necessary.
            inp_img = (inp_img - inp_img.min()) / (inp_img.max() - inp_img.min() + 1e-5)
            rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min() + 1e-5)
            axes[i][0].imshow(inp_img)
            axes[i][0].set_title('Input')
            axes[i][0].axis('off')
            axes[i][1].imshow(rec_img)
            axes[i][1].set_title('Reconstruction')
            axes[i][1].axis('off')
        fig.tight_layout()
        return fig
