from __future__ import annotations
# Based on https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
# MIT License

# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

# - Adopted to support newer torch/lightning versions
# - added docstrings and type annotations
# - adopted for anysensor inputs
# - manual optimization according to newer lightning version


import os

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchvision.datasets.utils import download_url

from .modules.distributions import DiagonalGaussianDistribution


class AutoencoderKL(LightningModule):
    # that checkpoint works together with this config
    # https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml
    hf_url = 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt'

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        embed_dim: int,
        ckpt_path: str | None = None,
        ignore_keys: list[str] = [],
        image_key: str = 'image',
        learning_rate: float = 1e-5,
        rgb_channel_indices: list[int] = [0, 1, 2],
        colorize_nlabels: int | None = None,
        monitor: str | None = None,
    ) -> None:
        """Initialize the KL-regularized Autoencoder.

        Args:
            encoder: Encoder module to use for encoding input into latent space
            decoder: Decoder module to use for decoding latent representation
            loss_fn: Loss function to use for training
            embed_dim: Dimension of the latent embedding space
            ckpt_path: Path to checkpoint file for loading pretrained weights
            ignore_keys: List of keys to ignore when loading checkpoint
            image_key: Key used to access images in the input batch
            learning_rate: Learning rate for training
            rgb_channel_indices: Indices of RGB channels in input images, used for logging
                samples during training
            colorize_nlabels: Number of labels for colorization feature
            monitor: Metric to monitor during training
        """
        super().__init__()
        self.image_key = image_key
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fn

        self.z_channels = self.encoder.z_channels
        self.quant_conv = torch.nn.Conv2d(2 * self.z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.z_channels, 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.rgb_channel_indices = rgb_channel_indices
        self.learning_rate = learning_rate

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys=list()) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file containing model weights
            ignore_keys: List of keys to ignore when loading the state dict
        """
        if path is None:
            path = 'vae-ft-mse-840000-ema-pruned.ckpt'
            # get the Huggingface checkpoint
            if not os.path.exists(path):
                download_url(self.hf_url, '.', path)

        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f'Deleting key {k} from state_dict.')
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f'Restored from {path}')

    def encode(
        self, x: torch.Tensor, wvs: torch.Tensor
    ) -> DiagonalGaussianDistribution:
        """Encode input tensor to latent representation.

        Args:
            x: Input tensor to encode [B, C, H, W]
            wvs: wavelengths values fo channels

        Returns:
            Posterior distribution in latent space
        """
        h = self.encoder(x, wvs)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor, wvs: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image space.

        Args:
            z: Latent representation to decode [B, D, H, W]

        Returns:
            Decoded tensor in image space [B, C, H, W]
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z, wvs)
        return dec

    def forward(
        self, input: torch.Tensor, wvs: torch.Tensor, sample_posterior: bool = True
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Forward pass through the autoencoder.

        Args:
            input: Input tensor [B, C, H, W]
            sample_posterior: Whether to sample from posterior or take mode

        Returns:
            Tuple of (decoded tensor, posterior distribution)
        """
        posterior = self.encode(input, wvs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, wvs)
        return dec, posterior

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for autoencoder.

        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch

        Returns:
            Loss value for current step
        """
        # Retrieve optimizers
        opt_ae, opt_disc = self.optimizers()

        inputs = batch[self.image_key]
        wvs = batch['wvs'][0, :]
        reconstructions, posterior = self(inputs, wvs)

        # Compute losses for autoencoder and discriminator
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            wvs=wvs,
            last_layer=self.get_last_layer(),
            split='train',
        )
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            wvs=wvs,
            last_layer=self.get_last_layer(),
            split='train',
        )

        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#gradient-accumulation
        # Optimize autoencoder
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        # clip gradients
        self.clip_gradients(
            opt_ae, gradient_clip_val=1.0, gradient_clip_algorithm='norm'
        )
        opt_ae.step()

        # Optimize discriminator
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        # clip gradients
        self.clip_gradients(
            opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm='norm'
        )
        opt_disc.step()

        self.log(
            'aeloss', aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        self.log(
            'discloss',
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )


    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Validation step for autoencoder.

        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch

        Returns:
            Dictionary of logged metrics
        """
        inputs = batch[self.image_key]
        wvs = batch['wvs'][0, :]
        reconstructions, posterior = self(inputs, wvs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            wvs=wvs,
            last_layer=self.get_last_layer(),
            split='val',
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            wvs=wvs,
            last_layer=self.get_last_layer(),
            split='val',
        )

        self.log('val/rec_loss', log_dict_ae['val/rec_loss'])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """Configure optimizers for autoencoder and discriminator.

        Returns:
            Tuple of (list of optimizers, list of schedulers)
        """
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        # Both optimizers use a fixed learning rate; no schedulers are configured.
        return [opt_ae, opt_disc], []

    def get_last_layer(self) -> torch.Tensor:
        """Get weights of the last decoder layer.

        Returns:
            Weight tensor of final convolution layer
        """
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(
        self, batch: dict[str, torch.Tensor], only_inputs: bool = False, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Generate images for logging during training.

        Args:
            batch: Input batch dictionary
            only_inputs: If True, only log input images
            **kwargs: Additional arguments

        Returns:
            Dictionary containing input, reconstructed and sampled images
        """
        # This visualization uses random RGB projection for multispectral inputs;
        # it does not define a sensor-specific display mapping.
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log['samples'] = self.decode(torch.randn_like(posterior.sample()))
            log['reconstructions'] = xrec
        log['inputs'] = x
        return log

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert segmentation masks to RGB images.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            RGB tensor [B, 3, H, W] with values in [-1, 1]
        """
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer('colorize', torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
