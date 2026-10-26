from __future__ import annotations
import math
import os
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from safetensors import safe_open
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .modules.distributions import DiagonalGaussianDistribution


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float,
    final_lr: float,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        base_lr: Initial learning rate after warmup
        final_lr: Final learning rate after decay
        num_cycles: Number of cosine cycles (0.5 for half cycle)

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale between base_lr and final_lr
        lr_scale = (base_lr - final_lr) * cosine_decay + final_lr
        return lr_scale / base_lr  # Return ratio for scheduler

    return LambdaLR(optimizer, lr_lambda)


class FluxAutoencoderKL(LightningModule):
    valid_modes = ['distill', 'finetune', 'flow-refine']

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        denoiser: torch.nn.Module | None = None,
        # schedule: torch.nn.Module | None = None,
        sampler: torch.nn.Module | None = None,
        ckpt_path: str | None = None,
        training_mode: str = 'finetune',  # 'distill' or 'finetune', or 'flow-refine'
        ignore_keys: list[str] = [],
        image_key: str = 'image',
        freeze_body: bool = True,
        base_lr: float = 1e-4,
        final_lr_sched: float | None = None,
        warmup_epochs: int | None = None,
        decay_end_epoch: int | None = None,
        clip_grad: float | None = None,
        # --- EQ-VAE Hyperparameters ---
        p_prior: float = 0.0,  # Probability of latent regularization
        p_prior_s: float = 0.0,  # Probability of image-level prior preservation
        anisotropic: bool = False,  # Allow different x/y scaling
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.training_mode = training_mode
        self.image_key = image_key
        self.warmup_epochs = warmup_epochs
        self.decay_end_epoch = decay_end_epoch
        self.base_lr = base_lr
        self.final_lr_sched = final_lr_sched
        self.clip_grad = clip_grad

        # eq-vae specific
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        self.anisotropic = anisotropic

        self.automatic_optimization = False

        assert training_mode in self.valid_modes, (
            f'Invalid training mode: {training_mode}, must be one of {self.valid_modes}'
        )

        # for flow refinement denoiser and scheduler are required
        if self.training_mode == 'flow-refine':
            assert denoiser is not None, (
                'Denoiser model must be provided for flow-refine mode.'
            )
            # assert schedule is not None, "Schedule model must be provided for flow-refine mode."
            assert sampler is not None, 'Sampler must be provided for flow-refine mode.'
            self.refiner = denoiser
            # self.schedule = schedule
            self.sampler = sampler

        # --- FLUX Specific Latent Stats ---
        self.ps = [2, 2]
        self.bn_eps = 1e-4
        # Calculate expected latent dim based on patch size
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * encoder.z_channels,
            affine=False,
            track_running_stats=True,
        )

        self.freeze_body = freeze_body
        if self.freeze_body:
            self._freeze_main_body()

        # --- Initialization ---
        self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # =========================================================
    #  INITIALIZATION & LOADING
    # =========================================================
    def _freeze_main_body(self):
        """Locks the pre-trained VAE body (ResNets, Attentions, QuantConv).
        Only keeps the Dynamic Input/Output layers trainable.
        """
        print('--- FREEZING VAE BODY (Phase 2 Mode) ---')

        # 1. Freeze EVERYTHING first
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        # 2. Unfreeze ONLY Dynamic Layers
        if self.encoder.use_dynamic_ops:
            for p in self.encoder.conv_in.parameters():
                p.requires_grad = True
            print(' -> Encoder Dynamic Input: UNLOCKED')

        if self.decoder.use_dynamic_ops:
            for p in self.decoder.conv_out.parameters():
                p.requires_grad = True
            print(' -> Decoder Dynamic Output: UNLOCKED')

    def init_from_ckpt(self, path, ignore_keys=list()) -> None:
        if not path or not os.path.exists(path):
            return

        print(f'Loading {self.training_mode} weights from {path}...')

        # 1. Load State Dict (Handle .safetensors and .ckpt/.pt)
        sd = {}
        if path.endswith('safetensors'):
            with safe_open(path, framework='pt', device='cpu') as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        else:
            sd = torch.load(path, map_location='cpu')
            sd = sd.get('state_dict', sd)  # Unwrap if nested

        # 2. Distillation Setup: Intercept Teacher Weights
        if self.training_mode == 'distill':
            self._register_teacher_weights(sd)

        # 3. Filter Keys (Remove static layers if dynamic, remove ignore_keys)
        keys = list(sd.keys())
        for k in keys:
            # Skip static input/output weights ONLY if they look like static weights
            # (i.e. they are NOT part of the new dynamic sub-modules)
            if self.encoder.use_dynamic_ops and 'encoder.conv_in' in k:
                if 'weight_generator' not in k and 'fclayer' not in k:
                    del sd[k]
                    continue

            if self.decoder.use_dynamic_ops and 'decoder.conv_out' in k:
                if 'weight_generator' not in k and 'fclayer' not in k:
                    del sd[k]
                    continue

            # User-specified ignore keys
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
                    break

        # 4. Load
        missing, unexpected = self.load_state_dict(sd, strict=False)

        # 5. RUN VERIFICATION
        self._verify_loading(missing, unexpected, ignore_keys)

    def _verify_loading(self, missing_keys, unexpected_keys, user_ignore_keys):
        """Ensures that 'missing_keys' ONLY contains the dynamic layers we expected to miss,
        and that the VAE Body (ResNets, Attentions) was loaded correctly.
        """
        critical_errors = []

        # Define what we EXPECT to be missing based on config
        allowed_missing_prefixes = []

        if self.encoder.use_dynamic_ops:
            allowed_missing_prefixes.append('encoder.conv_in')

        if self.decoder.use_dynamic_ops:
            allowed_missing_prefixes.append('decoder.conv_out')

        # Teacher buffers may be absent in pretrained checkpoints.
        # Since we manually registered/filled them before loading,
        # they are not required in the loaded state_dict.
        allowed_missing_prefixes.append('teacher_')

        # Add user defined ignores
        allowed_missing_prefixes.extend(user_ignore_keys)

        # Check every missing key
        for k in missing_keys:
            # logic: is this key allowed to be missing?
            is_allowed = False
            for p in allowed_missing_prefixes:
                if k.startswith(p):
                    is_allowed = True
                    break

            # If it's NOT in our allowed list, it's a critical error.
            if not is_allowed:
                critical_errors.append(k)

        if len(critical_errors) > 0:
            raise RuntimeError(
                f'FATAL: The checkpoint failed to load critical weights!\n'
                f'The following keys are missing but were expected:\n{critical_errors[:20]}...\n'
                f'(Total {len(critical_errors)} missing keys). Check your checkpoint path or architecture.'
            )

        print('Weights loaded successfully.')
        print(
            f' - Missing keys (Expected): {len(missing_keys)} (Dynamic Layers + Teachers)'
        )
        print(f' - Unexpected keys (Ignored): {len(unexpected_keys)}')

    def _register_teacher_weights(self, sd):
        """Extracts RGB weights AND BIASES from checkpoint for distillation."""

        # Helper to safely get tensor or None
        def get_tensor(key):
            return sd.get(key, None)

        enc_w, enc_b = (
            get_tensor('encoder.conv_in.weight'),
            get_tensor('encoder.conv_in.bias'),
        )
        dec_w, dec_b = (
            get_tensor('decoder.conv_out.weight'),
            get_tensor('decoder.conv_out.bias'),
        )

        if enc_w is None or dec_w is None:
            raise ValueError(
                'Distillation requires a checkpoint with standard RGB weights!'
            )

        self.register_buffer('teacher_enc_w', enc_w)
        self.register_buffer('teacher_dec_w', dec_w)

        if enc_b is not None:
            self.register_buffer('teacher_enc_b', enc_b)
        if dec_b is not None:
            self.register_buffer('teacher_dec_b', dec_b)

        print('Teacher weights & biases registered.')

    def configure_optimizers(self):
        # MODE A: DISTILLATION
        if self.training_mode == 'distill':
            params = list(self.encoder.conv_in.parameters()) + list(
                self.decoder.conv_out.parameters()
            )
            opt = torch.optim.Adam(params, lr=self.base_lr)
            return opt

        elif self.training_mode == 'flow-refine':
            # Only optimize the refiner
            opt = torch.optim.AdamW(self.refiner.parameters(), lr=1e-4)
            return opt

        # MODE B: FINETUNING (Phase 2 & 3)
        else:
            # Smart Filter: Only pass parameters that require gradients
            # This handles both freeze_body=True and freeze_body=False automatically

            ae_params = [p for p in self.encoder.parameters() if p.requires_grad] + [
                p for p in self.decoder.parameters() if p.requires_grad
            ]

            opt_ae = torch.optim.Adam(ae_params, lr=self.base_lr)

            optimizers = [opt_ae]
            scheduler_list = []

            # Check if loss_fn has discriminator
            if hasattr(self.loss_fn, 'discriminator'):
                disc_params = self.loss_fn.discriminator.parameters()
                opt_disc = torch.optim.Adam(disc_params, lr=self.base_lr)
                optimizers.append(opt_disc)

            # Build schedulers
            if (
                self.final_lr_sched is not None
                and self.warmup_epochs is not None
                and self.decay_end_epoch is not None
            ):
                steps_per_epoch = 2000  # Estimate or compute from dataloader

                num_warmup_steps = self.warmup_epochs * steps_per_epoch
                num_training_steps = self.decay_end_epoch * steps_per_epoch

                sch_ae = get_cosine_schedule_with_warmup(
                    opt_ae,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    base_lr=self.base_lr,
                    final_lr=self.final_lr_sched,
                )
                scheduler_list.append({'scheduler': sch_ae, 'interval': 'step'})

                if len(optimizers) > 1:  # Has disc
                    sch_disc = get_cosine_schedule_with_warmup(
                        opt_disc,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        base_lr=self.base_lr,
                        final_lr=self.final_lr_sched,
                    )
                    scheduler_list.append({'scheduler': sch_disc, 'interval': 'step'})

            if len(scheduler_list) > 0:
                return optimizers, scheduler_list
            else:
                return optimizers

    # =========================================================
    #  FORWARD PASS (Encode -> Shuffle -> Decode)
    # =========================================================

    def encode(
        self, x: torch.Tensor, wvs: torch.Tensor
    ) -> DiagonalGaussianDistribution:
        moments = self.encoder(x, wvs)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor, wvs: torch.Tensor) -> torch.Tensor:
        # Flux Process: Inverse Normalize -> Unshuffle -> Decode
        z = self.inv_normalize_latent(z)
        z = rearrange(
            z, '... (c pi pj) i j -> ... c (i pi) (j pj)', pi=self.ps[0], pj=self.ps[1]
        )
        return self.decoder(z, wvs)

    def forward(
        self,
        input: torch.Tensor,
        wvs: torch.Tensor,
        sample_posterior: bool = True,
        scale: float | tuple[float, float] | None = None,
        angle: int | None = None,
        use_refiner=False,
    ):
        posterior = self.encode(input, wvs)
        z = posterior.sample() if sample_posterior else posterior.mode()

        # EQ-VAE: Apply transformations
        if scale is not None:
            h, w = z.shape[-2:]
            # Calculate new dims that are multiples of the patch size (self.ps)
            if isinstance(scale, (tuple, list)):
                new_h = round(h * scale[0] / self.ps[0]) * self.ps[0]
                new_w = round(w * scale[1] / self.ps[1]) * self.ps[1]
            else:
                new_h = round(h * scale / self.ps[0]) * self.ps[0]
                new_w = round(w * scale / self.ps[1]) * self.ps[1]

            # Use explicit 'size' instead of 'scale_factor' to guarantee divisibility
            z = F.interpolate(
                z, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

        if angle is not None:
            z = torch.rot90(z, k=angle, dims=[-1, -2])

        # Now rearrange will always work because new_h and new_w are multiples of 2
        z_shuffled = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        z_normalized = self.normalize_latent(z_shuffled)

        dec = self.decode(z_normalized, wvs)

        if use_refiner:
            # Pass through flow refiner
            dec = self.refine(dec, wvs=wvs)

        return dec, posterior

    def normalize_latent(self, z):
        if self.training:
            self.bn.train()
        else:
            self.bn.eval()
        return self.bn(z)

    def inv_normalize_latent(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to normalized latent space."""
        posterior = self.encode(x, wvs)
        z = posterior.mode()
        z_shuffled = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        return self.normalize_latent(z_shuffled)

    @torch.no_grad()
    def encode_spatial_normalized(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to spatially-structured normalized latent.

        Process:
        1. Encode -> z
        2. Shuffle -> z_shuffled
        3. BN using VAE stats -> z_norm
        4. Unshuffle -> z_spatial

        Returns: [B, C, H, W] where C=32, preserving spatial layout but with VAE normalization applied.
        """
        # Get normalized packed latent [B, 128, H/16, W/16]
        z_norm = self.encode_to_latent(x, wvs)

        # Unshuffle back to spatial [B, 32, H/8, W/8]
        z_spatial = rearrange(
            z_norm,
            '... (c pi pj) i j -> ... c (i pi) (j pj)',
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return z_spatial

    @torch.no_grad()
    def decode_spatial_normalized(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode from spatially-structured normalized latent.

        Process:
        1. Shuffle -> z_packed
        2. Inverse BN (handled by decode)
        3. Unshuffle & Decode (handled by decode)
        """
        # Shuffle to packed format [B, 128, H/16, W/16]
        z_packed = rearrange(
            z, '... c (i pi) (j pj) -> ... (c pi pj) i j', pi=self.ps[0], pj=self.ps[1]
        )
        # Decode expects packed normalized latent
        return self.decode(z_packed, wvs)

    # =========================================================
    #  TRAINING & VALIDATION ROUTING
    # =========================================================

    def training_step(self, batch, batch_idx):
        if self.training_mode == 'distill':
            return self._training_step_distill(batch)
        elif self.training_mode == 'flow-refine':
            return self._training_step_flow_refinement(batch)
        return self._training_step_finetune(batch)

    def validation_step(self, batch, batch_idx):
        if self.training_mode == 'distill':
            return self._validation_step_distill(batch)
        elif self.training_mode == 'flow-refine':
            return self._validation_step_flow_refinement(batch)
        return self._validation_step_finetune(batch)

    # =========================================================
    #  MODE A: DISTILLATION
    # =========================================================

    def _training_step_distill(self, batch):
        opt = self.optimizers()
        opt.zero_grad()
        loss, logs = self._compute_distill_loss()
        self.manual_backward(loss)
        opt.step()
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def _validation_step_distill(self, batch):
        loss, logs = self._compute_distill_loss(split='val')
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def _compute_distill_loss(self, split='train'):
        # RGB Wavelengths in Microns (match SD weights Ch0=R, Ch1=G, Ch2=B)
        rgb_wvs = torch.tensor([0.665, 0.560, 0.490], device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        loss_calculated = False

        # --- Encoder Distillation ---
        if self.encoder.use_dynamic_ops:
            # Use helper from DynamicConv to get exact weight/bias layout
            s_enc_w, s_enc_b = self.encoder.conv_in.get_distillation_weight(rgb_wvs)

            total_loss += F.mse_loss(s_enc_w, self.teacher_enc_w)
            if s_enc_b is not None and hasattr(self, 'teacher_enc_b'):
                total_loss += F.mse_loss(s_enc_b, self.teacher_enc_b)
            loss_calculated = True

        # --- Decoder Distillation ---
        if self.decoder.use_dynamic_ops:
            s_dec_w, s_dec_b = self.decoder.conv_out.get_distillation_weight(rgb_wvs)

            total_loss += F.mse_loss(s_dec_w, self.teacher_dec_w)
            if s_dec_b is not None and hasattr(self, 'teacher_dec_b'):
                total_loss += F.mse_loss(s_dec_b, self.teacher_dec_b)
            loss_calculated = True

        # Safety: Ensure loss has grad even if dynamic ops disabled (prevents crash)
        if not loss_calculated and not total_loss.requires_grad:
            total_loss.requires_grad_(True)

        return total_loss, {f'{split}/distill_loss': total_loss}

    # =========================================================
    #  MODE B: FINETUNING
    # =========================================================
    def _training_step_finetune(self, batch):
        opts = self.optimizers()
        if isinstance(opts, list):
            opt_gen = opts[0]
            opt_disc = opts[1] if len(opts) > 1 else None
        else:
            opt_gen = opts
            opt_disc = None

        schs = self.lr_schedulers()
        if isinstance(schs, list):
            sch_gen = schs[0] if schs else None
            sch_disc = schs[1] if len(schs) > 1 else None
        else:
            sch_gen = schs
            sch_disc = None

        images = self.get_input(batch, self.image_key)  # Original full-res [B, C, H, W]
        wvs = batch['wvs']

        # EQ-VAE Hyperparams / State
        # Discrete bins [8/32, 16/32, 24/32] -> [0.25, 0.5, 0.75]
        # This prevents the 'new shape every batch' slowdown
        scale_bins = [0.375, 0.5, 0.75]

        # =========================================================
        # 1. EQ-VAE REGULARIZATION & FORWARD PASS
        # =========================================================

        if random.random() < self.p_prior:
            angle = random.choice([1, 2, 3])
            if self.anisotropic:
                scale = (random.choice(scale_bins), random.choice(scale_bins))
            else:
                scale = random.choice(scale_bins)

            recon, posterior = self.forward(images, wvs, scale=scale, angle=angle)

            # Match ground truth to the transformed latent output
            with torch.no_grad():
                target_images = F.interpolate(
                    images, size=recon.shape[-2:], mode='area'
                )
                target_images = torch.rot90(target_images, k=angle, dims=[-1, -2])

        elif random.random() < self.p_prior_s:
            scale = random.choice(scale_bins)

            recon, posterior = self.forward(images, wvs, scale=scale)

            with torch.no_grad():
                target_images = F.interpolate(
                    images, size=recon.shape[-2:], mode='area'
                )

        else:
            # Standard full-resolution reconstruction
            recon, posterior = self.forward(images, wvs)
            target_images = images

        # =========================================================
        # 2. GENERATOR TRAINING
        # =========================================================
        opt_gen.zero_grad()
        if opt_disc is not None and hasattr(self.loss_fn, 'discriminator'):
            self.loss_fn.discriminator.eval()

        # Pass the correctly sized target_images (matching recon)
        gen_loss, log_dict_gen = self.loss_fn(
            inputs=target_images,
            wvs=wvs,
            reconstructions=recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split='train',
        )

        self.manual_backward(gen_loss)

        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt_gen.param_groups[0]['params'], self.clip_grad
            )

        opt_gen.step()
        if sch_gen is not None:
            sch_gen.step()

        # =========================================================
        # 3. DISCRIMINATOR TRAINING
        # =========================================================
        train_disc = (
            opt_disc is not None
            and self.global_step >= self.loss_fn.disc_start
            and self.loss_fn.disc_weight > 0.0
        )

        if train_disc:
            if hasattr(self.loss_fn, 'discriminator'):
                self.loss_fn.discriminator.train()

            opt_disc.zero_grad()
            # Use target_images so the discriminator learns on the same transformations
            recon_detached = recon.detach()

            disc_loss, log_dict_disc = self.loss_fn(
                inputs=target_images,
                wvs=wvs,
                reconstructions=recon_detached,
                optimizer_idx=1,
                global_step=self.global_step,
                last_layer=None,
                split='train',
            )

            self.manual_backward(disc_loss)
            opt_disc.step()

            if sch_disc is not None:
                sch_disc.step()

        # =========================================================
        # 4. LOGGING
        # =========================================================
        log_dict = {**log_dict_gen}
        log_dict['train/lr_gen'] = opt_gen.param_groups[0]['lr']

        if train_disc:
            log_dict['train/lr_disc'] = opt_disc.param_groups[0]['lr']
            log_dict.update(log_dict_disc)

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=images.shape[0],
        )

        return gen_loss

    def _validation_step_finetune(self, batch):
        images = self.get_input(batch, self.image_key)
        wvs = batch['wvs']
        recon, _ = self.forward(images, wvs)

        # Compute loss (generator branch only)
        val_loss, log_dict = self.loss_fn(
            inputs=images,
            wvs=wvs,
            reconstructions=recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=None,
            split='val',
        )

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )

        return val_loss

    # =========================================================
    #  STAGE 3
    # =========================================================

    def _training_step_flow_refinement(self, batch):
        opt = self.optimizers()
        opt.zero_grad()

        x_target = self.get_input(batch, self.image_key)  # Clean 'x'
        wvs = batch['wvs']

        # Get Source 'z' (Reconstruction) from frozen VAE
        with torch.no_grad():
            x_recon, _ = self.forward(x_target, wvs)
            x_recon = x_recon.detach()

        # Sample time t uniformly [0, 1]
        t = torch.rand(x_target.shape[0], device=self.device)

        # Compute JiT loss using Azula components
        loss = self.refiner.loss(
            x=x_target,
            z=x_recon,
            t=t,
            wvs=wvs,  # Passed as kwargs to backbone
        )

        self.manual_backward(loss)
        opt.step()

        self.log('train/loss_rec', loss, prog_bar=True)
        return loss

    def _validation_step_flow_refinement(self, batch):
        images = self.get_input(batch, self.image_key)  # Ground Truth
        wvs = batch['wvs']

        # 1. Base VAE Reconstruction (Frozen Path)
        with torch.no_grad():
            x_recon, _ = self.forward(images, wvs)
            x_recon = x_recon.detach()

        # 2. Refined Reconstruction (Sampler Path)
        # This executes the actual flow matching inference
        x_refined = self.refine(x_recon, wvs, steps=20)

        # 3. Compute Metrics
        base_mse = F.mse_loss(x_recon, images)
        refined_mse = F.mse_loss(x_refined, images)

        self.log_dict(
            {
                'val/loss_rec': refined_mse,
                'val/refinement_gain': base_mse - refined_mse,
            },
            prog_bar=True,
            on_epoch=True,
        )

        return refined_mse

    @torch.no_grad()
    def refine(self, x_recon: Tensor, wvs: Tensor, steps: int = 25) -> Tensor:
        """Runs the EulerSampler from t=0 (recon) to t=1 (target)."""
        sampler = self.sampler(denoiser=self.refiner, steps=steps)
        return sampler(x=x_recon, wvs=wvs)

    # =========================================================
    #  UTILITIES & LOGGING
    # =========================================================

    def get_last_layer(self) -> torch.Tensor:
        # Handles generic output vs dynamic wrapper output
        return self.decoder.conv_out.weight

    def get_input(self, batch, k):
        return batch[k]
