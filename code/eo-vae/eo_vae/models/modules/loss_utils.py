from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_conv import DynamicConv


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize tensor by its L2 norm."""
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor
) -> torch.Tensor:
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )
    return d_loss


def vanilla_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.softplus(-logits_fake))


class NetLinLayer(nn.Module):
    """A single linear layer for computing weighted feature distances."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []

        # 1. Create layer
        conv = nn.Conv1d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)

        # 2. Initialize weights to positive values.
        # Since we don't have perceptual labels, we treat all features as equally
        # important initially.
        nn.init.constant_(conv.weight, 1.0 / chn_in)

        layers += [conv]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DOFALPIPS(nn.Module):
    """DOFA-LPIPS: Learned Perceptual Image Patch Similarity using a frozen DOFA backbone.
    Calculates perceptual distance in the multispectral feature space.
    """

    def __init__(self, dofa_net: nn.Module, use_dropout: bool = True):
        super().__init__()
        self.net = dofa_net

        # Freeze DOFA backbone
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        # Auto-detect dimensions from the pretrained model
        self.embed_dim = getattr(dofa_net, 'embed_dim', 768)
        # Using 4 layers (standard for Swin/ViT hierarchical features) or depth
        self.num_layers = 4

        # Learnable 1x1 convs to weight the differences in feature channels
        self.lin_layers = nn.ModuleList(
            [
                NetLinLayer(self.embed_dim, use_dropout=use_dropout)
                for _ in range(self.num_layers)
            ]
        )


    def forward(
        self, input: torch.Tensor, target: torch.Tensor, wvs: torch.Tensor
    ) -> torch.Tensor:
        # with torch.no_grad():
        feats_in = self.net.forward_features(input, wvs)
        feats_tgt = self.net.forward_features(target, wvs)

        val = torch.tensor(0.0, device=input.device)

        for k, (f_in, f_tgt) in enumerate(zip(feats_in, feats_tgt)):
            if k >= len(self.lin_layers):
                break

            if f_in.shape[-1] == self.embed_dim:
                f_in = f_in.transpose(1, 2)
                f_tgt = f_tgt.transpose(1, 2)

            f_in = normalize_tensor(f_in)
            f_tgt = normalize_tensor(f_tgt)

            # Squared difference is ALWAYS positive
            diff = (f_in - f_tgt) ** 2

            # Weighted sum. If lin_layers weights are positive, this result is positive.
            val += self.lin_layers[k](diff).mean()

        return val


class DOFADiscriminator(nn.Module):
    """DOFA-Discriminator: Uses frozen DOFA features with lightweight trainable heads.
    Efficiently discriminates multispectral data without projecting to RGB.
    """

    def __init__(
        self, dofa_net: nn.Module, hidden_dim: int = 256, norm_type: str = 'bn'
    ):
        super().__init__()
        self.net = dofa_net

        # Freeze backbone
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        self.embed_dim = getattr(dofa_net, 'embed_dim', 768)

        # Discriminator Heads (one per feature scale/layer)
        # Using 4 layers to capture multi-scale artifacts
        self.num_layers = 4
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.embed_dim, hidden_dim, kernel_size=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(hidden_dim, 1, kernel_size=1),
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor | None, wvs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Extract features [B, N, D]
        fake_feats = self.net.forward_features(fake, wvs)
        real_feats = self.net.forward_features(real, wvs) if real is not None else None

        logits_fake = []
        logits_real = []

        for k, head in enumerate(self.heads):
            if k >= len(fake_feats):
                break

            # Process Fake
            f_feat = fake_feats[k].transpose(1, 2)  # [B, D, N]
            logits_fake.append(head(f_feat).view(fake.shape[0], -1))

            # Process Real
            if real is not None:
                r_feat = real_feats[k].transpose(1, 2)
                logits_real.append(head(r_feat).view(real.shape[0], -1))

        # Concatenate logits from all scales
        logits_fake = torch.cat(logits_fake, dim=1)
        logits_real = torch.cat(logits_real, dim=1) if real is not None else None

        return logits_fake, logits_real


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d
        use_bias = False

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main_net = nn.Sequential(*sequence)

        # add dynamic convolution input layer to process input
        self.conv_in = DynamicConv(
            wv_planes=128,
            inter_dim=128,
            kernel_size=3,
            stride=1,
            padding=1,
            embed_dim=input_nc,
        )

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor | None, wvs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass matching DinoDisc API.

        Args:
            fake: Fake/generated images [B, C, H, W]
            real: Real images [B, C, H, W] (can be None)
            wvs: Wavelengths [C] or [B, C]

        Returns:
            Tuple of (logits_fake, logits_real)
        """
        logits_fake = self.main_net(self.conv_in(fake, wvs))
        logits_real = (
            self.main_net(self.conv_in(real, wvs)) if real is not None else None
        )

        return logits_fake, logits_real
