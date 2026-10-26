from __future__ import annotations
# MIT License

# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

# Based on https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py

# - Adopted to support newer torch version
# - added docstrings and type annotations
# - adopted for anysensor inputs

import numpy as np
import torch


# Related diagonal Gaussian implementations:
# https://github.com/huggingface/diffusers/blob/aad69ac2f323734a083d66fa89197bf7d88e5a57/src/diffusers/models/autoencoders/vae.py#L691
class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        """Initialize diagonal Gaussian distribution.

        Args:
            parameters: Tensor containing mean and logvar
            deterministic: If True, sets variance to 0
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution.

        Returns:
            Sampled tensor
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(
        self, other: type['DiagonalGaussianDistribution'] | None = None
    ) -> torch.Tensor:
        """Compute KL divergence to another distribution.

        Args:
            other: Distribution to compute KL divergence to

        Returns:
            KL divergence value
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: list[int] = [1, 2, 3]) -> torch.Tensor:
        """Compute negative log likelihood of sample.

        Args:
            sample: Sample to compute likelihood for
            dims: Dimensions to sum over

        Returns:
            Negative log likelihood
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """Get distribution mode.

        Returns:
            Mode (mean) of distribution
        """
        return self.mean


"""
source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
Compute the KL divergence between two gaussians.
Shapes are automatically broadcasted, so batches can be compared to
scalars, among other use cases.
"""


def normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between two normal distributions.

    Args:
        mean1: Mean of first distribution
        logvar1: Log variance of first distribution
        mean2: Mean of second distribution
        logvar2: Log variance of second distribution

    Returns:
        KL divergence value
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, 'at least one argument must be a Tensor'

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
