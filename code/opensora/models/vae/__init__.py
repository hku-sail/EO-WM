from .autoencoder_2d import AutoEncoderFlux
from .discriminator import N_LAYER_DISCRIMINATOR_3D


# Register the Earth observation VAE adapter.
try:
    from .eo_vae_wrapper import EOVAEWrapper
    from opensora.registry import MODELS
    MODELS.register_module("eo_vae", module=EOVAEWrapper)
    print("[opensora] EOVAEWrapper registered as 'eo_vae'")
except Exception as e:
    import warnings
    warnings.warn(f"[opensora] Failed to register eo_vae: {e}")