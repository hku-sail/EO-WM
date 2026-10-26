# Bundled EO-VAE

EO-WM uses a modified copy of [EO-VAE](https://github.com/nilsleh/eo-vae), imported through `eo_vae.models.new_autoencoder.EOFluxVAE` and adapted by `opensora/models/vae/eo_vae_wrapper.py`.

Install from `EO-WM/code`:

```bash
python -m pip install -e ./eo-vae --no-deps
```

The [upstream weights](https://huggingface.co/nilsleh/eo-vae) provide the base model. EO-WM training and evaluation use an EarthNet2021-finetuned checkpoint; see the repository [model table](../../README.md#model-checkpoints) for release status and paths. Put `eo-vae.ckpt` and its matching `model_config.yaml` in the same directory and select it with `AE_PATH`.

Upstream attribution and the [Apache license](LICENSE) are retained. The torchvision dependency is aligned with the EO-WM environment; install shared dependencies using `../install_env.sh`.
