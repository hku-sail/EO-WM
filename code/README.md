# EO-WM implementation

Training and inference code adapted from Open-Sora 2.0, with a bundled EO-VAE package.

Follow the repository [README](../README.md) for installation, weights, data preparation, and commands. All EO-WM commands in that guide run from this directory.

| Entry | Purpose |
| --- | --- |
| `install_env.sh` | Linux/CUDA environment |
| `train_earthnet.sh` | Main-model training |
| `eval_earthnet.sh` | Forecasting and evaluation on the paper benchmarks |
| `scripts/data_tools/prepare_release_data.py` | Generate local manifests from relative dataset paths |
| `configs/diffusion/train/eo_earthnet_stage1_cond_in10out20_neweovae.py` | Main configuration |
| `configs/vae/earthnet/model_config.yaml` | Released finetuned EO-VAE architecture |

See [configuration reference](../docs/configuration.md) and [data documentation](data/README.md). Upstream notices are retained in [LICENSE](LICENSE) and [NOTICE](../NOTICE).
