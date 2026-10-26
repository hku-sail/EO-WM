# EO-WM configuration reference

Follow the [main README](../README.md) for installation and commands. Paths below are relative to `EO-WM/code`.

## Main configuration

The main configuration is `configs/diffusion/train/eo_earthnet_stage1_cond_in10out20_neweovae.py`.

| Setting | Value |
| --- | --- |
| Input / forecast | 10 / 20 Sentinel-2 RGBN frames, 128 × 128 |
| Transformer | Hidden size 768; 12 heads; 12 double-stream + 14 single-stream blocks |
| Conditioning | Observed imagery, DEM, tile location, and meteorology |
| Weather features | Climatology/anomaly decomposition and cumulative heat/water stress |
| Initialization | Random DiT; frozen finetuned EO-VAE and text encoders |
| Training | 4 GPUs × batch 64; seed 42; 18,000 optimizer updates |
| Optimizer | AdamW; lr 2e-4; weight decay 0.01; 500 warmup steps |
| Schedule | Cosine, with `epochs=1000`; `MAX_STEPS` sets the stopping budget |
| Precision | bf16, ColossalAI ZeRO-2 |
| Sampling | 30 steps; guidance 1.0; 128 × 128 |
| Sampling members | 5 for benchmark evaluation |

## Local paths

| Variable | Default |
| --- | --- |
| `AE_PATH` | `weights/eo-vae` |
| `T5_PATH` | `google/t5-v1_1-xxl` |
| `CLIP_PATH` | `openai/clip-vit-large-patch14` |
| `CLIM_PATH` | `data/climatology/era5_climatology_train.pt` |

For local text encoders, use directories such as `weights/google/t5-v1_1-xxl` and `weights/openai/clip-vit-large-patch14`. After downloading all model assets, optional offline execution uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The launchers also accept `CONFIG`, `DATA_CSV` (training), `BENCH` (evaluation), `GPU_IDS`, `NPROC`, and output-directory overrides.

## Checkpoints and outputs

Evaluation reads `master.safetensors`. Training may save EMA weights separately. The main configuration omits optimizer state, so its default checkpoints are intended for inference rather than an exact optimizer-state resume.

Evaluation writes `metrics.json`, per-sample or per-pair CSVs, and optional visualizations. The launcher uses the reference Earthformer metric with raw EarthNet2021 masks. See [Earthformer reference evaluation](earthformer_reference.md) for the baseline examples.
