# configs/diffusion/train/earthnet_image.py
"""
EarthNet2021 基础训练配置（Open-Sora + EO-VAE）。

默认约定：
1. CSV 由 tool_trans_earthnet_to_csv.py 生成（默认 in=10, out=20）
2. 数据集类型为 EarthNetVideoDataset
3. 4 通道光学输入映射到 EO-VAE: B02/B03/B04/B08
"""

_base_ = []

import os

# ── Dataset ──────────────────────────────────────────────────
dataset = dict(
    type="EarthNetVideoDataset",
    # 建议通过命令行覆盖:
    # --dataset.data_path /path/to/earthnet2021_train_meta.csv
    data_path="data/metadata/earthnet2021_train_in10out20_refine-caption_multi-p_corrected-temp50.csv",
    # tile_id → (lon,lat) 位置索引; 随仓库分发, 相对 repo 根目录
    tile_location_json="data/tiles/earthnet2021_tile_location_index.json",
    load_meso=True,
    load_dem=True,
    normalize_optical=True,
    normalize_meso=True,
    normalize_dem=True,
    cloud_prob_thresh=0.3,  # 与 CSV 生成脚本默认 cloud_thresh 对齐
    era5_out_channels=8,    # 与现有 EO 条件注入分支对齐
    resize_meso_to_highres=True,  # 将 80*80 resize 到 128*128, 否则 img+dem+era5 维度不匹配
    meso_ratio=5,
    min_frames=1,
    fps_max=1.0,
)

grad_ckpt_settings = (8, 100)

# ── Bucket（默认 10+20=30 帧） ───────────────────────────────
bucket_config = {
    "128px": {
        30: (1.0, 8),
    },
}

# ── AE ───────────────────────────────────────────────────────
ae = dict(
    type="eo_vae",
    pretrained_path=os.environ.get("AE_PATH", "weights/eo-vae"),
    repo_id="nilsleh/eo-vae",
    ckpt_filename="eo-vae.ckpt",
    config_filename="model_config.yaml",
    s2_band_names=["B02", "B03", "B04", "B08"],
    freeze_encoder=False,
)
is_causal_vae = False
is_eo_vae = True

# ── DiT model ────────────────────────────────────────────────
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=128,  # latent_channels(32) * 2 * 2
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=1152,
    mlp_ratio=4.0,
    num_heads=16,
    depth=14,
    depth_single_blocks=28,
    axes_dim=[16, 28, 28],  # sum=72 = hidden_size / num_heads
    theta=10_000,
    qkv_bias=True,
)

# ── Text encoders ────────────────────────────────────────────
# 文本编码器默认从 HuggingFace Hub 下载 (首次运行自动缓存);
# 已有本地权重时, 直接把 from_pretrained 指向本地目录即可。
t5 = dict(
    type="text_embedder",
    from_pretrained=os.environ.get("T5_PATH", "google/t5-v1_1-xxl"),
    cache_dir=None,
    max_length=128,
    shardformer=False,
)

clip = dict(
    type="text_embedder",
    from_pretrained=os.environ.get("CLIP_PATH", "openai/clip-vit-large-patch14"),
    cache_dir=None,
    max_length=77,
)


dropout_ratio = {
    "t5": 0.1,
    "clip": 0.1,
}

# ── Optimization ─────────────────────────────────────────────
lr = 1e-4
eps = 1e-15
optim = dict(
    cls="AdamW",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
)
warmup_steps = 500
update_warmup_steps = True
grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# ── Acceleration ─────────────────────────────────────────────
prefetch_factor = 2
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"
grad_checkpoint = True
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)
async_io = False

# ── Others ───────────────────────────────────────────────────
seed = 42
outputs = "outputs/earthnet_stage1"
epochs = 1000
log_every = 10
ckpt_every = 500
keep_n_latest = 10
wandb_project = "earthnet_dit"

save_master_weights = True
load_master_weights = True
