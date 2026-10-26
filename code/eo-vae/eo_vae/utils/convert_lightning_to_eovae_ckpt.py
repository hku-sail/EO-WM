# convert_lightning_to_eovae_ckpt.py
"""
将 Lightning checkpoint 转换为 EOFluxVAE.from_pretrained() 可直接读取的格式。

使用方法：
    python /path/to/projects/eo-vae/eo_vae/utils/convert_lightning_to_eovae_ckpt.py \
        --lightning_ckpt /path/to/eo-vae_exp_results/mmdc_128_4GPU_just_s2_multiframe/eo-vae-mmdc-ft-128_03-24-2026_00-19-31/last.ckpt \
        --output_dir /path/to/eo-vae-finetuned_just_s2_multiframe_25k \
        --pretrained_dir /path/to/models--nilsleh--eo-vae

python /path/to/projects/eo-vae/eo_vae/utils/convert_lightning_to_eovae_ckpt.py \
    --lightning_ckpt /path/to/epoch=25-step=40066.ckpt \
    --output_dir /path/to/eo-vae-earthnet-finetuned \
    --pretrained_dir /path/to/models--nilsleh--eo-vae

--lightning_ckpt /path/to/eo-vae_exp_results/earthnet2021_128_finetune/epoch=9-step=15410.ckpt \
"""

import argparse
import shutil
from pathlib import Path

import torch


def convert(
    lightning_ckpt: str,
    output_dir: str,
    pretrained_dir: str,
    verify: bool = True,
):
    lightning_ckpt = Path(lightning_ckpt)
    output_dir     = Path(output_dir)
    pretrained_dir = Path(pretrained_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取 Lightning checkpoint，提取 state_dict ──────────
    print(f"Loading Lightning ckpt: {lightning_ckpt}")
    ckpt   = torch.load(lightning_ckpt, map_location="cpu")
    ft_sd  = ckpt.get("state_dict", ckpt)
    print(f"  Keys in Lightning ckpt : {len(ft_sd)}")

    # ── 2. 打印 Lightning 元数据（方便存档）────────────────────
    meta_keys = [k for k in ckpt.keys() if k != "state_dict"]
    print(f"  Lightning meta fields  : {meta_keys}")
    if "epoch" in ckpt:
        print(f"  Trained epoch          : {ckpt['epoch']}")
    if "global_step" in ckpt:
        print(f"  Global step            : {ckpt['global_step']}")

    # ── 3. 验证键名与预训练格式完全一致 ────────────────────────
    if verify:
        pt_ckpt_path = pretrained_dir / "eo-vae.ckpt"
        print(f"\nVerifying against pretrained: {pt_ckpt_path}")
        pt_raw = torch.load(pt_ckpt_path, map_location="cpu")
        pt_sd  = pt_raw.get("state_dict", pt_raw)

        ft_keys = set(ft_sd.keys())
        pt_keys = set(pt_sd.keys())

        only_ft = ft_keys - pt_keys
        only_pt = pt_keys - ft_keys

        if only_ft:
            print(f"  ⚠ Keys only in finetuned ({len(only_ft)}): {sorted(only_ft)[:5]}")
        if only_pt:
            print(f"  ⚠ Keys only in pretrained ({len(only_pt)}): {sorted(only_pt)[:5]}")
        if not only_ft and not only_pt:
            print(f"  ✓ Key sets identical ({len(ft_keys)} keys)")

        # 统计实际更新了多少权重
        changed = sum(
            1 for k in ft_keys & pt_keys
            if not torch.allclose(ft_sd[k].float(), pt_sd[k].float(), atol=1e-6)
        )
        print(f"  ✓ Changed keys: {changed} / {len(ft_keys)}")

    # ── 4. 保存为纯 state_dict 格式（与预训练 eo-vae.ckpt 格式一致）
    out_ckpt_path = output_dir / "eo-vae.ckpt"
    # EOFluxVAE.from_pretrained 读取时用 ckpt.get("state_dict", ckpt)
    # 所以两种格式都兼容：直接存 state_dict 或存 {"state_dict": sd} 都行
    # 这里存纯 state_dict（与原始 eo-vae.ckpt 格式保持一致）
    torch.save(ft_sd, out_ckpt_path)
    print(f"\n✓ Saved converted ckpt : {out_ckpt_path}")
    print(f"  File size: {out_ckpt_path.stat().st_size / 1024**2:.1f} MB")

    # ── 5. 拷贝 model_config.yaml（模型结构不变，直接复用）──────
    src_config = pretrained_dir / "model_config.yaml"
    dst_config = output_dir / "model_config.yaml"
    if src_config.exists():
        shutil.copy2(src_config, dst_config)
        print(f"✓ Copied model_config.yaml : {dst_config}")
    else:
        print(f"⚠ model_config.yaml not found in {pretrained_dir}, skipping.")

    # ── 6. 拷贝其他辅助文件（.pt 权重初始化文件、README 等）────
    for fname in [
        "decoder_dconv_weight_generator_init_0.01_er50k.pt",
        "encoder_dconv_weight_generator_init_0.01_er50k.pt",
        "README.md",
    ]:
        src = pretrained_dir / fname
        dst = output_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied auxiliary file  : {dst}")

    # ── 7. 快速验证：用 from_pretrained 实际加载一次 ───────────
    print(f"\n{'='*55}")
    print(f"  Final verification: loading with from_pretrained ...")
    print(f"{'='*55}")
    try:
        from eo_vae.models.new_autoencoder import EOFluxVAE
        model = EOFluxVAE.from_pretrained(
            repo_id="nilsleh/eo-vae",
            ckpt_filename="eo-vae.ckpt",
            ckpt_path=str(output_dir),
            config_filename="model_config.yaml",
            device="cpu",
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Total parameters: {total_params:,}")

        # 抽查第一个键的值是否与转换后文件一致
        first_key = list(ft_sd.keys())[0]
        model_val = dict(model.named_parameters()).get(first_key)
        if model_val is not None:
            match = torch.allclose(
                model_val.cpu().float(),
                ft_sd[first_key].float(),
                atol=1e-6
            )
            print(f"  ✓ Weight spot-check [{first_key[:50]}]: match={match}")
        else:
            # 可能是 buffer 不在 named_parameters 里
            model_sd  = model.state_dict()
            match = torch.allclose(
                model_sd[first_key].float(),
                ft_sd[first_key].float(),
                atol=1e-6
            )
            print(f"  ✓ Buffer spot-check [{first_key[:50]}]: match={match}")

    except Exception as e:
        print(f"  ⚠ Verification failed: {e}")
        print(f"    (This is non-fatal if eo_vae is not installed in current env)")

    print(f"\n{'='*55}")
    print(f"  转换完成！输出目录：{output_dir}")
    print(f"  使用方式：")
    print(f"    EOFluxVAE.from_pretrained(")
    print(f"        repo_id='nilsleh/eo-vae',")
    print(f"        ckpt_path='{output_dir}',")
    print(f"    )")
    print(f"  或在 EOVAEWrapper 中：")
    print(f"    EOVAEWrapper(pretrained_path='{output_dir}')")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lightning_ckpt",
        type=str,
        required=True,
        help="Path to Lightning last.ckpt or epoch=xx-step=xx.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted checkpoint",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="/path/to/models--nilsleh--eo-vae",
        help="Original pretrained directory (for config/aux files and verification)",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip key-name verification against pretrained ckpt",
    )
    args = parser.parse_args()
    convert(
        lightning_ckpt=args.lightning_ckpt,
        output_dir=args.output_dir,
        pretrained_dir=args.pretrained_dir,
        verify=not args.no_verify,
    )

'''
# 转换
python convert_lightning_to_eovae_ckpt.py \
    --lightning_ckpt /path/to/eo-vae_exp_results/mmdc_128_4GPU/\
eo-vae-mmdc-ft-128_03-23-2026_20-51-48/last.ckpt \
    --output_dir     /path/to/models--nilsleh--eo-vae-mmdc-finetuned \
    --pretrained_dir /path/to/models--nilsleh--eo-vae
'''