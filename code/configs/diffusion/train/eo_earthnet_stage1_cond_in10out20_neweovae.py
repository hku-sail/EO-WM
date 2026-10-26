"""EO-WM training configuration with weather anomalies and cumulative stress."""
_base_ = ['eo_earthnet_image.py']
import os

grad_ckpt_settings = (4, 100)
dataset = dict(data_path='data/metadata/earthnet2021_train_in10out20_refine-caption_multi-p_corrected-temp50.csv',
               cloud_prob_thresh=0.2,
               climatology_path=os.environ.get('CLIM_PATH', 'data/climatology/era5_climatology_train.pt'),
               use_cumstress=True,
               stress_variant='directional_product')
ae = dict(pretrained_path=os.environ.get('AE_PATH', 'weights/eo-vae'))
bucket_config = {'_delete_': True, '128px': {30: (1.0, 64)}}
model = dict(grad_ckpt_settings=grad_ckpt_settings,
             cond_embed=True,
             hidden_size=768,
             num_heads=12,
             depth=12,
             depth_single_blocks=14,
             axes_dim=[16, 24, 24],
             cond_reinject_every_n=4)
lr = 0.0002
optim = dict(lr=lr, weight_decay=0.01)
use_cosine_scheduler = True
warmup_steps = 500
ema_decay = 0.9999
grad_clip = 1.0
ckpt_every = 1000
keep_n_latest = 20
outputs = 'outputs/earthnet'
wandb_project = 'earthnet'
condition_config = dict(in_len=10, v2v=0.9, i2v=0.05, t2v=0.05)
quality_mask_area_thresh = 0.5
train_shift_alpha = 2.0
is_eo_vae = True
no_i2v_ref_loss = True
era5_drop_prob = 0.15
skip_optimizer_save = True
