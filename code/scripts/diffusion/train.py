import gc
import math
import os
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
gc.disable()
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig
from tqdm import tqdm
from opensora.acceleration.checkpoint import GLOBAL_ACTIVATION_MANAGER, set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.aspect import bucket_to_shapes
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import CheckpointIO, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import NsysProfiler, Timers, all_reduce_mean, create_tensorboard_writer, is_log_process, is_pipeline_enabled, log_cuda_max_memory, log_cuda_memory, log_model_params, print_mem, to_torch_dtype
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.sampling import get_res_lin_function, pack, prepare, prepare_ids, time_shift
from opensora.utils.train import build_training_target, create_colossalai_plugin, dropout_condition, get_batch_loss, prepare_visual_condition_causal, prepare_visual_condition_uncausal, prepare_visual_condition_eo, set_eps, set_lr, setup_device, update_ema, warmup_ae

torch.backends.cudnn.benchmark = False
os.environ['TORCHDYNAMO_DISABLE'] = '1'


def main():
    cfg = parse_configs()
    dtype = to_torch_dtype(cfg.get('dtype', 'bf16'))
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get('grad_ckpt_buffer_size', 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get('seed', 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get('pin_memory_cache_pre_alloc_numels', None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels
    plugin_type = cfg.get('plugin', 'zero2')
    plugin_config = cfg.get('plugin_config', {})
    plugin_kwargs = {}
    if plugin_type == 'hybrid':
        plugin_kwargs['custom_policy'] = MMDiTPolicy
    plugin = create_colossalai_plugin(plugin=plugin_type,
                                      dtype=cfg.get('dtype', 'bf16'),
                                      grad_clip=cfg.get('grad_clip', 0),
                                      **plugin_config,
                                      **plugin_kwargs)
    booster = Booster(plugin=plugin)
    seq_align = plugin_config.get('sp_size', 1)
    exp_name, exp_dir = create_experiment_workspace(cfg.get('outputs', './outputs'),
                                                    model_name=config_to_name(cfg),
                                                    config=cfg.to_dict(),
                                                    exp_name=cfg.get('exp_name', None))
    logger = create_logger(exp_dir)
    logger.info('Training configuration:\n %s', pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get('wandb', False):
            wandb.init(project=cfg.get('wandb_project', 'Open-Sora'), name=exp_name, config=cfg.to_dict(), dir=exp_dir)
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg['plugin_config'].get('tp_size', 1)
    sp_size = cfg['plugin_config'].get('sp_size', 1)
    pp_size = cfg['plugin_config'].get('pp_size', 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info('Number of GPUs: %s', num_gpus)
    logger.info('Number of groups: %s', num_groups)
    logger.info('Building dataset...')
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info('Dataset contains %s samples.', len(dataset))
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(dataset=dataset,
                           batch_size=cfg.get('batch_size', None),
                           num_workers=cfg.get('num_workers', 4),
                           seed=cfg.get('seed', 1024),
                           shuffle=True,
                           drop_last=True,
                           pin_memory=True,
                           process_group=get_data_parallel_group(),
                           prefetch_factor=cfg.get('prefetch_factor', None),
                           cache_pin_memory=cache_pin_memory,
                           num_groups=num_groups)
    print_mem('before prepare_dataloader')
    dataloader, sampler = prepare_dataloader(bucket_config=cfg.get('bucket_config', None),
                                             num_bucket_build_workers=cfg.get('num_bucket_build_workers', 1),
                                             **dataloader_args)
    print_mem('after prepare_dataloader')
    num_steps_per_epoch = len(dataloader)
    dataset.to_efficient()
    logger.info('Building models...')
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get('grad_checkpoint', True):
        set_grad_checkpoint(model)
    log_cuda_memory('diffusion')
    log_model_params(model)
    use_lora = cfg.get('lora_config', None) is not None
    if cfg.get('ema_decay', None) is not None and (not use_lora):
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info('EMA model created.')
    else:
        ema = ema_shape_dict = None
        logger.info('No EMA model created.')
    log_cuda_memory('EMA')
    if use_lora:
        lora_config = LoraConfig(**cfg.get('lora_config', None))
        model = booster.enable_lora(model=model,
                                    lora_config=lora_config,
                                    pretrained_dir=cfg.get('lora_checkpoint', None))
        log_cuda_memory('lora')
        log_model_params(model)
    if not cfg.get('cached_video', False):
        model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        del model_ae.decoder
        log_cuda_memory('autoencoder')
        log_model_params(model_ae)
    if not cfg.get('cached_text', False):
        model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory('t5')
        log_model_params(model_t5)
        model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory('clip')
        log_model_params(model_clip)
    optimizer = create_optimizer(model, cfg.optim)
    lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                       num_steps_per_epoch=num_steps_per_epoch,
                                       epochs=cfg.get('epochs', 1000),
                                       warmup_steps=cfg.get('warmup_steps', None),
                                       use_cosine_scheduler=cfg.get('use_cosine_scheduler', False))
    log_cuda_memory('optimizer')
    if cfg.get('cached_text', False):
        null_txt = torch.load('/path/to/cached_text/null_t5.pt', map_location=device)
        null_vec = torch.load('/path/to/cached_text/null_clip.pt', map_location=device)
    else:
        null_txt = model_t5('')
        null_vec = model_clip('')
    logger.info('Preparing for distributed training...')
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model=model,
                                                                  optimizer=optimizer,
                                                                  lr_scheduler=lr_scheduler,
                                                                  dataloader=dataloader)
    torch.set_default_dtype(torch.float)
    logger.info('Boosted model for distributed training')
    log_cuda_memory('boost')
    cfg_epochs = cfg.get('epochs', 1000)
    log_step = acc_step = 0
    running_loss = 0.0
    timers = Timers(record_time=cfg.get('record_time', False), record_barrier=cfg.get('record_barrier', False))
    nsys = NsysProfiler(warmup_steps=cfg.get('nsys_warmup_steps', 2),
                        num_steps=cfg.get('nsys_num_steps', 2),
                        enabled=cfg.get('nsys', False))
    logger.info('Training for %s epochs with %s steps per epoch', cfg_epochs, num_steps_per_epoch)
    load_master_weights = cfg.get('load_master_weights', False)
    save_master_weights = cfg.get('save_master_weights', False)
    start_epoch = cfg.get('start_epoch', None)
    start_step = cfg.get('start_step', None)
    if cfg.get('load', None) is not None:
        logger.info('Loading checkpoint from %s', cfg.load)
        lr_scheduler_to_load = lr_scheduler
        if cfg.get('update_warmup_steps', False):
            lr_scheduler_to_load = None
        ret = checkpoint_io.load(booster,
                                 cfg.load,
                                 model=model,
                                 ema=ema,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler_to_load,
                                 sampler=None if start_step is not None else sampler,
                                 include_master_weights=load_master_weights)
        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]
        logger.info('Loaded checkpoint %s at epoch %s step %s', cfg.load, ret[0], ret[1])
        if ema is not None and checkpoint_io.ema_load_failed:
            logger.info('Re-initializing EMA from loaded model weights (EMA checkpoint was corrupted)')
            unwrapped = model.unwrap() if hasattr(model, 'unwrap') else model
            ema.load_state_dict(unwrapped.state_dict(), strict=False)
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get('initial_lr', None))
        set_eps(optimizer, cfg.optim.eps)
        if not cfg.get('update_warmup_steps', False) and getattr(lr_scheduler, 'finished', False) and hasattr(
                lr_scheduler, '_last_lr') and lr_scheduler._last_lr:
            for _g, _last in zip(optimizer.param_groups, lr_scheduler._last_lr):
                _g['lr'] = _last
            _lr_scale = float(cfg.get('lr_resume_scale', 1.0))
            if abs(_lr_scale - 1.0) > 1e-09:
                for _g in optimizer.param_groups:
                    _g['lr'] = _g['lr'] * _lr_scale
                logger.warning('lr_resume_scale=%.4f applied: resumed lr %s -> %s', _lr_scale, lr_scheduler._last_lr,
                               optimizer.param_groups[0]['lr'])
            logger.info('Resumed lr synced to scheduler _last_lr=%s (cosine recursion anchor restored)',
                        optimizer.param_groups[0]['lr'])
        if cfg.get('update_warmup_steps', False):
            assert cfg.get(
                'warmup_steps',
                None) is not None, 'you need to set warmup_steps in order to pass --update-warmup-steps True'
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info('The learning rate starts from %s', optimizer.param_groups[0]['lr'])
    if start_step is not None:
        if start_step >= num_steps_per_epoch:
            start_epoch = start_epoch + start_step // num_steps_per_epoch if start_epoch is not None else start_step // num_steps_per_epoch
            start_step = start_step % num_steps_per_epoch
    else:
        start_step = 0
    sampler.set_step(start_step)
    start_epoch = start_epoch if start_epoch is not None else 0
    logger.info('Starting from epoch %s step %s', start_epoch, start_step)
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory('sharding EMA')
    if cfg.get('warmup_ae', False):
        shapes = bucket_to_shapes(cfg.get('bucket_config', None), batch_size=cfg.ae.batch_size)
        warmup_ae(model_ae, shapes, device, dtype)
    sigma_min = cfg.get('sigma_min', 1e-05)
    accumulation_steps = cfg.get('accumulation_steps', 1)
    ckpt_every = cfg.get('ckpt_every', 0)
    if cfg.get('is_eo_vae', False):
        prepare_visual_condition = prepare_visual_condition_eo
    elif cfg.get('is_causal_vae', False):
        prepare_visual_condition = prepare_visual_condition_causal
    else:
        prepare_visual_condition = prepare_visual_condition_uncausal

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch.pop('video')
        y = batch.pop('text')
        bs = x.shape[0]
        lat = batch.pop('lat', None)
        lon = batch.pop('lon', None)
        s2_time = batch.pop('s2_time', None)
        frame_mask = batch.pop('frame_mask', None)
        quality_mask = batch.pop('quality_mask', None)
        if quality_mask is not None:
            quality_mask = quality_mask.to(device=device, dtype=torch.float32)
            quality_mask = torch.nan_to_num(quality_mask, nan=0.0, posinf=0.0, neginf=0.0)
            quality_mask = torch.clamp(quality_mask, 0.0, 1.0)
        if True and lat is not None and (lon is not None) and (s2_time is not None):
            lat = lat.to(device=device, dtype=torch.float32)
            lon = lon.to(device=device, dtype=torch.float32)
            s2_time = s2_time.to(device=device, dtype=torch.float32)
            lat = torch.nan_to_num(lat, nan=0.0)
            lon = torch.nan_to_num(lon, nan=0.0)
            s2_time = torch.nan_to_num(s2_time, nan=0.0)
            lat_rad = lat * math.pi / 180.0
            lon_rad = lon * math.pi / 180.0
            x_coord = torch.cos(lat_rad) * torch.cos(lon_rad)
            y_coord = torch.cos(lat_rad) * torch.sin(lon_rad)
            z_coord = torch.sin(lat_rad)
            t0 = s2_time[:, 0]
            doy = t0 % 365.25
            sin_doy = torch.sin(2 * math.pi * doy / 365.25)
            cos_doy = torch.cos(2 * math.pi * doy / 365.25)
            year = 1990.0 + t0 / 365.25
            year_norm = 2.0 * ((year - 1950.0) / (2050.0 - 1950.0)) - 1.0
            eo_info = torch.stack([x_coord, y_coord, z_coord, sin_doy, cos_doy, year_norm], dim=-1)
            inp['eo_info'] = eo_info.to(device=device, dtype=dtype)
            doy_all = torch.remainder(s2_time, 365.25)
            sin_doy_all = torch.sin(2 * math.pi * doy_all / 365.25)
            cos_doy_all = torch.cos(2 * math.pi * doy_all / 365.25)
            year_all = 1990.0 + s2_time / 365.25
            year_norm_all = 2.0 * ((year_all - 1950.0) / (2050.0 - 1950.0)) - 1.0
            eo_time_info = torch.stack([sin_doy_all, cos_doy_all, year_norm_all], dim=-1)
            if frame_mask is not None:
                valid = frame_mask.to(device=device, dtype=eo_time_info.dtype).unsqueeze(-1)
                eo_time_info = eo_time_info * valid
            inp['eo_time_info'] = eo_time_info.to(device=device, dtype=dtype)
        dem = batch.pop('dem', None)
        if dem is not None:
            inp['dem'] = dem.to(device=device, dtype=dtype)
            inp['dem_mask'] = batch.pop('dem_mask').to(device=device, dtype=dtype)
        era5 = batch.pop('era5', None)
        if era5 is not None:
            inp['era5'] = era5.to(device=device, dtype=dtype)
            inp['era5_mask'] = batch.pop('era5_mask').to(device=device, dtype=dtype)
        era5_anom = batch.pop('era5_anom', None)
        if era5_anom is not None:
            inp['era5_anom'] = era5_anom.to(device=device, dtype=dtype)
        era5_clim = batch.pop('era5_clim', None)
        if era5_clim is not None:
            inp['era5_clim'] = era5_clim.to(device=device, dtype=dtype)
        era5_clim_std = batch.pop('era5_clim_std', None)
        if era5_clim_std is not None:
            inp['era5_clim_std'] = era5_clim_std.to(device=device, dtype=dtype)
        cumstress = batch.pop('cumstress', None)
        if cumstress is not None:
            inp['cumstress'] = cumstress.to(device=device, dtype=dtype)
        era5_drop_prob = cfg.get('era5_drop_prob', 0.0)
        if era5_drop_prob > 0.0:
            drop_ids = torch.rand(bs, device=device) < era5_drop_prob
            drop_5d = drop_ids.view(bs, 1, 1, 1, 1).to(dtype=dtype)
            if 'era5_anom' in inp:
                inp['era5_anom'] = inp['era5_anom'] * (1.0 - drop_5d)
            else:
                inp['era5'] = inp['era5'] * (1.0 - drop_5d)
                inp['era5_mask'] = inp['era5_mask'] * (1.0 - drop_5d)
            if 'cumstress' in inp:
                drop_3d = drop_ids.view(bs, 1, 1).to(dtype=dtype)
                inp['cumstress'] = inp['cumstress'] * (1.0 - drop_3d)
        with nsys.range('encode_video'), timers['encode_video']:
            if cfg.get('condition_config', None) is not None:
                x_0, cond = prepare_visual_condition(x, cfg.condition_config, model_ae)
                if cfg.get('no_i2v_ref_loss', False):
                    inp['masks'] = cond[:, 0, :, :, :]
                cond = pack(cond, patch_size=cfg.get('patch_size', 2))
                inp['cond'] = cond
                inp['n_cond_frames'] = cfg.condition_config.get('in_len', 0)
            elif cfg.get('cached_video', False):
                x_0 = batch.pop('video_latents').to(device=device, dtype=dtype)
            else:
                x_0 = model_ae.encode(x)
        if quality_mask is not None:
            quality_mask_thresh = float(cfg.get('quality_mask_area_thresh', 0.95))
            quality_mask_thresh = max(0.0, min(1.0, quality_mask_thresh))
            quality_mask_latent = F.interpolate(quality_mask.unsqueeze(1).to(dtype=torch.float32),
                                                size=(x_0.shape[-3], x_0.shape[-2], x_0.shape[-1]),
                                                mode='area').squeeze(1)
            quality_mask_latent = (quality_mask_latent >= quality_mask_thresh).to(dtype=x_0.dtype)
            if frame_mask is not None and frame_mask.shape[1] == quality_mask_latent.shape[1]:
                quality_mask_latent = quality_mask_latent * frame_mask.to(
                    device=device, dtype=quality_mask_latent.dtype).unsqueeze(-1).unsqueeze(-1)
            inp['quality_mask'] = quality_mask_latent
        _cfg_shift_alpha = cfg.get('train_shift_alpha', None)
        if _cfg_shift_alpha is not None:
            shift_alpha = float(_cfg_shift_alpha)
        else:
            shift_alpha = get_res_lin_function()(x_0.shape[-1] * x_0.shape[-2] // 4)
            shift_alpha *= math.sqrt(x_0.shape[-3])
        t = torch.sigmoid(torch.randn(bs, device=device))
        t = time_shift(shift_alpha, t).to(dtype)
        _t_floor = cfg.get('t_floor', None)
        if _t_floor is not None:
            t = t.clamp(min=float(_t_floor))
        snr_weight = None
        if cfg.get('cached_text', False):
            t5_embedding = batch.pop('text_t5').to(device=device, dtype=dtype)
            clip_embedding = batch.pop('text_clip').to(device=device, dtype=dtype)
            with nsys.range('encode_text'), timers['encode_text']:
                inp_ = prepare_ids(x_0, t5_embedding, clip_embedding)
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get('patch_size', 2))
        else:
            with nsys.range('encode_text'), timers['encode_text']:
                inp_ = prepare(model_t5,
                               model_clip,
                               x_0,
                               prompt=y,
                               seq_align=seq_align,
                               patch_size=cfg.get('patch_size', 2))
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get('patch_size', 2))
        if cfg.get('dropout_ratio', None) is not None:
            cur_null_txt = null_txt
            num_pad_null_txt = inp['txt'].shape[1] - cur_null_txt.shape[1]
            if num_pad_null_txt > 0:
                cur_null_txt = torch.cat([cur_null_txt] + [cur_null_txt[:, -1:]] * num_pad_null_txt, dim=1)
            inp['txt'] = dropout_condition(cfg.dropout_ratio.get('t5', 0.0), inp['txt'], cur_null_txt)
            inp['y_vec'] = dropout_condition(cfg.dropout_ratio.get('clip', 0.0), inp['y_vec'], null_vec)
        x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
        inp['img'] = x_t
        inp['timesteps'] = t.to(dtype)
        inp['guidance'] = torch.full((x_t.shape[0], ), cfg.get('guidance', 4), device=x_t.device, dtype=x_t.dtype)
        return (inp, x_0, x_1, snr_weight)

    def run_iter(inp, x_0, x_1, snr_weight, actual_update_step):
        iter_stats = {}
        if is_pipeline_enabled(plugin_type, plugin_config):
            inp['target'] = build_training_target(x_0, x_1, sigma_min=sigma_min)
            with nsys.range('forward-backward'), timers['forward-backward']:
                data_iter = iter([inp])

                def _unwrap_vpred(out):
                    return out['v_pred'] if isinstance(out, dict) else out

                if cfg.get('no_i2v_ref_loss', False):
                    loss_fn = lambda out, input_: get_batch_loss(_unwrap_vpred(out),
                                                                 input_['target'],
                                                                 input_.pop('masks', None),
                                                                 input_.pop('quality_mask', None),
                                                                 snr_weight=snr_weight) / accumulation_steps
                else:
                    loss_fn = lambda out, input_: get_batch_loss(_unwrap_vpred(out),
                                                                 input_['target'],
                                                                 None,
                                                                 input_.pop('quality_mask', None),
                                                                 snr_weight=snr_weight) / accumulation_steps
                loss = booster.execute_pipeline(data_iter, model, loss_fn, optimizer)['loss']
                loss = loss * accumulation_steps if loss is not None else loss
                loss_item = all_reduce_mean(loss.data.clone().detach())
        else:
            with nsys.range('forward'), timers['forward']:
                model_pred = model(**inp)
                latent_loss = None
                loss = None
                if model_pred is None:
                    raise RuntimeError("Diffusion training requires model output 'v_pred'.")
                v_t = build_training_target(x_0, x_1, sigma_min=sigma_min)
                if cfg.get('no_i2v_ref_loss', False):
                    latent_loss = get_batch_loss(model_pred,
                                                 v_t,
                                                 inp.pop('masks', None),
                                                 inp.pop('quality_mask', None),
                                                 snr_weight=snr_weight)
                else:
                    latent_loss = get_batch_loss(model_pred,
                                                 v_t,
                                                 None,
                                                 inp.pop('quality_mask', None),
                                                 snr_weight=snr_weight)
                loss = latent_loss
                if loss is None:
                    raise RuntimeError('The diffusion model did not produce a training loss.')
            loss_item = all_reduce_mean(loss.data.clone().detach()).item()
            dist.barrier()
            with nsys.range('backward'), timers['backward']:
                ctx = booster.no_sync(model, optimizer) if cfg.get('plugin', 'zero2') in (
                    'zero1', 'zero1-seq') and (step + 1) % accumulation_steps != 0 else nullcontext()
                with ctx:
                    booster.backward(loss=loss / accumulation_steps, optimizer=optimizer)
        with nsys.range('optim'), timers['optim']:
            if (step + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        if ema is not None:
            with nsys.range('update_ema'), timers['update_ema']:
                update_ema(ema, model.unwrap(), optimizer=optimizer, decay=cfg.get('ema_decay', 0.9999))
        return (loss_item, iter_stats)

    max_steps = cfg.get('max_steps', None)
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError('max_steps must be a positive integer or None')
    if max_steps is not None and (start_epoch * num_steps_per_epoch + start_step) // accumulation_steps >= max_steps:
        logger.info('Checkpoint already reached max_steps=%s', max_steps)
        return
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info('Beginning epoch %s...', epoch)
        with tqdm(enumerate(dataloader_iter, start=start_step),
                  desc=f'Epoch {epoch}',
                  disable=not is_log_process(plugin_type, plugin_config),
                  initial=start_step,
                  total=num_steps_per_epoch) as pbar:
            pbar_iter = iter(pbar)

            def fetch_data():
                step, batch = next(pbar_iter)
                pinned_video = batch['video']
                batch['video'] = pinned_video.to(device, dtype, non_blocking=True)
                return (batch, step, pinned_video)

            batch_, step_, pinned_video_ = fetch_data()
            for _ in range(start_step, num_steps_per_epoch):
                nsys.step()
                with nsys.range('load_data'), timers['load_data']:
                    batch, step, pinned_video = (batch_, step_, pinned_video_)
                    if step + 1 < num_steps_per_epoch:
                        batch_, step_, pinned_video_ = fetch_data()
                with nsys.range('iter'), timers['iter']:
                    global_step = epoch * num_steps_per_epoch + step
                    actual_update_step = (global_step + 1) // accumulation_steps
                    inp, x_0, x_1, snr_weight = prepare_inputs(batch)
                    if cache_pin_memory:
                        dataloader_iter.remove_cache(pinned_video)
                    loss, iter_stats = run_iter(inp, x_0, x_1, snr_weight, actual_update_step)
                if loss is not None:
                    running_loss += loss
                global_step = epoch * num_steps_per_epoch + step
                actual_update_step = (global_step + 1) // accumulation_steps
                log_step += 1
                acc_step += 1
                if (global_step + 1) % accumulation_steps == 0:
                    if actual_update_step % cfg.get('log_every', 1) == 0:
                        if is_log_process(plugin_type, plugin_config):
                            avg_loss = running_loss / log_step
                            pbar.set_postfix({
                                'loss': avg_loss,
                                'global_grad_norm': optimizer.get_grad_norm(),
                                'step': step,
                                'global_step': global_step,
                                'lr': optimizer.param_groups[0]['lr']
                            })
                            if tb_writer is not None:
                                tb_writer.add_scalar('loss', loss, actual_update_step)
                            if cfg.get('wandb', False):
                                wandb_dict = {
                                    'iter': global_step,
                                    'acc_step': acc_step,
                                    'epoch': epoch,
                                    'loss': loss,
                                    'avg_loss': avg_loss,
                                    'lr': optimizer.param_groups[0]['lr'],
                                    'eps': optimizer.param_groups[0]['eps'],
                                    'global_grad_norm': optimizer.get_grad_norm()
                                }
                                if cfg.get('record_time', False):
                                    wandb_dict.update(timers.to_dict())
                                wandb.log(wandb_dict, step=actual_update_step)
                        running_loss = 0.0
                        log_step = 0
                reached_max_steps = max_steps is not None and actual_update_step >= max_steps and (
                    (global_step + 1) % accumulation_steps == 0)
                with nsys.range('checkpoint'), timers['checkpoint']:
                    if (global_step + 1) % accumulation_steps == 0 and (
                            ckpt_every > 0 and actual_update_step % ckpt_every == 0 or reached_max_steps):
                        gc.collect()
                        if dist.get_rank() == 0:
                            no_grad_params = []
                            for name, param in model.named_parameters():
                                if param.requires_grad and param.grad is None:
                                    no_grad_params.append(name)
                            if no_grad_params:
                                print(f'[DEBUG ckpt] {len(no_grad_params)} params have grad=None:')
                                for n in no_grad_params:
                                    print(f'  - {n}')
                            else:
                                print('[DEBUG ckpt] All params have gradients ✓')
                        _skip_optim = cfg.get('skip_optimizer_save', False)
                        save_dir = checkpoint_io.save(
                            booster,
                            exp_dir,
                            model=model,
                            ema=ema,
                            optimizer=optimizer if save_master_weights else None if _skip_optim else optimizer,
                            lr_scheduler=None if _skip_optim else lr_scheduler,
                            sampler=sampler,
                            epoch=epoch,
                            step=step + 1,
                            global_step=global_step + 1,
                            batch_size=cfg.get('batch_size', None),
                            lora=use_lora,
                            actual_update_step=actual_update_step,
                            ema_shape_dict=ema_shape_dict,
                            async_io=cfg.get('async_io', False),
                            include_master_weights=save_master_weights,
                            skip_optimizer_state=_skip_optim)
                        logger.info('Saved checkpoint at epoch %s, step %s, global_step %s to %s', epoch, step + 1,
                                    actual_update_step, save_dir)
                        rm_checkpoints(exp_dir, keep_n_latest=cfg.get('keep_n_latest', -1))
                        logger.info('Removed old checkpoints and kept %s latest ones.', cfg.get('keep_n_latest', -1))
                if cfg.get('record_time', False):
                    print(timers.to_str(epoch, step))
                if reached_max_steps:
                    logger.info('Reached max_steps=%s; final checkpoint saved', max_steps)
                    log_cuda_max_memory('final')
                    return
        sampler.reset()
        start_step = 0
    log_cuda_max_memory('final')


if __name__ == '__main__':
    main()
