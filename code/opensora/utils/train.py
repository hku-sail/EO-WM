from __future__ import annotations
import random
import warnings
from collections import OrderedDict
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from opensora.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group, set_tensor_parallel_group
from opensora.utils.optimizer import LinearWarmupLR


def set_lr(optimizer: torch.optim.Optimizer, lr_scheduler: _LRScheduler, lr: float, initial_lr: float = None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group.get('base_lr', lr)
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.base_lrs = [param_group.get('base_lr', lr) for param_group in optimizer.param_groups]
        if initial_lr is not None:
            lr_scheduler.initial_lr = initial_lr


def set_warmup_steps(lr_scheduler: _LRScheduler, warmup_steps: int):
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.warmup_steps = warmup_steps


def set_eps(optimizer: torch.optim.Optimizer, eps: float = None):
    if eps is not None:
        for param_group in optimizer.param_groups:
            param_group['eps'] = eps


def setup_device() -> tuple[torch.device, DistCoordinator]:
    """
    Setup the device and the distributed coordinator.

    Returns:
        tuple[torch.device, DistCoordinator]: The device and the distributed coordinator.
    """
    assert torch.cuda.is_available(), 'Training currently requires at least one GPU.'
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    coordinator = DistCoordinator()
    device = get_current_device()
    return (device, coordinator)


def create_colossalai_plugin(plugin: str, dtype: str, grad_clip: float,
                             **kwargs) -> LowLevelZeroPlugin | HybridParallelPlugin:
    """
    Create a ColossalAI plugin.

    Args:
        plugin (str): The plugin name.
        dtype (str): The data type.
        grad_clip (float): The gradient clip value.

    Returns:
        LowLevelZeroPlugin |  HybridParallelPlugin: The plugin.
    """
    plugin_kwargs = dict(precision=dtype,
                         initial_scale=2**16,
                         max_norm=grad_clip,
                         overlap_allgather=True,
                         cast_inputs=False,
                         reduce_bucket_size_in_m=20)
    plugin_kwargs.update(kwargs)
    sp_size = plugin_kwargs.get('sp_size', 1)
    if plugin == 'zero1' or plugin == 'zero2':
        assert sp_size == 1, 'Zero plugin does not support sequence parallelism'
        stage = 1 if plugin == 'zero1' else 2
        plugin = LowLevelZeroPlugin(stage=stage, **plugin_kwargs)
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == 'hybrid':
        plugin_kwargs['find_unused_parameters'] = True
        reduce_bucket_size_in_m = plugin_kwargs.pop('reduce_bucket_size_in_m')
        if 'zero_bucket_size_in_m' not in plugin_kwargs:
            plugin_kwargs['zero_bucket_size_in_m'] = reduce_bucket_size_in_m
        plugin_kwargs.pop('cast_inputs')
        plugin_kwargs['enable_metadata_cache'] = False
        custom_policy = plugin_kwargs.pop('custom_policy', None)
        if custom_policy is not None:
            custom_policy = custom_policy()
        plugin = HybridParallelPlugin(custom_policy=custom_policy, **plugin_kwargs)
        set_tensor_parallel_group(plugin.tp_group)
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f'Unknown plugin {plugin}')
    return plugin


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module,
               model: torch.nn.Module,
               optimizer=None,
               decay: float = 0.9999,
               sharded: bool = True):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (torch.nn.Module): The EMA model.
        model (torch.nn.Module): The current model.
        optimizer (torch.optim.Optimizer): The optimizer.
        decay (float): The decay rate.
        sharded (bool): Whether the model is sharded.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name == 'pos_embed':
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def dropout_condition(prob: float, txt: torch.Tensor, null_txt: torch.Tensor) -> torch.Tensor:
    """
    Apply dropout to the text tensor.

    Args:
        prob (float): The dropout probability.
        txt (torch.Tensor): The text tensor.
        null_txt (torch.Tensor): The null text tensor.

    Returns:
        torch.Tensor: The text tensor with dropout applied.
    """
    if prob == 0:
        warnings.warn('Dropout probability is 0, skipping dropout')
    drop_ids = torch.rand(txt.shape[0], device=txt.device) < prob
    drop_ids = drop_ids.view((drop_ids.shape[0], ) + (1, ) * (txt.ndim - 1))
    new_txt = torch.where(drop_ids, null_txt, txt)
    return new_txt


def prepare_visual_condition_uncausal(x: torch.Tensor,
                                      condition_config: dict,
                                      model_ae: torch.nn.Module,
                                      pad: bool = False) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])
    masks = torch.zeros(B, 1, T, H, W).to(x.device, x.dtype)
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:
        if T <= 32 // model_ae.time_compression_ratio:
            condition_config.pop('v2v_head', None)
            condition_config.pop('v2v_tail', None)
            condition_config.pop('v2v_head_easy', None)
            condition_config.pop('v2v_tail_easy', None)
        if T <= 64 // model_ae.time_compression_ratio:
            condition_config.pop('v2v_head_easy', None)
            condition_config.pop('v2v_tail_easy', None)
        mask_cond_options = list(condition_config.keys())
        mask_cond_weights = list(condition_config.values())
        for i in range(B):
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            if mask_cond == 'i2v_head':
                masks[i, :, 0, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = torch.cat([x[i, :, :1]] * pad_num + [x[i, :, :-pad_num]], dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encode(padded_x)[0]
                else:
                    x_0[i] = model_ae.encode(x[i:i + 1])[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
            elif mask_cond == 'i2v_loop':
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = torch.cat([x[i, :, :1]] * pad_num + [x[i, :, :-pad_num * 2]] +
                                         [x[i, :, -pad_num * 2 - 1].unsqueeze(1)] * pad_num,
                                         dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encode(padded_x)[0]
                    latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :,
                                                                -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encode(x[i:i + 1])[0]
                    latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == 'i2v_tail':
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = torch.cat([x[i, :, pad_num:]] + [x[i, :, -1:]] * pad_num, dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encode(padded_x)[0]
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :,
                                                                -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encode(x[i:i + 1])[0]
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == 'v2v_head':
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == 'v2v_tail':
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            elif mask_cond == 'v2v_head_easy':
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == 'v2v_tail_easy':
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            else:
                assert mask_cond == 't2v', f'Unknown mask condition {mask_cond}'
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:
        x_0 = model_ae.encode(x)
    latent = masks * latent
    cond = torch.cat((masks, latent), dim=1)
    return (x_0, cond)


def prepare_visual_condition_causal(x: torch.Tensor, condition_config: dict,
                                    model_ae: torch.nn.Module) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])
    masks = torch.zeros(B, 1, T, H, W).to(x.device, x.dtype)
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:
        if T <= 32 // model_ae.time_compression_ratio + 1:
            condition_config.pop('v2v_head', None)
            condition_config.pop('v2v_tail', None)
            condition_config.pop('v2v_head_easy', None)
            condition_config.pop('v2v_tail_easy', None)
        if T <= 64 // model_ae.time_compression_ratio + 1:
            condition_config.pop('v2v_head_easy', None)
            condition_config.pop('v2v_tail_easy', None)
        mask_cond_options = list(condition_config.keys())
        mask_cond_weights = list(condition_config.values())
        for i in range(B):
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            if mask_cond == 'i2v_head':
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
            elif mask_cond == 'i2v_loop':
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == 'i2v_tail':
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif 'v2v_head' in mask_cond:
                ref_t = 33 if not 'easy' in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = model_ae.encode(x[i, :, :ref_t, :, :].unsqueeze(0))
            elif 'v2v_tail' in mask_cond:
                ref_t = 33 if not 'easy' in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = model_ae.encode(x[i, :, -ref_t:, :, :].unsqueeze(0))
            else:
                assert mask_cond == 't2v', f'Unknown mask condition {mask_cond}'
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:
        x_0 = model_ae.encode(x)
    latent = masks * latent
    cond = torch.cat((masks, latent), dim=1)
    return (x_0, cond)


def get_batch_loss(model_pred, v_t, masks=None, quality_mask=None, snr_weight=None):
    """
    支持两类mask:
      - masks: i2v/v2v 条件帧mask（1=条件帧, 0=生成帧）
      - quality_mask: 空间质量mask（1=有效像素, 0=无效像素/云区）
      - snr_weight: SNR权重 [B, 1, 1] 或 [B]，用于MinSNR Loss加权
    """
    need_unpack = masks is not None or quality_mask is not None
    if not need_unpack:
        if snr_weight is not None:
            return (snr_weight * (model_pred.float() - v_t.float())**2).mean()
        return F.mse_loss(model_pred.float(), v_t.float(), reduction='mean')
    ref_mask = masks if masks is not None else quality_mask
    num_frames, height, width = ref_mask.shape[-3:]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f'get_batch_loss expects even latent H/W for patch_size=2, got H={height}, W={width}')
    model_pred = rearrange(model_pred,
                           'b (t h w) (c ph pw) -> b c t (h ph) (w pw)',
                           h=height // 2,
                           w=width // 2,
                           t=num_frames,
                           ph=2,
                           pw=2)
    v_t = rearrange(v_t,
                    'b (t h w) (c ph pw) -> b c t (h ph) (w pw)',
                    h=height // 2,
                    w=width // 2,
                    t=num_frames,
                    ph=2,
                    pw=2)
    frame_cond_mask = None
    if masks is not None:
        frame_cond_mask = masks[:, :, 0, 0]
    if quality_mask is not None:
        quality_mask = torch.nan_to_num(quality_mask, nan=0.0, posinf=0.0, neginf=0.0)
        quality_mask = torch.clamp(quality_mask, 0.0, 1.0)
    if frame_cond_mask is not None:
        batch_loss = model_pred.new_tensor(0.0)
        valid_count = 0
        for i in range(model_pred.size(0)):
            pred_val = model_pred[i]
            target_val = v_t[i]
            m = frame_cond_mask[i]
            gen_idx = (m == 0).nonzero(as_tuple=True)[0]
            if gen_idx.numel() == 0:
                continue
            pred_val = pred_val[:, gen_idx]
            target_val = target_val[:, gen_idx]
            if quality_mask is not None:
                q = quality_mask[i, gen_idx].unsqueeze(0).to(dtype=pred_val.dtype)
                pred_val = pred_val * q
                target_val = target_val * q
            sample_loss = F.mse_loss(pred_val.float(), target_val.float(), reduction='mean')
            if snr_weight is not None:
                w = snr_weight[i] if snr_weight.dim() == 1 else snr_weight[i, 0, 0]
                sample_loss = w * sample_loss
            batch_loss += sample_loss
            valid_count += 1
        if valid_count == 0:
            return model_pred.float().mean() * 0.0
        return batch_loss / valid_count
    if quality_mask is not None:
        q = quality_mask.unsqueeze(1).to(dtype=model_pred.dtype)
        if snr_weight is not None:
            return (snr_weight * (model_pred.float() * q.float() - v_t.float() * q.float())**2).mean()
        return F.mse_loss(model_pred.float() * q.float(), v_t.float() * q.float(), reduction='mean')
    if snr_weight is not None:
        return (snr_weight * (model_pred.float() - v_t.float())**2).mean()
    return F.mse_loss(model_pred.float(), v_t.float(), reduction='mean')


def build_training_target(x_0: torch.Tensor, x_1: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """
    Build the training target under the current v-parameterization.
    """
    return (1 - sigma_min) * x_1 - x_0


def prepare_visual_condition_eo(x: torch.Tensor,
                                condition_config: dict,
                                model_ae: torch.nn.Module,
                                dem: torch.Tensor = None,
                                era5: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EO 专用的 visual condition 准备函数。

    EO任务固定语义：
        - 前 in_len 帧 = 已知输入（条件），mask=1
        - 后 out_len 帧 = 预测目标，mask=0
        - 没有随机掩膜策略，condition_config 只需提供 in_len

    Args:
        x : (B, C, T_pixel, H, W) 像素空间视频
                          T_pixel = in_len + out_len = 30
                          前 in_len 帧为输入，后 out_len 帧为预测目标
        condition_config : dict，必须包含 "in_len" 键
                          例如 {"in_len": 10}
        model_ae: EOVAEWrapper 实例
                          time_compression_ratio=1（纯2D VAE）

    Returns:
        x_0  : (B, lC, T, lH, lW)  完整latent（扩散训练目标）
        cond : (B, lC+1, T, lH, lW) 条件tensor = cat([masks, masked_latent])
               masks 形状  (B, 1,  T, lH, lW)，前in_len_latent帧=1，其余=0
               latent 形状 (B, lC, T, lH, lW)，前in_len_latent帧=编码值，其余=0

    Cond 通道布局（与 prepare_visual_condition_uncausal 完全一致）：
        cond[:, 0:1,  ...]  = masks          (1通道)
        cond[:, 1:,   ...]  = masked_latent  (lC=32通道)
        共 lC+1 = 33 通道
    """
    B, C_pixel, T_pixel, H_pixel, W_pixel = x.shape
    T_lat, lH, lW = model_ae.get_latent_size((T_pixel, H_pixel, W_pixel))
    lC = model_ae.latent_channels
    x_0 = torch.zeros(B, lC, T_lat, lH, lW, device=x.device, dtype=x.dtype)
    for i in range(B):
        x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    masks = torch.zeros(B, 1, T_lat, lH, lW, device=x.device, dtype=x.dtype)
    latent = torch.zeros(B, lC, T_lat, lH, lW, device=x.device, dtype=x.dtype)
    in_len_lat = condition_config.get('in_len', None)
    assert in_len_lat is not None, f"[prepare_visual_condition_eo] condition_config 必须包含 'in_len' 键，当前 condition_config={condition_config}"
    task_keys = [k for k in condition_config.keys() if k != 'in_len']
    if len(task_keys) == 0:
        task_keys, task_weights = (['v2v'], [1.0])
    else:
        task_weights = [condition_config[k] for k in task_keys]
    for i in range(B):
        task = random.choices(task_keys, weights=task_weights, k=1)[0]
        if task == 'v2v':
            masks[i, :, :in_len_lat, :, :] = 1.0
            latent[i, :, :in_len_lat, :, :] = x_0[i, :, :in_len_lat, :, :]
        elif task == 'i2v':
            masks[i, :, :1, :, :] = 1.0
            latent[i, :, :1, :, :] = x_0[i, :, :1, :, :]
        elif task == 't2v':
            pass
        else:
            raise ValueError(f'Unknown EO mask condition: {task}')
    cond = torch.cat([masks, latent], dim=1)
    return (x_0, cond)


@torch.no_grad()
def warmup_ae(model_ae: nn.Module, shapes: list[tuple[int, ...]], device: torch.device, dtype: torch.dtype):
    progress_bar = tqdm(shapes, desc='Warmup AE', disable=dist.get_rank() != 0)
    for x_shape in progress_bar:
        x = torch.randn(*x_shape, device=device, dtype=dtype)
        _ = model_ae.encode(x)
