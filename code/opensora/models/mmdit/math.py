from __future__ import annotations
import torch
import torch.nn.functional as F
from einops import rearrange
try:
    from flash_attn import flash_attn_func as flash_attn_func_v2
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func_v2 = None
# from liger_kernel.ops.rope import LigerRopeFunction
from torch import Tensor
from typing import Tuple

# math.py 顶部，替换 liger rope 相关 import

# ── liger rope 纯 PyTorch 替换 ────────────────────────────────────
# q/k shape:  (batch, n_heads, seq_len, head_dim)
# cos/sin:    (batch, seq_len, head_dim)  ← 无 n_heads 维
import torch.autograd as _autograd
def _apply_rope(
    x: torch.Tensor,       # (B, H, L, D)
    cos: torch.Tensor,     # (B, L, D)
    sin: torch.Tensor,     # (B, L, D)
) -> torch.Tensor:
    # 插入 n_heads 维，变为 (B, 1, L, D)，broadcast 到 (B, H, L, D)
    cos = cos.unsqueeze(1)   # (B, 1, L, D)
    sin = sin.unsqueeze(1)   # (B, 1, L, D)
    head_dim = x.shape[-1]
    half = head_dim // 2
    # 标准 RoPE：前半和后半分别旋转
    x1 = x[..., :half]   # (B, H, L, D/2)
    x2 = x[..., half:]   # (B, H, L, D/2)
    cos1 = cos[..., :half]
    sin1 = sin[..., :half]
    # 旋转公式
    out = torch.cat([
        x1 * cos1 - x2 * sin1,
        x1 * sin1 + x2 * cos1,
    ], dim=-1)
    return out.to(x.dtype)
class LigerRopeFunction(_autograd.Function):
    """
    纯 PyTorch RoPE，替代 liger_kernel.ops.rope.LigerRopeFunction
    调用签名: q, k = LigerRopeFunction.apply(q, k, cos, sin)
    q/k:      (B, H, L, D)
    cos/sin:  (B, L, D)
    """
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out = _apply_rope(q, cos, sin)
        k_out = _apply_rope(k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out
    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        # RoPE 逆变换：sin 取反（旋转矩阵的转置）
        grad_q_in = _apply_rope(grad_q, cos, -sin)
        grad_k_in = _apply_rope(grad_k, cos, -sin)
        return grad_q_in, grad_k_in, None, None
# ────────────────────────────────────────────────────────────────

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    SUPPORT_FA3 = True
except:
    SUPPORT_FA3 = False


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if SUPPORT_FA3:
        return flash_attn_func_v3(q, k, v)[0]
    if HAS_FLASH_ATTN:
        return flash_attn_func_v2(q, k, v)
    q = rearrange(q, "B L H D -> B H L D")
    k = rearrange(k, "B L H D -> B H L D")
    v = rearrange(v, "B L H D -> B H L D")
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    return rearrange(x, "B H L D -> B L H D")


def attention(q: Tensor, k: Tensor, v: Tensor, pe, attn_mask: Tensor = None) -> Tensor:
    if isinstance(pe, torch.Tensor):
        q, k = apply_rope(q, k, pe)
    else:
        cos, sin = pe
        q, k = LigerRopeFunction.apply(q, k, cos, sin)

    if attn_mask is not None:
        # SDPA path: q,k,v are [B, H, L, D] after RoPE — SDPA expects this layout
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype), dropout_p=0.0)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x

    # flash_attn path: expects [B, L, H, D]
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    x = flash_attn_func(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")

    return x


def liger_rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return (cos, sin)


def rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rearrange_tensor(tensor):
    """
    Rearranges the last dimension (D) of the input tensor based on the specified mapping:
    2d -> d, 2d+1 -> D/2 + d.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        torch.Tensor: Tensor with rearranged last dimension, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the indices based on the mapping rule
    indices[:half_D] = torch.arange(0, D, 2, device=tensor.device)
    indices[half_D:] = torch.arange(1, D, 2, device=tensor.device)

    # Rearrange the tensor based on the computed indices
    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor):
    """
    Restores the original order of the last dimension (D) of the input tensor based on the reverse mapping:
    d -> 2d, D/2 + d -> 2d + 1.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        torch.Tensor: Tensor with restored original last dimension order, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    reverse_indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the reverse indices to restore the original order
    reverse_indices[::2] = torch.arange(half_D, device=tensor.device)
    reverse_indices[1::2] = torch.arange(half_D, D, device=tensor.device)

    # Rearrange the tensor based on the reverse indices
    return tensor.index_select(dim=-1, index=reverse_indices)
