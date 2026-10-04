"""PixelUp's building blocks: pixel encoder, Semantic Encoder stage attention, and the neighborhood decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.layers import CastRMSNorm, CrossAttention, low_res_heads, upsampled_neighborhood_attention, upsampling_dilation


def add_(x, y):
    """``x + y``, in place on ``x`` when that gives the same result (no graph recorded, no dtype promotion, e.g. under
    autocast); ``x`` must be a fresh tensor nothing else reads."""
    if (
        torch.is_grad_enabled()
        or torch.result_type(x, y) != x.dtype
        or x.shape != torch.broadcast_shapes(x.shape, y.shape)
    ):
        return x + y
    return x.add_(y)


class _ResBlock1x1(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class PixelEncoder(nn.Module):
    """Per-pixel query encoder: a 3x3 stem followed by 1x1 residual blocks."""

    def __init__(self, d: int = 384):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.r1, self.r2 = _ResBlock1x1(64), _ResBlock1x1(64)
        self.expand = nn.Conv2d(64, d, 1, bias=False)
        self.r3, self.r4 = _ResBlock1x1(d), _ResBlock1x1(d)

    def forward(self, x):
        x = self.stem(x)
        x = self.r2(self.r1(x))
        x = self.expand(x)
        return self.r4(self.r3(x))


class CrossAttentionStage(nn.Module):
    """Pre-norm cross-attention (queries -> Semantic Encoder stage features) + FFN, both residual."""

    def __init__(self, d: int = 384, c_kv: int = 384, num_heads: int = 6, kernel_size: int = 9, mlp_ratio: int = 4):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.norm_q, self.norm_kv = CastRMSNorm(d), CastRMSNorm(c_kv)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(c_kv, d, bias=False)
        self.v_proj = nn.Linear(c_kv, d, bias=False)
        self.norm_ffn = CastRMSNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * mlp_ratio, bias=False), nn.GELU(), nn.Linear(d * mlp_ratio, d, bias=False)
        )
        # Windowed attention over the nearest-upsampled k/v, dilated by the upsampling factor (no parameters)
        self.attend = CrossAttention(d, num_heads, kernel_size=(kernel_size, kernel_size))

    def attention(self, q, k, v, q_size, kv_size, global_attn):
        """Multi-head attention of tokens q [B, Nq, C] to k, v [B, Nk, C]: global, or windowed on the 2D grids."""
        if global_attn:
            qh, kh, vh = (rearrange(t, "b n (h e) -> b h n e", h=self.num_heads) for t in (q, k, v))
            out = F.scaled_dot_product_attention(qh, kh, vh, scale=self.scale)
            return rearrange(out, "b h n e -> b n (h e)")
        out = self.attend(
            rearrange(q, "b (h w) c -> b c h w", h=q_size[0]),
            rearrange(k, "b (h w) c -> b c h w", h=kv_size[0]),
            rearrange(v, "b (h w) c -> b c h w", h=kv_size[0]),
        )
        return rearrange(out, "b c h w -> b (h w) c")

    def forward(self, q_2d, kv_2d, global_attn: bool = False):
        q_size, kv_size = q_2d.shape[-2:], kv_2d.shape[-2:]
        q_tok = rearrange(q_2d, "b c h w -> b (h w) c")
        kv = self.norm_kv(rearrange(kv_2d, "b c h w -> b (h w) c"))
        out = self.attention(
            self.q_proj(self.norm_q(q_tok)), self.k_proj(kv), self.v_proj(kv), q_size, kv_size, global_attn
        )
        x = add_(out.contiguous(), q_tok)
        x = add_(x, self.ffn(self.norm_ffn(x)))
        return rearrange(x, "b (h w) c -> b c h w", h=q_size[0]).contiguous()


class NeighborhoodCrossAttention(nn.Module):
    """Decoder: every output pixel attends to a window of the low-res values; the softmax weights are averaged
    over heads and applied to all value channels at once (no value projection), so the output stays in the
    backbone's feature space."""

    def __init__(self, d_enc: int, value_dim: int, num_heads: int = 6, kernel_size: int = 9):
        super().__init__()
        assert d_enc % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d_enc // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.value_dim = value_dim  # informative: the weights apply to whatever values are given
        self.q_proj = nn.Linear(d_enc, d_enc, bias=False)
        self.k_proj = nn.Linear(d_enc, d_enc, bias=False)
        self.norm_q, self.norm_k = CastRMSNorm(d_enc), CastRMSNorm(d_enc)
        self.max_dilation: int | None = None
        # The original always attends in bf16
        self.compute_dtype = torch.bfloat16

    def normalize_queries(self, q_2d):
        """RMS-normalized queries, channels last: [B, Ho, Wo, C]."""
        return self.norm_q(rearrange(q_2d, "b c h w -> b h w c"))

    def project_queries(self, q):
        """Normalized queries [B, Ho, Wo, C] -> projected and scaled, in heads: [B, Ho, Wo, n, D], compute dtype."""
        q = self.q_proj(q)
        # Scaled before the cast, in place: for a power-of-2 scale (head dim 64) the same values as scaling after it
        q = q * self.scale if torch.is_grad_enabled() else q.mul_(self.scale)
        return q.to(self.compute_dtype).unflatten(-1, (self.num_heads, -1))

    def project_keys(self, k_2d):
        """Keys [B, C, Hk, Wk] -> normalized and projected, in heads: [B, Hk, Wk, n, D], compute dtype."""
        k = self.k_proj(self.norm_k(rearrange(k_2d, "b c h w -> b h w c")))
        return k.to(self.compute_dtype).unflatten(-1, (self.num_heads, -1))

    def attend(self, q, k, v_2d, out_dtype=None):
        """Attention of q, k (from ``project_queries`` / ``project_keys``) over the values v_2d [B, value_dim, Hk, Wk].

        Returns [B, value_dim, Ho, Wo] in ``out_dtype`` (default: the compute dtype).
        """
        ho, wo = q.shape[1:3]
        hk, wk = v_2d.shape[-2:]
        out, _ = upsampled_neighborhood_attention(
            q,
            k,
            low_res_heads(v_2d, 1, self.compute_dtype),  # one head: the head-averaged weights apply to every channel
            (self.kernel_size, self.kernel_size),
            upsampling_dilation((ho, wo), (hk, wk), self.max_dilation),
            1.0,
            need_weights=False,
            out_dtype=out_dtype,
        )
        return rearrange(out.squeeze(3), "b h w c -> b c h w")

    def forward(self, q_2d, k_2d, v_2d, out_dtype=None):
        """q_2d: [B, C, Ho, Wo]; k_2d: [B, C, Hk, Wk]; v_2d: [B, value_dim, Hk, Wk] -> [B, value_dim, Ho, Wo]."""
        q = self.project_queries(self.normalize_queries(q_2d))
        return self.attend(q, self.project_keys(k_2d), v_2d, out_dtype)
