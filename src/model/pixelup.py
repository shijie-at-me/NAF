"""PixelUp: zero-shot semantic feature upsampling (Singh, Nihal and Hoskere, arXiv:2608.02792).

Port of https://github.com/deepankkumar/PixelUp (MIT License, v0.1.0, ``pixelup/models/pixelup.py``) to this
repository's upsampler interface. The architecture and parameter names are unchanged, so the released checkpoints
load as they are. Differences from the original:

- Neighborhood attention goes through ``src.layers``: NATTEN when it can run, else the PyTorch fallback. The
  original needs NATTEN < 0.20 for the split ``na2d_qk`` / ``na2d_av`` kernels of its decoder, which averages the
  softmax weights over heads before applying them to the values; the fallback does that directly on the low-res
  values (no upsampled copy of them, no chunking over value channels).
- ``checkpoint`` may be a URL (downloaded once to the torch hub cache) and the Semantic Encoder weights must come
  from the checkpoint (the released ones carry them) or from a local Hugging Face snapshot dir.
- Without autograd, the full-resolution tensors are updated in place, each one is released as soon as the next
  exists, and the decoder writes its float32 output directly: the same results with a lower peak memory.

Checkpoint loading and architecture resolution are in ``src/model/pixelup_checkpoint.py``.

How it works: a frozen DINOv3 ConvNeXt (the Semantic Encoder, ``src/backbone/convnext.py``) encodes the image at
``semantic_scale`` x the output resolution. Pixel-encoder queries are refined coarse-to-fine through its four stages
(each: upsample the stage features, add them, windowed cross-attention to them), then a NAF-like decoder lets every
output pixel attend to a window of the backbone's low-res features (the values) and returns their weighted average.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.layers.attentions import CrossAttention, low_res_heads
from src.layers.neighborhood import upsampled_neighborhood_attention
from src.layers.spiral_rope import SpiralRoPE2D
from src.layers.upsample import build_up_module
from src.model.base import BaseUpsampler
from src.model.pixelup_checkpoint import build_semantic_encoder, load_checkpoint, resolve_arch, split_checkpoint

__all__ = ["PixelUp"]

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

# Semantic Encoder stages, coarse to fine (its outputs come fine to coarse: s0 at stride 4 ... s3 at stride 32)
STAGES = ("s3", "s2", "s1", "s0")
# Outputs whose side times semantic_scale is below this are computed larger and average-pooled back
MIN_SEMANTIC_SIDE = 576.0


# --- building blocks ------------------------------------------------------------------------------------------------


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


class _RMSNorm(nn.RMSNorm):
    def forward(self, x):
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)


class CrossAttentionStage(nn.Module):
    """Pre-norm cross-attention (queries -> Semantic Encoder stage features) + FFN, both residual."""

    def __init__(self, d: int = 384, c_kv: int = 384, num_heads: int = 6, kernel_size: int = 9, mlp_ratio: int = 4):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.norm_q, self.norm_kv = _RMSNorm(d), _RMSNorm(c_kv)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(c_kv, d, bias=False)
        self.v_proj = nn.Linear(c_kv, d, bias=False)
        self.norm_ffn = _RMSNorm(d)
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
        self.norm_q, self.norm_k = _RMSNorm(d_enc), _RMSNorm(d_enc)
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
        dy, dx = max(1, ho // hk), max(1, wo // wk)
        if self.max_dilation is not None:
            dy, dx = min(dy, self.max_dilation), min(dx, self.max_dilation)
        out, _ = upsampled_neighborhood_attention(
            q,
            k,
            low_res_heads(v_2d, 1, self.compute_dtype),  # one head: the head-averaged weights apply to every channel
            (self.kernel_size, self.kernel_size),
            (dy, dx),
            1.0,
            need_weights=False,
            out_dtype=out_dtype,
        )
        return rearrange(out.squeeze(3), "b h w c -> b c h w")

    def forward(self, q_2d, k_2d, v_2d, out_dtype=None):
        """q_2d: [B, C, Ho, Wo]; k_2d: [B, C, Hk, Wk]; v_2d: [B, value_dim, Hk, Wk] -> [B, value_dim, Ho, Wo]."""
        q = self.project_queries(self.normalize_queries(q_2d))
        return self.attend(q, self.project_keys(k_2d), v_2d, out_dtype)


# --- model ----------------------------------------------------------------------------------------------------------


class PixelUp(BaseUpsampler):
    aliases = ("pixel_up",)

    def __init__(
        self,
        checkpoint: str | None = "pixelup_convnext_s",
        value_dim: int | None = None,
        kernel_size: int = 9,
        semantic_scale: float = 2.0,
        semantic_encoder: str | dict | None = None,
        semantic_encoder_dir: str | None = None,
        name: str = "pixelup",
        feature_dim: int | None = None,
        **overrides,
    ):
        """``checkpoint``: path, URL or release asset name ("pixelup_convnext_s" / "pixelup_convnext_t"), or None for
        a randomly initialized upsampler. ``value_dim`` (alias ``feature_dim``) is only informative: the decoder
        applies its weights to whatever features it gets. ``overrides`` set fields of ``DEFAULT_ARCH``."""
        super().__init__()
        sd, embedded = (None, {}) if checkpoint is None else split_checkpoint(load_checkpoint(checkpoint))
        a = resolve_arch(sd, embedded, overrides)

        self.value_dim = value_dim or feature_dim
        self.name = name
        self.semantic_scale = semantic_scale
        self.pixel_encoder_stride = int(a["pixel_encoder_stride"])
        self.fusion = str(a["fusion"])
        self.chain_q_rope = bool(a["chain_q_rope"])
        spf = str(a["stage_pool_factors"] or "")
        self.stage_pool_factors = [int(x) for x in spf.split(",")] if spf else None
        if self.stage_pool_factors is not None and len(self.stage_pool_factors) != 4:
            raise ValueError("stage_pool_factors must have 4 entries, e.g. '8,4,2,1'")
        self.chain_global_attn = bool(a["chain_global_attn"])
        # Keep the queries at the pixel-encoder resolution instead of doubling them at every stage
        self.fixed_q = False
        self.arch = a

        d_enc = int(a["d_enc"])
        self.semantic_encoder = build_semantic_encoder(sd, embedded, semantic_encoder, semantic_encoder_dir)
        self.pixel_enc = PixelEncoder(d_enc)
        channels = dict(zip(("s0", "s1", "s2", "s3"), self.semantic_encoder.OUT_CH, strict=True))
        skipped = {s.strip() for s in str(a["skip_attention_stages"]).split(",") if s.strip()}
        for tag in STAGES:
            c_x = channels[tag]
            setattr(self, f"up_{tag}", build_up_module(str(a["up_method"]), c_x, d_enc, int(a["up_scale"])))
            fuse = nn.Conv2d(2 * d_enc, d_enc, 1, bias=False) if self.fusion in ("both", "fuse_only") else None
            setattr(self, f"fuse_{tag}", fuse)
            use_cross = self.fusion in ("both", "cross_only") and tag not in skipped
            cross = CrossAttentionStage(d_enc, c_x, int(a["num_heads_encoder"]), kernel_size) if use_cross else None
            setattr(self, f"cross_{tag}", cross)

        self.q_rope = SpiralRoPE2D(d_enc, int(a["rope_directions"])) if self.chain_q_rope else None
        self.q_norm = _RMSNorm(d_enc) if bool(a["q_norm_stages"]) else None

        if bool(a["use_smooth"]):
            # 1x1 conv initialized to the identity
            self.smooth = nn.Conv2d(d_enc, d_enc, 1, bias=False)
            with torch.no_grad():
                self.smooth.weight.copy_(torch.eye(d_enc).view(d_enc, d_enc, 1, 1))
        else:
            self.smooth = nn.Identity()
        self.rope = SpiralRoPE2D(d_enc, int(a["rope_directions"]))
        self.attention = NeighborhoodCrossAttention(
            d_enc, self.value_dim or d_enc, int(a["num_heads_decoder"]), kernel_size
        )

        self.register_buffer("_mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMNET_STD).view(1, 3, 1, 1), persistent=False)

        if sd is not None:
            self._load_weights(sd, checkpoint)

    def _load_weights(self, sd, path):
        skip = ("semantic_encoder.", "convnext.", "refiner.", "ce.", "proj_s")
        up_sd = {k: v for k, v in sd.items() if not k.startswith(skip)}
        model_sd = self.state_dict()
        bad = [
            f"{k}: ckpt{tuple(v.shape)} != model{tuple(model_sd[k].shape)}"
            for k, v in up_sd.items()
            if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)
        ]
        unexpected = [k for k in up_sd if k not in model_sd]
        missing = [k for k in model_sd if k not in up_sd and not k.startswith("semantic_encoder.")]
        if bad or unexpected or missing:
            raise RuntimeError(
                f"PixelUp arch/ckpt mismatch for {path}\n  shape: {bad[:4]}\n  "
                f"unexpected: {unexpected[:4]}\n  missing: {missing[:4]}"
            )
        self.load_state_dict(up_sd, strict=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.semantic_encoder.eval()
        return self

    # ---- query chain ----

    def _refine(self, q, tag, s, pinned):
        """One stage: add the upsampled stage features ``s`` (resized to the queries), then cross-attend to ``s``."""
        up, fuse, cross = getattr(self, f"up_{tag}"), getattr(self, f"fuse_{tag}"), getattr(self, f"cross_{tag}")
        s_up = up(s)
        if s_up.shape[-2:] != q.shape[-2:]:
            if pinned:
                s_up = F.adaptive_avg_pool2d(s_up, q.shape[-2:])
            else:
                s_up = F.interpolate(s_up, size=q.shape[-2:], mode="bilinear", align_corners=False)
        q = fuse(torch.cat([q, s_up], dim=1)) if fuse is not None else add_(q, s_up)
        if cross is not None:
            q = q + cross(q, s, global_attn=pinned or self.chain_global_attn)
        if self.q_norm is not None:
            q = self.q_norm(q.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return q

    def _chain(self, x_norm, sf, q_size):
        """Pixel-encoder queries refined coarse-to-fine by the four Semantic Encoder stages ``sf`` (fine to coarse).

        With ``stage_pool_factors`` ("pinned"), the queries stay at ``q_size`` and the stages attend globally;
        otherwise they start at 1 / ``pixel_encoder_stride`` of the image and double at every stage.
        """
        h, w = x_norm.shape[-2:]
        stride = self.pixel_encoder_stride
        q = self.pixel_enc(
            F.interpolate(x_norm, size=(max(1, h // stride), max(1, w // stride)), mode="bilinear", align_corners=False)
        )
        pinned = self.stage_pool_factors is not None
        if pinned and q.shape[-2:] != tuple(q_size):
            q = F.adaptive_avg_pool2d(q, q_size)
        if self.q_rope is not None:
            q = self.q_rope(q)

        for i, (tag, s) in enumerate(zip(STAGES, reversed(sf), strict=True)):
            if i > 0 and not pinned and not self.fixed_q:
                q = F.interpolate(q, scale_factor=2.0, mode="bilinear", align_corners=False)
                if self.q_rope is not None:
                    q = self.q_rope(q)
            q = self._refine(q, tag, s, pinned)
        return q

    # ---- forward ----

    def _semantic_input(self, image, output_size):
        """The guide clamped to [0, 1], resized to ``semantic_scale`` x the output size (multiple of 32), normalized."""
        image = (image * self._std + self._mean).clamp(0, 1)
        size = [max(32, int(round(s * self.semantic_scale / 32)) * 32) for s in output_size]
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        return (image - self._mean) / self._std

    @staticmethod
    def _decoder_grids(enc, feat_size, output_size):
        """Decoder queries at the output size and keys at the feature size, both from the encoded queries."""
        k_2d = enc if enc.shape[-2:] == feat_size else F.adaptive_avg_pool2d(enc, feat_size)
        if enc.shape[-2:] == output_size:
            q_2d = enc
        elif enc.shape[-2] >= output_size[0] and enc.shape[-1] >= output_size[1]:
            q_2d = F.adaptive_avg_pool2d(enc, output_size)
        else:
            q_2d = F.interpolate(enc, size=output_size, mode="bilinear", align_corners=False)
        return q_2d, k_2d

    def forward(self, image, features, output_size, *args, **kwargs):
        """``image``: ImageNet-normalized guide (any size); ``features``: [B, C, h, w] backbone features."""
        output_size = (int(output_size[0]), int(output_size[1]))
        # Small outputs are computed at a multiple of the size and average-pooled back
        min_side = math.ceil(MIN_SEMANTIC_SIDE / max(self.semantic_scale, 1e-6))
        if min(output_size) < min_side:
            k = math.ceil(min_side / min(output_size))
            big = self.forward(image, features, (output_size[0] * k, output_size[1] * k), *args, **kwargs)
            return F.adaptive_avg_pool2d(big, output_size)

        # The Semantic Encoder and the pixel encoder both see the image at semantic_scale x the output size
        x_norm = self._semantic_input(image, output_size)
        q = self._chain(x_norm, self.semantic_encoder(x_norm), output_size)
        enc = self.rope(self.smooth(q))
        del q, x_norm

        # The full-resolution intermediates are released as soon as the next one exists
        q_2d, k_2d = self._decoder_grids(enc, tuple(features.shape[-2:]), output_size)
        k = self.attention.project_keys(k_2d)
        q = self.attention.normalize_queries(q_2d)
        del enc, q_2d, k_2d
        q = self.attention.project_queries(q)
        return self.attention.attend(q, k, features, out_dtype=torch.float32)
