import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    NATTEN_RECENT = False
    from natten.functional import na2d_av, na2d_qk
except ImportError:
    NATTEN_RECENT = True
    from natten import na2d


def to_heads(x, num_heads):
    """[B, n * D, H, W] -> [B, H, W, n, D]."""
    return rearrange(x, "b (n d) h w -> b h w n d", n=num_heads)


def from_heads(x):
    """[B, H, W, n, D] -> [B, n * D, H, W]."""
    return rearrange(x, "b h w n d -> b (n d) h w")


def upsample_to_heads(x, size, num_heads, dtype):
    """Nearest-exact upsample [B, n * D, h, w] to ``size`` and split heads: [B, H, W, n, D] in ``dtype``.

    Casting and switching to channels-last happen at the low resolution, so the only full-resolution
    tensor allocated is the upsampled output, already in the [B, H, W, C] layout NATTEN reads (nearest
    interpolation only copies values, so the result is identical to casting afterwards).
    """
    x = x.to(dtype).contiguous(memory_format=torch.channels_last)
    # Autocast would run the interpolation in fp32; nearest only copies values, so keep ``dtype``
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = F.interpolate(x, size=size, mode="nearest-exact")
    return x.permute(0, 2, 3, 1).unflatten(-1, (num_heads, -1))


def legacy_attention(q, k, v, kernel_size, dilation, scale=1, return_weights=False):
    """Neighborhood attention with the pre-0.20 NATTEN ops; inputs and output are [B, H, W, n, D].

    With ``return_weights``, also returns the scaled pre-softmax scores [B, n, H, W, K*K].
    """
    q = rearrange(q * scale, "b h w n d -> b n h w d")  # scaling q is cheaper than scaling the K*K scores
    k = rearrange(k, "b h w n d -> b n h w d")
    v = rearrange(v, "b h w n d -> b n h w d")
    attn_scores = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)

    attn_weights = attn_scores.softmax(dim=-1)
    features = na2d_av(attn_weights, v, kernel_size=kernel_size, dilation=dilation)
    features = rearrange(features, "b n h w d -> b h w n d")

    if return_weights:
        return features, attn_scores
    return features


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9),
        backend="cutlass-fna",
        **kwargs,
    ):
        """``backend`` is passed to NATTEN >= 0.20 (``None`` lets NATTEN pick one that can run)."""
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.backend = backend

        self.scale = (dim // num_heads) ** -0.5

    def attend(self, q, k, v, dilation, return_weights=False):
        """Neighborhood attention on [B, H, W, n, D] tensors."""
        if return_weights:
            assert not NATTEN_RECENT, "Return weights not supported with recent natten versions"
            return legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale, return_weights=True)
        if NATTEN_RECENT:
            # Modern na2d uses head_dim ** -0.5, which equals self.scale
            return na2d(q, k, v, kernel_size=self.kernel_size, dilation=dilation, stride=1, backend=self.backend)
        return legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale)

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        """Each high-res query attends to a window of the low-res keys/values around its location.

        k and v are nearest-upsampled to the query resolution, and the window is dilated by the
        upsampling factor so that it covers ``kernel_size`` distinct low-res positions.
        """
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        self.dilation = dilation

        q = to_heads(q, self.num_heads)
        k = upsample_to_heads(k, (hq, wq), self.num_heads, q.dtype)
        v = upsample_to_heads(v, (hq, wq), self.num_heads, q.dtype)

        if return_weights:
            out, attn_weights = self.attend(q, k, v, dilation, return_weights=True)
            return from_heads(out), attn_weights
        return from_heads(self.attend(q, k, v, dilation))
