import torch.nn as nn


def same_conv(in_channels, out_channels, kernel_size, pad_mode="zeros", bias=True):
    """Conv2d with "same" padding.

    A 1x1 kernel has no padding, so it uses ``padding_mode="zeros"``: any other mode would still run a
    (zero-width) padding pass that copies the whole input, which costs time and activation memory.
    """
    padding = kernel_size // 2
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=pad_mode if padding else "zeros",
        bias=bias,
    )


# Convolutions
class EncBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        norm_kwargs=None,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        bias=True,
        residual=False,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(**norm_kwargs)
        self.conv1 = same_conv(in_channels, out_channels, kernel_size, pad_mode=pad_mode, bias=bias)
        self.norm2 = norm_fn(**norm_kwargs)
        self.conv2 = same_conv(out_channels, out_channels, kernel_size, pad_mode=pad_mode, bias=bias)
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = same_conv(in_channels, out_channels, 1, bias=bias)
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        if self.residual:
            return x + residual
        return x


def encoder(in_dim, hidden_dim, kernel_size=1, ks_res=1, num_layers=2, bias=True, num_groups=8, residual=False):
    return nn.Sequential(
        same_conv(in_dim, hidden_dim, kernel_size, pad_mode="reflect", bias=bias),
        *[
            EncBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=ks_res,
                pad_mode="reflect",
                norm_fn=nn.GroupNorm,
                norm_kwargs={"num_groups": num_groups, "num_channels": hidden_dim},
                activation_fn=nn.SiLU,
                use_conv_shortcut=False,
                bias=bias,
                residual=residual,
            )
            for _ in range(num_layers)
        ],
    )
