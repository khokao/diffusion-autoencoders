"""
The codes are modified.

Link:
    - [SinusoidalPossitionalEmbedding] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L6-L24
    - [Downsample] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L55-L74
    - [Upsample] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L36-L52
    - [ResnetBlock] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L77-L134
    - [AttentionBlock] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L137-L189
"""
import math

import torch
import torch.nn as nn


class SinusoidalPossitionalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): Number of embedded dimensions.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Embedded values.
                shape = (size, )
                dtype = torch.float32

        Returns:
            emb (toch.tensor): Sinusoidal embeddings.
                shape = (size, dim)
                dtype = torch.float32
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class Downsample(nn.Module):
    """Downsampling module for Unet.
    """
    def __init__(self, in_chans, use_conv):
        """
        Args:
            in_chans (int): Number of input channels.
            use_conv (bool): A flag to use convolution.
        """
        super().__init__()
        self.in_chans = in_chans
        self.use_conv = use_conv

        self.conv = nn.Conv2d(self.in_chans, self.in_chans, 4, 2, 1)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


class Upsample(nn.Module):
    """Upsampling module for Unet.
    """
    def __init__(self, in_chans, use_conv):
        """
        Args:
            in_chans (int): Number of input channels.
            use_conv (bool): A flag to use convolution.
        """
        super().__init__()
        self.in_chans = in_chans
        self.use_conv = use_conv

        self.conv = nn.Conv2d(self.in_chans, self.in_chans, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2.0)

    def forward(self, x):
        x = self.up(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """Residual network for Unet
    """
    def __init__(self, in_chans, out_chans, p_dropout, groups, time_emb_chans=None, style_emb_chans=None):
        """
        Args:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
            p_dropout (float): Probability of dropout.
            groups (int): Number of groups to separate the channels into.
            time_emb_chans (int): Number of channels for time embedding.
            style_emb_chans (int): Number of channels for style embedding.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.p_dropout = p_dropout
        self.groups = groups

        self.time_emb_chans = time_emb_chans
        self.style_emb_chans = style_emb_chans
        self.use_condition = all([self.time_emb_chans is not None, self.style_emb_chans is not None])

        self._create_network()

    def _create_network(self):
        act = nn.SiLU()
        dropout = nn.Dropout(self.p_dropout)

        norm1 = nn.GroupNorm(self.groups, self.in_chans)
        conv1 = nn.Conv2d(self.in_chans, self.out_chans, 3, 1, 1)

        norm2 = nn.GroupNorm(self.groups, self.out_chans)
        conv2 = nn.Conv2d(self.out_chans, self.out_chans, 3, 1, 1)

        if self.in_chans != self.out_chans:
            self.res_conv = nn.Conv2d(self.in_chans, self.out_chans, 1, 1, 0)
        else:
            self.res_conv = nn.Identity()

        self.block1 = nn.Sequential(
            norm1,
            act,
            conv1,
        )
        self.block2 = nn.Sequential(
            norm2,
            act,
            dropout,
            conv2,
        )

        if self.use_condition:
            # Double channels for scaling and shift
            time_linear = nn.Linear(self.time_emb_chans, 2 * self.out_chans)
            self.time_mlp = nn.Sequential(
                act,
                time_linear,
            )

            # For scaling only
            style_linear = nn.Linear(self.style_emb_chans, self.out_chans)
            self.style_mlp = nn.Sequential(
                act,
                style_linear,
            )

    def forward(self, x, time_emb=None, style_emb=None):
        #  ref: https://github.com/phizaz/diffae/blob/34c07c2fc3c2a8ad1ce1dfabbd1ef1ed43957ca3/model/blocks.py
        #  Resblock in Unet: norm -> act -> conv -> norm -> condition -> act -> dropout -> conv (+ residual connection)
        if self.use_condition:
            time_mlp_out = self.time_mlp(time_emb)[:, :, None, None]
            time_scale, time_shift = torch.chunk(time_mlp_out, chunks=2, dim=1)
            style_scale = self.style_mlp(style_emb)[:, :, None, None]

            identity = x
            out = self.block1(x)
            out = self.block2[0](out)
            out *= time_scale
            out += time_shift
            out *= style_scale
            out = self.block2[1:](out)
            out += self.res_conv(identity)

        #  Resblock in Semantic Encoder Network: norm -> act -> conv (+ residual connection)
        else:
            assert not any([time_emb is not None, style_emb is not None]), 'use_condition=False, but condition(s) given'
            identity = x
            out = self.block1(x)
            out += self.res_conv(identity)

        return out


class AttentionBlock(nn.Module):
    """Attention block in Unet.
    """
    def __init__(self, in_chans, groups):
        """
        Args:
            in_chans (int): Number of input channels.
            groups (int): Number of groups to separate the channels into.
        """
        super().__init__()
        self.scale = int(in_chans) ** (-0.5)

        self.norm = nn.GroupNorm(groups, in_chans)
        self.q_conv = nn.Conv2d(in_chans, in_chans, 1, 1, 0)
        self.k_conv = nn.Conv2d(in_chans, in_chans, 1, 1, 0)
        self.v_conv = nn.Conv2d(in_chans, in_chans, 1, 1, 0)
        self.to_out = nn.Conv2d(in_chans, in_chans, 1, 1, 0)

    def forward(self, x):
        identity = x

        x = self.norm(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        b, c, h, w = x.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # shape = (b, h * w, c)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn_weight = (torch.bmm(q, k) * self.scale).softmax(dim=2)
        attn_weight = attn_weight.permute(0, 2, 1)  # shape = (b, h * w, h * w)
        attn_out = torch.bmm(v, attn_weight)  # shape = (b, c, h * w)
        attn_out = attn_out.reshape(b, c, h, w)

        out = self.to_out(attn_out) + identity
        return out
