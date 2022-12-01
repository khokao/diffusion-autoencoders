"""
The codes are modified.

Link:
    - [SemanticEncoder]
        - https://github.com/ermongroup/ddim/
          blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L192-L341
        - https://github.com/phizaz/diffae/
          blob/6328174152cfa3361877ca61cb629aeab1b7f319/model/unet.py#L352-L535
"""
import torch.nn as nn

from .network_blocks import AttentionBlock, Downsample, ResnetBlock


class SemanticEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Semantic encoder summarizes input images into descriptive vectors.

        Args (dict): A dict of config.
        """
        super().__init__()
        self.network_cfg = cfg['model']['network']['encoder']
        self.image_size = self.network_cfg['image_size']
        self.in_chans = self.network_cfg['in_channels']
        self.out_chans = self.network_cfg['out_channels']
        self.model_chans = self.network_cfg['model_channels']
        self.emb_chans = self.network_cfg['emb_channels']
        self.chan_mults = self.network_cfg['channel_multipliers']
        self.n_res_blocks = self.network_cfg['num_resnet_blocks']
        self.res_dropout = self.network_cfg['resnet_dropout']
        self.attn_resolution = self.network_cfg['attn_resolution']
        self.use_conv_resample = self.network_cfg['use_conv_resample']
        self.groups = self.network_cfg['num_groups']

        self.time_emb_chans = self.emb_chans
        self.style_emb_chans = self.emb_chans
        self.n_resolutions = len(self.chan_mults)

        self._create_network()

    def _create_network(self):
        """
        Create semantic encoder networks.
        The architecture is same as that of the first half of Unet decoder.
        """
        # e.g.) model_chans = 64, chan_mults = (1, 2, 4, 8) ---> chans = [64, 128, 256, 512]
        chans = [self.model_chans, *map(lambda x: self.model_chans * x, self.chan_mults)]

        # e.g.) image_size = 64, len(chan_mults) = 4 ---> resolutions = [64, 32, 16, 8]
        resolutions = [self.image_size // (2**i) for i in range(self.n_resolutions)]

        self.init_conv = nn.Conv2d(self.in_chans, self.model_chans, 3, 1, 1)

        # downsampling
        self.downs = nn.ModuleList()
        for i_level in range(self.n_resolutions):
            in_c = chans[i_level]
            out_c = chans[i_level + 1]
            resolution = resolutions[i_level]

            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            for i_block in range(self.n_res_blocks):
                res_block.append(ResnetBlock(in_c, out_c, self.res_dropout, self.groups))
                if resolution in self.attn_resolution:
                    attn_block.append(AttentionBlock(out_c, self.groups))
                else:
                    attn_block.append(nn.Identity())
                in_c = out_c
            is_last = bool(i_level == self.n_resolutions - 1)
            if is_last:
                downsample = nn.Identity()
            else:
                downsample = Downsample(out_c, use_conv=self.use_conv_resample)

            down = nn.Module()
            down.res_block = res_block
            down.attn_block = attn_block
            down.downsample = downsample
            self.downs.append(down)

        # middle
        mid_chans = chans[-1]
        self.middle = nn.Module()
        self.middle.res_block1 = ResnetBlock(mid_chans, mid_chans, self.res_dropout, self.groups)
        self.middle.attn_block1 = AttentionBlock(mid_chans, self.groups)
        self.middle.res_block2 = ResnetBlock(mid_chans, mid_chans, self.res_dropout, self.groups)

        self.flatten_block = nn.Sequential(
            nn.GroupNorm(self.groups, mid_chans),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(mid_chans, self.style_emb_chans, 1, 1, 0),
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32

        Returns:
            out (torch.tensor): Style embedding.
                shape = (batch, style_emb_chans)
                dtype = torch.float32
        """
        out = self.init_conv(x)
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_res_blocks):
                out = self.downs[i_level].res_block[i_block](out)
                out = self.downs[i_level].attn_block[i_block](out)
            out = self.downs[i_level].downsample(out)

        out = self.middle.res_block1(out)
        out = self.middle.attn_block1(out)
        out = self.middle.res_block2(out)

        out = self.flatten_block(out)
        return out
