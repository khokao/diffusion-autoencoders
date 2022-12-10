import torch.nn as nn

from .encoder import SemanticEncoder
from .unet import Unet


class DiffusionAutoEncoders(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: A dict of config.
        """
        super().__init__()
        self.encoder = SemanticEncoder(cfg)
        self.unet = Unet(cfg)

    def forward(self, x0, xt, t):
        """
        Args:
            x0 (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            xt (torch.tensor): A tensor of x at time step t.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            t (torch.tensor): A tensor of time steps.
                shape = (batch, )
                dtype = torch.float32

        Returns:
            out (torch.tensor): A tensor of output.
                shape = (batch, channels, height, width)
                dtype = torch.float32
        """
        style_emb = self.encoder(x0)
        out = self.unet(xt, t, style_emb)
        return out
