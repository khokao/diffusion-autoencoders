"""
The codes are modified.

Link:
    - [SimpleLoss] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/losses.py#L4-L15
"""
import torch.nn as nn


class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, noises):
        """
        Args:
            outputs (torch.tensor): A tensor of predicted noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            noises (torch.tensor): A tensor of ground truth noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32

        Returns:
            loss (torch.tensor): A tensor of simple loss.
                shape = ()
                dtype = torch.flaot32
        """
        loss = (noises - outputs).square().sum(dim=(1, 2, 3)).mean(dim=0)
        return loss
