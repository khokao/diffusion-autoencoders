"""
The codes are modified.

Link:
    - [sample_testdata]
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/experiment.py#L120-L139
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/renderer.py#L43-L60
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L181-L215
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L716-L807
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L584-L631
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L274-L368
    - [encode_stochastic]
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/experiment.py#L147-L155
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L633-L714
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/diffusion/base.py#L274-L368
    - [interpolate]
        - https://github.com/phizaz/diffae/blob/master/interpolate.ipynb
"""
import warnings

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import get_betas


class Sampler:
    def __init__(self, model, cfg):
        """
        Args:
            model: Diffusion Autoencoder model.
            cfg (dict): A dict of config.
        """
        self.model = model
        self.cfg = cfg

        self.device = self.cfg['general']['device']
        self.model.to(self.device)

        self.num_timesteps = cfg['model']['timesteps']['num']
        self.betas = get_betas(cfg)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.zeros(1, device=self.device)], dim=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.lpips_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(self.device)

    def sample_testdata(self, test_dataset, eta=0.0):
        """Autoencode test data and calculate evaluation metrics.
        """
        test_dataset = test_dataset
        test_loader = DataLoader(test_dataset, **self.cfg['test']['dataloader'])

        scores = {
            'lpips': [],
            'mse': [],
        }
        self.model.eval()
        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            x0, _ = batch
            x0 = x0.to(self.device)
            batch_size = x0.shape[0]

            xt = self.encode_stochastic(x0, disable_tqdm=True)
            style_emb = self.model.encoder(x0)
            for _t in reversed(range(self.num_timesteps)):
                t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                e = self.model.unet(xt, t, style_emb)

                # Equation 12 of Denoising Diffusion Implicit Models
                x0_t = (
                    torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt
                    - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None] * e
                ).clamp(-1, 1)
                e = (
                    (torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                    / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None])
                )
                sigma = (
                    eta
                    * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                    / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
                )
                xt = (
                    torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * x0_t
                    + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2)[:, None, None, None] * e
                )
                xt = xt + torch.randn_like(x0) * sigma[:, None, None, None] if _t != 0 else xt

            norm_x0 = (x0 + 1) / 2
            norm_xt = (xt + 1) / 2
            scores['lpips'].append(self.lpips_fn_alex(x0, xt).view(-1))
            scores['mse'].append((norm_x0 - norm_xt).square().mean(dim=[1, 2, 3]))

        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).mean().item()

        return scores

    def sample_one_image(self, image, xt=None, style_emb=None, eta=0.0):
        """Get the result of autoencoding a single image
        """
        self.model.eval()

        x0 = image.unsqueeze(dim=0).to(self.device)
        batch_size = x0.shape[0]

        if xt is None:
            xt = self.encode_stochastic(x0)
        if style_emb is None:
            style_emb = self.model.encoder(x0)

        x0_preds = []
        xt_preds = []
        for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e = self.model.unet(xt, t, style_emb)

            # Equation 12 of Denoising Diffusion Implicit Models
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None] * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None])
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2)[:, None, None, None] * e
            )
            xt = xt + torch.randn_like(x0) * sigma if _t != 0 else xt

            x0_preds.append(x0_t[0])
            xt_preds.append(xt[0])

        result = {
            'x0_preds': x0_preds,
            'xt_preds': xt_preds,
            'input': x0[0],
            'output': xt_preds[-1],
        }
        return result

    def encode_stochastic(self, x0, disable_tqdm=False):
        """
        Get stochastic encoded tensor xT.
        It is necessary to obtain stochastic subcode for high-quality reconstruction, but not when training.
        See https://github.com/phizaz/diffae/issues/17 for more details.
        """
        batch_size = x0.shape[0]

        xt = x0.detach().clone()
        style_emb = self.model.encoder(x0)
        for _t in tqdm(range(self.num_timesteps), disable=disable_tqdm, desc='stochastic encoding...'):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e = self.model.unet(xt, t, style_emb)

            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None] * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_next[t])[:, None, None, None] * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_next[t])[:, None, None, None] * e
            )

        return xt

    def interpolate(self, xt_1, xt_2, style_emb_1, style_emb_2, alpha, eta=0.0):
        """Interpolation of 2 images.
        """
        def cos(a, b):
            a = a.view(-1)
            b = b.view(-1)
            a = torch.nn.functional.normalize(a, dim=0)
            b = torch.nn.functional.normalize(b, dim=0)
            return (a * b).sum()
        theta = torch.arccos(cos(xt_1, xt_2))

        self.model.eval()
        batch_size = xt_1.shape[0]

        xt = (
            torch.sin((1 - alpha) * theta) * xt_1.flatten() + torch.sin(alpha * theta) * xt_2.flatten()
        ) / torch.sin(theta)
        xt = xt.view(-1, *xt_1.shape[1:])
        style_emb = (1 - alpha) * style_emb_1 + alpha * style_emb_2

        x0_preds = []
        xt_preds = []
        for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e = self.model.unet(xt, t, style_emb)

            # Equation 12 of Denoising Diffusion Implicit Models
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None] * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None])
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2)[:, None, None, None] * e
            )
            xt = xt + torch.randn_like(xt_1) * sigma if _t != 0 else xt

            x0_preds.append(x0_t[0])
            xt_preds.append(xt[0])

        result = {
            'x0_preds': x0_preds,
            'xt_preds': xt_preds,
            'output': xt_preds[-1],
        }
        return result
