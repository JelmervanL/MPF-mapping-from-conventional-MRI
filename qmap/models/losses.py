"""Regularization terms of the training objective (paper Eq. 12) and image-similarity metrics."""

import torch
import torch.nn.functional as F


def pd_constraint_loss(PD, mask, threshold=0.65):
    """Soft lower bound on PD: mean over the brain of max(0, threshold - PD)."""
    PD = PD[:, None, :, :]
    if mask is None or not torch.any(mask):
        return torch.tensor(0.0, device=PD.device)
    return torch.clamp(threshold - PD[mask], min=0).mean()


def gaussian_prior_loss(pred, mask, mu, sigma):
    """Gaussian prior mean((x - mu)^2 / (2 sigma^2)) over the voxels in `mask`."""
    mask = mask.squeeze()
    vals = torch.masked_select(pred, mask)
    if vals.numel() == 0:
        return torch.tensor(0., device=pred.device)
    return torch.mean((vals - mu) ** 2 / (2 * sigma ** 2))


def pearson_loss(pred, target, eps=1e-8):
    """1 - Pearson correlation (reported as a metric only)."""
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    covariance = (pred_centered * target_centered).sum(dim=1)
    var_pred = (pred_centered ** 2).sum(dim=1)
    var_target = (target_centered ** 2).sum(dim=1)

    variance_product = torch.clamp(var_pred * var_target, min=eps)
    corr = covariance / torch.sqrt(variance_product)
    return (1.0 - corr).mean()


class SSIM(torch.nn.Module):
    """Masked SSIM with an 11x11 Gaussian window (reported as a metric only)."""
    def __init__(self, window_size=11, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('window', self._create_window(window_size, channels))

    @staticmethod
    def _create_window(window_size, channels):
        coords = torch.arange(window_size, dtype=torch.float)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g /= g.sum()
        g_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return g_2d.expand(channels, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2, mask=None):
        """SSIM of img1 against img2 ([B, C, H, W]); data range and average taken inside `mask`."""
        if mask is not None:
            mask_float = mask.float()
            valid_pixels = img2[mask.bool()]
            data_range = valid_pixels.max() - valid_pixels.min() if valid_pixels.numel() > 0 else 1.0
        else:
            data_range = img2.max() - img2.min()

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            return (ssim_map * mask_float).sum() / (mask_float.sum() + 1e-8)
        return ssim_map.mean()
