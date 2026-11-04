"""Batch augmentation on the GPU for aligned face crops.

Decoding is done once into uint8 arrays, so per-step augmentation runs on the GPU where it is cheap. The recipe
follows common practice for face models: small random scale and shift, horizontal flip, colour jitter and greyscale;
the identity recipe adds blur and low-resolution simulation, which DigiFace-1M's authors found important for closing the
gap between synthetic training faces and real photos.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    """Strength of each augmentation; 0 turns it off."""

    flip: float = 0.5
    scale: tuple[float, float] = (0.9, 1.1)
    shift: float = 0.05            # fraction of the image side
    rotate_deg: float = 10.0
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    grayscale: float = 0.1
    blur: float = 0.0              # probability of a Gaussian blur
    lowres: float = 0.0            # probability of down- then up-sampling
    erase: float = 0.0             # probability of a random erased rectangle


EXPRESSION_AUG = AugmentConfig(erase=0.25)
IDENTITY_AUG = AugmentConfig(scale=(0.9, 1.05), shift=0.04, rotate_deg=8.0, brightness=0.4, contrast=0.4,
                             saturation=0.4, grayscale=0.2, blur=0.2, lowres=0.2, erase=0.1)


def _uniform(n: int, low: float, high: float, generator: torch.Generator, device) -> torch.Tensor:
    return torch.rand(n, generator=generator, device=device) * (high - low) + low


def _bernoulli(n: int, p: float, generator: torch.Generator, device) -> torch.Tensor:
    return torch.rand(n, generator=generator, device=device) < p


def _gaussian_blur(x: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    radius = 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """Converts uint8 [B, 3, H, W] to float in [0, 1]."""
    return images.float().div_(255.0)


def augment_batch(images: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator) -> torch.Tensor:
    """Returns an augmented float batch in [0, 1] from a uint8 or float batch [B, 3, H, W] on any device."""
    x = to_float(images) if images.dtype == torch.uint8 else images.clone()
    n, device = x.shape[0], x.device
    # Geometry: one affine warp per image (scale, rotation, shift, flip).
    scale = _uniform(n, cfg.scale[0], cfg.scale[1], generator, device)
    angle = _uniform(n, -cfg.rotate_deg, cfg.rotate_deg, generator, device) * torch.pi / 180.0
    shift = _uniform(2 * n, -cfg.shift, cfg.shift, generator, device).view(n, 2) * 2.0
    flip = torch.where(_bernoulli(n, cfg.flip, generator, device), -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # Colour.
    if cfg.brightness:
        x = x * _uniform(n, 1 - cfg.brightness, 1 + cfg.brightness, generator, device).view(n, 1, 1, 1)
    if cfg.contrast:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * _uniform(n, 1 - cfg.contrast, 1 + cfg.contrast, generator, device).view(n, 1, 1, 1) + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    if cfg.saturation:
        factor = _uniform(n, 1 - cfg.saturation, 1 + cfg.saturation, generator, device).view(n, 1, 1, 1)
        x = (x - gray) * factor + gray
    if cfg.grayscale:
        pick = _bernoulli(n, cfg.grayscale, generator, device).view(n, 1, 1, 1)
        x = torch.where(pick, gray.expand_as(x), x)
    if cfg.blur:
        pick = _bernoulli(n, cfg.blur, generator, device)
        if pick.any():
            x[pick] = _gaussian_blur(x[pick])
    if cfg.lowres:
        pick = _bernoulli(n, cfg.lowres, generator, device)
        if pick.any():
            size = x.shape[-1]
            small = F.interpolate(x[pick], size=(size // 3, size // 3), mode="bilinear", align_corners=False)
            x[pick] = F.interpolate(small, size=(size, size), mode="bilinear", align_corners=False)
    if cfg.erase:
        pick = _bernoulli(n, cfg.erase, generator, device)
        size = x.shape[-1]
        for i in torch.nonzero(pick).flatten().tolist():
            side = int(size * float(_uniform(1, 0.15, 0.35, generator, device)))
            top = int(float(_uniform(1, 0, size - side, generator, device)))
            left = int(float(_uniform(1, 0, size - side, generator, device)))
            x[i, :, top:top + side, left:left + side] = float(_uniform(1, 0, 1, generator, device))
    return x.clamp_(0.0, 1.0)
