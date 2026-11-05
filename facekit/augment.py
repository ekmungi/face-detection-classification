"""Random changes to face crops during training, run on the GPU.

The training images are decoded once into uint8 arrays. Each training step then changes a batch on the GPU, where
this work is cheap. The changes follow common practice for face models: a small random zoom, shift and rotation, a
horizontal flip, colour changes and greyscale.

The identity model also gets blur and a low-resolution simulation. The authors of DigiFace-1M found that strong
appearance changes of this kind help a model trained on synthetic faces work on real photos.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    """Sets how strongly each random change is applied. A value of 0 turns that change off."""

    # Probability that an image is mirrored left to right.
    flip: float = 0.5

    # Range of the random zoom factor. A factor below 1 zooms out and a factor above 1 zooms in.
    scale: tuple[float, float] = (0.9, 1.1)

    # Largest random shift, as a fraction of the image side.
    shift: float = 0.05

    # Largest random rotation in degrees, in either direction.
    rotate_deg: float = 10.0

    # Largest relative change in brightness. A value of 0.3 multiplies the image by a factor between 0.7 and 1.3.
    brightness: float = 0.3

    # Largest relative change in contrast, applied around the image's mean value.
    contrast: float = 0.3

    # Largest relative change in colour saturation, applied around the image's grey version.
    saturation: float = 0.3

    # Probability that an image is turned to greyscale.
    grayscale: float = 0.1

    # Probability that an image is softened with a Gaussian blur.
    blur: float = 0.0

    # Probability that an image is shrunk to a third of its side and enlarged again, which removes fine detail.
    lowres: float = 0.0

    # Probability that a random square of the image is hidden under a flat grey value.
    erase: float = 0.0


# The expression model keeps the default changes and adds random erasing, which teaches it to cope with a hidden
# part of the face.
EXPRESSION_AUG = AugmentConfig(erase=0.25)

# The identity model uses a narrower zoom range and stronger colour changes. It also adds blur, low resolution and
# random erasing, which help a model trained on synthetic faces cope with real photos.
IDENTITY_AUG = AugmentConfig(
    scale=(0.9, 1.05), shift=0.04, rotate_deg=8.0, brightness=0.4, contrast=0.4,
    saturation=0.4, grayscale=0.2, blur=0.2, lowres=0.2, erase=0.1,
)


def _uniform(n: int, low: float, high: float, generator: torch.Generator, device) -> torch.Tensor:
    """Returns n random numbers spread evenly between low and high."""
    return torch.rand(n, generator=generator, device=device) * (high - low) + low


def _bernoulli(n: int, p: float, generator: torch.Generator, device) -> torch.Tensor:
    """Returns n random true-or-false values, each of which is true with probability p."""
    return torch.rand(n, generator=generator, device=device) < p


def _gaussian_blur(x: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """Blurs a float batch [B, C, H, W] with a 5 x 5 Gaussian kernel.

    The blur runs as two one-dimensional passes, first horizontally and then vertically. Two passes give the same
    result as one two-dimensional pass at a lower cost. The edges are padded by reflection, so the border of the
    image does not darken.
    """
    radius = 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # Each colour channel gets its own copy of the kernel, so the channels are blurred separately.
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """Converts a uint8 batch [B, 3, H, W] to float32 values between 0 and 1."""
    return images.float().div_(255.0)


def augment_batch(images: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator) -> torch.Tensor:
    """Returns a randomly changed copy of a batch [B, 3, H, W] as float32 values between 0 and 1.

    The batch may hold uint8 or float values and may sit on any device. Mixed precision is switched off while the
    function runs, because under mixed precision the blur would return bfloat16 values, which cannot be written back
    into the float32 batch.
    """
    with torch.autocast(device_type=images.device.type, enabled=False):
        return _augment(images, cfg, generator)


def _augment(images: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator) -> torch.Tensor:
    """Applies the random changes set in `cfg` to a batch. `augment_batch` describes the input and the output."""
    x = to_float(images) if images.dtype == torch.uint8 else images.float().clone()
    n, device = x.shape[0], x.device

    # Geometry. One affine warp per image combines the zoom, rotation, shift and flip, so each image is resampled
    # only once. Reflection padding fills the corners that the warp exposes.
    scale = _uniform(n, cfg.scale[0], cfg.scale[1], generator, device)
    angle = _uniform(n, -cfg.rotate_deg, cfg.rotate_deg, generator, device) * torch.pi / 180.0
    shift = _uniform(2 * n, -cfg.shift, cfg.shift, generator, device).view(n, 2) * 2.0  # the warp spans -1 to 1
    flip = torch.where(_bernoulli(n, cfg.flip, generator, device), -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # Brightness. Each image is multiplied by one random factor.
    if cfg.brightness:
        x = x * _uniform(n, 1 - cfg.brightness, 1 + cfg.brightness, generator, device).view(n, 1, 1, 1)

    # Contrast. Each pixel's distance from the image's mean value is stretched or shrunk.
    if cfg.contrast:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * _uniform(n, 1 - cfg.contrast, 1 + cfg.contrast, generator, device).view(n, 1, 1, 1) + mean

    # Saturation. Each pixel's distance from its grey value is stretched or shrunk. The grey version uses the
    # standard luminance weights for red, green and blue, and the greyscale step below reuses it.
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    if cfg.saturation:
        factor = _uniform(n, 1 - cfg.saturation, 1 + cfg.saturation, generator, device).view(n, 1, 1, 1)
        x = (x - gray) * factor + gray

    # Greyscale. Some images are replaced by their grey version.
    if cfg.grayscale:
        pick = _bernoulli(n, cfg.grayscale, generator, device).view(n, 1, 1, 1)
        x = torch.where(pick, gray.expand_as(x), x)

    # Blur. Some images are softened, as an out-of-focus or heavily compressed photo would be.
    if cfg.blur:
        pick = _bernoulli(n, cfg.blur, generator, device)
        if pick.any():
            x[pick] = _gaussian_blur(x[pick])

    # Low resolution. Some images are shrunk to a third of their side and enlarged again. This removes fine detail
    # in the way a small face in a large photo loses it.
    if cfg.lowres:
        pick = _bernoulli(n, cfg.lowres, generator, device)
        if pick.any():
            size = x.shape[-1]
            small = F.interpolate(x[pick], size=(size // 3, size // 3), mode="bilinear", align_corners=False)
            x[pick] = F.interpolate(small, size=(size, size), mode="bilinear", align_corners=False)

    # Random erasing. Each picked image gets one random square, with a side of 15 to 35% of the image side, filled
    # with a flat grey value. The square is built as a mask on the GPU, so the GPU never waits for the CPU.
    if cfg.erase:
        pick = _bernoulli(n, cfg.erase, generator, device).view(n, 1, 1, 1)
        size = x.shape[-1]
        side = (_uniform(n, 0.15, 0.35, generator, device) * size).view(n, 1, 1, 1)
        top = (_uniform(n, 0, 1, generator, device).view(n, 1, 1, 1) * (size - side))
        left = (_uniform(n, 0, 1, generator, device).view(n, 1, 1, 1) * (size - side))
        rows = torch.arange(size, device=device).view(1, 1, size, 1)
        cols = torch.arange(size, device=device).view(1, 1, 1, size)
        inside = (rows >= top) & (rows < top + side) & (cols >= left) & (cols < left + side) & pick
        fill = _uniform(n, 0, 1, generator, device).view(n, 1, 1, 1)
        x = torch.where(inside, fill, x)

    return x.clamp_(0.0, 1.0)
