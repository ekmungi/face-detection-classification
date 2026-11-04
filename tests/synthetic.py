"""Small synthetic image datasets for tests, so no download and no GPU is needed."""
import torch
from torch.utils.data import TensorDataset


def make_dataset(size=64, seed=0, n_classes=2):
    """Returns a TensorDataset of 1x28x28 images in `n_classes` clearly separable classes (labels 0..n_classes-1)."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(size) % n_classes
    images = torch.randn(size, 1, 28, 28, generator=generator) * 0.1
    for k in range(n_classes):
        images[labels == k, 0, k * 2:k * 2 + 4, :] += 1.0  # a bright horizontal band whose position encodes the class
    return TensorDataset(images, labels)
