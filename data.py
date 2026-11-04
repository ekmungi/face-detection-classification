"""Data helpers: reproducible non-overlapping splits, MNIST loading in fetch or store mode, and the transforms."""
import torch
from torch.utils.data import Subset

import augment
import datastore
import paths

# The dataset folder name; torchvision also uses it, so <root>/MNIST/raw holds the files.
MNIST_NAME = "MNIST"


def split_indices(n_items, train_size, val_size, seed):
    """Returns (train_indices, val_indices) as lists drawn from range(n_items) with no overlap, repeatable for a seed."""
    if train_size + val_size > n_items:
        raise ValueError(f"{train_size} + {val_size} items requested but only {n_items} exist")
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:train_size], order[train_size:train_size + val_size]


def fetch_mnist(folder):
    """Downloads MNIST with torchvision into <folder>/raw. The only code here that uses the network (fetch mode)."""
    from torchvision import datasets  # imported here so the splitting helper works without torchvision

    datasets.MNIST(folder.parent, train=True, download=True)
    datasets.MNIST(folder.parent, train=False, download=True)


def _mnist_root(data_dir, work_dir, mode):
    """Returns the folder that holds MNIST/raw for this mode, fetching or unpacking it when needed.

    fetch: downloaded into work_dir (the machine's disk) and nothing is written to data_dir.
    store: read from the kept copy in data_dir (a zip is unpacked into work_dir); a missing copy raises DatasetMissing.
    """
    folder = datastore.resolve(MNIST_NAME, mode, data_dir, work_dir or paths.work_data_dir(), fetch=fetch_mnist)
    return folder.parent


def load_mnist(data_dir, augment_train=False, work_dir=None, mode="fetch"):
    """Returns (train_full, val_source, test) MNIST datasets, getting the files as `mode` says.

    val_source holds the same images as train_full without augmentation, so validation never sees random changes.
    """
    from torchvision import datasets

    root = _mnist_root(data_dir, work_dir, mode)
    train_full = datasets.MNIST(root, train=True, download=False, transform=augment.build_transform(augment_train))
    val_source = datasets.MNIST(root, train=True, download=False, transform=augment.build_transform(False))
    test = datasets.MNIST(root, train=False, download=False, transform=augment.build_transform(False))
    return train_full, val_source, test


def load_mnist_raw(data_dir, work_dir=None, mode="fetch"):
    """Returns the training set as plain pictures with no transform, for showing images in a notebook."""
    from torchvision import datasets

    return datasets.MNIST(_mnist_root(data_dir, work_dir, mode), train=True, download=False)


def make_sets(train_full, val_source, train_size, val_size, seed):
    """Returns (train_set, val_set, train_indices, val_indices): two Subsets over disjoint items."""
    train_idx, val_idx = split_indices(len(train_full), train_size, val_size, seed)
    return Subset(train_full, train_idx), Subset(val_source, val_idx), train_idx, val_idx
