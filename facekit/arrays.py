"""Image arrays on disk: decode a dataset once into a uint8 memory-mapped array, then train from it at disk speed.

An array set is three files in one folder: `<name>.u8` (N x S x S x 3 RGB bytes), `<name>.labels.npy` and
`<name>.json` (count, size, class names and any per-item metadata such as source ids, used by the split checks).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def array_paths(folder: str | Path, name: str) -> tuple[Path, Path, Path]:
    folder = Path(folder)
    return folder / f"{name}.u8", folder / f"{name}.labels.npy", folder / f"{name}.json"


def array_exists(folder: str | Path, name: str) -> bool:
    """True when all three files of the array set exist (a half-written set has no .json yet)."""
    return all(p.exists() for p in array_paths(folder, name))


def write_array(folder: str | Path, name: str, items: Iterable[tuple[np.ndarray, int, str]], count: int, size: int,
                meta: dict | None = None) -> dict:
    """Writes an array set from (image uint8 S x S x 3 RGB, label, source id) items and returns its description.

    `count` is an upper bound; the array is truncated to the number of items actually written. The .json file is
    written last, so a crash leaves no set that array_exists() accepts.
    """
    data_path, labels_path, meta_path = array_paths(folder, name)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)
    data = np.lib.format.open_memmap(str(data_path), mode="w+", dtype=np.uint8, shape=(count, size, size, 3))
    labels = np.empty(count, dtype=np.int64)
    ids: list[str] = []
    n = 0
    for image, label, source_id in items:
        if n >= count:
            raise ValueError(f"more than {count} items were given")
        if image.shape != (size, size, 3):
            raise ValueError(f"item {n} has shape {image.shape}, expected {(size, size, 3)}")
        data[n] = image
        labels[n] = label
        ids.append(source_id)
        n += 1
    data.flush()
    del data
    np.save(labels_path, labels[:n])
    description = {"name": name, "count": n, "size": size, "ids": ids, **(meta or {})}
    meta_path.write_text(json.dumps(description), encoding="utf-8")
    return description


def read_array(folder: str | Path, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (images memmap N x S x S x 3, labels, description) of an array set."""
    data_path, labels_path, meta_path = array_paths(folder, name)
    description = json.loads(meta_path.read_text(encoding="utf-8"))
    images = np.load(data_path, mmap_mode="r")[: description["count"]]   # the file may hold unused capacity
    return images, np.load(labels_path), description


class ArrayDataset(Dataset):
    """Serves (uint8 tensor [3, S, S], label) from an images array; `indices` selects a subset."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: Sequence[int] | None = None):
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        j = int(self.indices[i])
        return torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1), int(self.labels[j])


class EpochSampler(Sampler[int]):
    """A shuffle that depends only on (seed, epoch) and can start part-way, so a resumed run sees the same order."""

    def __init__(self, length: int, seed: int, epoch: int, start: int = 0, weights: np.ndarray | None = None):
        self.length, self.seed, self.epoch, self.start = length, seed, epoch, start
        self.weights = weights

    def order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        if self.weights is None:
            return rng.permutation(self.length)
        p = self.weights / self.weights.sum()
        return rng.choice(self.length, size=self.length, replace=True, p=p)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order()[self.start:].tolist())

    def __len__(self) -> int:
        return self.length - self.start


def worker_init(_worker_id: int) -> None:
    """Keeps each data-loader worker single-threaded: with 8 workers, OpenCV's and torch's own thread pools would
    otherwise oversubscribe the CPU many times over (the cause of a slow first full run)."""
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def epoch_loader(dataset: Dataset, batch_size: int, seed: int, epoch: int, start_batch: int = 0,
                 num_workers: int = 4, weights: np.ndarray | None = None, drop_last: bool = True) -> DataLoader:
    """Returns the training loader for one epoch, starting after `start_batch` batches."""
    sampler = EpochSampler(len(dataset), seed, epoch, start_batch * batch_size, weights)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last, persistent_workers=False, worker_init_fn=worker_init)


def eval_loader(dataset: Dataset, batch_size: int = 512, num_workers: int = 4) -> DataLoader:
    """Returns an ordered loader for evaluation."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                      worker_init_fn=worker_init)


def stratified_split(labels: np.ndarray, fractions: tuple[float, float, float], seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits item indices into train, validation and test with the same class proportions in each."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for cls in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == cls))
        n_train = int(round(fractions[0] * len(members)))
        n_val = int(round(fractions[1] * len(members)))
        parts[0] += members[:n_train].tolist()
        parts[1] += members[n_train:n_train + n_val].tolist()
        parts[2] += members[n_train + n_val:].tolist()
    return tuple(np.sort(np.array(p, dtype=np.int64)) for p in parts)


def grouped_split(labels: np.ndarray, groups: Sequence[str], fractions: tuple[float, float, float], seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits items into train, validation and test so that all items of one group (for example one generation
    prompt) land in the same split; groups are allocated per class (the group's most common label) by `fractions`."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(str(g), []).append(i)
    by_class: dict[int, list[str]] = {}
    for g, idx in members.items():
        cls = int(np.bincount(labels[idx]).argmax())
        by_class.setdefault(cls, []).append(g)
    parts: list[list[int]] = [[], [], []]
    for cls in sorted(by_class):
        names = sorted(by_class[cls])
        names = [names[i] for i in rng.permutation(len(names))]
        n_train = int(round(fractions[0] * len(names)))
        n_val = int(round(fractions[1] * len(names)))
        for part, chosen in enumerate((names[:n_train], names[n_train:n_train + n_val], names[n_train + n_val:])):
            for g in chosen:
                parts[part] += members[g]
    return tuple(np.sort(np.array(p, dtype=np.int64)) for p in parts)


def split_overlap(*groups: Iterable[str]) -> int:
    """Returns how many ids appear in more than one group (0 means the splits are disjoint)."""
    seen: dict[str, int] = {}
    for g, ids in enumerate(groups):
        for i in set(ids):
            seen.setdefault(i, -1)
            seen[i] = g if seen[i] in (-1, g) else -2
    return sum(1 for v in seen.values() if v == -2)


def parallel_map(fn: Callable, items: Sequence, workers: int = 8, chunksize: int = 64) -> Iterator:
    """Maps fn over items in worker processes, keeping order (used to decode images once)."""
    if workers <= 1:
        yield from map(fn, items)
        return
    import multiprocessing as mp
    import sys
    context = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    with context.Pool(workers) as pool:
        yield from pool.imap(fn, items, chunksize=chunksize)
