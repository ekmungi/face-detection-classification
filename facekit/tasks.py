"""Training tasks for the expression and identity models: batches, one training step, and validation.

Each task plugs into facekit.engine.fit. The detector task lives in facekit.det_task because its data and loss differ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from facekit import metrics
from facekit.arrays import ArrayDataset, epoch_loader, eval_loader
from facekit.augment import EXPRESSION_AUG, IDENTITY_AUG, AugmentConfig, augment_batch, to_float


def _generator(cache: dict, device: torch.device, seed: int) -> torch.Generator:
    key = str(device)
    if key not in cache:
        cache[key] = torch.Generator(device=device).manual_seed(seed)
    return cache[key]


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-square-root class frequency, scaled to mean 1: a common, gentle correction for class imbalance."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model: nn.Module, dataset: ArrayDataset, device: torch.device, batch_size: int = 256,
                   num_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [N, C], labels [N]) for a dataset of uint8 crops."""
    model.eval()
    outputs, labels = [], []
    for x, y in eval_loader(dataset, batch_size, num_workers):
        outputs.append(model(to_float(x.to(device, non_blocking=True))).float().cpu())
        labels.append(y)
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def embed(embedder: nn.Module, dataset: ArrayDataset, device: torch.device, batch_size: int = 512,
          num_workers: int = 4, flip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (L2-normalised embeddings [N, D], labels [N]); with flip, the embedding of the mirrored face is added,
    the usual test-time step for face recognition."""
    embedder.eval()
    outputs, labels = [], []
    for x, y in eval_loader(dataset, batch_size, num_workers):
        x = to_float(x.to(device, non_blocking=True))
        e = embedder(x).float()
        if flip:
            e = e + embedder(torch.flip(x, dims=[3])).float()
        outputs.append(F.normalize(e, dim=1).cpu())
        labels.append(y)
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy()


class ExpressionTask:
    """Fine-tunes an ExpressionNet on aligned crops with label smoothing and class weights; picks the best macro-F1."""

    monitor = "macro_f1"
    monitor_mode = "max"

    def __init__(self, model: nn.Module, images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                 val_idx: np.ndarray, num_classes: int, batch_size: int = 64, num_workers: int = 4, seed: int = 0,
                 aug: AugmentConfig = EXPRESSION_AUG, label_smoothing: float = 0.1):
        self.model = model
        self.num_classes = num_classes
        self.train_set = ArrayDataset(images, labels, train_idx)
        self.val_set = ArrayDataset(images, labels, val_idx)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.steps_per_epoch = len(self.train_set) // batch_size
        self.aug = aug
        self.label_smoothing = label_smoothing
        self.weights = class_weights(labels[train_idx], num_classes)
        self._generators: dict = {}

    def train_batches(self, epoch: int, start_batch: int):
        return epoch_loader(self.train_set, self.batch_size, self.seed, epoch, start_batch, self.num_workers)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), y, weight=self.weights.to(logits.device),
                               label_smoothing=self.label_smoothing)

    def training_step(self, batch, device: torch.device):
        x, y = batch
        x = augment_batch(x.to(device, non_blocking=True), self.aug, _generator(self._generators, device, self.seed))
        y = y.to(device, non_blocking=True)
        logits = self.model(x)
        return self.loss(logits, y), {"acc": (logits.argmax(1) == y).float().mean()}

    def validate(self, device: torch.device) -> dict[str, float]:
        with torch.autocast(device_type=torch.device(device).type, enabled=False):   # same precision as the test
            logits, y = predict_logits(self.model, self.val_set, device, 256, self.num_workers)
        pred = logits.argmax(1)
        loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item()
        report = metrics.classification_report(y, pred, self.num_classes, n_boot=1)
        return {"loss": loss, "accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}


class IdentityTask:
    """Trains an IdentityTrainer (embedder + ArcFace head) and selects on held-out identities' TAR at FAR 1e-3."""

    monitor = "tar@far=0.001"
    monitor_mode = "max"

    def __init__(self, model: nn.Module, images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                 val_idx: np.ndarray, batch_size: int = 256, num_workers: int = 6, seed: int = 0,
                 aug: AugmentConfig = IDENTITY_AUG, n_genuine: int = 3000, n_impostor: int = 30000):
        self.model = model
        self.train_set = ArrayDataset(images, labels, train_idx)
        self.val_set = ArrayDataset(images, labels, val_idx)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.steps_per_epoch = len(self.train_set) // batch_size
        self.aug = aug
        self.val_pairs, self.val_same = metrics.verification_pairs(labels[val_idx], n_genuine, n_impostor, seed)
        self._generators: dict = {}

    def train_batches(self, epoch: int, start_batch: int):
        return epoch_loader(self.train_set, self.batch_size, self.seed, epoch, start_batch, self.num_workers)

    def training_step(self, batch, device: torch.device):
        x, y = batch
        x = augment_batch(x.to(device, non_blocking=True), self.aug, _generator(self._generators, device, self.seed))
        loss, accuracy = self.model(x, y.to(device, non_blocking=True))
        return loss, {"acc": accuracy}

    def validate(self, device: torch.device) -> dict[str, float]:
        # Full precision, so the similarity scores behind the 1e-3 false-accept threshold are not rounded.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            embeddings, _ = embed(self.model.embedder, self.val_set, device, 512, self.num_workers)
        scores = metrics.cosine_scores(embeddings, self.val_pairs)
        report = metrics.verification_report(scores, self.val_same, fars=(1e-3, 1e-4))
        return {k: report[k] for k in ("auc", "eer", "tar@far=0.001", "tar@far=0.0001", "threshold@far=0.001")}
