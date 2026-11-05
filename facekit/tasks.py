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
    """Returns the random number generator for `device`, creating it with `seed` the first time it is needed."""
    key = str(device)
    if key not in cache:
        cache[key] = torch.Generator(device=device).manual_seed(seed)
    return cache[key]


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Returns one weight per class: the inverse square root of the class frequency, scaled to a mean of 1.

    This is a common and gentle correction for class imbalance. A class with no examples is treated as if it had one.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model: nn.Module, dataset: ArrayDataset, device: torch.device, batch_size: int = 256,
                   num_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [N, C], labels [N]) for a dataset of uint8 crops. The logits are the raw class scores."""
    model.eval()
    outputs, labels = [], []
    for x, y in eval_loader(dataset, batch_size, num_workers):
        outputs.append(model(to_float(x.to(device, non_blocking=True))).float().cpu())
        labels.append(y)
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def embed(embedder: nn.Module, dataset: ArrayDataset, device: torch.device, batch_size: int = 512,
          num_workers: int = 4, flip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (L2-normalised embeddings [N, D], labels [N]) for a dataset of uint8 crops.

    L2-normalised means that every embedding is scaled to a length of 1. With `flip` set, the embedding of the
    mirrored face is added to the embedding of the original face before normalising. This is the usual test-time step
    for face recognition.
    """
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
    """Fine-tunes an ExpressionNet on aligned crops with label smoothing and class weights.

    The best epoch is the one with the highest macro-F1 (the F1 score averaged over the classes with equal weight) on
    the validation images.
    """

    monitor = "macro_f1"
    monitor_mode = "max"

    def __init__(self, model: nn.Module, images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                 val_idx: np.ndarray, num_classes: int, batch_size: int = 64, num_workers: int = 4, seed: int = 0,
                 aug: AugmentConfig = EXPRESSION_AUG, label_smoothing: float = 0.1):
        """Splits the images into a training set and a validation set by index and prepares the loss weights.

        `images` and `labels` hold the whole dataset, and `train_idx` and `val_idx` pick the rows of each split.
        `aug` sets the random image changes applied to training batches, and `label_smoothing` softens the targets
        of the loss.
        """
        self.model = model
        self.num_classes = num_classes
        self.train_set = ArrayDataset(images, labels, train_idx)
        self.val_set = ArrayDataset(images, labels, val_idx)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed

        # The number of full batches in one epoch. A final partial batch is not counted.
        self.steps_per_epoch = len(self.train_set) // batch_size
        self.aug = aug
        self.label_smoothing = label_smoothing

        # Classes with fewer training images get a larger weight in the loss.
        self.weights = class_weights(labels[train_idx], num_classes)
        self._generators: dict = {}

    def train_batches(self, epoch: int, start_batch: int):
        """Returns the training batches of an epoch, starting at `start_batch` so that a run can resume."""
        return epoch_loader(self.train_set, self.batch_size, self.seed, epoch, start_batch, self.num_workers)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the class-weighted cross-entropy loss with label smoothing, computed in float32."""
        return F.cross_entropy(logits.float(), y, weight=self.weights.to(logits.device),
                               label_smoothing=self.label_smoothing)

    def training_step(self, batch, device: torch.device):
        """Applies the random image changes to one batch and runs it through the model.

        Returns (loss, {"acc": accuracy}). The accuracy is the share of the batch that the model classifies correctly.
        """
        x, y = batch
        x = augment_batch(x.to(device, non_blocking=True), self.aug, _generator(self._generators, device, self.seed))
        y = y.to(device, non_blocking=True)
        logits = self.model(x)
        return self.loss(logits, y), {"acc": (logits.argmax(1) == y).float().mean()}

    def validate(self, device: torch.device) -> dict[str, float]:
        """Returns the validation loss, accuracy and macro-F1."""
        # Mixed precision is switched off, so that validation runs at the same precision as the final test.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            logits, y = predict_logits(self.model, self.val_set, device, 256, self.num_workers)
        pred = logits.argmax(1)
        loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item()

        # Only the point values are used here, so a single bootstrap round (one resampling of the validation set, used
        # for confidence intervals) keeps the report fast.
        report = metrics.classification_report(y, pred, self.num_classes, n_boot=1)
        return {"loss": loss, "accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}


class IdentityTask:
    """Trains an IdentityTrainer (the embedding network and its margin head) on identity labels.

    The best epoch is chosen on identities that were held out of training. The score is the TAR (true accept rate) at
    a FAR (false accept rate) of 1e-3 on pairs of validation images, which means the share of same-person pairs that
    are accepted when only one in a thousand different-person pairs is accepted.
    """

    monitor = "tar@far=0.001"
    monitor_mode = "max"

    def __init__(self, model: nn.Module, images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                 val_idx: np.ndarray, batch_size: int = 256, num_workers: int = 6, seed: int = 0,
                 aug: AugmentConfig = IDENTITY_AUG, n_genuine: int = 3000, n_impostor: int = 30000):
        """Splits the images into a training set and a validation set and samples the validation pairs once.

        `n_genuine` and `n_impostor` are the numbers of same-person and different-person pairs drawn from the
        validation images. The pairs stay fixed, so the validation scores are comparable between epochs.
        """
        self.model = model
        self.train_set = ArrayDataset(images, labels, train_idx)
        self.val_set = ArrayDataset(images, labels, val_idx)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed

        # The number of full batches in one epoch. A final partial batch is not counted.
        self.steps_per_epoch = len(self.train_set) // batch_size
        self.aug = aug
        self.val_pairs, self.val_same = metrics.verification_pairs(labels[val_idx], n_genuine, n_impostor, seed)
        self._generators: dict = {}

    def train_batches(self, epoch: int, start_batch: int):
        """Returns the training batches of an epoch, starting at `start_batch` so that a run can resume."""
        return epoch_loader(self.train_set, self.batch_size, self.seed, epoch, start_batch, self.num_workers)

    def training_step(self, batch, device: torch.device):
        """Applies the random image changes to one batch and runs it through the trainer.

        Returns (loss, {"acc": accuracy}). The accuracy is the share of the batch whose nearest class centre is correct.
        """
        x, y = batch
        x = augment_batch(x.to(device, non_blocking=True), self.aug, _generator(self._generators, device, self.seed))
        loss, accuracy = self.model(x, y.to(device, non_blocking=True))
        return loss, {"acc": accuracy}

    def validate(self, device: torch.device) -> dict[str, float]:
        """Returns the verification metrics of the validation pairs, which include the monitored TAR at FAR 1e-3."""
        # Mixed precision is switched off, so that the similarity scores behind the 1e-3 false-accept threshold are
        # not rounded.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            embeddings, _ = embed(self.model.embedder, self.val_set, device, 512, self.num_workers)
        scores = metrics.cosine_scores(embeddings, self.val_pairs)
        report = metrics.verification_report(scores, self.val_same, fars=(1e-3, 1e-4))
        return {k: report[k] for k in ("auc", "eer", "tar@far=0.001", "tar@far=0.0001", "threshold@far=0.001")}
