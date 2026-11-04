"""The detector training task for facekit.engine.fit, and the detector's evaluation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from facekit import det_eval
from facekit.arrays import EpochSampler
from facekit.det_data import WiderFaceTrain, detection_collate, read_rgb
from facekit.det_model import FaceDetNet, decode, detect_image, detection_loss


def split_records(records: list[dict], val_fraction: float = 0.05, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Splits WIDER train records into (train, validation) by image, grouped by event folder share, with a fixed seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_val = int(round(val_fraction * len(records)))
    val = [records[i] for i in sorted(order[:n_val])]
    train = [records[i] for i in sorted(order[n_val:])]
    return train, val


class DetectorTask:
    """Trains FaceDetNet on augmented WIDER FACE crops; validates with AP at IoU 0.5 on held-out training images."""

    monitor = "ap"
    monitor_mode = "max"

    def __init__(self, model: FaceDetNet, train_records: list[dict], val_records: list[dict], image_root: str | Path,
                 size: int = 640, batch_size: int = 32, num_workers: int = 6, seed: int = 0,
                 val_images: int = 300, val_max_side: int = 640):
        self.model = model
        self.image_root = Path(image_root)
        self.train_set = WiderFaceTrain(train_records, image_root, size=size, augment=True, seed=seed)
        self.val_records = val_records[:val_images]
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.steps_per_epoch = len(self.train_set) // batch_size
        self.val_max_side = val_max_side

    def train_batches(self, epoch: int, start_batch: int):
        sampler = EpochSampler(len(self.train_set), self.seed, epoch, start_batch * self.batch_size)
        return DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
                          collate_fn=detection_collate, pin_memory=True, drop_last=True)

    def training_step(self, batch, device: torch.device):
        images, targets = batch
        outputs = self.model(images.to(device, non_blocking=True).float().div_(255.0))
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        loss, stats = detection_loss(outputs, targets)
        return loss, {k: v for k, v in stats.items() if k != "total"}

    def validate(self, device: torch.device) -> dict[str, float]:
        # Full precision: box coordinates in bfloat16 would be off by several pixels on large images.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            predictions, truths = predict_records(self.model, self.val_records, self.image_root, device,
                                                  max_side=self.val_max_side)
        return {"ap": det_eval.average_precision(predictions, truths)}


def predict_records(model: FaceDetNet, records: list[dict], image_root: Path, device, max_side: int = 640,
                    score_threshold: float = 0.02) -> tuple[list, list]:
    """Runs the detector on each record's image and returns (predictions, ground truths) for det_eval."""
    predictions, truths = [], []
    for record in records:
        image = read_rgb(Path(image_root) / record["path"])
        faces = detect_image(model, image, device=device, max_side=max_side, score_threshold=score_threshold)
        boxes = np.array([f["box"] for f in faces], dtype=np.float32).reshape(-1, 4)
        scores = np.array([f["score"] for f in faces], dtype=np.float32)
        predictions.append((boxes, scores))
        truths.append({"boxes": np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)})
    return predictions, truths


def choose_score_threshold(predictions: list, truths: list, iou: float = 0.5) -> dict:
    """Picks the detection score threshold that maximises F1 on validation images (never on the test set)."""
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    n_true = sum(len(t["boxes"]) for t in truths)
    for threshold in np.arange(0.2, 0.9, 0.05):
        tp = fp = 0
        for (boxes, scores), truth in zip(predictions, truths):
            keep = scores >= threshold
            matched = det_eval.match_boxes(boxes[keep], truth["boxes"], iou) if keep.any() else []
            n_match = len(matched[0]) if isinstance(matched, tuple) else len(matched)
            tp += n_match
            fp += int(keep.sum()) - n_match
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, n_true)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": float(round(threshold, 2)), "f1": f1, "precision": precision, "recall": recall}
    return best


@torch.no_grad()
def raw_outputs(model: FaceDetNet, images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Runs the network on a float batch in [0, 1] and returns its four output maps (used by the ONNX parity check)."""
    model.eval()
    return model(images)


__all__ = ["DetectorTask", "split_records", "predict_records", "choose_score_threshold", "raw_outputs", "decode"]
