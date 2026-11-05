"""Evaluates the face detector with average precision, face size buckets and landmark error.

Boxes are given as (x1, y1, x2, y2) pixel coordinates. A ground truth is a dict with ``boxes`` [N, 4] and an optional
``ignore`` array [N] of true-or-false flags. An ignored face, for example a WIDER FACE ``invalid`` face or a face below
a size bucket, does not count as a positive and does not penalise a detection that lands on it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor | Sequence


def _to_numpy(values: ArrayLike, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    """Converts a tensor or array-like to a float64 numpy array with the given trailing shape."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1, *shape_tail)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Returns the [A, B] matrix of IoU values for two sets of (x1, y1, x2, y2) boxes.

    IoU (intersection over union) is the overlapping area of two boxes divided by the area that they cover together.
    It is 0 for boxes that do not touch and 1 for identical boxes.
    """
    # The overlap of every pair of boxes. Pairs that do not overlap get an area of zero.
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)

    # The union is the sum of the two box areas minus the overlap, which would otherwise be counted twice.
    area_a = np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def _ignore_flags(ground_truth: dict, count: int) -> np.ndarray:
    """Returns the ground truth's ignore flags, or all False when it has none."""
    ignore = ground_truth.get("ignore")
    if ignore is None:
        return np.zeros(count, dtype=bool)
    if isinstance(ignore, torch.Tensor):
        ignore = ignore.detach().cpu().numpy()
    return np.asarray(ignore, dtype=bool).reshape(count)


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Returns the all-point interpolated area under the precision-recall curve, as in Pascal VOC 2010 and later."""
    # Add a start point and an end point, then make the precision never rise as the recall grows.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Sum the area of the rectangles under the curve at every point where the recall changes.
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def _match_outcomes(
    predictions: Sequence[tuple[ArrayLike, ArrayLike]], ground_truths: Sequence[dict], iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Matches the predictions of a whole dataset to its ground truth with the greedy Pascal VOC rule.

    The predictions are visited from the highest score to the lowest across all images.

    Returns the scores and the outcomes of all predictions in score order, and the number of ground-truth faces that
    are not ignored. An outcome is 1 for a true positive, 0 for a false positive and -1 for a prediction that landed on
    an ignored face.
    """
    # Collect the overlaps between every prediction and every ground-truth face, image by image. Also build one list
    # of (score, image, prediction index) entries so that the predictions of all images can be sorted together.
    entries, ious, gts_ignore, matched = [], [], [], []
    num_positive = 0
    for image, ((pred_boxes, pred_scores), ground_truth) in enumerate(zip(predictions, ground_truths)):
        boxes = _to_numpy(pred_boxes, (4,))
        scores = _to_numpy(pred_scores)
        gt_boxes = _to_numpy(ground_truth["boxes"], (4,))
        ignore = _ignore_flags(ground_truth, len(gt_boxes))
        num_positive += int((~ignore).sum())
        ious.append(box_iou(boxes, gt_boxes))
        gts_ignore.append(ignore)
        matched.append(np.zeros(len(gt_boxes), dtype=bool))
        entries.extend((float(score), image, k) for k, score in enumerate(scores))
    entries.sort(key=lambda entry: -entry[0])

    # Visit the predictions from the highest score down. A prediction that overlaps no ground-truth face by at least
    # the threshold is a false positive. Otherwise it is a true positive if its best-overlapping face is free, a false
    # positive if that face was already matched by a higher-scoring prediction, and ignored if that face is ignored.
    outcomes = np.zeros(len(entries), dtype=np.int8)
    for position, (_, image, k) in enumerate(entries):
        overlaps = ious[image][k]
        if overlaps.size == 0 or overlaps.max() < iou_threshold:
            continue
        best = int(overlaps.argmax())
        if gts_ignore[image][best]:
            outcomes[position] = -1
        elif not matched[image][best]:
            matched[image][best] = True
            outcomes[position] = 1
    scores = np.array([entry[0] for entry in entries], dtype=np.float64)
    return scores, outcomes, num_positive


def average_precision(
    predictions: Sequence[tuple[ArrayLike, ArrayLike]], ground_truths: Sequence[dict], iou_threshold: float = 0.5
) -> float:
    """Returns the Pascal VOC style average precision over a dataset.

    ``predictions`` holds one entry per image, which is a pair of boxes [K, 4] and scores [K]. ``ground_truths`` holds
    one dict per image, with ``boxes`` [N, 4] and an optional ``ignore`` array [N].

    Each prediction, taken in descending score order over the whole dataset, is compared with the ground-truth face that
    it overlaps most. When the overlap reaches ``iou_threshold``, the prediction is a true positive if that face is not
    yet matched, a false positive if the face is already matched, and neither if the face is ignored. When the overlap
    stays below ``iou_threshold``, the prediction is a false positive.

    Returns NaN when there are no faces that are not ignored. Raises ValueError when the two lists differ in length.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"{len(predictions)} prediction lists for {len(ground_truths)} images")
    _, outcomes, num_positive = _match_outcomes(predictions, ground_truths, iou_threshold)
    if num_positive == 0:
        return float("nan")

    # Build the precision-recall curve from the predictions that are not ignored, in score order, and measure the area
    # under it.
    counted = outcomes[outcomes >= 0]
    true_positives = np.cumsum(counted == 1)
    false_positives = np.cumsum(counted == 0)
    recall = true_positives / num_positive
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    return _interpolated_ap(recall, precision)


def size_buckets(ground_truths: Sequence[dict], min_side: float) -> list[dict]:
    """Returns copies of the ground truths in which faces whose shorter side is below ``min_side`` are also ignored."""
    bucketed = []
    for ground_truth in ground_truths:
        boxes = _to_numpy(ground_truth["boxes"], (4,))
        shorter = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        ignore = _ignore_flags(ground_truth, len(boxes)) | (shorter < min_side)
        bucketed.append({**ground_truth, "boxes": boxes.copy(), "ignore": ignore})
    return bucketed


def match_boxes(
    pred_boxes: ArrayLike, gt_boxes: ArrayLike, iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs predicted boxes with ground-truth boxes and returns (prediction indices, ground-truth indices).

    The predictions are visited in the order given, so pass them sorted by score. Each prediction takes the
    ground-truth box that it overlaps most among those not yet taken, as long as the IoU reaches `iou_threshold`.
    The pairs are useful for picking the landmarks that :func:`landmark_nme` compares.
    """
    iou = box_iou(_to_numpy(pred_boxes, (4,)), _to_numpy(gt_boxes, (4,)))
    pred_index, gt_index, taken = [], [], set()
    for k in range(iou.shape[0]):
        candidates = [(value, n) for n, value in enumerate(iou[k]) if n not in taken and value >= iou_threshold]
        if candidates:
            _, best = max(candidates)
            taken.add(best)
            pred_index.append(k)
            gt_index.append(best)
    return np.array(pred_index, dtype=np.int64), np.array(gt_index, dtype=np.int64)


def landmark_nme(pred_landmarks: ArrayLike, gt_landmarks: ArrayLike, boxes: ArrayLike) -> float:
    """Returns the normalised mean error (NME) of the predicted landmarks, ignoring missing ground truth.

    Each landmark error is the distance to the ground-truth point, divided by sqrt(width * height) of the
    ground-truth box of that face. Ground-truth landmarks that are NaN are skipped.

    All three inputs are matched pairs: the predicted landmarks [N, 5, 2], the ground-truth landmarks [N, 5, 2] and
    the ground-truth boxes [N, 4]. Returns NaN when no ground-truth landmark is annotated.
    """
    pred = _to_numpy(pred_landmarks, (5, 2))
    gt = _to_numpy(gt_landmarks, (5, 2))
    gt_boxes = _to_numpy(boxes, (4,))

    # The size of each face, used to make the error independent of how large the face is in the image.
    sides = np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0, None)
    norm = np.sqrt(sides[:, 0] * sides[:, 1])
    errors = np.linalg.norm(pred - gt, axis=2) / np.maximum(norm, 1e-12)[:, None]

    # Average only over the landmarks that have a ground-truth position.
    valid = np.isfinite(gt).all(axis=2)
    if not valid.any():
        return float("nan")
    return float(errors[valid].mean())
