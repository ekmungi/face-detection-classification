"""Offline tests for facekit.det_eval: average precision, size buckets, matching and landmark error."""
from __future__ import annotations

import math

import numpy as np
import torch

from facekit.det_eval import average_precision, landmark_nme, match_boxes, size_buckets

FACE_A = [10.0, 10.0, 50.0, 50.0]
FACE_B = [100.0, 100.0, 160.0, 170.0]
FAR_AWAY = [300.0, 300.0, 340.0, 340.0]


def preds(boxes: list[list[float]], scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(scores, dtype=np.float32)


def test_perfect_predictions_give_ap_one() -> None:
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}, {"boxes": np.array([FACE_B])}]
    predictions = [preds([FACE_A, FACE_B], [0.9, 0.8]), (torch.tensor([FACE_B]), torch.tensor([0.7]))]
    assert average_precision(predictions, ground_truths) == 1.0


def test_hand_computed_half_with_a_missed_face() -> None:
    # Two faces; the top prediction finds one, the second is a false positive: recall 0.5 at precision 1.
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}]
    predictions = [preds([FACE_A, FAR_AWAY], [0.9, 0.8])]
    assert math.isclose(average_precision(predictions, ground_truths), 0.5)


def test_hand_computed_half_with_a_false_positive_first() -> None:
    # One face; a higher-scored false positive comes first, so full recall is reached at precision 0.5.
    ground_truths = [{"boxes": np.array([FACE_A])}, {"boxes": np.zeros((0, 4))}]
    predictions = [preds([FACE_A], [0.8]), preds([FAR_AWAY], [0.9])]
    assert math.isclose(average_precision(predictions, ground_truths), 0.5)


def test_duplicate_detection_is_a_false_positive() -> None:
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}]
    predictions = [preds([FACE_A, FACE_A, FACE_B], [0.9, 0.8, 0.7])]
    # Precision 1, 0.5, 0.67 at recall 0.5, 0.5, 1: AP = 0.5 * 1 + 0.5 * 2/3.
    assert math.isclose(average_precision(predictions, ground_truths), 0.5 + 0.5 * 2 / 3)


def test_ignored_faces_are_neither_positives_nor_false_positives() -> None:
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B]), "ignore": np.array([False, True])}]
    # A missed ignored face does not lower recall.
    assert average_precision([preds([FACE_A], [0.9])], ground_truths) == 1.0
    # A top-scored detection on the ignored face is not a false positive.
    assert average_precision([preds([FACE_B, FACE_A], [0.95, 0.9])], ground_truths) == 1.0
    # Without the ignore flag the same miss costs half the recall.
    assert math.isclose(average_precision([preds([FACE_A], [0.9])], [{"boxes": ground_truths[0]["boxes"]}]), 0.5)


def test_ap_is_nan_without_positive_faces() -> None:
    assert math.isnan(average_precision([preds([FACE_A], [0.9])], [{"boxes": np.zeros((0, 4))}]))


def test_size_buckets_mark_small_faces_ignored_without_mutating() -> None:
    small, large, large_ignored = [0, 0, 10, 40], [0, 0, 40, 40], [50, 50, 100, 100]
    ground_truths = [{"boxes": np.array([small, large, large_ignored]), "ignore": np.array([False, False, True])}]
    bucketed = size_buckets(ground_truths, min_side=16)
    assert bucketed[0]["ignore"].tolist() == [True, False, True]
    assert ground_truths[0]["ignore"].tolist() == [False, False, True]
    assert size_buckets([{"boxes": np.array([small, large])}], 16)[0]["ignore"].tolist() == [True, False]


def test_match_boxes_pairs_by_overlap() -> None:
    pred_index, gt_index = match_boxes(np.array([FACE_B, FAR_AWAY, FACE_A]), np.array([FACE_A, FACE_B]))
    assert pred_index.tolist() == [0, 2] and gt_index.tolist() == [1, 0]


def test_landmark_nme_normalises_by_box_and_skips_nan() -> None:
    gt = np.tile(np.array([[20.0, 20.0]]), (5, 1))[None]
    pred = gt + np.array([3.0, 4.0])  # error 5 everywhere
    box = np.array([[0.0, 0.0, 10.0, 10.0]])  # sqrt(w * h) = 10
    assert math.isclose(landmark_nme(pred, gt, box), 0.5)
    gt_partial = gt.copy()
    gt_partial[0, 1:] = np.nan
    pred_partial = pred.copy()
    pred_partial[0, 1:] += 100.0  # would raise the error if the NaN points were counted
    assert math.isclose(landmark_nme(pred_partial, gt_partial, box), 0.5)
    assert math.isnan(landmark_nme(pred, np.full_like(gt, np.nan), box))
