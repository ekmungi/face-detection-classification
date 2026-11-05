"""Offline tests for facekit.det_eval, which scores face detections and facial landmarks.

The tests cover four things. Average precision (AP, a single score from 0 to 1 that rewards finding every face with
confident, correct boxes) is checked against values worked out by hand. Size buckets mark very small faces as ignored.
Box matching pairs predicted boxes with true boxes by their overlap. The landmark error (NME, the normalised mean
error, which is the distance between predicted and true landmarks divided by the face size) is checked on a case with
a known answer.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from facekit.det_eval import average_precision, landmark_nme, match_boxes, size_buckets

# Three boxes written as [left, top, right, bottom] in pixels. FAR_AWAY overlaps neither of the two faces.
FACE_A = [10.0, 10.0, 50.0, 50.0]
FACE_B = [100.0, 100.0, 160.0, 170.0]
FAR_AWAY = [300.0, 300.0, 340.0, 340.0]


def preds(boxes: list[list[float]], scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns the predicted boxes and their scores of one image as the pair of float arrays that the scorer expects."""
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(scores, dtype=np.float32)


def test_perfect_predictions_give_ap_one() -> None:
    # The second image passes its prediction as torch tensors, because the scorer accepts both kinds of array.
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}, {"boxes": np.array([FACE_B])}]
    predictions = [preds([FACE_A, FACE_B], [0.9, 0.8]), (torch.tensor([FACE_B]), torch.tensor([0.7]))]
    assert average_precision(predictions, ground_truths) == 1.0


def test_hand_computed_half_with_a_missed_face() -> None:
    # There are two faces. The top prediction finds one and the second prediction is a false positive. Recall is
    # therefore 0.5 at a precision of 1.
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}]
    predictions = [preds([FACE_A, FAR_AWAY], [0.9, 0.8])]
    assert math.isclose(average_precision(predictions, ground_truths), 0.5)


def test_hand_computed_half_with_a_false_positive_first() -> None:
    # There is one face. A false positive with a higher score comes first, so full recall is reached only at a
    # precision of 0.5.
    ground_truths = [{"boxes": np.array([FACE_A])}, {"boxes": np.zeros((0, 4))}]
    predictions = [preds([FACE_A], [0.8]), preds([FAR_AWAY], [0.9])]
    assert math.isclose(average_precision(predictions, ground_truths), 0.5)


def test_duplicate_detection_is_a_false_positive() -> None:
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B])}]
    predictions = [preds([FACE_A, FACE_A, FACE_B], [0.9, 0.8, 0.7])]

    # The precision values are 1, 0.5 and 0.67 at the recall values 0.5, 0.5 and 1. The AP is therefore
    # 0.5 * 1 + 0.5 * 2/3.
    assert math.isclose(average_precision(predictions, ground_truths), 0.5 + 0.5 * 2 / 3)


def test_ignored_faces_are_neither_positives_nor_false_positives() -> None:
    ground_truths = [{"boxes": np.array([FACE_A, FACE_B]), "ignore": np.array([False, True])}]

    # A missed ignored face does not lower the recall.
    assert average_precision([preds([FACE_A], [0.9])], ground_truths) == 1.0

    # A top-scored detection on the ignored face is not counted as a false positive.
    assert average_precision([preds([FACE_B, FACE_A], [0.95, 0.9])], ground_truths) == 1.0

    # Without the ignore flag, the same miss costs half of the recall.
    assert math.isclose(average_precision([preds([FACE_A], [0.9])], [{"boxes": ground_truths[0]["boxes"]}]), 0.5)


def test_ap_is_nan_without_positive_faces() -> None:
    assert math.isnan(average_precision([preds([FACE_A], [0.9])], [{"boxes": np.zeros((0, 4))}]))


def test_size_buckets_mark_small_faces_ignored_without_mutating() -> None:
    # The shorter side of the first box is 10 pixels, which is below the minimum side of 16 pixels. The third box is
    # already marked as ignored.
    small, large, large_ignored = [0, 0, 10, 40], [0, 0, 40, 40], [50, 50, 100, 100]
    ground_truths = [{"boxes": np.array([small, large, large_ignored]), "ignore": np.array([False, False, True])}]
    bucketed = size_buckets(ground_truths, min_side=16)
    assert bucketed[0]["ignore"].tolist() == [True, False, True]

    # The input must stay unchanged, because the function returns copies.
    assert ground_truths[0]["ignore"].tolist() == [False, False, True]

    # A ground truth without an ignore flag gets one.
    assert size_buckets([{"boxes": np.array([small, large])}], 16)[0]["ignore"].tolist() == [True, False]


def test_match_boxes_pairs_by_overlap() -> None:
    # Prediction 0 overlaps true face 1, prediction 2 overlaps true face 0, and prediction 1 overlaps nothing.
    pred_index, gt_index = match_boxes(np.array([FACE_B, FAR_AWAY, FACE_A]), np.array([FACE_A, FACE_B]))
    assert pred_index.tolist() == [0, 2] and gt_index.tolist() == [1, 0]


def test_landmark_nme_normalises_by_box_and_skips_nan() -> None:
    # The five true landmarks all sit at the same point. Every predicted landmark is off by 3 pixels in x and 4 pixels
    # in y, which is a distance of 5. The box is 10 by 10, so its size sqrt(w * h) is 10 and the error is 5 / 10 = 0.5.
    gt = np.tile(np.array([[20.0, 20.0]]), (5, 1))[None]
    pred = gt + np.array([3.0, 4.0])
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert math.isclose(landmark_nme(pred, gt, box), 0.5)

    # Four true landmarks are missing (NaN). Their predictions are moved far away, which would raise the error if the
    # missing points were counted. The result must stay the same.
    gt_partial = gt.copy()
    gt_partial[0, 1:] = np.nan
    pred_partial = pred.copy()
    pred_partial[0, 1:] += 100.0
    assert math.isclose(landmark_nme(pred_partial, gt_partial, box), 0.5)

    # With no true landmark at all, there is nothing to score and the result is NaN.
    assert math.isnan(landmark_nme(pred, np.full_like(gt, np.nan), box))
