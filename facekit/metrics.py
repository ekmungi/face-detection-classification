"""Evaluation metrics for classification, face verification and face identification.

Classification is reported with bootstrap confidence intervals (ranges estimated by resampling the test set).
Verification asks whether two photos show the same person, and is reported with a ROC curve (the trade-off between
accepting true matches and accepting false ones). Identification asks which person in a gallery a probe photo shows.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Returns the matrix of counts [true class, predicted class]."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, n_boot: int = 1000,
                          seed: int = 0) -> dict:
    """Returns accuracy and macro-F1 with 95% bootstrap intervals, per-class recall, and the confusion matrix.

    Macro-F1 is the F1 score (a balance of precision and recall) averaged over the classes with equal weight. Each
    bootstrap round draws a new test set of the same size, with replacement, and the interval runs from the 2.5th to
    the 97.5th percentile of the rounds. `n_boot` is the number of rounds, and `seed` makes the draws repeatable.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    # Recompute both scores on resampled test sets to estimate how much they would vary.
    rng = np.random.default_rng(seed)
    boot_acc, boot_f1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        boot_acc.append((y_true[idx] == y_pred[idx]).mean())
        boot_f1.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))

    # Recall of a class is the share of its images that were predicted correctly.
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    recall = matrix.diagonal() / np.maximum(1, matrix.sum(1))
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "accuracy_ci95": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        "macro_f1": macro_f1,
        "macro_f1_ci95": [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))],
        "per_class_recall": [float(r) for r in recall],
        "confusion_matrix": matrix.tolist(),
    }


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    """Returns the expected calibration error of the top-class probability (0 is perfectly calibrated).

    The predictions are grouped into `bins` equal-width bins by confidence. In each bin the gap between the average
    confidence and the actual accuracy is weighted by the share of predictions in that bin, and the weighted gaps are
    added up.
    """
    confidence = probs.max(1)
    correct = probs.argmax(1) == np.asarray(y_true)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def verification_pairs(labels: np.ndarray, n_genuine: int, n_impostor: int, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Samples pairs of image indices and returns (pairs [P, 2], same [P]).

    A genuine pair shows the same identity and an impostor pair shows two different identities. `same` is 1 for a
    genuine pair and 0 for an impostor pair, and the genuine pairs come first. Only identities with at least two images
    can form a genuine pair.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    # Group the image indices by identity, keeping only identities that have at least two images.
    by_id: dict[int, np.ndarray] = {}
    for identity in np.unique(labels):
        members = np.flatnonzero(labels == identity)
        if len(members) >= 2:
            by_id[int(identity)] = members
    ids = np.array(sorted(by_id))

    # Genuine pairs. Pick an identity at random, then two different images of it.
    genuine = []
    for _ in range(n_genuine):
        members = by_id[int(rng.choice(ids))]
        a, b = rng.choice(members, 2, replace=False)
        genuine.append((a, b))

    # Impostor pairs. Draw random pairs of images and keep those with different identities.
    impostor = []
    while len(impostor) < n_impostor:
        a, b = rng.integers(0, len(labels), 2)
        if labels[a] != labels[b]:
            impostor.append((a, b))
    pairs = np.array(genuine + impostor, dtype=np.int64)
    same = np.array([1] * len(genuine) + [0] * len(impostor), dtype=np.int64)
    return pairs, same


def cosine_scores(embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Returns the cosine similarity of each pair of rows."""
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    return np.sum(e[pairs[:, 0]] * e[pairs[:, 1]], axis=1)


def verification_report(scores: np.ndarray, same: np.ndarray, fars: tuple[float, ...] = (1e-2, 1e-3, 1e-4)) -> dict:
    """Returns ROC AUC, equal error rate, TAR and the threshold at each FAR, and the best-accuracy threshold.

    AUC is the area under the ROC curve. The equal error rate (EER) is the error at the point where the false accept
    rate equals the false reject rate. TAR (true accept rate) is the share of genuine pairs that are accepted, and FAR
    (false accept rate) is the share of impostor pairs that are accepted. For each requested FAR, the report holds the
    TAR at the loosest threshold that keeps the FAR within that limit, together with that threshold. The best
    threshold is the one with the highest balanced accuracy, which weights genuine and impostor pairs equally.
    """
    # Keep every point of the ROC curve, because the EER needs the exact point where the two error rates cross.
    fpr, tpr, thresholds = roc_curve(same, scores, drop_intermediate=False)
    report: dict = {"auc": float(roc_auc_score(same, scores)), "n_pairs": int(len(same))}
    eer_index = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    report["eer"] = float((fpr[eer_index] + 1 - tpr[eer_index]) / 2)

    # For each FAR limit, find the last ROC point whose FPR (false positive rate, the same as the FAR) stays within
    # it. Index 0 is skipped because it is scikit-learn's "reject everything" threshold. When no point stays within
    # the limit, the strictest real threshold at index 1 is reported instead.
    for far in fars:
        ok = np.flatnonzero(fpr[1:] <= far) + 1
        i = int(ok[-1]) if len(ok) else 1
        report[f"tar@far={far:g}"] = float(tpr[i])
        report[f"threshold@far={far:g}"] = float(min(thresholds[i], 1.0))

    # The threshold with the best balanced accuracy, which is the average of the TAR and the true reject rate.
    balanced = 0.5 * (tpr[1:] + 1.0 - fpr[1:])
    best = int(np.argmax(balanced))
    report["best_balanced_accuracy"] = float(balanced[best])
    report["best_balanced_accuracy_threshold"] = float(thresholds[1:][best])

    # The ROC curve is thinned to every n-th point, which leaves between 400 and 800 points of a long curve. That is
    # enough for a plot and keeps the report small.
    report["roc"] = {"fpr": fpr[:: max(1, len(fpr) // 400)].tolist(), "tpr": tpr[:: max(1, len(tpr) // 400)].tolist()}
    return report


def identification_report(gallery: np.ndarray, gallery_ids: np.ndarray, probes: np.ndarray, probe_ids: np.ndarray,
                          threshold: float | None = None) -> dict:
    """Closed-set rank-1 and rank-5 accuracy of probes against a gallery, and the open-set rate at `threshold`.

    The rank-1 accuracy is the share of probes whose best match is the right person. The rank-5 accuracy is the share
    whose right person is among the five best matches.

    The open-set rate counts a probe as correct only when its best match is the right person AND scores at least the
    threshold, which is how the pipeline decides between a name and "unknown". It is reported only when a threshold
    is given.
    """
    # Compare every probe with every gallery entry by cosine similarity, and sort the gallery for each probe.
    g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-12)
    p = probes / np.linalg.norm(probes, axis=1, keepdims=True).clip(1e-12)
    sims = p @ g.T
    order = np.argsort(-sims, axis=1)
    ranked = np.asarray(gallery_ids)[order]
    probe_ids = np.asarray(probe_ids)
    report = {
        "n_gallery": int(len(gallery_ids)),
        "n_probes": int(len(probe_ids)),
        "rank1": float((ranked[:, 0] == probe_ids).mean()),
        "rank5": float((ranked[:, :5] == probe_ids[:, None]).any(1).mean()),
    }
    if threshold is not None:
        best = sims[np.arange(len(p)), order[:, 0]]
        report["open_set_correct_at_threshold"] = float(((ranked[:, 0] == probe_ids) & (best >= threshold)).mean())
        report["threshold"] = float(threshold)
    return report
