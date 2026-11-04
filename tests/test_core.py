"""Offline CPU tests for alignment, labels, models, arrays, metrics, the gallery and augmentation."""
import json

import numpy as np
import pytest
import torch

from facekit import align, arrays, metrics
from facekit.augment import EXPRESSION_AUG, IDENTITY_AUG, augment_batch
from facekit.expression import CLASSES, ExpressionNet, emonet_binary_class, emonet_binary_target, emonet_hq_class
from facekit.gallery import Gallery
from facekit.identity import ArcFaceHead, IdentityTrainer, IResNet


# --- alignment ----------------------------------------------------------------------------------------------


def test_align_maps_template_landmarks_to_themselves():
    image = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    out = align.align_face(image, align.ARCFACE_TEMPLATE_112, 112)
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 2


def test_align_undoes_scale_and_shift():
    matrix = align.estimate_transform(align.ARCFACE_TEMPLATE_112 * 2 + 30, 112)
    mapped = (np.c_[align.ARCFACE_TEMPLATE_112 * 2 + 30, np.ones(5)] @ matrix.T)
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE_112, atol=0.5)


def test_align_scales_template_for_larger_crops():
    assert np.allclose(align.template_for(224), align.ARCFACE_TEMPLATE_112 * 2)


def test_align_rejects_bad_landmarks():
    with pytest.raises(ValueError):
        align.estimate_transform(np.zeros((4, 2)))


# --- EmoNet labels --------------------------------------------------------------------------------------------


BINARY_LABEL = ("[{'human-4': {'Amusement': True}}, {'human-6': {'Amusement': True}}, "
                "{'human-1': {'Amusement': False}}, {'human-3': {'Distress': False}}]")


def test_binary_target_is_the_most_rated_category_with_majority():
    assert emonet_binary_target(BINARY_LABEL) == ("Amusement", True)
    assert emonet_binary_class(BINARY_LABEL) == CLASSES.index("happy")


def test_binary_unconfirmed_or_unmapped_is_dropped():
    unconfirmed = "[{'a': {'Fear': False}}, {'b': {'Fear': False}}, {'c': {'Fear': True}}]"
    assert emonet_binary_class(unconfirmed) is None
    unmapped = "[{'a': {'Jealousy & Envy': True}}, {'b': {'Jealousy & Envy': True}}]"
    assert emonet_binary_class(unmapped) is None


def test_hq_class_takes_clear_top_class_or_neutral():
    clear = "[{'h1': {'Negative High-Energy Emotions|Anger': 6, 'Negative High-Energy Emotions|Fear': 1}}]"
    assert emonet_hq_class(clear) == CLASSES.index("anger")
    calm = "[{'h1': {'Negative High-Energy Emotions|Anger': 0, 'Positive|Amusement': 1}}]"
    assert emonet_hq_class(calm) == CLASSES.index("neutral")
    mixed = "[{'h1': {'Negative High-Energy Emotions|Anger': 5, 'Negative High-Energy Emotions|Fear': 5}}]"
    assert emonet_hq_class(mixed) is None


# --- models ---------------------------------------------------------------------------------------------------


def test_expression_net_outputs_logits_for_any_input_size():
    model = ExpressionNet(backbone="resnet18", pretrained=False, input_size=96).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 112, 112)).shape == (2, len(CLASSES))


def test_iresnet_embedding_shape():
    model = IResNet("iresnet18").eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 112, 112)).shape == (2, 512)


def test_arcface_margin_lowers_the_true_class_logit():
    head = ArcFaceHead(8, 4, scale=1.0, margin=0.5)
    emb = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 2])
    logits = head(emb, labels)
    cosine = head.cosine(emb)
    rows = torch.arange(3)
    assert torch.all(logits[rows, labels] < cosine[rows, labels])
    other = torch.ones_like(logits, dtype=torch.bool)
    other[rows, labels] = False
    assert torch.allclose(logits[other], cosine[other])


def test_identity_trainer_first_loss_is_near_log_classes_times_scale_effect():
    torch.manual_seed(0)
    model = IdentityTrainer(num_identities=10, depth="iresnet18", scale=1.0, margin=0.0)
    loss, acc = model(torch.rand(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
    assert abs(loss.item() - np.log(10)) < 0.3
    assert 0.0 <= acc.item() <= 1.0


# --- arrays and splits ----------------------------------------------------------------------------------------


def test_write_and_read_array_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    items = [(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), i % 2, f"id{i}") for i in range(5)]
    description = arrays.write_array(tmp_path, "toy", iter(items), count=7, size=8)
    assert description["count"] == 5
    assert arrays.array_exists(tmp_path, "toy")
    images, labels, meta = arrays.read_array(tmp_path, "toy")
    assert images.shape == (5, 8, 8, 3)
    assert np.array_equal(images[3], items[3][0])
    assert labels.tolist() == [0, 1, 0, 1, 0]
    assert meta["ids"][4] == "id4"
    dataset = arrays.ArrayDataset(images, labels, [1, 3])
    x, y = dataset[1]
    assert x.shape == (3, 8, 8) and x.dtype == torch.uint8 and y == 1


def test_epoch_sampler_is_reproducible_and_resumable():
    full = list(arrays.EpochSampler(20, seed=1, epoch=3))
    assert sorted(full) == list(range(20))
    assert list(arrays.EpochSampler(20, seed=1, epoch=3, start=8)) == full[8:]
    assert full != list(arrays.EpochSampler(20, seed=1, epoch=4))


def test_stratified_split_is_disjoint_and_keeps_proportions():
    labels = np.array([0] * 100 + [1] * 20)
    train, val, test = arrays.stratified_split(labels, (0.7, 0.15, 0.15), seed=0)
    assert len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 and len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == 120
    assert abs((labels[test] == 1).mean() - 20 / 120) < 0.05


def test_split_overlap_counts_shared_ids():
    assert arrays.split_overlap(["a", "b"], ["c"], ["d"]) == 0
    assert arrays.split_overlap(["a", "b"], ["b", "c"], ["a"]) == 2


# --- metrics --------------------------------------------------------------------------------------------------


def test_classification_report_perfect_and_confusion():
    y = np.array([0, 1, 2, 2])
    report = metrics.classification_report(y, y, 3, n_boot=50)
    assert report["accuracy"] == 1.0 and report["macro_f1"] == 1.0
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_verification_report_separable_scores():
    same = np.array([1] * 100 + [0] * 1000)
    scores = np.r_[np.linspace(0.6, 0.9, 100), np.linspace(-0.2, 0.3, 1000)]
    report = metrics.verification_report(scores, same)
    assert report["auc"] == 1.0
    assert report["tar@far=0.001"] == 1.0
    assert report["eer"] == 0.0


def test_verification_pairs_are_correctly_labelled():
    labels = np.repeat(np.arange(10), 5)
    pairs, same = metrics.verification_pairs(labels, 50, 200, seed=0)
    assert np.all((labels[pairs[:, 0]] == labels[pairs[:, 1]]) == same.astype(bool))
    assert np.all(pairs[same == 1, 0] != pairs[same == 1, 1])


def test_identification_report_rank1():
    gallery = np.eye(3)
    probes = np.array([[0.9, 0.1, 0], [0, 1, 0], [0.6, 0.0, 0.4]])
    report = metrics.identification_report(gallery, np.array([0, 1, 2]), probes, np.array([0, 1, 2]), threshold=0.95)
    assert report["rank1"] == pytest.approx(2 / 3)
    assert report["open_set_correct_at_threshold"] == pytest.approx(2 / 3)


def test_expected_calibration_error_zero_for_perfect_confidence():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.expected_calibration_error(probs, np.array([0, 1])) == 0.0


# --- gallery --------------------------------------------------------------------------------------------------


def test_gallery_enroll_identify_unknown_and_roundtrip(tmp_path):
    gallery = Gallery(threshold=0.5)
    gallery.enroll("ana", np.array([[1.0, 0, 0], [0.9, 0.1, 0]]))
    gallery.enroll("ben", np.array([[0, 1.0, 0]]))
    assert gallery.identify(np.array([1.0, 0.05, 0]))[0] == "ana"
    assert gallery.identify(np.array([0, 0, 1.0]))[0] is None
    gallery.enroll("ana", np.array([[1.0, 0, 0]]))
    assert gallery.counts == [3, 1]
    path = tmp_path / "g.npz"
    gallery.save(path)
    loaded = Gallery.load(path)
    assert loaded.names == ["ana", "ben"] and loaded.threshold == 0.5
    assert np.allclose(loaded.templates, gallery.templates)


# --- augmentation ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [EXPRESSION_AUG, IDENTITY_AUG])
def test_augment_keeps_shape_and_range(cfg):
    x = torch.randint(0, 255, (6, 3, 32, 32), dtype=torch.uint8)
    out = augment_batch(x, cfg, torch.Generator().manual_seed(0))
    assert out.shape == (6, 3, 32, 32) and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_augment_under_autocast_returns_float32_with_blur():
    x = torch.randint(0, 255, (8, 3, 32, 32), dtype=torch.uint8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = augment_batch(x, IDENTITY_AUG, torch.Generator().manual_seed(0))
    assert out.dtype == torch.float32
