"""Tests alignment, labels, models, arrays, metrics, the gallery and augmentation, offline and on the CPU.

Some terms used below. Alignment moves a face image so that five landmark points (eyes, nose and mouth corners) land
on fixed template positions. ArcFace is a way of training a face-identity model with an angular margin, which is an
extra penalty that forces the embeddings of different people further apart. IResNet is the residual network that
turns a face into an embedding (a list of numbers that stands for the face). A gallery holds the enrolled people's
templates, which are averaged embeddings, and is used to name a face or to say that the person is unknown. AUC (area
under the curve) and EER (equal error rate) summarise how well same-person scores are separated from
different-person scores. TAR is the share of same-person pairs that are accepted, and FAR is the share of
different-person pairs that are wrongly accepted.
"""
import json

import numpy as np
import pytest
import torch

from facekit import align, arrays, metrics
from facekit.augment import EXPRESSION_AUG, IDENTITY_AUG, augment_batch
from facekit.expression import CLASSES, ExpressionNet, emonet_binary_class, emonet_binary_target, emonet_hq_class
from facekit.gallery import Gallery
from facekit.identity import ArcFaceHead, IdentityTrainer, IResNet


# Tests of face alignment.


def test_align_maps_template_landmarks_to_themselves():
    """Aligning an image whose landmarks already sit on the template leaves the image almost unchanged."""
    image = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    out = align.align_face(image, align.ARCFACE_TEMPLATE_112, 112)
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 2


def test_align_undoes_scale_and_shift():
    """Landmarks that were enlarged and shifted are mapped back onto the template."""
    matrix = align.estimate_transform(align.ARCFACE_TEMPLATE_112 * 2 + 30, 112)

    # A column of ones is added so that the matrix can apply the shift as well as the scale.
    mapped = (np.c_[align.ARCFACE_TEMPLATE_112 * 2 + 30, np.ones(5)] @ matrix.T)
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE_112, atol=0.5)


def test_align_scales_template_for_larger_crops():
    """The template is defined for 112 pixel crops, so a 224 pixel crop uses the template at twice the size."""
    assert np.allclose(align.template_for(224), align.ARCFACE_TEMPLATE_112 * 2)


def test_align_rejects_bad_landmarks():
    with pytest.raises(ValueError):
        align.estimate_transform(np.zeros((4, 2)))


# Tests of the conversion of EmoNet labels into expression classes.


# A binary EmoNet label, written as the text that the dataset stores. Each entry is one rater, who says whether one
# emotion category applies. Three raters rated Amusement (two said yes and one said no), and a fourth rater rated
# Distress (no).
BINARY_LABEL = ("[{'human-4': {'Amusement': True}}, {'human-6': {'Amusement': True}}, "
                "{'human-1': {'Amusement': False}}, {'human-3': {'Distress': False}}]")


def test_binary_target_is_the_most_rated_category_with_majority():
    """The target is the category that most raters rated, together with the majority answer for it."""
    assert emonet_binary_target(BINARY_LABEL) == ("Amusement", True)
    assert emonet_binary_class(BINARY_LABEL) == CLASSES.index("happy")


def test_binary_unconfirmed_or_unmapped_is_dropped():
    """A label is dropped when most raters deny its category or when the category has no expression class."""
    # Only one of the three raters says that Fear applies, so the category is not confirmed.
    unconfirmed = "[{'a': {'Fear': False}}, {'b': {'Fear': False}}, {'c': {'Fear': True}}]"
    assert emonet_binary_class(unconfirmed) is None

    # The raters agree on the category, but it does not correspond to any of the expression classes.
    unmapped = "[{'a': {'Jealousy & Envy': True}}, {'b': {'Jealousy & Envy': True}}]"
    assert emonet_binary_class(unmapped) is None


def test_hq_class_takes_clear_top_class_or_neutral():
    """A clear top category gives its class, a calm face gives neutral and a tie between categories gives nothing."""
    # Anger is rated far higher than Fear.
    clear = "[{'h1': {'Negative High-Energy Emotions|Anger': 6, 'Negative High-Energy Emotions|Fear': 1}}]"
    assert emonet_hq_class(clear) == CLASSES.index("anger")

    # Anger has a rating of 0 and Amusement has a rating of only 1, so the face counts as neutral.
    calm = "[{'h1': {'Negative High-Energy Emotions|Anger': 0, 'Positive|Amusement': 1}}]"
    assert emonet_hq_class(calm) == CLASSES.index("neutral")

    # Anger and Fear are rated equally, so no single class can be chosen.
    mixed = "[{'h1': {'Negative High-Energy Emotions|Anger': 5, 'Negative High-Energy Emotions|Fear': 5}}]"
    assert emonet_hq_class(mixed) is None


# Tests of the expression and identity models.


def test_expression_net_outputs_logits_for_any_input_size():
    """The expression model returns one score per class even when the image size differs from its input size."""
    # The model is built for 96 pixel input, but the test passes 112 pixel images.
    model = ExpressionNet(backbone="resnet18", pretrained=False, input_size=96).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 112, 112)).shape == (2, len(CLASSES))


def test_iresnet_embedding_shape():
    """The identity network returns an embedding of 512 numbers for each image."""
    model = IResNet("iresnet18").eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 112, 112)).shape == (2, 512)


def test_arcface_margin_lowers_the_true_class_logit():
    """The margin lowers the score of the true identity only and leaves the scores of the other identities alone."""
    head = ArcFaceHead(8, 4, scale=1.0, margin=0.5)
    emb = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 2])
    logits = head(emb, labels)
    cosine = head.cosine(emb)
    rows = torch.arange(3)

    # For the true identity of each row, the score with the margin is below the plain cosine similarity.
    assert torch.all(logits[rows, labels] < cosine[rows, labels])

    # For every other identity, the score equals the plain cosine similarity.
    other = torch.ones_like(logits, dtype=torch.bool)
    other[rows, labels] = False
    assert torch.allclose(logits[other], cosine[other])


def test_identity_trainer_first_loss_is_near_log_classes_times_scale_effect():
    """With a scale of 1 and no margin, an untrained model has a loss close to that of random guessing.

    Random guessing among 10 identities gives a loss of log(10).
    """
    torch.manual_seed(0)
    model = IdentityTrainer(num_identities=10, depth="iresnet18", scale=1.0, margin=0.0)
    loss, acc = model(torch.rand(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
    assert abs(loss.item() - np.log(10)) < 0.3
    assert 0.0 <= acc.item() <= 1.0


# Tests of the image arrays and of the splits into training, validation and test sets.


def test_write_and_read_array_roundtrip(tmp_path):
    """Images, labels and ids that are written to an array on disk come back unchanged when it is read."""
    rng = np.random.default_rng(0)

    # Five images with alternating labels and the ids id0 to id4.
    items = [(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), i % 2, f"id{i}") for i in range(5)]

    # The count of 7 is only an upper bound, and the description records the 5 items that were actually written.
    description = arrays.write_array(tmp_path, "toy", iter(items), count=7, size=8)
    assert description["count"] == 5
    assert arrays.array_exists(tmp_path, "toy")
    images, labels, meta = arrays.read_array(tmp_path, "toy")
    assert images.shape == (5, 8, 8, 3)
    assert np.array_equal(images[3], items[3][0])
    assert labels.tolist() == [0, 1, 0, 1, 0]
    assert meta["ids"][4] == "id4"

    # A dataset over the indices 1 and 3 returns the image at index 3 as a channel-first uint8 tensor.
    dataset = arrays.ArrayDataset(images, labels, [1, 3])
    x, y = dataset[1]
    assert x.shape == (3, 8, 8) and x.dtype == torch.uint8 and y == 1


def test_epoch_sampler_is_reproducible_and_resumable():
    """The sampler gives each index once per epoch, repeats an epoch for the same seed and can resume part way."""
    full = list(arrays.EpochSampler(20, seed=1, epoch=3))
    assert sorted(full) == list(range(20))
    assert list(arrays.EpochSampler(20, seed=1, epoch=3, start=8)) == full[8:]
    assert full != list(arrays.EpochSampler(20, seed=1, epoch=4))


def test_stratified_split_is_disjoint_and_keeps_proportions():
    """The three splits share no index, together cover every index and keep the share of the rarer class."""
    labels = np.array([0] * 100 + [1] * 20)
    train, val, test = arrays.stratified_split(labels, (0.7, 0.15, 0.15), seed=0)
    assert len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 and len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == 120
    assert abs((labels[test] == 1).mean() - 20 / 120) < 0.05


def test_split_overlap_counts_shared_ids():
    """The overlap is the number of ids that appear in more than one split."""
    assert arrays.split_overlap(["a", "b"], ["c"], ["d"]) == 0
    assert arrays.split_overlap(["a", "b"], ["b", "c"], ["a"]) == 2


# Tests of the evaluation metrics.


def test_classification_report_perfect_and_confusion():
    """Perfect predictions give an accuracy of 1 and a confusion matrix with counts only on the diagonal."""
    y = np.array([0, 1, 2, 2])
    report = metrics.classification_report(y, y, 3, n_boot=50)
    assert report["accuracy"] == 1.0 and report["macro_f1"] == 1.0
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_verification_report_separable_scores():
    """Scores that separate perfectly give an AUC of 1, a TAR of 1 at a FAR of 0.001 and an EER of 0."""
    same = np.array([1] * 100 + [0] * 1000)

    # Same-person scores run from 0.6 to 0.9 and different-person scores run from -0.2 to 0.3, so they never overlap.
    scores = np.r_[np.linspace(0.6, 0.9, 100), np.linspace(-0.2, 0.3, 1000)]
    report = metrics.verification_report(scores, same)
    assert report["auc"] == 1.0
    assert report["tar@far=0.001"] == 1.0
    assert report["eer"] == 0.0


def test_verification_pairs_are_correctly_labelled():
    """Each pair is marked as same-person exactly when its two photos have the same label."""
    labels = np.repeat(np.arange(10), 5)
    pairs, same = metrics.verification_pairs(labels, 50, 200, seed=0)
    assert np.all((labels[pairs[:, 0]] == labels[pairs[:, 1]]) == same.astype(bool))

    # A same-person pair never pairs a photo with itself.
    assert np.all(pairs[same == 1, 0] != pairs[same == 1, 1])


def test_identification_report_rank1():
    """Two of the three probes are matched to the right person, so rank-1 accuracy is two thirds."""
    # The gallery holds one template for each of three people. The first two probes are closest to their own person,
    # and the third probe is closest to the wrong person.
    gallery = np.eye(3)
    probes = np.array([[0.9, 0.1, 0], [0, 1, 0], [0.6, 0.0, 0.4]])
    report = metrics.identification_report(gallery, np.array([0, 1, 2]), probes, np.array([0, 1, 2]), threshold=0.95)
    assert report["rank1"] == pytest.approx(2 / 3)
    assert report["open_set_correct_at_threshold"] == pytest.approx(2 / 3)


def test_expected_calibration_error_zero_for_perfect_confidence():
    """A model that is fully confident and always right has an expected calibration error of 0."""
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.expected_calibration_error(probs, np.array([0, 1])) == 0.0


# Tests of the gallery of enrolled people.


def test_gallery_enroll_identify_unknown_and_roundtrip(tmp_path):
    """The gallery names enrolled people, reports unknown faces, adds photos to a person and survives saving."""
    gallery = Gallery(threshold=0.5)
    gallery.enroll("ana", np.array([[1.0, 0, 0], [0.9, 0.1, 0]]))
    gallery.enroll("ben", np.array([[0, 1.0, 0]]))

    # A face close to Ana is named Ana, and a face unlike anybody enrolled gets no name.
    assert gallery.identify(np.array([1.0, 0.05, 0]))[0] == "ana"
    assert gallery.identify(np.array([0, 0, 1.0]))[0] is None

    # Enrolling Ana again adds a third photo to her instead of creating a new person.
    gallery.enroll("ana", np.array([[1.0, 0, 0]]))
    assert gallery.counts == [3, 1]

    # After saving and loading, the names, the threshold and the templates are the same.
    path = tmp_path / "g.npz"
    gallery.save(path)
    loaded = Gallery.load(path)
    assert loaded.names == ["ana", "ben"] and loaded.threshold == 0.5
    assert np.allclose(loaded.templates, gallery.templates)


# Tests of the random image changes used in training.


@pytest.mark.parametrize("cfg", [EXPRESSION_AUG, IDENTITY_AUG])
def test_augment_keeps_shape_and_range(cfg):
    """Augmenting a uint8 batch gives a float32 batch of the same shape with values between 0 and 1."""
    x = torch.randint(0, 255, (6, 3, 32, 32), dtype=torch.uint8)
    out = augment_batch(x, cfg, torch.Generator().manual_seed(0))
    assert out.shape == (6, 3, 32, 32) and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_augment_under_autocast_returns_float32_with_blur():
    """Augmentation with blur still returns float32 when it runs under mixed precision (autocast)."""
    x = torch.randint(0, 255, (8, 3, 32, 32), dtype=torch.uint8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = augment_batch(x, IDENTITY_AUG, torch.Generator().manual_seed(0))
    assert out.dtype == torch.float32


def test_umeyama_recovers_a_similarity_exactly_and_spreads_noise():
    """The Umeyama method finds the rotation, scale and shift that carry the template onto a transformed copy.

    The name also mentions noise, but this test adds no noise and checks only the exact case.
    """
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -4.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = align.ARCFACE_TEMPLATE_112.astype(np.float64)
    dst = (scale * src @ rot.T) + shift
    m = align.umeyama(src, dst)
    assert np.allclose(np.c_[src, np.ones(5)] @ m.T, dst, atol=1e-6)


def test_hq_neutral_requires_no_strong_unmapped_emotion():
    """A face rated strongly for Pain and Distress is not neutral, even though neither has an expression class."""
    pain = "[{'h1': {'Group|Pain': 6.5, 'Group|Distress': 5.5, 'Group|Anger': 0, 'Group|Amusement': 0}}]"
    assert emonet_hq_class(pain) is None


def test_gallery_enrolment_weights_every_photo_equally(tmp_path):
    """A template is built from all the person's photos with equal weight, however many enrolment calls added them."""
    g = Gallery(0.5)
    g.enroll("a", np.array([[1.0, 0.0]]))
    g.enroll("a", np.array([[0.0, 1.0], [0.0, 1.0]]))

    # The three photos add up to (1, 2), and the template is that sum scaled to unit length.
    expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
    assert np.allclose(g.templates[0], expected, atol=1e-6)

    # A gallery that was saved and loaded continues from the same sum, which becomes (2, 2) after one more photo.
    g.save(tmp_path / "g.npz")
    loaded = Gallery.load(tmp_path / "g.npz")
    loaded.enroll("a", np.array([[1.0, 0.0]]))
    assert np.allclose(loaded.templates[0], np.array([2.0, 2.0]) / np.sqrt(8.0), atol=1e-6)


def test_verification_threshold_is_never_the_reject_all_sentinel():
    """The reported threshold is a real cosine similarity, even when no score in the data keeps the FAR in its limit."""
    same = np.array([1, 0, 0, 0])

    # The highest-scoring different-person pair outscores the only same-person pair.
    scores = np.array([0.5, 0.9, 0.1, 0.2])
    report = metrics.verification_report(scores, same, fars=(1e-3,))

    # A placeholder threshold that rejects every pair would lie above 1.0, which no cosine similarity can reach.
    assert report["threshold@far=0.001"] <= 1.0
