"""Tests for the split helpers and the image transforms."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import augment  # noqa: E402
import data  # noqa: E402
from PIL import Image  # noqa: E402
from synthetic import make_dataset  # noqa: E402


def test_split_indices_are_disjoint_the_right_size_and_repeatable():
    train, val = data.split_indices(100, 60, 20, seed=3)
    assert len(train) == 60 and len(val) == 20 and not set(train) & set(val)
    assert data.split_indices(100, 60, 20, seed=3) == (train, val)
    assert data.split_indices(100, 60, 20, seed=4) != (train, val)


def test_split_indices_rejects_more_items_than_exist():
    with pytest.raises(ValueError):
        data.split_indices(10, 8, 5, seed=0)


def test_make_sets_returns_disjoint_subsets_of_the_requested_sizes():
    source = make_dataset(50)
    train_set, val_set, train_idx, val_idx = data.make_sets(source, source, 30, 10, seed=0)
    assert len(train_set) == 30 and len(val_set) == 10 and not set(train_idx) & set(val_idx)


def test_plain_transform_gives_a_normalised_tensor_and_is_deterministic():
    image = Image.new("L", (28, 28), color=255)
    transform = augment.build_transform(False)
    first, second = transform(image), transform(image)
    assert first.shape == (1, 28, 28) and torch.equal(first, second)
    assert first.max().item() == pytest.approx((1.0 - augment.MNIST_MEAN) / augment.MNIST_STD)


def test_the_augmenting_transform_adds_one_random_step_and_keeps_the_shape():
    assert len(augment.build_transform(True).transforms) == len(augment.build_transform(False).transforms) + 1
    assert augment.build_transform(True)(Image.new("L", (28, 28), color=255)).shape == (1, 28, 28)
