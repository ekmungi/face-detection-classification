"""Tests for models.py and quicktrain.py; skipped when torch is not installed."""
import pytest

torch = pytest.importorskip("torch")

import models  # noqa: E402  (imported after the torch check on purpose)
import quicktrain  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def test_cnn_and_mlp_produce_ten_logits_per_image():
    batch = torch.zeros(4, 1, 28, 28)
    for name in models.MODEL_NAMES:
        assert models.build_model(name)(batch).shape == (4, 10)


def test_parameter_counts_match_the_docstrings():
    assert parameter_count(models.Net()) == 1_199_882
    assert parameter_count(models.MLP()) == 235_146


def test_the_wide_mlp_from_the_second_script_has_the_expected_size_and_output():
    import models_extra

    model = models_extra.WideMLP()
    assert parameter_count(model) == 936_330
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        models.build_model("transformer")


def make_dataset(size=64):
    """Two linearly separable classes in a 1x28x28 shape, so a few steps are enough to learn them."""
    torch.manual_seed(0)
    labels = torch.arange(size) % 2
    images = torch.randn(size, 1, 28, 28) * 0.1
    images[labels == 1] += 1.0
    return TensorDataset(images, labels)


def test_train_steps_lowers_the_loss_and_returns_one_loss_per_step():
    loader = DataLoader(make_dataset(), batch_size=16)
    losses = quicktrain.train_steps(models.MLP(), loader, steps=30, device="cpu")
    assert len(losses) == 30
    assert losses[-1] < losses[0]


def test_accuracy_is_high_after_training_on_separable_data():
    data = make_dataset()
    loader = DataLoader(data, batch_size=16)
    model = models.MLP()
    quicktrain.train_steps(model, loader, steps=40, device="cpu")
    assert quicktrain.accuracy(model, loader, "cpu") > 0.9


def test_subset_loader_has_the_requested_size_and_repeats_with_the_same_seed():
    data = make_dataset(64)
    first = [y.tolist() for _, y in quicktrain.subset_loader(data, 20, 5, seed=1)]
    again = [y.tolist() for _, y in quicktrain.subset_loader(data, 20, 5, seed=1)]
    assert sum(len(batch) for batch in first) == 20
    assert first == again


def test_evaluate_returns_loss_and_accuracy_and_leaves_the_model_in_train_mode():
    loader = DataLoader(make_dataset(), batch_size=16)
    model = models.MLP()
    loss, acc = quicktrain.evaluate(model, loader, "cpu")
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert model.training
