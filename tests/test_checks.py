"""Tests for the stage 4 logic checks."""
import pytest

torch = pytest.importorskip("torch")

import checks  # noqa: E402
import models  # noqa: E402
from synthetic import make_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def loader(n_classes=10):
    return DataLoader(make_dataset(64, n_classes=n_classes), batch_size=32)


def test_initial_loss_passes_for_a_fresh_model_and_fails_for_an_overconfident_one():
    torch.manual_seed(0)
    assert checks.check_initial_loss(models.MLP(), loader(), 10, "cpu").ok
    bad = models.MLP()
    for p in bad.parameters():
        p.data.mul_(40.0)  # huge logits make the first loss far above ln(10)
    assert not checks.check_initial_loss(bad, loader(), 10, "cpu").ok


def test_overfit_one_batch_passes_normally_and_fails_when_given_too_few_steps():
    torch.manual_seed(0)
    assert checks.check_overfit_batch(models.MLP, loader(), "cpu", steps=80, target=0.05).ok
    assert not checks.check_overfit_batch(models.MLP, loader(), "cpu", steps=1, target=0.01).ok


def test_no_overlap_names_the_number_of_shared_items():
    assert checks.check_no_overlap([1, 2, 3], [4, 5]).ok
    result = checks.check_no_overlap([1, 2, 3], [3, 4])
    assert not result.ok and "1 items" in result.detail


def test_loss_decreases_compares_the_first_and_last_tenth():
    assert checks.check_loss_decreases({"step_loss": [2.0] * 10 + [1.0] * 80 + [0.5] * 10}).ok
    assert not checks.check_loss_decreases({"step_loss": [1.0] * 100}).ok


def test_finite_catches_nan_and_infinity():
    assert checks.check_finite({"step_loss": [1.0, 0.5], "val_acc": [0.9]}).ok
    assert not checks.check_finite({"step_loss": [1.0, float("nan")], "val_acc": [float("inf")]}).ok


def test_above_chance_uses_the_last_validation_accuracy():
    assert checks.check_above_chance({"val_acc": [0.2, 0.8]}, 10).ok
    assert not checks.check_above_chance({"val_acc": [0.8, 0.11]}, 10).ok


def test_all_passed_and_format_results():
    results = [checks.CheckResult("a", True, "fine"), checks.CheckResult("b", False, "broken")]
    assert not checks.all_passed(results) and checks.all_passed(results[:1])
    assert checks.format_results(results).splitlines() == ["PASS  a: fine", "FAIL  b: broken"]
