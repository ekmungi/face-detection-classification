"""Logic checks for stage 4: they show that training is correct and not only that it ran without a crash."""
import dataclasses
import math

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """The outcome of one check: its name, whether it passed, and a sentence saying what was measured."""

    name: str
    ok: bool
    detail: str


def check_initial_loss(model, loader, n_classes, device, tolerance=0.5):
    """An untrained classifier should score about ln(n_classes) on its first batch (2.30 for 10 classes)."""
    x, y = next(iter(loader))
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(model(x.to(device)), y.to(device)).item()
    model.train()
    expected = math.log(n_classes)
    return CheckResult("initial loss", abs(loss - expected) <= tolerance,
                       f"first-batch loss {loss:.3f}, expected about {expected:.3f}")


def check_overfit_batch(model_factory, loader, device, steps=80, lr=1e-3, target=0.05):
    """A fresh model trained repeatedly on one batch should drive its loss to almost zero; if not, the loop or data is broken."""
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    return CheckResult("overfit one batch", loss.item() < target,
                       f"loss after {steps} steps on one batch: {loss.item():.4f} (target below {target})")


def check_no_overlap(train_indices, val_indices):
    """No item may be in both the training and the validation set."""
    shared = set(train_indices) & set(val_indices)
    return CheckResult("no train/validation overlap", not shared, f"{len(shared)} items are in both sets")


def check_loss_decreases(history, min_drop=0.2):
    """The mean of the last tenth of the step losses should be at least `min_drop` below the mean of the first tenth."""
    losses = history["step_loss"]
    tenth = max(1, len(losses) // 10)
    first, last = sum(losses[:tenth]) / tenth, sum(losses[-tenth:]) / tenth
    return CheckResult("loss decreases", first - last >= min_drop, f"mean loss {first:.3f} at the start, {last:.3f} at the end")


def check_finite(history):
    """Every recorded loss and accuracy must be a finite number."""
    values = [v for series in history.values() for v in series]
    bad = sum(1 for v in values if not math.isfinite(v))
    return CheckResult("no NaN or infinity", bad == 0, f"{bad} of {len(values)} recorded values are not finite")


def check_above_chance(history, n_classes, margin=0.2):
    """The last validation accuracy should beat random guessing by at least `margin`."""
    accuracy = history["val_acc"][-1]
    chance = 1.0 / n_classes
    return CheckResult("validation above chance", accuracy >= chance + margin,
                       f"validation accuracy {accuracy:.3f}, chance is {chance:.3f}")


def all_passed(results):
    """True when every CheckResult in `results` passed."""
    return all(result.ok for result in results)


def format_results(results):
    """Returns one line per check, with PASS or FAIL, for printing in a notebook or a log."""
    return "\n".join(f"{'PASS' if r.ok else 'FAIL'}  {r.name}: {r.detail}" for r in results)
