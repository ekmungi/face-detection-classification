"""The one training loop shared by the notebook and train.py, with exact resume and callbacks for checkpoints and heartbeats.

Because both the notebook and the script call fit(), a loop that works in the notebook is the loop that runs in the script.
"""
import dataclasses
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import quicktrain


@dataclasses.dataclass(frozen=True)
class TrainState:
    """Where training stands: the global step, the epoch counted from 0, and the batches already done in that epoch."""

    step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0


class Callback:
    """Base class for hooks; override the ones you need. Each hook receives the current TrainState."""

    def on_step(self, state, loss, model, optimizer):
        """Called after every optimizer step with the training loss of that step."""

    def on_epoch_end(self, state, metrics, model, optimizer):
        """Called after every epoch with a dict of train_loss, train_acc (when measured), val_loss and val_acc."""


def epoch_loader(train_set, batch_size, seed, epoch):
    """Returns the loader for one epoch; the shuffle depends only on seed and epoch, so a resumed run sees the same order."""
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))


def fit(model, optimizer, train_set, val_set, epochs, batch_size, device, seed=0, state=None,
        callbacks=(), max_steps=0, step_sleep=0.0, eval_train=True):
    """Trains model for `epochs` epochs, starting from `state` when resuming, and returns (history, final state, stopped_early).

    Args:
        model, optimizer: the model and its optimizer, already on `device`; a resumed run passes ones loaded from a checkpoint.
        train_set, val_set: map-style datasets.
        epochs: total epochs to reach, counting epochs already done in `state`.
        batch_size, seed: the shuffle of epoch k depends on seed + k only.
        state: a TrainState to resume from, or None to start at step 0.
        callbacks: Callback objects called after each step and each epoch.
        max_steps: stop after this global step (0 = no limit); used by smoke tests.
        step_sleep: seconds to sleep after each step, to make a short run last longer.
        eval_train: also measure loss and accuracy over the whole training set at each epoch end.

    The history holds "step_loss" (one value per step taken in this call) and one value per epoch for
    "train_loss", "train_acc", "val_loss" and "val_acc".
    """
    state = state or TrainState()
    history = {"step_loss": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    val_loader = DataLoader(val_set, batch_size=1000)
    train_eval_loader = DataLoader(train_set, batch_size=1000) if eval_train else None
    stopped_early = False
    model.train()
    while state.epoch < epochs and not stopped_early:
        for index, (x, y) in enumerate(epoch_loader(train_set, batch_size, seed, state.epoch)):
            if index < state.batch_in_epoch:
                continue  # batches already trained before the checkpoint
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            state = dataclasses.replace(state, step=state.step + 1, batch_in_epoch=index + 1)
            history["step_loss"].append(loss.item())
            if step_sleep:
                time.sleep(step_sleep)
            for callback in callbacks:
                callback.on_step(state, history["step_loss"][-1], model, optimizer)
            if max_steps and state.step >= max_steps:
                stopped_early = True
                break
        if stopped_early:
            break
        state = dataclasses.replace(state, epoch=state.epoch + 1, batch_in_epoch=0)
        metrics = {}
        for prefix, loader in (("train", train_eval_loader), ("val", val_loader)):
            if loader is not None:
                metrics[f"{prefix}_loss"], metrics[f"{prefix}_acc"] = quicktrain.evaluate(model, loader, device)
                history[f"{prefix}_loss"].append(metrics[f"{prefix}_loss"])
                history[f"{prefix}_acc"].append(metrics[f"{prefix}_acc"])
        for callback in callbacks:
            callback.on_epoch_end(state, metrics, model, optimizer)
    return history, state, stopped_early
