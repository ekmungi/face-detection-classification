"""Callbacks for the shared training loop: rolling checkpoints with exact resume, and a heartbeat file for observers."""
import random
import time

import torch

import runlib
from trainer import Callback, TrainState


class CheckpointCallback(Callback):
    """Saves a checkpoint every `every` steps into run_dir, keeping only the newest `keep`."""

    def __init__(self, run_dir, every=100, keep=2):
        self.run_dir = run_dir
        self.every = every
        self.keep = keep
        self.last = None  # file name of the newest checkpoint written by this object

    def on_step(self, state, loss, model, optimizer):
        if state.step % self.every == 0:
            self.save(state, model, optimizer)

    def save(self, state, model, optimizer):
        """Writes a checkpoint for `state` atomically, prunes old ones and returns the new path."""
        path = runlib.checkpoint_path(self.run_dir, state.step)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": state.step, "epoch": state.epoch, "batch_in_epoch": state.batch_in_epoch,
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        }, tmp)
        tmp.replace(path)
        runlib.prune_checkpoints(self.run_dir, self.keep)
        self.last = path.name
        return path

    @staticmethod
    def load(path, model, optimizer, device):
        """Restores model, optimizer and random-number state from a checkpoint and returns its TrainState.

        weights_only=True refuses to unpickle arbitrary objects, so a tampered checkpoint cannot run code.
        """
        saved = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        return TrainState(saved["step"], saved["epoch"], saved["batch_in_epoch"])


class HeartbeatCallback(Callback):
    """Writes the heartbeat file at most every `every_s` seconds, and once more when finish() is called."""

    def __init__(self, path, every_s=5.0, last_checkpoint=lambda: None, clock=time.time):
        self.path = path
        self.every_s = every_s
        self.last_checkpoint = last_checkpoint  # a function returning the newest checkpoint name
        self.clock = clock                      # replaced in tests
        self._last_write = float("-inf")

    def on_step(self, state, loss, model, optimizer):
        now = self.clock()
        if now - self._last_write >= self.every_s:
            runlib.write_heartbeat(self.path, state.step, state.epoch, loss, self.last_checkpoint(), "running", now=now)
            self._last_write = now

    def finish(self, state, loss, status="done"):
        """Writes the final heartbeat with status "done" or "failed"."""
        runlib.write_heartbeat(self.path, state.step, state.epoch, loss, self.last_checkpoint(), status, now=self.clock())
