"""The shared training loop: it trains, records its run files, resumes exactly, and skips a finished run."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from facekit import engine
from facekit.arrays import ArrayDataset, epoch_loader


class ToyTask:
    """Two separable classes of 4x4 images; the model must learn them."""

    monitor = "accuracy"
    monitor_mode = "max"

    def __init__(self, seed=0, n=64):
        rng = np.random.default_rng(seed)
        labels = np.arange(n) % 2
        images = (rng.random((n, 4, 4, 3)) * 60 + labels[:, None, None, None] * 150).astype(np.uint8)
        self.dataset = ArrayDataset(images, labels)
        torch.manual_seed(seed)   # independent of whatever ran before
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
        self.steps_per_epoch = n // 8

    def train_batches(self, epoch, start_batch):
        return epoch_loader(self.dataset, 8, 0, epoch, start_batch, num_workers=0)

    def training_step(self, batch, device):
        x, y = batch
        logits = self.model(x.float().to(device) / 255)
        return F.cross_entropy(logits, y.to(device)), {}

    def validate(self, device):
        x = torch.stack([self.dataset[i][0] for i in range(len(self.dataset))]).float() / 255
        y = torch.tensor([self.dataset[i][1] for i in range(len(self.dataset))])
        return {"accuracy": float((self.model(x).argmax(1) == y).float().mean())}


def _cfg(**kw):
    base = dict(epochs=3, lr=0.05, weight_decay=0.0, warmup_epochs=0.5, amp=False, checkpoint_every=5, log_every=100)
    base.update(kw)
    return engine.TrainConfig(**base)


def test_fit_trains_and_writes_run_files(tmp_path):
    task = ToyTask()
    result = engine.fit(task, _cfg(), tmp_path / "run", device="cpu", log=lambda s: None)
    assert result.status == "done"
    assert result.best_value == 1.0
    run = tmp_path / "run"
    for name in ("config.json", "env.json", "git.txt", "metrics.json", "DONE", "best.pt", "final.pt", "heartbeat.json"):
        assert (run / name).exists(), name
    assert len(engine.load_history(run)) == 3
    assert json.loads((run / "heartbeat.json").read_text())["status"] == "done"


def test_finished_run_is_not_trained_again(tmp_path):
    engine.fit(ToyTask(), _cfg(), tmp_path / "run", device="cpu", log=lambda s: None)
    again = engine.fit(ToyTask(), _cfg(), tmp_path / "run", device="cpu", log=lambda s: None)
    assert again.status == "already-done"
    assert len(again.history) == 3


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    torch.manual_seed(0)
    full_task = ToyTask()
    engine.fit(full_task, _cfg(), tmp_path / "full", device="cpu", log=lambda s: None)
    torch.manual_seed(0)
    part_task = ToyTask()
    stopped = engine.fit(part_task, _cfg(max_steps=13), tmp_path / "part", device="cpu", log=lambda s: None)
    assert stopped.status == "stopped-max-steps"
    torch.manual_seed(0)
    resumed_task = ToyTask()
    engine.fit(resumed_task, _cfg(), tmp_path / "part", device="cpu", log=lambda s: None)
    for a, b in zip(full_task.model.parameters(), resumed_task.model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_lr_factor_warms_up_then_decays():
    assert engine.lr_factor(0, 100, 10, 0.0) == 0.1
    assert engine.lr_factor(9, 100, 10, 0.0) == 1.0
    assert engine.lr_factor(100, 100, 10, 0.0) == 0.0
    assert 0.0 < engine.lr_factor(55, 100, 10, 0.0) < 1.0


def test_no_weight_decay_on_biases_and_norms():
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    decay, no_decay = engine.parameter_groups(model, 0.1)
    assert len(decay["params"]) == 1 and len(no_decay["params"]) == 3


def test_overfit_one_batch_drives_loss_down():
    torch.manual_seed(0)
    x, y = torch.randn(8, 5), torch.randint(0, 3, (8,))
    losses = engine.overfit_one_batch(nn.Linear(5, 3), lambda m: F.cross_entropy(m(x), y), steps=200, lr=0.05)
    assert losses[-1] < 0.1 * losses[0]


def test_time_schedule_anneals_and_marks_done(tmp_path):
    task = ToyTask()
    cfg = _cfg(epochs=50, max_minutes=0.02, time_schedule=True)   # 1.2 seconds
    result = engine.fit(task, cfg, tmp_path / "run", device="cpu", log=lambda s: None)
    metrics_json = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert result.status in ("schedule-complete", "done")
    assert metrics_json["annealed"] is True
    assert metrics_json["final_lr"] <= 0.05 * 0.011
    assert (tmp_path / "run" / "DONE").exists()


def test_resume_keeps_the_saved_configuration(tmp_path):
    engine.fit(ToyTask(), _cfg(max_steps=13), tmp_path / "run", device="cpu", log=lambda s: None)
    lines = []
    engine.fit(ToyTask(), _cfg(epochs=6), tmp_path / "run", device="cpu", log=lines.append)
    assert any("saved configuration" in line for line in lines)
    assert json.loads((tmp_path / "run" / "config.json").read_text())["train"]["epochs"] == 3
