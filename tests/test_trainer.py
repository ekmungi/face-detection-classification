"""Tests for trainer.fit: history shape, early stop, callbacks, and exact resume from a checkpoint."""
import pytest

torch = pytest.importorskip("torch")

import callbacks  # noqa: E402
import models  # noqa: E402
import trainer  # noqa: E402
from synthetic import make_dataset  # noqa: E402


def build(seed=0):
    torch.manual_seed(seed)
    model = models.MLP()
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


def test_fit_records_one_loss_per_step_and_one_metric_per_epoch():
    data = make_dataset(64)
    model, optimizer = build()
    history, state, stopped = trainer.fit(model, optimizer, data, data, epochs=2, batch_size=16, device="cpu")
    assert len(history["step_loss"]) == 8
    assert all(len(history[k]) == 2 for k in ("train_loss", "train_acc", "val_loss", "val_acc"))
    assert state == trainer.TrainState(step=8, epoch=2, batch_in_epoch=0)
    assert not stopped


def test_max_steps_stops_early_and_records_the_position():
    data = make_dataset(64)
    model, optimizer = build()
    history, state, stopped = trainer.fit(model, optimizer, data, data, 3, 16, "cpu", max_steps=5)
    assert stopped and state == trainer.TrainState(step=5, epoch=1, batch_in_epoch=1)
    assert len(history["step_loss"]) == 5


def test_eval_train_false_skips_the_training_set_metrics():
    data = make_dataset(32)
    model, optimizer = build()
    history, _, _ = trainer.fit(model, optimizer, data, data, 1, 16, "cpu", eval_train=False)
    assert history["train_acc"] == [] and len(history["val_acc"]) == 1


def test_callbacks_are_called_after_every_step_and_epoch():
    class Counter(trainer.Callback):
        def __init__(self):
            self.steps, self.epochs = [], []

        def on_step(self, state, loss, model, optimizer):
            self.steps.append(state.step)

        def on_epoch_end(self, state, metrics, model, optimizer):
            self.epochs.append(state.epoch)

    data, counter = make_dataset(32), Counter()
    model, optimizer = build()
    trainer.fit(model, optimizer, data, data, 2, 16, "cpu", callbacks=[counter])
    assert counter.steps == [1, 2, 3, 4]
    assert counter.epochs == [1, 2]


def test_a_run_resumed_from_a_checkpoint_matches_an_uninterrupted_run_exactly(tmp_path):
    data = make_dataset(64)
    model_a, opt_a = build()
    history_a, _, _ = trainer.fit(model_a, opt_a, data, data, 3, 16, "cpu")  # 12 steps, uninterrupted

    model_b, opt_b = build()
    saver = callbacks.CheckpointCallback(tmp_path, every=5, keep=2)
    trainer.fit(model_b, opt_b, data, data, 3, 16, "cpu", callbacks=[saver], max_steps=7)  # "crashes" at step 7

    model_c, opt_c = build(seed=99)  # a different initialisation, to prove the checkpoint restores everything
    state = callbacks.CheckpointCallback.load(tmp_path / saver.last, model_c, opt_c, "cpu")
    assert state == trainer.TrainState(step=5, epoch=1, batch_in_epoch=1)
    history_c, final, _ = trainer.fit(model_c, opt_c, data, data, 3, 16, "cpu", state=state)

    assert history_c["step_loss"] == history_a["step_loss"][5:]
    assert final.step == 12
    assert all(torch.equal(a, c) for a, c in zip(model_a.state_dict().values(), model_c.state_dict().values()))
