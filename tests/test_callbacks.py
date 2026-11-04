"""Tests for the checkpoint and heartbeat callbacks."""
import pytest

torch = pytest.importorskip("torch")

import callbacks  # noqa: E402
import models  # noqa: E402
import runlib  # noqa: E402
import trainer  # noqa: E402
from synthetic import make_dataset  # noqa: E402


def build():
    torch.manual_seed(0)
    model = models.MLP()
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


def test_checkpoints_are_saved_at_the_interval_and_old_ones_pruned(tmp_path):
    data = make_dataset(64)
    model, optimizer = build()
    saver = callbacks.CheckpointCallback(tmp_path, every=3, keep=1)
    trainer.fit(model, optimizer, data, data, 2, 16, "cpu", callbacks=[saver])  # 8 steps: saves at 3 and 6
    assert [s for s, _ in runlib.list_checkpoints(tmp_path)] == [6]
    assert saver.last == "checkpoint-00000006.pt"


def test_load_restores_the_weights_and_returns_the_position(tmp_path):
    model, optimizer = build()
    saver = callbacks.CheckpointCallback(tmp_path)
    path = saver.save(trainer.TrainState(step=40, epoch=2, batch_in_epoch=7), model, optimizer)
    other, other_opt = build()
    for p in other.parameters():
        p.data.add_(1.0)
    state = callbacks.CheckpointCallback.load(path, other, other_opt, "cpu")
    assert state == trainer.TrainState(40, 2, 7)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))


def test_heartbeat_is_written_at_most_once_per_interval_and_finished_with_a_status(tmp_path):
    times = iter([0.0, 1.0, 6.0, 7.0])
    beat = callbacks.HeartbeatCallback(tmp_path / "hb.json", every_s=5.0, last_checkpoint=lambda: "checkpoint-00000002.pt",
                                       clock=lambda: next(times))
    state = trainer.TrainState
    beat.on_step(state(1, 0, 1), 0.9, None, None)  # written (first call)
    beat.on_step(state(2, 0, 2), 0.8, None, None)  # skipped, only 1 s later
    assert runlib.read_heartbeat(tmp_path / "hb.json")["step"] == 1
    beat.on_step(state(3, 0, 3), 0.7, None, None)  # written, 6 s after the first
    heartbeat = runlib.read_heartbeat(tmp_path / "hb.json")
    assert heartbeat["step"] == 3 and heartbeat["last_checkpoint"] == "checkpoint-00000002.pt"
    beat.finish(state(4, 1, 0), 0.6, "done")
    assert runlib.read_heartbeat(tmp_path / "hb.json")["status"] == "done"
