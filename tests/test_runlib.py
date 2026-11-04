"""Tests for runlib: atomic JSON writes, heartbeat age, checkpoint discovery and pruning, spec loading."""
import json

import pytest

import runlib


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "metrics.json"
    runlib.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]


def test_heartbeat_round_trip_and_age(tmp_path):
    path = tmp_path / "heartbeat.json"
    runlib.write_heartbeat(path, step=10, epoch=0, loss=0.5, last_checkpoint=None, status="running", now=1000.0)
    heartbeat = runlib.read_heartbeat(path)
    assert heartbeat["step"] == 10 and heartbeat["status"] == "running"
    assert runlib.heartbeat_age(heartbeat, now=1042.5) == pytest.approx(42.5)


def test_read_heartbeat_returns_none_when_missing_or_corrupt(tmp_path):
    assert runlib.read_heartbeat(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert runlib.read_heartbeat(bad) is None
    assert runlib.heartbeat_age(None) is None


def test_latest_checkpoint_orders_by_step_number_not_by_text(tmp_path):
    for step in (900, 1000, 20):
        runlib.checkpoint_path(tmp_path, step).write_bytes(b"x")
    (tmp_path / "checkpoint-notes.txt").write_text("ignored")
    assert runlib.latest_checkpoint(tmp_path).name == "checkpoint-00001000.pt"


def test_latest_checkpoint_is_none_for_an_empty_folder(tmp_path):
    assert runlib.latest_checkpoint(tmp_path) is None


def test_prune_checkpoints_keeps_only_the_newest(tmp_path):
    for step in (100, 200, 300, 400):
        runlib.checkpoint_path(tmp_path, step).write_bytes(b"x")
    removed = runlib.prune_checkpoints(tmp_path, keep=2)
    assert [p.name for p in removed] == ["checkpoint-00000100.pt", "checkpoint-00000200.pt"]
    assert [s for s, _ in runlib.list_checkpoints(tmp_path)] == [300, 400]


def test_prune_checkpoints_rejects_keep_below_one(tmp_path):
    with pytest.raises(ValueError):
        runlib.prune_checkpoints(tmp_path, keep=0)


def test_load_spec_reads_toml(tmp_path):
    spec = tmp_path / "spec.toml"
    spec.write_text('[train]\nepochs = 3\n')
    assert runlib.load_spec(spec)["train"]["epochs"] == 3
