"""Tests for runlib, the helpers that a training run uses to record its progress and to resume after a stop.

They cover atomic JSON writes (a file is either complete or absent, never half written), the heartbeat file that shows
a run is alive and how old it is, finding and pruning checkpoints (saved states of a model from which a run can resume),
and loading a run specification from a TOML file.
"""
import json

import pytest

import runlib


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "metrics.json"
    runlib.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}

    # The folder must hold only the final file, so the temporary file used during the write has been removed.
    assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]


def test_heartbeat_round_trip_and_age(tmp_path):
    path = tmp_path / "heartbeat.json"
    runlib.write_heartbeat(path, step=10, epoch=0, loss=0.5, last_checkpoint=None, status="running", now=1000.0)
    heartbeat = runlib.read_heartbeat(path)
    assert heartbeat["step"] == 10 and heartbeat["status"] == "running"

    # The age is the time in seconds between the heartbeat's own time (1000.0) and the time given as now.
    assert runlib.heartbeat_age(heartbeat, now=1042.5) == pytest.approx(42.5)


def test_read_heartbeat_returns_none_when_missing_or_corrupt(tmp_path):
    assert runlib.read_heartbeat(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert runlib.read_heartbeat(bad) is None
    assert runlib.heartbeat_age(None) is None


def test_latest_checkpoint_orders_by_step_number_not_by_text(tmp_path):
    # The folder holds checkpoints from steps 900, 1000 and 20. The latest one is the one with the highest step number.
    for step in (900, 1000, 20):
        runlib.checkpoint_path(tmp_path, step).write_bytes(b"x")

    # A file that merely starts with the same word is not a checkpoint and must be ignored.
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
