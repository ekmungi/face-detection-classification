"""Parity tests: the script path (train.run) behaves like calling the shared fit() directly, resumes exactly, and writes the run history files."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import models  # noqa: E402
import paths  # noqa: E402
import quicktrain  # noqa: E402
import runlib  # noqa: E402
import trainer  # noqa: E402
import train  # noqa: E402
from synthetic import make_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def quiet(message):
    """Swallows the log lines so test output stays clean."""


def args_for(tmp_path, run_id, *extra, exp="exp-001"):
    return train.parse_args(["--artifact-root", str(tmp_path), "--exp", exp, "--run-id", run_id, "--model", "mlp",
                             "--epochs", "3", "--batch-size", "16", "--checkpoint-every", "5", "--seed", "0", *extra])


def test_the_script_path_matches_a_direct_call_to_fit(tmp_path):
    data = make_dataset(64)
    metrics = train.run(args_for(tmp_path, "run-001"), data, data, data, say=quiet)

    train.set_seeds(0)
    model = models.build_model("mlp")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, state, _ = trainer.fit(model, optimizer, data, data, 3, 16, "cpu", seed=0, eval_train=False)
    expected = quicktrain.accuracy(model, DataLoader(data, batch_size=1000), "cpu")

    assert metrics["steps"] == state.step == 12
    assert metrics["val_accuracy"] == expected


def test_a_script_run_resumed_after_a_stop_gives_the_same_result_as_an_uninterrupted_run(tmp_path):
    data = make_dataset(64)
    whole = train.run(args_for(tmp_path, "run-001"), data, data, data, quiet)

    first = train.run(args_for(tmp_path, "run-002", "--max-steps", "7"), data, data, data, quiet)
    assert first["stopped_early"] and first["steps"] == 7
    second = train.run(args_for(tmp_path, "run-002"), data, data, data, quiet)

    assert second["resumed_from"] == "checkpoint-00000005.pt"
    assert second["steps"] == whole["steps"] == 12
    assert (second["val_accuracy"], second["test_accuracy"]) == (whole["val_accuracy"], whole["test_accuracy"])


def test_the_run_folder_holds_the_files_the_convention_names(tmp_path):
    data = make_dataset(64)
    train.run(args_for(tmp_path, "run-001"), data, data, data, quiet)
    run_dir = paths.run_dir("exp-001", "run-001", tmp_path)
    for name in ("config.json", "env.json", "git.txt", "attempts.jsonl", "metrics_history.jsonl", "heartbeat.json",
                 "metrics.json", "DONE"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "checkpoints").is_dir()
    assert [e["event"] for e in runlib.read_jsonl(run_dir / "attempts.jsonl")] == ["start", "finish"]
    assert [e["epoch"] for e in runlib.read_jsonl(run_dir / "metrics_history.jsonl")] == [1, 2, 3]


def test_a_resume_appends_history_and_leaves_the_written_once_files_unchanged(tmp_path):
    data = make_dataset(64)
    run_dir = paths.run_dir("exp-001", "run-002", tmp_path)
    train.run(args_for(tmp_path, "run-002", "--max-steps", "7"), data, data, data, quiet)
    config_before = (run_dir / "config.json").read_text()
    train.run(args_for(tmp_path, "run-002", "--max-steps", "0"), data, data, data, quiet)
    assert (run_dir / "config.json").read_text() == config_before
    assert [e["event"] for e in runlib.read_jsonl(run_dir / "attempts.jsonl")] == ["start", "stopped", "resume", "finish"]


def test_smoke_runs_must_live_in_scratch(tmp_path):
    data = make_dataset(32)
    with pytest.raises(ValueError):
        train.run(args_for(tmp_path, "smoke-001"), data, data, data, quiet)
    metrics = train.run(args_for(tmp_path, "smoke-001", "--max-steps", "3", exp="scratch"), data, data, data, quiet)
    assert metrics["steps"] == 3
    assert (tmp_path / "scratch" / "smoke-001" / "metrics.json").is_file()
