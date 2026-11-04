"""Tests for run_eval.py: the evaluation entry point writes the run folder, the heartbeat and metrics.json, and trains nothing."""
import sys

import pytest

import run_eval
import runlib

SPEC = ('[experiment]\nid = "exp-002"\npipeline = "evaluate"\n\n[evaluate]\nmodel = "encoder-x"\nmetric = "recall_at_5"\n'
        'eval_split = "test"\nbaseline = "mean"\nseed = 0\nbatch_size = 4\ncache_outputs = false\n')


def make_args(tmp_path, spec_text=SPEC, extra=()):
    """Parses a command line for a smoke run whose specification and artifacts folder live in tmp_path."""
    spec = tmp_path / "spec.toml"
    spec.write_text(spec_text, encoding="utf-8")
    argv = ["--spec", str(spec), "--exp", "scratch", "--run-id", "smoke-001", "--artifact-root", str(tmp_path / "art"), *extra]
    return run_eval.parse_args(argv)


def run_folder(tmp_path):
    return tmp_path / "art" / "scratch" / "smoke-001"


def test_a_run_writes_the_run_folder_the_scores_and_the_done_marker(tmp_path):
    seen = []

    def score(table, progress):
        seen.append(table)
        return {"recall_at_5": 0.5}

    metrics = run_eval.run(make_args(tmp_path), score, say=lambda message: None)
    folder = run_folder(tmp_path)
    assert seen[0]["model"] == "encoder-x" and seen[0]["batch_size"] == 4
    assert metrics["recall_at_5"] == 0.5 and metrics["run_id"] == "smoke-001" and metrics["exp"] == "scratch"
    assert runlib.read_heartbeat(folder / "heartbeat.json")["status"] == "done"
    assert all((folder / name).is_file() for name in ("config.json", "env.json", "git.txt", "metrics.json", "DONE"))
    assert not (folder / "checkpoints").exists()


def test_progress_moves_the_heartbeat_to_the_number_of_items_scored(tmp_path):
    during = []

    def score(table, progress):
        progress(2, 8)
        during.append(runlib.read_heartbeat(run_folder(tmp_path) / "heartbeat.json"))
        return {}

    run_eval.run(make_args(tmp_path), score, say=lambda message: None)
    assert during[0]["step"] == 2 and during[0]["status"] == "running"
    assert runlib.read_heartbeat(run_folder(tmp_path) / "heartbeat.json")["step"] == 2


def test_a_failing_score_leaves_a_failed_heartbeat_and_no_done_marker(tmp_path):
    def score(table, progress):
        progress(3, 8)
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_eval.run(make_args(tmp_path), score, say=lambda message: None)
    beat = runlib.read_heartbeat(run_folder(tmp_path) / "heartbeat.json")
    assert beat["status"] == "failed" and beat["step"] == 3
    assert not (run_folder(tmp_path) / "DONE").exists() and not (run_folder(tmp_path) / "metrics.json").exists()


@pytest.mark.parametrize("result", [
    None, 0.5, [("recall_at_5", 0.5)], {"recall_at_5": "high"}, {"recall_at_5": True}, {"recall_at_5": None},
    {"recall_at_5": float("nan")}, {"recall_at_5": float("inf")}, {1: 0.5},
])
def test_a_score_result_that_is_not_a_dict_of_numbers_is_a_failed_evaluation(tmp_path, result):
    """A wrong result is the evaluation's failure, so the heartbeat must say failed and nothing may claim the run finished."""
    with pytest.raises((TypeError, ValueError), match="score"):
        run_eval.run(make_args(tmp_path), lambda table, progress: result, say=lambda message: None)
    folder = run_folder(tmp_path)
    assert runlib.read_heartbeat(folder / "heartbeat.json")["status"] == "failed"
    assert not (folder / "DONE").exists() and not (folder / "metrics.json").exists()


def test_a_score_result_with_whole_and_decimal_numbers_is_accepted(tmp_path):
    metrics = run_eval.run(make_args(tmp_path), lambda table, progress: {"hits": 3, "recall_at_5": 0.5}, say=lambda message: None)
    assert metrics["hits"] == 3 and (run_folder(tmp_path) / "DONE").is_file()


def test_a_specification_without_an_evaluate_table_is_refused(tmp_path):
    args = make_args(tmp_path, spec_text='[experiment]\nid = "exp-001"\n\n[train]\nepochs = 1\n')
    with pytest.raises(SystemExit, match=r"\[evaluate\]"):
        run_eval.run(args, lambda table, progress: {}, say=lambda message: None)


def test_the_first_attempt_describes_the_run_and_a_second_attempt_keeps_it(tmp_path):
    run_eval.run(make_args(tmp_path), lambda table, progress: {"a": 1}, say=lambda message: None)
    first = (run_folder(tmp_path) / "config.json").read_text(encoding="utf-8")
    run_eval.run(make_args(tmp_path, spec_text=SPEC.replace("encoder-x", "encoder-y")), lambda table, progress: {"a": 2}, say=lambda message: None)
    assert (run_folder(tmp_path) / "config.json").read_text(encoding="utf-8") == first and "encoder-x" in first


def test_the_arguments_are_the_ones_train_py_takes_for_a_run(tmp_path):
    args = make_args(tmp_path)
    assert (args.exp, args.run_id) == ("scratch", "smoke-001") and args.spec.endswith("spec.toml") and args.artifact_root
    with pytest.raises(SystemExit) as stop:
        run_eval.parse_args(["--exp", "scratch", "--run-id", "smoke-001"])
    assert stop.value.code == 2


def test_a_badly_named_run_is_refused_before_anything_is_written(tmp_path):
    args = make_args(tmp_path, extra=["--exp", "exp-1"])
    with pytest.raises(ValueError, match="exp-NNN"):
        run_eval.run(args, lambda table, progress: {}, say=lambda message: None)
    assert not (tmp_path / "art").exists()


def test_main_runs_the_projects_own_score_and_logs_to_evaluate_log(tmp_path, monkeypatch):
    (tmp_path / "evaluation.py").write_text("def score(table, progress):\n    progress(1, 1)\n    return {'recall_at_5': 1.0}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "evaluation", raising=False)
    spec = tmp_path / "spec.toml"
    spec.write_text(SPEC, encoding="utf-8")
    code = run_eval.main(["--spec", str(spec), "--exp", "scratch", "--run-id", "smoke-001", "--artifact-root", str(tmp_path / "art")])
    log = (run_folder(tmp_path) / "run_eval.log").read_text(encoding="utf-8")
    assert code == 0 and "scored 1 of 1" in log and "finished" in log and (run_folder(tmp_path) / "DONE").is_file()
    monkeypatch.delitem(sys.modules, "evaluation", raising=False)


def test_main_says_what_is_missing_when_the_project_has_no_evaluation_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "evaluation", None)
    spec = tmp_path / "spec.toml"
    spec.write_text(SPEC, encoding="utf-8")
    with pytest.raises(SystemExit, match=r"evaluation\.py"):
        run_eval.main(["--spec", str(spec), "--exp", "scratch", "--run-id", "smoke-001", "--artifact-root", str(tmp_path / "art")])
