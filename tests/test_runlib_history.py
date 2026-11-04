"""Tests for the run-history helpers in runlib: write-once files, append-only JSON lines, code version, environment."""
import runlib


def test_write_once_writes_text_and_objects_and_refuses_to_overwrite(tmp_path):
    assert runlib.write_once(tmp_path / "git.txt", "abc123")
    assert (tmp_path / "git.txt").read_text() == "abc123\n"
    assert not runlib.write_once(tmp_path / "git.txt", "different")
    assert (tmp_path / "git.txt").read_text() == "abc123\n"
    assert runlib.write_once(tmp_path / "config.json", {"lr": 0.001})
    assert not runlib.write_once(tmp_path / "config.json", {"lr": 5})
    assert '"lr": 0.001' in (tmp_path / "config.json").read_text()


def test_append_jsonl_only_grows_and_read_jsonl_returns_every_line_in_order(tmp_path):
    path = tmp_path / "attempts.jsonl"
    assert runlib.read_jsonl(path) == []
    runlib.append_jsonl(path, {"event": "start"})
    runlib.append_jsonl(path, {"event": "resume", "step": 5})
    assert runlib.read_jsonl(path) == [{"event": "start"}, {"event": "resume", "step": 5}]


def test_code_version_prefers_the_marker_file_then_git_then_unknown(tmp_path):
    assert runlib.read_code_version(tmp_path) == "unknown"
    (tmp_path / "code_version.txt").write_text("deadbeef dirty\n")
    assert runlib.read_code_version(tmp_path) == "deadbeef dirty"


def test_environment_info_has_python_and_platform_and_the_extra_entries():
    info = runlib.environment_info({"device": "cpu"})
    assert set(info) == {"python", "platform", "device"} and info["device"] == "cpu"
