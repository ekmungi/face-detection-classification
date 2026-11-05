"""Tests for runlib.claim_run_mode, which makes a run folder belong to one notebook mode.

The trainings reuse the plans they find in their run folder, so a medium run started in a smoke run's folder would
train on the smoke run's schedule. The notebook therefore records its mode in the run folder and stops when the folder
holds a run of another mode.
"""
import json

import pytest

import runlib


def test_a_new_run_folder_records_its_mode(tmp_path):
    runlib.claim_run_mode(tmp_path, "medium")
    assert (tmp_path / runlib.RUN_MODE_FILE).read_text(encoding="utf-8").strip() == "medium"


def test_a_run_of_the_same_mode_may_continue_in_its_folder(tmp_path):
    runlib.claim_run_mode(tmp_path, "medium")
    runlib.claim_run_mode(tmp_path, "medium")
    assert (tmp_path / runlib.RUN_MODE_FILE).read_text(encoding="utf-8").strip() == "medium"


def test_a_folder_of_another_mode_stops_the_run_with_a_plain_message(tmp_path):
    runlib.claim_run_mode(tmp_path, "smoke")
    with pytest.raises(RuntimeError) as error:
        runlib.claim_run_mode(tmp_path, "medium")

    # The message must name both modes and tell the user to choose another RUN_ID. The folder must keep its mode.
    message = str(error.value)
    assert "smoke" in message and "medium" in message and "RUN_ID" in message
    assert (tmp_path / runlib.RUN_MODE_FILE).read_text(encoding="utf-8").strip() == "smoke"


def test_an_earlier_run_without_a_mode_file_is_known_by_its_summary(tmp_path):
    # This folder comes from a run made before the mode file existed, so only its summary file names the mode.
    (tmp_path / "summary.json").write_text(json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        runlib.claim_run_mode(tmp_path, "medium")

    # The mode that the summary names may continue, and the claim then writes the mode file.
    runlib.claim_run_mode(tmp_path, "smoke")
    assert (tmp_path / runlib.RUN_MODE_FILE).read_text(encoding="utf-8").strip() == "smoke"


def test_training_plans_from_a_run_of_unknown_mode_stop_the_run(tmp_path):
    # A plan file exists, but neither a mode file nor a summary says which mode made it.
    (tmp_path / "detector").mkdir()
    (tmp_path / "detector" / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        runlib.claim_run_mode(tmp_path, "medium")

    # The run must stop without claiming the folder.
    assert "not recorded" in str(error.value)
    assert not (tmp_path / runlib.RUN_MODE_FILE).exists()
