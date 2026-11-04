"""Tests for paths.py: where the artifacts folder is, how the synced Drive folder is found, and the folder naming rules."""
from pathlib import Path

import pytest

import paths


def make_mount(tmp_path, *relative_dirs):
    for relative in relative_dirs:
        (tmp_path / relative).mkdir(parents=True)
    return tmp_path


def test_the_environment_variable_wins():
    assert paths.artifact_root({"ARTIFACT_ROOT": "/somewhere"}, on_colab=True) == Path("/somewhere")


def test_drive_artifacts_are_found_under_mydrive(tmp_path):
    mount = make_mount(tmp_path, f"MyDrive/ResearchArtifacts/{paths.PROJECT}")
    assert paths.drive_artifacts_dir({}, mount) == mount / "MyDrive" / "ResearchArtifacts" / paths.PROJECT


def test_drive_artifacts_are_found_in_a_computer_folder_synced_by_the_drive_app(tmp_path):
    mount = make_mount(tmp_path, f"Othercomputers/MY-PC/ResearchArtifacts/{paths.PROJECT}")
    assert paths.drive_artifacts_dir({}, mount) == mount / "Othercomputers" / "MY-PC" / "ResearchArtifacts" / paths.PROJECT


def test_a_synced_computer_folder_wins_over_mydrive_when_both_exist(tmp_path):
    mount = make_mount(tmp_path, f"MyDrive/ResearchArtifacts/{paths.PROJECT}",
                       f"Othercomputers/MY-PC/ResearchArtifacts/{paths.PROJECT}")
    assert paths.drive_artifacts_dir({}, mount) == mount / "Othercomputers" / "MY-PC" / "ResearchArtifacts" / paths.PROJECT


def test_with_several_synced_computers_the_first_by_name_wins(tmp_path):
    mount = make_mount(tmp_path, f"Othercomputers/ZED/ResearchArtifacts/{paths.PROJECT}",
                       f"Othercomputers/ALPHA/ResearchArtifacts/{paths.PROJECT}",
                       f"MyDrive/ResearchArtifacts/{paths.PROJECT}")
    assert paths.drive_artifacts_dir({}, mount) == mount / "Othercomputers" / "ALPHA" / "ResearchArtifacts" / paths.PROJECT


def test_the_drive_environment_variable_beats_both_drive_folders(tmp_path):
    mount = make_mount(tmp_path, f"MyDrive/ResearchArtifacts/{paths.PROJECT}",
                       f"Othercomputers/MY-PC/ResearchArtifacts/{paths.PROJECT}")
    assert paths.drive_artifacts_dir({"DRIVE_ARTIFACTS_DIR": "/x/y"}, mount) == Path("/x/y")


def test_on_colab_artifacts_follow_the_synced_computer_folder_when_both_exist(tmp_path):
    mount = make_mount(tmp_path, f"MyDrive/ResearchArtifacts/{paths.PROJECT}",
                       f"Othercomputers/MY-PC/ResearchArtifacts/{paths.PROJECT}")
    assert paths.artifact_root({}, on_colab=True, mount=mount) == mount / "Othercomputers" / "MY-PC" / "ResearchArtifacts" / paths.PROJECT


def test_the_drive_environment_variable_overrides_the_search(tmp_path):
    assert paths.drive_artifacts_dir({"DRIVE_ARTIFACTS_DIR": "/x/y"}, tmp_path) == Path("/x/y")


def test_drive_artifacts_are_none_when_nothing_matches(tmp_path):
    assert paths.drive_artifacts_dir({}, tmp_path) is None


def test_on_colab_with_the_folder_synced_artifacts_are_there(tmp_path):
    mount = make_mount(tmp_path, f"Othercomputers/MY-PC/ResearchArtifacts/{paths.PROJECT}")
    assert paths.artifact_root({}, on_colab=True, mount=mount) == mount / "Othercomputers" / "MY-PC" / "ResearchArtifacts" / paths.PROJECT


def test_on_colab_without_the_folder_the_machine_disk_is_used(tmp_path):
    assert paths.artifact_root({}, on_colab=True, mount=tmp_path) == Path("/content/artifacts")


def test_on_this_computer_the_research_artifacts_folder_is_used_when_it_exists(tmp_path):
    shared = make_mount(tmp_path, f"ResearchArtifacts/{paths.PROJECT}") / "ResearchArtifacts" / paths.PROJECT
    assert paths.artifact_root({}, on_colab=False, home=tmp_path) == shared


def test_on_this_computer_without_that_folder_artifacts_sits_in_the_project(tmp_path):
    assert paths.artifact_root({}, on_colab=False, home=tmp_path) == paths.project_root() / "artifacts"


def test_run_dir_builds_the_expected_path(tmp_path):
    assert paths.run_dir("exp-002", "run-007", tmp_path) == tmp_path / "exp-002" / "run-007"
    assert paths.run_dir("scratch", "smoke-001", tmp_path) == tmp_path / "scratch" / "smoke-001"


@pytest.mark.parametrize("exp, run_id", [("exp-1", "run-001"), ("exp-001", "run-1"), ("exp-001", "smoke-001"),
                                         ("scratch", "run-001"), ("experiment", "run-001")])
def test_bad_names_are_rejected(exp, run_id):
    with pytest.raises(ValueError):
        paths.check_names(exp, run_id)


def test_the_work_data_folder_is_the_machine_disk_on_colab():
    assert paths.work_data_dir({}, on_colab=True) == Path("/content/data")


def test_the_work_data_folder_on_this_computer_is_a_cache_outside_the_artifacts(tmp_path):
    assert paths.work_data_dir({}, on_colab=False, home=tmp_path) == tmp_path / ".cache" / "research-data" / paths.PROJECT


def test_the_work_data_environment_variable_wins():
    assert paths.work_data_dir({"WORK_DATA_DIR": "/somewhere/else"}, on_colab=True) == Path("/somewhere/else")


def test_data_and_figure_folders(tmp_path):
    assert paths.data_dir(tmp_path) == tmp_path / "data"
    assert paths.figure_dir("02_pipeline", tmp_path) == tmp_path / "scratch" / "figures" / "02_pipeline"
