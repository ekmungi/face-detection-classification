"""Tests for the setup cell of each notebook: it must say which artifacts folder is in use and warn when results would go to a Colab machine's own disk."""
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
NAMES = ["01_helpers", "02_pipeline"]

# Phrases the warning must carry: where results go, that they are lost, and why (Drive not mounted or not synced).
WARNING_PHRASES = ["WARNING", "machine's own disk", "lost when", "Drive is not mounted", "not synced"]


def setup_cell_source(name):
    """Returns the source of the notebook's setup cell: the code cell that defines FIG_DIR."""
    notebook = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    (source,) = [c for c in cells if "FIG_DIR = paths.figure_dir(" in c]
    return source


@pytest.fixture
def run_setup_cell(monkeypatch, capsys):
    """Returns a function that runs a notebook's setup cell as if on (or off) Colab and returns what it printed.

    /content is faked through os.path.isdir, ARTIFACT_ROOT chooses the artifacts folder, and nothing is created on disk
    under /content.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    real_isdir, real_makedirs = os.path.isdir, os.makedirs

    def run(name, on_colab, artifact_root):
        """Runs notebook `name`'s setup cell with ARTIFACT_ROOT=artifact_root and /content present or absent."""
        monkeypatch.setenv("ARTIFACT_ROOT", str(artifact_root))
        monkeypatch.setattr(os.path, "isdir", lambda p: on_colab if os.fspath(p) == "/content" else real_isdir(p))
        monkeypatch.setattr(os, "makedirs", lambda p, *a, **k: None if os.fspath(p).replace("\\", "/").startswith("/content")
                            else real_makedirs(p, *a, **k))
        capsys.readouterr()
        exec(compile(setup_cell_source(name), f"{name}.ipynb setup cell", "exec"), {"CODE_DIRS": []})
        return capsys.readouterr().out

    return run


@pytest.mark.parametrize("name", NAMES)
def test_the_setup_cell_carries_the_machine_disk_warning(name):
    source = setup_cell_source(name)
    for phrase in WARNING_PHRASES:
        assert phrase in source, f"{name}: the setup cell does not contain {phrase!r}"
    assert "/content/artifacts" in source


@pytest.mark.parametrize("name", NAMES)
def test_on_colab_with_results_going_to_the_machine_disk_the_cell_warns(run_setup_cell, name):
    out = run_setup_cell(name, on_colab=True, artifact_root="/content/artifacts")
    assert "artifacts folder: " in out and "content" in out
    assert all(phrase in out for phrase in WARNING_PHRASES)


@pytest.mark.parametrize("name", NAMES)
def test_on_colab_with_the_drive_folder_the_cell_names_it_and_does_not_warn(run_setup_cell, tmp_path, name):
    drive = tmp_path / "drive" / "Othercomputers" / "PC" / "ResearchArtifacts" / "proj"
    out = run_setup_cell(name, on_colab=True, artifact_root=drive)
    assert f"artifacts folder: {drive}" in out
    assert "WARNING" not in out


@pytest.mark.parametrize("name", NAMES)
def test_on_this_computer_the_cell_names_the_folder_and_does_not_warn(run_setup_cell, tmp_path, name):
    out = run_setup_cell(name, on_colab=False, artifact_root=tmp_path / "artifacts")
    assert f"artifacts folder: {tmp_path / 'artifacts'}" in out
    assert "WARNING" not in out


@pytest.mark.parametrize("name", NAMES)
def test_off_colab_the_warning_never_shows_even_for_a_content_artifacts_folder(run_setup_cell, name):
    out = run_setup_cell(name, on_colab=False, artifact_root="/content/artifacts")
    assert "artifacts folder: " in out and "WARNING" not in out
