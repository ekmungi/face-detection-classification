"""Tests for the end-to-end notebook's bootstrap cell: clone the code on Colab, stop on a Colab machine without a GPU.

The cell runs before the project can be imported, so it is plain inline code in tools/build_notebook.py. These tests run its
source with stand-ins for the three calls that touch the machine: os.path.isdir, shutil.which and subprocess.run.
"""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parent.parent
MARKER = "# Bootstrap"
CLONE = "/content/face-detection-classification"
REPO_URL = "https://github.com/ekmungi/face-detection-classification"


def builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_source():
    cells = [cell for cell in builder().CELLS if cell[0] == "code" and cell[1].startswith(MARKER)]
    assert len(cells) == 1, "the notebook needs exactly one bootstrap cell"
    return cells[0][1]


def run_bootstrap(monkeypatch, *, on_colab, gpu, cloned):
    """Runs the bootstrap cell on a pretend machine and returns the subprocess.run calls it made."""
    calls = []
    real_isdir = os.path.isdir

    def isdir(path):
        text = str(path).replace("\\", "/")
        if text == "/content":
            return on_colab
        if text == CLONE:
            return cloned
        return real_isdir(path)

    monkeypatch.setattr(os.path, "isdir", isdir)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi" if gpu and name == "nvidia-smi" else None)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    exec(compile(bootstrap_source(), "bootstrap-cell", "exec"), {})
    return calls


def test_bootstrap_does_nothing_on_the_laptop(monkeypatch):
    assert run_bootstrap(monkeypatch, on_colab=False, gpu=False, cloned=False) == []


def test_bootstrap_stops_on_a_colab_machine_without_a_gpu(monkeypatch):
    with pytest.raises(RuntimeError) as error:
        run_bootstrap(monkeypatch, on_colab=True, gpu=False, cloned=False)
    message = str(error.value)
    assert "GPU" in message and "T4, L4 or A100" in message and "Remove" in message


def test_bootstrap_clones_on_a_colab_gpu_machine_without_the_code(monkeypatch):
    calls = run_bootstrap(monkeypatch, on_colab=True, gpu=True, cloned=False)
    assert calls == [(["git", "clone", "--depth", "1", REPO_URL, CLONE], {"check": True})]


def test_bootstrap_leaves_an_existing_clone_alone(monkeypatch):
    assert run_bootstrap(monkeypatch, on_colab=True, gpu=True, cloned=True) == []


def test_bootstrap_runs_after_the_parameters_and_before_the_setup():
    cells = builder().CELLS
    parameters = next(i for i, cell in enumerate(cells) if cell[0] == "code" and "parameters" in cell[2])
    bootstrap = next(i for i, cell in enumerate(cells) if cell[0] == "code" and cell[1].startswith(MARKER))
    setup = next(i for i, cell in enumerate(cells) if cell[0] == "code" and 'os.environ.get("FACEKIT_REPO"' in cell[1])
    assert parameters < bootstrap < setup


def test_committed_notebook_matches_the_builder():
    committed = json.loads((ROOT / "notebooks" / "03_end_to_end.ipynb").read_text(encoding="utf-8"))["cells"]
    built = builder().CELLS
    as_text = lambda source: source if isinstance(source, str) else "".join(source)  # noqa: E731
    assert [as_text(cell["source"]) for cell in committed] == [cell[1] for cell in built], (
        "notebooks/03_end_to_end.ipynb is out of date: run python tools/build_notebook.py")
