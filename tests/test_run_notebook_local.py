"""Tests for tools/run_notebook_local.py: cell-by-cell execution, output rendering, default output path and failure status."""
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import run_notebook_local  # noqa: E402


def write_notebook(path, sources):
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    nbformat.write(notebook, path)


def test_render_output_handles_stream_result_and_error():
    assert run_notebook_local.render_output({"output_type": "stream", "text": "hi\n"}) == "hi\n"
    assert run_notebook_local.render_output({"output_type": "execute_result", "data": {"text/plain": "3"}}) == "3"
    assert "ValueError: bad" in run_notebook_local.render_output({"output_type": "error", "ename": "ValueError", "evalue": "bad"})
    assert run_notebook_local.render_output({"output_type": "unknown"}) == ""


def test_default_out_path_is_under_the_artifact_root_scratch_notebook_out(tmp_path):
    out = run_notebook_local.default_out_path(tmp_path / "notebooks" / "02_pipeline.ipynb", root=tmp_path / "art")
    assert out == tmp_path / "art" / "scratch" / "notebook-out" / "02_pipeline.local.ipynb"


def test_cells_share_one_kernel_and_the_output_notebook_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "art"))
    notebook = tmp_path / "notebooks" / "nb.ipynb"
    write_notebook(notebook, ["value = 41", "print(value + 1)"])
    lines = []
    status = run_notebook_local.run_notebook(notebook, tmp_path, echo=lines.append)
    assert status == 0
    assert "42" in lines
    assert (tmp_path / "art" / "scratch" / "notebook-out" / "nb.local.ipynb").exists()


def test_a_failing_cell_stops_the_run_and_returns_status_one(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "art"))
    notebook = tmp_path / "notebooks" / "bad.ipynb"
    write_notebook(notebook, ["print('first')", "raise ValueError('boom')", "print('never')"])
    lines = []
    assert run_notebook_local.run_notebook(notebook, tmp_path, echo=lines.append) == 1
    assert any("ValueError: boom" in line for line in lines)
    assert "never" not in lines
