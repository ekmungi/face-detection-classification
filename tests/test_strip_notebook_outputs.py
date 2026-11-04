"""Tests for tools/strip_notebook_outputs.py: committed notebooks must carry no outputs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import strip_notebook_outputs as tool  # noqa: E402

NOTEBOOK = {"cells": [
    {"cell_type": "markdown", "source": ["# title"]},
    {"cell_type": "code", "source": ["print(1)"], "execution_count": 4, "metadata": {},
     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]},
], "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def write(tmp_path, data=NOTEBOOK, name="n.ipynb"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_outputs_and_execution_counts_are_removed_and_the_source_is_kept():
    stripped, changed = tool.strip_notebook(NOTEBOOK)
    code = stripped["cells"][1]
    assert changed and code["outputs"] == [] and code["execution_count"] is None and code["source"] == ["print(1)"]
    assert stripped["metadata"] == NOTEBOOK["metadata"]


def test_the_original_dictionary_is_not_modified():
    tool.strip_notebook(NOTEBOOK)
    assert NOTEBOOK["cells"][1]["outputs"]


def test_a_clean_notebook_is_reported_unchanged():
    clean, _ = tool.strip_notebook(NOTEBOOK)
    assert tool.strip_notebook(clean)[1] is False


def test_main_rewrites_the_file(tmp_path):
    path = write(tmp_path)
    assert tool.main([str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["cells"][1]["outputs"] == []


def test_check_mode_reports_and_changes_nothing(tmp_path, capsys):
    path = write(tmp_path)
    before = path.read_text(encoding="utf-8")
    assert tool.main(["--check", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == before and "has outputs" in capsys.readouterr().out


def test_check_mode_passes_for_a_clean_notebook(tmp_path):
    clean, _ = tool.strip_notebook(NOTEBOOK)
    assert tool.main(["--check", str(write(tmp_path, clean))]) == 0
