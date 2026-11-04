"""Two rules for this repository.

1. Every code cell of the end-to-end notebook has a markdown explanation directly before it, so a reader can follow
   what each part does, why, and what in its output shows that it worked.
2. No tracked file carries an internal record number (a research, feature, decision, issue, specification, plan or
   task number from the project's private notes). The code says the reason itself instead.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "03_end_to_end.ipynb"
RECORD_NUMBER = re.compile(r"\b(?:R|F|DEC|ISS|SPEC|P|T)-\d{3}(?:_[A-Z]+)?\b")
MIN_EXPLANATION = 150   # characters of explanation besides any heading line: a sentence or two at the least


def explanation_text(markdown: str) -> str:
    return " ".join(line for line in markdown.splitlines() if not line.lstrip().startswith("#")).strip()


def test_every_code_cell_has_an_explanation_before_it():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    missing = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        before = cells[i - 1] if i > 0 else None
        if before is None or before["cell_type"] != "markdown" or len(explanation_text(source(before))) < MIN_EXPLANATION:
            missing.append(f"cell {i}: {source(cell).splitlines()[0][:60]}")
    assert missing == [], "code cells without a markdown explanation before them:\n" + "\n".join(missing)


def test_no_tracked_file_carries_a_record_number():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    found = []
    for name in filter(None, tracked):
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if RECORD_NUMBER.search(line):
                found.append(f"{name}:{number}: {RECORD_NUMBER.search(line).group(0)}")
    assert found == [], "record numbers belong in the private notes, not the repository:\n" + "\n".join(found)


def test_the_notebook_carries_the_pipeline_figure_from_docs_figures():
    import base64
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    figure = cells[1]
    assert figure["cell_type"] == "markdown"
    source = figure["source"] if isinstance(figure["source"], str) else "".join(figure["source"])
    assert "attachment:pipeline_flow.png" in source
    attached = base64.b64decode(figure["attachments"]["pipeline_flow.png"]["image/png"])
    assert attached == (ROOT / "docs" / "figures" / "pipeline_flow.png").read_bytes()
