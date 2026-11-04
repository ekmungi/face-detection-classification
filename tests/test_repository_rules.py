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


def test_the_notebook_links_the_pipeline_figure_in_docs_figures_and_embeds_nothing():
    """The figure is linked, never embedded: Colab cannot show an attachment, and an embedded picture bloats the file.

    VS Code and GitHub show the linked picture; Colab shows the alt text, so the alt text is a full sentence.
    """
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    figure = cells[1]
    assert figure["cell_type"] == "markdown"
    source = figure["source"] if isinstance(figure["source"], str) else "".join(figure["source"])
    link = re.search(r"!\[([^\]]+)\]\((\.\./docs/figures/[^)]+\.png)\)", source)
    assert link, "the second cell links a PNG in docs/figures"
    alt, target = link.groups()
    assert (NOTEBOOK.parent / target).resolve() == (ROOT / "docs" / "figures" / "pipeline_flow.png").resolve()
    assert (ROOT / "docs" / "figures" / "pipeline_flow.png").exists()
    assert len(alt) >= 80 and alt.endswith("."), "the alt text is a full sentence that says what the picture shows"
    assert all(not cell.get("attachments") for cell in cells if cell["cell_type"] == "markdown")


LABELS = ("**What it does and why**", "**What goes in**", "**What comes out**", "**How to tell it worked**")
NUMBERED_STEP = re.compile(r"^(\d+)\. \*\*(.+?)\.?\*\*")


def explanations():
    """Yields (cell number, explaining markdown, code) for every code cell of the notebook."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and i > 0 and cells[i - 1]["cell_type"] == "markdown":
            yield i, text(cells[i - 1]), text(cell)


def labelled_parts(markdown: str) -> dict[str, list[tuple[str, str]]]:
    """Splits an explaining cell into its four labelled parts, each a list of blocks."""
    parts: dict[str, list[tuple[str, str]]] = {}
    current = None
    for kind, text in blocks(markdown):
        if text in LABELS:
            current = text
            parts[current] = []
        elif current is not None:
            parts[current].append((kind, text))
    return parts


def test_every_explanation_has_a_step_heading_and_the_four_labelled_parts_in_order():
    problems = []
    for i, markdown, _ in explanations():
        if not any(line.startswith("### ") for line in markdown.splitlines()):
            problems.append(f"cell {i - 1}: no ### heading naming the step")
        found = [line.strip() for line in markdown.splitlines() if line.strip() in LABELS]
        if found != list(LABELS):
            problems.append(f"cell {i - 1}: labels {found}")
    assert problems == [], "\n".join(problems)


def test_each_labelled_part_is_a_list_of_sentences_never_a_paragraph():
    """A short line ending in a colon may introduce a list; a table or a picture may sit beside it."""
    problems = []
    for i, markdown, _ in explanations():
        for label, parts in labelled_parts(markdown).items():
            if not any(kind == "item" for kind, _ in parts):
                problems.append(f"cell {i - 1}: {label} has no list")
            problems += [f"cell {i - 1}: {label} has a paragraph: {text[:60]}" for kind, text in parts
                         if kind == "paragraph" and not text.endswith(":")]
    assert problems == [], "\n".join(problems)


def test_each_numbered_step_opens_with_a_matching_comment_in_the_code():
    """The numbered steps under "What it does and why" are the numbered comments in the code cell below."""
    problems = []
    for i, markdown, code in explanations():
        steps = [NUMBERED_STEP.match(text) for kind, text in labelled_parts(markdown).get(LABELS[0], [])
                 if kind == "item" and NUMBERED_STEP.match(text)]
        comments = [line.strip() for line in code.splitlines() if re.match(r"^\s*# \d+\. ", line)]
        expected = [f"# {m.group(1)}. {m.group(2)}" for m in steps]
        if [c[: len(e)] for c, e in zip(comments, expected)] != expected or len(comments) != len(expected):
            problems.append(f"cell {i}: markdown steps {expected} but code comments {comments}")
    assert problems == [], "\n".join(problems)


LEAD_MAX = 260        # characters in an explanation's first paragraph: a short lead sentence
BLOCK_MAX = 300       # characters in any paragraph or list item, so that no explanation is a wall of text
LIST_ITEM = re.compile(r"^\s*(?:-|\d+\.)\s")


def blocks(markdown: str) -> list[tuple[str, str]]:
    """Splits markdown into ("heading" | "image" | "table" | "item" | "paragraph", text) blocks.

    A list item runs until the next item, a blank line or a line that starts another kind of block; an indented line
    continues the item above it.
    """
    found: list[tuple[str, str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        kind = ("heading" if stripped.startswith("#") else "table" if stripped.startswith("|")
                else "image" if stripped.startswith("![") else "item" if LIST_ITEM.match(line) else "text")
        if not stripped:
            found.append(("blank", ""))
        elif kind == "text" and found and found[-1][0] in ("item", "paragraph") and (line.startswith(" ") or found[-1][0] == "paragraph"):
            found[-1] = (found[-1][0], found[-1][1] + " " + stripped)
        elif kind == "table" and found and found[-1][0] == "table":
            found[-1] = ("table", found[-1][1] + "\n" + stripped)
        else:
            found.append(("paragraph" if kind == "text" else kind, stripped))
    return [block for block in found if block[0] != "blank"]


def markdown_sources():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    return [(i, text(cell)) for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"]


def test_every_explanation_opens_with_a_short_lead_then_steps_points_or_a_table():
    problems = []
    for i, source in markdown_sources():
        parts = [block for block in blocks(source) if block[0] not in ("heading", "image")]
        if not parts or parts[0][0] != "paragraph" or len(parts[0][1]) > LEAD_MAX:
            problems.append(f"cell {i}: the first paragraph is missing or longer than {LEAD_MAX} characters")
        if not any(kind in ("item", "table") for kind, _ in parts):
            problems.append(f"cell {i}: no numbered steps, bullet points or table")
    assert problems == [], "\n".join(problems)


def test_no_paragraph_or_list_item_in_the_notebook_is_a_wall_of_text():
    long = [f"cell {i}: {text[:70]}... ({len(text)} characters)" for i, source in markdown_sources()
            for kind, text in blocks(source) if kind in ("item", "paragraph") and len(text) > BLOCK_MAX]
    assert long == [], f"paragraphs or list items longer than {BLOCK_MAX} characters:\n" + "\n".join(long)


def test_the_notebook_compares_the_three_modes_side_by_side_in_a_table():
    headers = [text.splitlines()[0].lower() for _, source in markdown_sources()
               for kind, text in blocks(source) if kind == "table"]
    assert any(all(mode in header for mode in ("smoke", "medium", "full")) for header in headers)


def test_the_notebook_text_names_no_run_folder():
    """Run and experiment folder names (such as the real run's own folder) are internal, like record numbers."""
    folder_name = re.compile(r"\b(?:exp|run|smoke|medium|eval-london)-\d{3}\b")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    found = [f"cell {i}: {m.group(0)}" for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"
             for m in folder_name.finditer(text(cell))]
    assert found == [], "run folder names in the notebook's explanations:\n" + "\n".join(found)
