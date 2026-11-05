"""Checks two rules for this repository, and the layout of the end-to-end notebook that the first rule covers.

1. Every code cell of the end-to-end notebook has a markdown explanation directly before it. A reader can then follow
   what each part does, why it does it, and what in its output shows that it worked.
2. No tracked file carries an internal record number. A record number is a research, feature, decision, issue,
   specification, plan or task number from the project's private notes. The code states the reason itself instead.
"""
import json
import re
import subprocess
from pathlib import Path

# The repository folder, which is the parent of the folder that holds the tests.
ROOT = Path(__file__).resolve().parent.parent

# The end-to-end notebook that the notebook rules apply to.
NOTEBOOK = ROOT / "notebooks" / "03_end_to_end.ipynb"

# Matches a record number: the letters of one record kind, a hyphen and three digits. An optional suffix of an
# underscore and capital letters may follow.
RECORD_NUMBER = re.compile(r"\b(?:R|F|DEC|ISS|SPEC|P|T)-\d{3}(?:_[A-Z]+)?\b")

# The fewest characters that an explanation may have, not counting heading lines. This is about one or two sentences.
MIN_EXPLANATION = 150


def explanation_text(markdown: str) -> str:
    """Returns the markdown joined into one line of text, with every heading line left out."""
    return " ".join(line for line in markdown.splitlines() if not line.lstrip().startswith("#")).strip()


def test_every_code_cell_has_an_explanation_before_it():
    """Each code cell follows a markdown cell with enough explanation text, and the failures name the code cells."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    missing = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue

        # The cell before a code cell must be a markdown cell with at least the minimum amount of explanation.
        before = cells[i - 1] if i > 0 else None
        if (before is None or before["cell_type"] != "markdown"
                or len(explanation_text(source(before))) < MIN_EXPLANATION):
            missing.append(f"cell {i}: {source(cell).splitlines()[0][:60]}")
    assert missing == [], "code cells without a markdown explanation before them:\n" + "\n".join(missing)


def test_no_tracked_file_carries_a_record_number():
    """No file that git tracks contains a record number, and the failure lists the file, line and number."""
    # Git lists every tracked file of the repository.
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    found = []
    for name in filter(None, tracked):
        path = ROOT / name

        # Files that are not text, such as images, and files that were deleted but not yet committed are skipped.
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue

        # Every line with a record number is reported with its file name and line number.
        for number, line in enumerate(text.splitlines(), 1):
            if RECORD_NUMBER.search(line):
                found.append(f"{name}:{number}: {RECORD_NUMBER.search(line).group(0)}")
    assert found == [], "record numbers belong in the private notes, not the repository:\n" + "\n".join(found)


# The start of the full GitHub address from which the notebook must show its figure. It points to the main branch of
# the repository.
FIGURE_ADDRESS = "https://raw.githubusercontent.com/ekmungi/face-detection-classification/main/"

# Patterns that find the address of a picture in markdown. Each one has a single group.
PICTURE_TARGETS = (
    # An inline markdown picture, whose address is written in brackets right after the alt text.
    re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)"),

    # A reference-style picture, which gives a label that is defined elsewhere in the cell.
    re.compile(r"!\[[^\]]*\]\[([^\]]*)\]"),

    # A picture written as an HTML image tag.
    re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']?([^\"'\s>]+)", re.I),
)


def picture_targets(markdown: str) -> list:
    """Returns every address from which a markdown cell shows a picture.

    A reference-style picture is resolved through its label, using the link definitions in the same cell. A label
    without a definition gives the text "unresolved reference" followed by the label.
    """
    # The link definitions of the cell, such as "[fig]: address", keyed by their lower-case label.
    definitions = {m.group(1).lower(): m.group(2)
                   for m in re.finditer(r"^\s*\[([^\]]+)\]:\s*<?(\S+?)>?(?:\s|$)", markdown, re.M)}
    targets = []
    for pattern in PICTURE_TARGETS:
        for found in pattern.finditer(markdown):
            target = found.group(1)

            # For a reference-style picture the group is a label. An empty label means that the alt text is the label.
            if pattern is PICTURE_TARGETS[1]:
                label = target or re.match(r"!\[([^\]]*)\]", found.group(0)).group(1)
                target = definitions.get(label.lower(), "unresolved reference " + label)
            targets.append(target)
    return targets


def test_the_notebook_shows_the_pipeline_figure_from_its_github_address_and_embeds_nothing():
    """The figure is shown from its full GitHub address, and no cell embeds a picture or shows one by a relative path.

    A relative path such as ../docs/figures resolves only where the notebook can be traced to the repository. That is
    the case for the file in a checkout, on GitHub, and in Colab when it opens the notebook from GitHub. A copy opened
    from Google Drive or uploaded has no repository to resolve the path against. The executed copy that an unattended
    run saves in its run folder is such a copy. The full address works in all of these whenever the computer is
    online. A picture embedded in the cell more than doubled the notebook's size. The alt text is a full sentence for
    a reader who cannot load the picture.
    """
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731

    # The second cell of the notebook shows the pipeline figure.
    figure = cells[1]
    assert figure["cell_type"] == "markdown"
    link = re.search(r"!\[([^\]]+)\]\((https://[^)\s]+\.png)\)", text(figure))
    assert link, "the second cell shows a PNG by its full address"
    alt, target = link.groups()

    # The address must point to the PNG file on the main branch, and that file must exist in the repository.
    assert target.startswith(FIGURE_ADDRESS), "the address is the PNG on the repository's main branch"
    assert target[len(FIGURE_ADDRESS):] == "docs/figures/pipeline_flow.png"
    assert (ROOT / "docs" / "figures" / "pipeline_flow.png").exists()
    assert len(alt) >= 80 and alt.endswith("."), "the alt text is a full sentence that says what the picture shows"

    # No markdown cell may embed a picture as an attachment, and every picture must be shown from a full https address.
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        assert not cell.get("attachments"), f"cell {i} embeds a picture as an attachment"
        for shown in picture_targets(text(cell)):
            assert shown.startswith("https://"), (
                f"cell {i} shows a picture from {shown[:60]!r}, not a full https address")


def test_picture_targets_finds_every_way_a_cell_can_show_a_picture():
    """The helper finds inline, reference-style, HTML and embedded pictures, and it ignores plain links."""
    # Each markdown text is paired with the address that the helper must find in it.
    cases = {
        # Inline pictures, with and without a title, with angle brackets, with a capital-letter extension and with
        # the picture data itself written into the address.
        "![a](../docs/figures/x.png)": "../docs/figures/x.png",
        '![a](../docs/figures/x.png "title")': "../docs/figures/x.png",
        "![a](<../docs/figures/x.png>)": "../docs/figures/x.png",
        "![a](../x.GIF)": "../x.GIF",
        "![a](data:image/png;base64,AAAA)": "data:image/png;base64,AAAA",

        # A picture written as an HTML image tag.
        '<img src="../docs/figures/x.png" width="400">': "../docs/figures/x.png",

        # Reference-style pictures, with a label, with an empty label and with a label that has no definition.
        "![a][fig]\n\n[fig]: ../docs/figures/x.png": "../docs/figures/x.png",
        "![fig][]\n\n[fig]: <../x.webp>": "../x.webp",
        "![a][missing]": "unresolved reference missing",

        # A picture that is shown from a full web address.
        "![a](https://example.org/x.png)": "https://example.org/x.png",
    }
    for markdown, expected in cases.items():
        assert picture_targets(markdown) == [expected], markdown
    assert picture_targets("[the PNG file](../docs/figures/x.png)") == [], "a plain link is not a picture"


# The four bold labels that every explaining cell must have, in this order.
LABELS = ("**What it does and why**", "**What goes in**", "**What comes out**", "**How to tell it worked**")

# Matches a numbered step such as "1. **Title.**". The first group is the number and the second is the title.
NUMBERED_STEP = re.compile(r"^(\d+)\. \*\*(.+?)\.?\*\*")


def explanations():
    """Yields (cell number, explaining markdown, code) for each code cell that directly follows a markdown cell."""
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
        # A block that is exactly a label starts a new part, and the blocks after it belong to that part.
        if text in LABELS:
            current = text
            parts[current] = []
        elif current is not None:
            parts[current].append((kind, text))
    return parts


def test_every_explanation_has_a_step_heading_and_the_four_labelled_parts_in_order():
    problems = []
    for i, markdown, _ in explanations():
        # The explanation needs a level-three heading that names the step.
        if not any(line.startswith("### ") for line in markdown.splitlines()):
            problems.append(f"cell {i - 1}: no ### heading naming the step")

        # The four labels must all be present and must appear in the fixed order.
        found = [line.strip() for line in markdown.splitlines() if line.strip() in LABELS]
        if found != list(LABELS):
            problems.append(f"cell {i - 1}: labels {found}")
    assert problems == [], "\n".join(problems)


def test_each_labelled_part_is_a_list_of_sentences_never_a_paragraph():
    """A short line that ends in a colon may introduce a list. A table or a picture may sit beside the list."""
    problems = []
    for i, markdown, _ in explanations():
        for label, parts in labelled_parts(markdown).items():
            # Each part needs at least one list item, and any paragraph in it must end in a colon to introduce a list.
            if not any(kind == "item" for kind, _ in parts):
                problems.append(f"cell {i - 1}: {label} has no list")
            problems += [f"cell {i - 1}: {label} has a paragraph: {text[:60]}" for kind, text in parts
                         if kind == "paragraph" and not text.endswith(":")]
    assert problems == [], "\n".join(problems)


def test_each_numbered_step_opens_with_a_matching_comment_in_the_code():
    """The numbered steps under "What it does and why" are the numbered comments in the code cell below."""
    problems = []
    for i, markdown, code in explanations():
        # The numbered steps in the markdown, and the numbered comments in the code cell below it.
        steps = [NUMBERED_STEP.match(text) for kind, text in labelled_parts(markdown).get(LABELS[0], [])
                 if kind == "item" and NUMBERED_STEP.match(text)]
        comments = [line.strip() for line in code.splitlines() if re.match(r"^\s*# \d+\. ", line)]
        expected = [f"# {m.group(1)}. {m.group(2)}" for m in steps]

        # Each comment must start with the number and title of its step, and the two lists must have the same length.
        if [c[: len(e)] for c, e in zip(comments, expected)] != expected or len(comments) != len(expected):
            problems.append(f"cell {i}: markdown steps {expected} but code comments {comments}")
    assert problems == [], "\n".join(problems)


# The most characters that the first paragraph of an explanation may have. It should be one short lead sentence.
LEAD_MAX = 260

# The most characters that any paragraph or list item may have, so that no explanation is a wall of text.
BLOCK_MAX = 300

# Matches the start of a list item, which is a hyphen or a number with a full stop, followed by a space.
LIST_ITEM = re.compile(r"^\s*(?:-|\d+\.)\s")


def blocks(markdown: str) -> list[tuple[str, str]]:
    """Splits markdown into blocks, each a pair of a kind and its text.

    The kind is one of "heading", "image", "table", "item" or "paragraph". A list item runs until the next item, a
    blank line or a line that starts another kind of block. An indented line continues the item above it.
    """
    found: list[tuple[str, str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()

        # Decide what kind of line this is from the way it starts. Any other line is plain text.
        kind = ("heading" if stripped.startswith("#") else "table" if stripped.startswith("|")
                else "image" if stripped.startswith("![") else "item" if LIST_ITEM.match(line) else "text")
        if not stripped:
            # A blank line is recorded, so that it ends the block above it. It is removed again at the end.
            found.append(("blank", ""))
        elif (kind == "text" and found and found[-1][0] in ("item", "paragraph")
                and (line.startswith(" ") or found[-1][0] == "paragraph")):
            # A text line continues the paragraph above it, or the list item above it when it is indented.
            found[-1] = (found[-1][0], found[-1][1] + " " + stripped)
        elif kind == "table" and found and found[-1][0] == "table":
            # A table row joins the table above it, with one table row on each line.
            found[-1] = ("table", found[-1][1] + "\n" + stripped)
        else:
            # Anything else starts a new block.
            found.append(("paragraph" if kind == "text" else kind, stripped))
    return [block for block in found if block[0] != "blank"]


def markdown_sources():
    """Returns (cell number, text) for every markdown cell of the notebook."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    return [(i, text(cell)) for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"]


def test_every_explanation_opens_with_a_short_lead_then_steps_points_or_a_table():
    problems = []
    for i, source in markdown_sources():
        # Headings and pictures are ignored, so that the first remaining block is the lead.
        parts = [block for block in blocks(source) if block[0] not in ("heading", "image")]
        if not parts or parts[0][0] != "paragraph" or len(parts[0][1]) > LEAD_MAX:
            problems.append(f"cell {i}: the first paragraph is missing or longer than {LEAD_MAX} characters")

        # After the lead there must be a list or a table.
        if not any(kind in ("item", "table") for kind, _ in parts):
            problems.append(f"cell {i}: no numbered steps, bullet points or table")
    assert problems == [], "\n".join(problems)


def test_no_paragraph_or_list_item_in_the_notebook_is_a_wall_of_text():
    long = [f"cell {i}: {text[:70]}... ({len(text)} characters)" for i, source in markdown_sources()
            for kind, text in blocks(source) if kind in ("item", "paragraph") and len(text) > BLOCK_MAX]
    assert long == [], f"paragraphs or list items longer than {BLOCK_MAX} characters:\n" + "\n".join(long)


def test_the_notebook_compares_the_three_modes_side_by_side_in_a_table():
    """Some table in the notebook has a header row that names the smoke, medium and full modes."""
    headers = [text.splitlines()[0].lower() for _, source in markdown_sources()
               for kind, text in blocks(source) if kind == "table"]
    assert any(all(mode in header for mode in ("smoke", "medium", "full")) for header in headers)


def test_the_notebook_text_names_no_run_folder():
    """Run and experiment folder names, such as the real run's own folder, are internal, like record numbers."""
    folder_name = re.compile(r"\b(?:exp|run|smoke|medium|eval-london)-\d{3}\b")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = lambda cell: cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])  # noqa: E731
    found = [f"cell {i}: {m.group(0)}" for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"
             for m in folder_name.finditer(text(cell))]
    assert found == [], "run folder names in the notebook's explanations:\n" + "\n".join(found)
