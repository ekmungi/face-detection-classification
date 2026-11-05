"""Removes cell outputs and execution counts from notebooks.

A notebook that was run in VS Code or Colab can then be committed without its results, local paths or account names.

Usage: python tools/strip_notebook_outputs.py [notebook ...] [--check]

With no notebook given, the script handles every file that matches notebooks/*.ipynb. With --check, it changes nothing
and exits with code 1 when a notebook still has outputs, so it can run as a pre-commit check.
"""
import argparse
import json
import sys
from pathlib import Path


def strip_notebook(data: dict) -> tuple[dict, bool]:
    """Returns a copy of the notebook without outputs and execution counts, and whether anything was removed.

    The result is a pair. The first item is the cleaned copy, and the second item is True when the notebook had any
    output or execution count to remove. The notebook given as data is not changed.
    """
    changed = False
    cells = []
    for cell in data.get("cells", []):
        cell = dict(cell)

        # Only code cells have outputs. Every other cell is copied as it is.
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                changed = True
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {**data, "cells": cells}, changed


def main(argv: list[str] | None = None) -> int:
    """Strips the given notebooks, or with --check only reports them, and returns the exit code.

    The exit code is 0 when the notebooks are clean or have been stripped. It is 1 when --check finds a notebook with
    outputs. The argv argument is a list of command-line arguments for tests. It defaults to the real command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--check", action="store_true", help="report notebooks that have outputs and change nothing")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.notebooks] or sorted(Path("notebooks").glob("*.ipynb"))

    # Strip each notebook, and write it back only when something changed and --check was not given.
    dirty = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        stripped, changed = strip_notebook(data)
        if changed:
            dirty.append(str(path))
            if not args.check:
                path.write_text(json.dumps(stripped, indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8", newline="\n")

    # List the notebooks that had outputs. With --check, a non-empty list makes the exit code 1.
    for name in dirty:
        sys.stdout.write(("has outputs: " if args.check else "stripped: ") + name + "\n")
    return 1 if (args.check and dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
