"""Removes cell outputs and execution counts from notebooks, so a notebook run in VS Code or Colab can be committed without
its results, local paths or account names. Usage: python tools/strip_notebook_outputs.py [notebook ...] [--check]

With no notebook given it handles every notebooks/*.ipynb. --check changes nothing and exits 1 when a notebook still has
outputs, so it can run as a pre-commit check."""
import argparse
import json
import sys
from pathlib import Path


def strip_notebook(data: dict) -> tuple[dict, bool]:
    """Returns (a copy of the notebook without outputs and execution counts, whether anything was removed)."""
    changed = False
    cells = []
    for cell in data.get("cells", []):
        cell = dict(cell)
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                changed = True
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {**data, "cells": cells}, changed


def main(argv: list[str] | None = None) -> int:
    """Strips (or with --check only reports) the given notebooks; returns 0 when clean or stripped, 1 when --check finds outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--check", action="store_true", help="report notebooks that have outputs and change nothing")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.notebooks] or sorted(Path("notebooks").glob("*.ipynb"))
    dirty = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        stripped, changed = strip_notebook(data)
        if changed:
            dirty.append(str(path))
            if not args.check:
                path.write_text(json.dumps(stripped, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for name in dirty:
        sys.stdout.write(("has outputs: " if args.check else "stripped: ") + name + "\n")
    return 1 if (args.check and dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
