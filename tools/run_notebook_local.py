"""Runs a notebook on a local Jupyter kernel one cell at a time, printing each cell's output, so problems show up before the notebook goes to Colab."""
import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so the project's paths.py can be imported
import paths  # noqa: E402


def default_out_path(notebook, label="local", root=None):
    """Returns <artifact root>/scratch/notebook-out/<name>.<label>.ipynb, where the artifact root is paths.artifact_root()."""
    base = Path(root) if root else paths.artifact_root()
    return base / "scratch" / "notebook-out" / f"{Path(notebook).stem}.{label}.ipynb"


def render_output(output):
    """Returns the printable text of one notebook cell output (stream text, a result, or an error message)."""
    kind = output.get("output_type")
    if kind == "stream":
        return output.get("text", "")
    if kind in ("execute_result", "display_data"):
        return output.get("data", {}).get("text/plain", "")
    if kind == "error":
        return f"{output.get('ename')}: {output.get('evalue')}\n"
    return ""


def run_notebook(path, cwd, out_path=None, timeout=300, echo=print):
    """Executes every code cell of the notebook at `path` in order on one kernel and returns 0, or 1 when a cell fails.

    Args:
        path: notebook file to run.
        cwd: working directory of the kernel (relative imports and data paths resolve from here).
        out_path: where to write the executed copy; defaults to default_out_path(path).
        timeout: seconds allowed per cell.
        echo: function used to print progress; replaced in tests.
    """
    path = Path(path)
    out_path = Path(out_path) if out_path else default_out_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    status = 0
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            echo(f"--- cell {index + 1}")
            try:
                client.execute_cell(cell, index)
            except CellExecutionError:
                status = 1
            for output in cell.outputs:
                echo(render_output(output).rstrip("\n"))
            if status:
                break
    nbformat.write(notebook, out_path)
    echo(f"wrote {out_path}")
    return status


def main(argv=None):
    """Command line entry point: run_notebook_local.py notebook.ipynb [--cwd DIR] [--out FILE]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--cwd", default=None, help="kernel working directory; defaults to the notebook's folder")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cwd = args.cwd or str(Path(args.notebook).resolve().parent)
    return run_notebook(args.notebook, cwd, args.out)


if __name__ == "__main__":
    sys.exit(main())
