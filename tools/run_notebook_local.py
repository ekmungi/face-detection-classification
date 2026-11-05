"""Runs a notebook on a local Jupyter kernel one cell at a time and prints the output of each cell.

Running the notebook on this computer first shows problems before the notebook goes to Colab. The executed copy of the
notebook, with all its outputs, is saved so that it can be inspected afterwards.

Usage: python tools/run_notebook_local.py notebook.ipynb [--cwd DIR] [--out FILE]
"""
import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# The repository folder is put on the import path, so that the project's paths.py can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402


def default_out_path(notebook, label="local", root=None):
    """Returns the default path for the executed copy of a notebook.

    The path is <artifact root>/scratch/notebook-out/<name>.<label>.ipynb, where the artifact root is the one that
    paths.artifact_root() returns unless a root is given.
    """
    base = Path(root) if root else paths.artifact_root()
    return base / "scratch" / "notebook-out" / f"{Path(notebook).stem}.{label}.ipynb"


def render_output(output):
    """Returns the printable text of one notebook cell output.

    The output is either printed stream text, a result or an error message. An output of any other kind gives an empty
    string.
    """
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

    The run stops at the first cell that fails. The executed copy is written to disk also when a cell fails with an
    error. A cell that runs past the timeout, or a kernel that dies, raises an exception instead, and then no copy is
    written.

    Args:
        path: The notebook file to run.
        cwd: The working directory of the kernel. Relative imports and data paths are resolved from this folder.
        out_path: Where to write the executed copy. It defaults to default_out_path(path).
        timeout: The number of seconds allowed for each cell.
        echo: The function used to print progress. The tests replace it.
    """
    path = Path(path)
    out_path = Path(out_path) if out_path else default_out_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(cwd)}})
    status = 0

    # All cells share one kernel, so a later cell sees the variables of the earlier ones. Markdown cells are skipped.
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            echo(f"--- cell {index + 1}")

            # A failing cell is recorded as a failed run, and its error output is still printed below.
            try:
                client.execute_cell(cell, index)
            except CellExecutionError:
                status = 1
            for output in cell.outputs:
                echo(render_output(output).rstrip("\n"))

            # Stop after the first failing cell, because later cells depend on the earlier ones.
            if status:
                break

    # Save the executed copy, including the outputs, so that the run can be inspected afterwards.
    nbformat.write(notebook, out_path)
    echo(f"wrote {out_path}")
    return status


def main(argv=None):
    """Runs the notebook named on the command line and returns the exit code.

    The command line is: run_notebook_local.py notebook.ipynb [--cwd DIR] [--out FILE]. The argv argument is a list of
    command-line arguments for tests. It defaults to the real command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--cwd", default=None, help="kernel working directory; defaults to the notebook's folder")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    # Without --cwd, the kernel starts in the folder that holds the notebook.
    cwd = args.cwd or str(Path(args.notebook).resolve().parent)
    return run_notebook(args.notebook, cwd, args.out)


if __name__ == "__main__":
    sys.exit(main())
