"""Starts the end-to-end notebook on the Colab machine as a detached papermill process, after updating the clone.

This script runs on the Colab machine itself. Papermill is a tool that runs a notebook from the command line.

The script reads these environment variables: MODE (smoke, medium or full), EXP and RUN_ID. It also reads
DET_BUDGET_MIN, ID_BUDGET_MIN and EXPR_BUDGET_MIN when they are set. These are the time budgets in minutes of the
detector, identity and expression trainings.

The process runs in its own session, so it survives the `colab exec` command that started it. A run that is already
running is not started twice. The executed notebook is written next to the run's other files.
"""
import os
import subprocess
import sys
from pathlib import Path

# The clone of the repository on the Colab machine.
REPO = Path("/content/face-detection-classification")

# Which run to start. The defaults describe a smoke run in the scratch folder.
mode, exp, run_id = (os.environ.get("MODE", "smoke"), os.environ.get("EXP", "scratch"),
                     os.environ.get("RUN_ID", "smoke-001"))

# Update the code to the newest version, print the version in use, and install what the notebook run needs.
subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "-q"], check=False)
print("code version:", subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True).stdout.strip())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill", "onnx", "onnxruntime"], check=True)

# Stop here when a papermill process for this run id is already running, so that the run is not started twice.
listing = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
if any("papermill" in line and f"RUN_ID {run_id}" in line for line in listing.splitlines()):
    print(f"{exp}/{run_id} is already running; not starting a second copy")
    sys.exit(0)

# Make the clone importable. The import caches are cleared because the kernel may have looked for the clone before it
# existed.
sys.path.insert(0, str(REPO))
import importlib  # noqa: E402
importlib.invalidate_caches()
import paths  # noqa: E402

# Create the run folder. The executed notebook is saved in it.
run = paths.run_dir(exp, run_id)
run.mkdir(parents=True, exist_ok=True)
out_nb = run / f"03_end_to_end.{mode}.executed.ipynb"

# Build the papermill command. The parameters override the values in the parameters cell of the notebook.
# USE_DRIVE is False because nobody can approve the Drive page in an unattended run.
command = ["papermill", str(REPO / "notebooks" / "03_end_to_end.ipynb"), str(out_nb), "--cwd", str(REPO),
           "--log-output", "--progress-bar", "--request-save-on-cell-execute",
           "-p", "MODE", mode, "-p", "EXP", exp, "-p", "RUN_ID", run_id,
           "-p", "USE_DRIVE", "False"]

# Pass on the optional training time budgets only when they are set.
for key in ("DET_BUDGET_MIN", "ID_BUDGET_MIN", "EXPR_BUDGET_MIN"):
    if os.environ.get(key):
        command += ["-p", key, os.environ[key]]

# Start papermill detached, in a new session, with its output appended to a log file. The unbuffered output makes the
# log show progress as it happens.
log = open(f"/content/{exp}_{run_id}.papermill.log", "a")
env = dict(os.environ, PYTHONUNBUFFERED="1", FACEKIT_REPO=str(REPO))
process = subprocess.Popen(command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=env,
                           start_new_session=True)
print("launched papermill, process", process.pid)
print("executed notebook:", out_nb)
print("log: /content/%s_%s.papermill.log" % (exp, run_id))
