"""Runs ON the Colab machine: updates the clone, then starts the end-to-end notebook as a detached papermill process.

Environment: MODE (smoke or full), EXP, RUN_ID, and optionally DET_BUDGET_MIN, ID_BUDGET_MIN, EXPR_BUDGET_MIN.
The process runs in its own session, so it survives the `colab exec` command that started it. A run that is already
running is not started twice. The executed notebook is written next to the run's other files.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/face-detection-classification")
mode, exp, run_id = os.environ.get("MODE", "smoke"), os.environ.get("EXP", "scratch"), os.environ.get("RUN_ID", "smoke-001")
subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "-q"], check=False)
print("code version:", subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True).stdout.strip())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill", "onnx", "onnxruntime"], check=True)

listing = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
if any("papermill" in line and f"RUN_ID {run_id}" in line for line in listing.splitlines()):
    print(f"{exp}/{run_id} is already running; not starting a second copy")
    sys.exit(0)

sys.path.insert(0, str(REPO))
import importlib  # noqa: E402
importlib.invalidate_caches()   # the kernel may have looked for the clone before it existed
import paths  # noqa: E402

run = paths.run_dir(exp, run_id)
run.mkdir(parents=True, exist_ok=True)
out_nb = run / f"03_end_to_end.{mode}.executed.ipynb"
command = ["papermill", str(REPO / "notebooks" / "03_end_to_end.ipynb"), str(out_nb), "--cwd", str(REPO),
           "--log-output", "--progress-bar", "--request-save-on-cell-execute",
           "-p", "MODE", mode, "-p", "EXP", exp, "-p", "RUN_ID", run_id]
for key in ("DET_BUDGET_MIN", "ID_BUDGET_MIN", "EXPR_BUDGET_MIN"):
    if os.environ.get(key):
        command += ["-p", key, os.environ[key]]
log = open(f"/content/{exp}_{run_id}.papermill.log", "a")
env = dict(os.environ, PYTHONUNBUFFERED="1", FACEKIT_REPO=str(REPO))
process = subprocess.Popen(command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
print("launched papermill, process", process.pid)
print("executed notebook:", out_nb)
print("log: /content/%s_%s.papermill.log" % (exp, run_id))
