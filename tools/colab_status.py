"""Prints the state of a notebook run on the Colab machine.

The report covers the processes of the run, the end of its log, its heartbeats, the GPU use and the disk use. This
script runs on the Colab machine itself and reads the environment variables EXP and RUN_ID. Reading the files on the
machine is the only reliable view, because Drive is slow to deliver files that are rewritten often.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# The clone of the repository on the Colab machine. It is made importable so that paths.py can be used.
REPO = Path("/content/face-detection-classification")
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

# Which run to look at. The defaults describe a smoke run in the scratch folder.
exp, run_id = os.environ.get("EXP", "scratch"), os.environ.get("RUN_ID", "smoke-001")
run = paths.run_dir(exp, run_id)

# List the processes that belong to this run: the papermill process and the separate identity training.
listing = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout.splitlines()
mine = [line for line in listing if ("papermill" in line and run_id in line) or "train.py --task identity" in line]
print("processes:", *(line[:160] for line in mine) or ["none"], sep="\n  ")

# Show the end of the papermill log, leaving out the lines that only name the cell being executed.
log = Path(f"/content/{exp}_{run_id}.papermill.log")
if log.exists():
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip() and "Executing:" not in l]
    print("papermill log (last 12 lines):", *(l[-220:] for l in lines[-12:]), sep="\n  ")

# For each training, show the latest heartbeat and the last line of its metrics history.
for name in ("detector", "expression", "identity"):
    beat = run / name / "heartbeat.json"
    if beat.exists():
        hb = json.loads(beat.read_text())
        print(f"{name}: status={hb['status']} epoch={hb['epoch']} step={hb['step']} loss={hb['loss']} "
              f"age={time.time() - hb['time']:.0f}s ckpt={hb['last_checkpoint']}")
        hist = run / name / "metrics_history.jsonl"
        if hist.exists():
            rows = hist.read_text().splitlines()
            if rows:
                print(f"  last epoch: {rows[-1][:300]}")

# The identity training runs as a separate process with its own log.
ilog = run / "identity_train.log"
if ilog.exists():
    print("identity log (last 3 lines):", *(l[-220:] for l in ilog.read_text(errors="replace").splitlines()[-3:]),
          sep="\n  ")

# Report whether the summary file of the run exists yet.
summary = run / "summary.json"
print("summary.json:", "present" if summary.exists() else "not yet")

# Show the GPU use and the free disk space.
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1])
