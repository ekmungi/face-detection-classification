"""Runs ON the Colab machine: prints the state of a notebook run (process, log tail, heartbeats, GPU, disk).

Environment: EXP and RUN_ID. Reading on the machine is the only reliable view: Drive delivers often-rewritten files late.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/face-detection-classification")
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

exp, run_id = os.environ.get("EXP", "scratch"), os.environ.get("RUN_ID", "smoke-001")
run = paths.run_dir(exp, run_id)
listing = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout.splitlines()
mine = [line for line in listing if ("papermill" in line and run_id in line) or "train.py --task identity" in line]
print("processes:", *(line[:160] for line in mine) or ["none"], sep="\n  ")
log = Path(f"/content/{exp}_{run_id}.papermill.log")
if log.exists():
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip() and "Executing:" not in l]
    print("papermill log (last 12 lines):", *(l[-220:] for l in lines[-12:]), sep="\n  ")
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
ilog = run / "identity_train.log"
if ilog.exists():
    print("identity log (last 3 lines):", *(l[-220:] for l in ilog.read_text(errors="replace").splitlines()[-3:]), sep="\n  ")
summary = run / "summary.json"
print("summary.json:", "present" if summary.exists() else "not yet")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1])
