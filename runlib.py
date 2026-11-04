"""Small helpers for run folders, heartbeat files, checkpoint discovery and run history, with no torch dependency so they can be tested anywhere."""
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

# Matches checkpoint file names such as checkpoint-00000300.pt and captures the step number.
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt$")


def write_json_atomic(path, payload):
    """Writes payload as JSON to path through a temporary file, so a reader never sees a half-written file.

    Args:
        path: destination file path (str or Path).
        payload: any JSON-serialisable object.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_once(path, content):
    """Writes content (text, or any JSON-serialisable object) to path only if it does not exist; returns True when written.

    Used for the files that describe a run and must never change after it starts: config, environment, code version.
    """
    path = Path(path)
    if path.exists():
        return False
    if isinstance(content, str):
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    else:
        write_json_atomic(path, content)
    return True


def append_jsonl(path, payload):
    """Appends payload as one JSON line to path, creating the file when needed; history files only ever grow."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def read_jsonl(path):
    """Returns the list of objects in a JSON-lines file, or an empty list when it does not exist."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def read_code_version(code_dir):
    """Returns the code version: the contents of code_version.txt in code_dir, else the git commit, else "unknown"."""
    marker = Path(code_dir) / "code_version.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or "unknown"
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def environment_info(extra=None):
    """Returns a dict describing the machine: Python version, platform, and any extra entries (torch version, device)."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    info.update(extra or {})
    return info


def write_heartbeat(path, step, epoch, loss, last_checkpoint, status, now=None):
    """Writes the heartbeat file that tells an observer what the run is doing and how recently.

    Args:
        path: heartbeat file path.
        step: global training step reached.
        epoch: current epoch number, counting from 0.
        loss: latest training loss, or None before the first step.
        last_checkpoint: file name of the newest checkpoint, or None.
        status: "running", "done" or "failed".
        now: unix time to record; defaults to the current time (a parameter so tests can fix it).
    """
    write_json_atomic(path, {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "last_checkpoint": last_checkpoint,
        "status": status,
        "time": time.time() if now is None else now,
    })


def read_heartbeat(path):
    """Returns the heartbeat dict, or None when the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def heartbeat_age(heartbeat, now=None):
    """Returns seconds since the heartbeat was written, or None when there is no heartbeat."""
    if heartbeat is None:
        return None
    return (time.time() if now is None else now) - heartbeat["time"]


def checkpoint_path(run_dir, step):
    """Returns the checkpoint path for a step; the zero padding keeps names sorted by step."""
    return Path(run_dir) / f"checkpoint-{step:08d}.pt"


def list_checkpoints(run_dir):
    """Returns (step, path) pairs for every checkpoint in run_dir, oldest first."""
    found = []
    for entry in Path(run_dir).glob("checkpoint-*.pt"):
        match = CHECKPOINT_PATTERN.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    return sorted(found)


def latest_checkpoint(run_dir):
    """Returns the newest checkpoint path in run_dir, or None when there is none."""
    found = list_checkpoints(run_dir)
    return found[-1][1] if found else None


def prune_checkpoints(run_dir, keep):
    """Deletes all but the newest `keep` checkpoints and returns the removed paths.

    Args:
        run_dir: folder holding the checkpoints.
        keep: how many of the newest checkpoints to keep; must be at least 1.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    found = list_checkpoints(run_dir)
    removed = [path for _, path in found[:-keep]]
    for path in removed:
        path.unlink()
    return removed


def load_spec(path):
    """Reads an experiment specification written in TOML and returns it as a dict."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)
