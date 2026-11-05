"""Small helpers for run folders, heartbeat files, checkpoint discovery and run history.

The helpers have no dependency on torch, so they can be tested on any machine. A run folder holds everything one
training run writes: its configuration, its heartbeat file, its checkpoints and its history files.
"""
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

    The payload is first written to a temporary file beside the destination. The temporary file then replaces the
    destination in one step, which cannot leave a partly written file behind.

    Args:
        path: The destination file path, as a str or a Path.
        payload: Any object that can be written as JSON. Values that JSON cannot hold are written as text.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_once(path, content):
    """Writes content to path only if the file does not exist yet, and returns True when it wrote the file.

    The content is either text or any object that can be written as JSON. This is used for the files that describe a
    run and must never change after the run starts: the configuration, the environment and the code version.
    """
    path = Path(path)
    if path.exists():
        return False

    # Text is written as it is, with a final newline added when it lacks one. Anything else is written as JSON.
    if isinstance(content, str):
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    else:
        write_json_atomic(path, content)
    return True


# The file in a run folder that names the notebook mode (smoke, medium or full) the run belongs to.
RUN_MODE_FILE = "run_mode.txt"


def claim_run_mode(run_dir, mode):
    """Records the notebook mode in a run folder, and stops when the folder holds a run of another mode.

    The trainings reuse the plans they find in their run folder, so a run started in the folder of a run of another
    mode would train on that run's schedule. A folder from before the mode file existed is known by the mode in its
    summary.json file. A folder with training plans but no record of its mode also stops the run.

    Raises:
        RuntimeError: If the folder holds a run of another mode, or a run whose mode is not recorded.
    """
    run_dir = Path(run_dir)
    marker = run_dir / RUN_MODE_FILE
    summary = run_dir / "summary.json"

    # The recorded mode comes from the mode file first and from the summary of an older run second.
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
    elif summary.exists():
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("mode")
    else:
        recorded = None

    # Training plans in the folder show that an earlier run has already used it.
    has_plans = any((run_dir / part / "plan.json").exists() for part in ("detector", "identity", "expression"))

    # Stop the run when the folder belongs to an earlier run of an unknown mode or of a different mode.
    if recorded is None and has_plans:
        raise RuntimeError(
            f"The run folder {run_dir} holds an earlier run whose mode is not recorded, and this is a {mode} run. "
            "The trainings reuse the plans in their run folder, so set RUN_ID in the parameters cell to a name that "
            "no earlier run used, then run the notebook again.")
    if recorded is not None and recorded != mode:
        raise RuntimeError(
            f"The run folder {run_dir} already holds a {recorded} run, and this is a {mode} run. The trainings reuse "
            "the plans in their run folder, so set RUN_ID in the parameters cell to a name that no earlier run used, "
            "then run the notebook again.")

    # A folder without a mode file gets one now, so that a later run of another mode is stopped.
    if not marker.exists():
        marker.write_text(mode + "\n", encoding="utf-8")


def append_jsonl(path, payload):
    """Appends payload as one JSON line to path, creating the file when needed.

    History files only ever grow, so lines are added at the end and never changed.
    """
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
    """Returns the version of the code in code_dir as a string.

    The version is the text of code_version.txt in code_dir when that file exists. Otherwise it is the current git
    commit. When neither is available, it is "unknown".
    """
    marker = Path(code_dir) / "code_version.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or "unknown"

    # Without the marker file, ask git for the current commit. A missing git program or a failed call is not an error.
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def environment_info(extra=None):
    """Returns a dict describing the machine: the Python version, the platform and any extra entries.

    The extra entries are supplied by the caller, for example the torch version and the device in use.
    """
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    info.update(extra or {})
    return info


def write_heartbeat(path, step, epoch, loss, last_checkpoint, status, now=None):
    """Writes the heartbeat file that tells an observer what the run is doing and how recently.

    Args:
        path: The path of the heartbeat file.
        step: The global training step reached.
        epoch: The current epoch number, counting from 0.
        loss: The latest training loss, or None before the first step.
        last_checkpoint: The file name of the newest checkpoint, or None when there is none yet.
        status: The state of the run, which is "running", "done" or "failed".
        now: The Unix time in seconds to record. It defaults to the current time, and exists so that tests can fix it.
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
    """Returns the heartbeat as a dict, or None when the file is missing or cannot be read as JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def heartbeat_age(heartbeat, now=None):
    """Returns the number of seconds since the heartbeat was written, or None when there is no heartbeat.

    The now argument is the Unix time to measure against. It defaults to the current time.
    """
    if heartbeat is None:
        return None
    return (time.time() if now is None else now) - heartbeat["time"]


def checkpoint_path(run_dir, step):
    """Returns the checkpoint path for a step in run_dir.

    The step number is padded with zeros to eight digits, so sorting the file names also sorts them by step.
    """
    return Path(run_dir) / f"checkpoint-{step:08d}.pt"


def list_checkpoints(run_dir):
    """Returns a list of (step, path) pairs for every checkpoint in run_dir, with the oldest first."""
    found = []

    # The pattern rejects files that only resemble a checkpoint name, such as a name without a step number.
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
        run_dir: The folder that holds the checkpoints.
        keep: How many of the newest checkpoints to keep. It must be at least 1.

    Raises:
        ValueError: If keep is smaller than 1.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")

    # The list is sorted with the oldest first, so everything before the last `keep` entries is removed.
    found = list_checkpoints(run_dir)
    removed = [path for _, path in found[:-keep]]
    for path in removed:
        path.unlink()
    return removed


def load_spec(path):
    """Reads an experiment specification written in TOML and returns it as a dict."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)
