"""Command-line training run for the Colab workflow (stage 5): builds data and model, calls the shared fit(), writes the run folder.

Everything it writes goes to <artifact-root>/<exp>/<run-id>/: config.json, env.json, git.txt (written once),
attempts.jsonl and metrics_history.jsonl (only appended to), heartbeat.json (overwritten), train.log,
checkpoints/, metrics.json and a DONE marker. Running it again with the same run id resumes from the newest checkpoint.
"""
import argparse
import datetime
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import callbacks
import data
import datastore
import models
import paths
import quicktrain
import runlib
import trainer


def spec_data_mode(spec, default="fetch"):
    """Returns the data mode that the specification's [data] table, or its list of [[data]] tables, asks for.

    A table with no `mode` means `default`. With several tables they must all agree, because one training run uses one
    mode; otherwise this raises SystemExit naming each table's mode and telling the user to pass --data-mode.
    """
    tables = spec.get("data", {})
    tables = tables if isinstance(tables, list) else [tables]
    labelled = [(t["name"] if isinstance(t.get("name"), str) else f"data[{i}]", t.get("mode", default))
                for i, t in enumerate(tables) if isinstance(t, dict)]
    modes = {mode for _, mode in labelled}
    if len(modes) > 1:
        listing = ", ".join(f"{label}: {mode}" for label, mode in labelled)
        raise SystemExit(f"the specification's [[data]] tables use different modes ({listing}), and a run uses one. "
                         f"Pass --data-mode fetch or --data-mode store to choose the mode for this run.")
    return modes.pop() if modes else default


def parse_bool(text):
    """Reads true/false (also yes/no, on/off, 1/0, any case) from a command-line value; raises ArgumentTypeError otherwise."""
    lowered = str(text).strip().lower()
    if lowered in ("true", "yes", "on", "1"):
        return True
    if lowered in ("false", "no", "off", "0"):
        return False
    raise argparse.ArgumentTypeError(f"{text!r} is not true or false")


def train_table(spec):
    """Returns the specification's [train] table with every key written with underscores.

    A hyphen and an underscore mean the same option (max-steps is max_steps). Raises SystemExit when [train] is not a table,
    when a key is written both ways (the message names both spellings), or when a key cannot be an option name.
    """
    table = spec.get("train", {})
    if not isinstance(table, dict):
        raise SystemExit(f"the specification's train entry must be a [train] table, not {type(table).__name__}")
    keys, result = {}, {}
    for key, value in table.items():
        name = key.replace("-", "_")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise SystemExit(f"[train] key {key!r} cannot be a command-line option: use letters, digits, hyphens and "
                             f"underscores, starting with a letter")
        if name in result:
            raise SystemExit(f"[train] has both {keys[name]!r} and {key!r}, which are the same option "
                             f"(hyphens and underscores are interchangeable). Keep one of them.")
        keys[name], result[name] = key, value
    return result


def add_train_option(parser, name, value):
    """Adds the option --name for a [train] value, taking its type from the value.

    true/false is a flag (--name alone means true, --name false turns it off), a number or text is a single value, and a
    non-empty list of one kind of number or text takes one or more values. Anything else raises SystemExit naming the key.
    """
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        parser.add_argument(flag, type=parse_bool, nargs="?", const=True, default=value)
    elif type(value) in (int, float, str):
        parser.add_argument(flag, type=type(value), default=value)
    elif isinstance(value, list) and value and len({type(v) for v in value}) == 1 and type(value[0]) in (int, float, str):
        parser.add_argument(flag, type=type(value[0]), nargs="+", default=list(value))
    else:
        raise SystemExit(f"[train] {name} = {value!r} cannot be a command-line option. Use true/false, a number, text, "
                         f"or a non-empty list of numbers or of text.")


def parse_args(argv=None):
    """Parses the command line; values from --spec's [train] table are used as defaults.

    A [train] key that matches a fixed option (model, augment, exp, ...) becomes that option's default; any other key
    becomes a new option of its own (see add_train_option). Hyphens and underscores in [train] keys are interchangeable.
    --data-mode on the command line beats the specification's [data] mode.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--spec", default=None)
    pre.add_argument("--data-mode", default=None)   # only read here, to know whether the user chose a mode; checked below
    known, _ = pre.parse_known_args(argv)
    defaults = dict(epochs=3, batch_size=128, lr=1e-3, seed=0, checkpoint_every=100, heartbeat_every_s=5.0,
                    train_size=55000, val_size=5000)
    data_mode = "fetch"
    if known.spec:
        spec = runlib.load_spec(known.spec)
        defaults.update(train_table(spec))
        if known.data_mode is None:
            data_mode = spec_data_mode(spec, data_mode)
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=None, help="experiment specification (TOML); its [train] values are the defaults")
    parser.add_argument("--data-mode", choices=datastore.MODES, default=data_mode,
                        help="fetch: download the dataset onto the machine's disk each session; store: read the kept copy in artifacts/data")
    parser.add_argument("--exp", default="exp-001", help="experiment folder, exp-NNN, or scratch for smoke runs")
    parser.add_argument("--run-id", default="run-001", help="run-NNN, or smoke-NNN inside scratch")
    parser.add_argument("--artifact-root", default=None, help="defaults to paths.artifact_root()")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit), for smoke tests")
    parser.add_argument("--step-sleep", type=float, default=0.0, help="seconds to sleep per step, to make a short run last longer")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--model", choices=models.MODEL_NAMES, default="cnn")
    parser.add_argument("--augment", type=parse_bool, nargs="?", const=True, default=False,
                        help="random shift and rotation on the training images; --augment false turns off what the specification turns on")
    fixed = set(vars(parser.parse_args([]))) | {"help"}
    for name, value in defaults.items():
        if name in fixed:
            parser.set_defaults(**{name: value})
        else:
            add_train_option(parser, name, value)
    args = parser.parse_args(argv)
    # A default does not go through `choices`, so a bad value from the specification is checked here.
    if args.model not in models.MODEL_NAMES:
        parser.error(f"model {args.model!r} is not one of {list(models.MODEL_NAMES)}")
    if args.data_mode not in datastore.MODES:
        parser.error(f"data mode {args.data_mode!r} is not one of {list(datastore.MODES)}")
    return args


def set_seeds(seed):
    """Seeds Python, NumPy and torch so a run with the same arguments repeats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now_iso():
    """Returns the current UTC time as an ISO 8601 string, for the history files."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class LogCallback(trainer.Callback):
    """Prints and logs one line at the end of each epoch, and appends it to metrics_history.jsonl."""

    def __init__(self, say, history_path):
        self.say = say
        self.history_path = history_path

    def on_epoch_end(self, state, metrics, model, optimizer):
        self.say(f"epoch {state.epoch} done step={state.step} val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f}")
        runlib.append_jsonl(self.history_path, {"time": now_iso(), "epoch": state.epoch, "step": state.step, **metrics})


def run(args, train_set, val_set, test_set, say=print):
    """Trains (or resumes) on the given datasets and returns the metrics dict that was written to metrics.json.

    The datasets are passed in, so tests can use small synthetic ones and the script uses MNIST.
    """
    root = Path(args.artifact_root) if args.artifact_root else paths.artifact_root()
    run_dir = paths.run_dir(args.exp, args.run_id, root)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Files that describe the run are written on the first attempt only and never changed afterwards.
    runlib.write_once(run_dir / "config.json", {k: v for k, v in vars(args).items()})
    runlib.write_once(run_dir / "env.json", runlib.environment_info(
        {"torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}))
    runlib.write_once(run_dir / "git.txt", runlib.read_code_version(paths.project_root()))

    state, resume_from = trainer.TrainState(), runlib.latest_checkpoint(checkpoint_dir)
    checkpoints = callbacks.CheckpointCallback(checkpoint_dir, args.checkpoint_every, args.keep_checkpoints)
    if resume_from is not None:
        state = callbacks.CheckpointCallback.load(resume_from, model, optimizer, device)
        checkpoints.last = resume_from.name
        say(f"resumed from {resume_from.name} at step {state.step}")
    heartbeat = callbacks.HeartbeatCallback(run_dir / "heartbeat.json", args.heartbeat_every_s, lambda: checkpoints.last)
    say(f"start device={device} torch={torch.__version__} model={args.model} epochs={args.epochs} batch={args.batch_size} seed={args.seed}")
    runlib.append_jsonl(run_dir / "attempts.jsonl", {
        "time": now_iso(), "event": "resume" if resume_from else "start", "step": state.step,
        "checkpoint": resume_from.name if resume_from else None, "device": str(device)})

    started = time.time()
    history, state, stopped_early = trainer.fit(
        model, optimizer, train_set, val_set, args.epochs, args.batch_size, device, args.seed, state,
        [checkpoints, heartbeat, LogCallback(say, run_dir / "metrics_history.jsonl")], args.max_steps, args.step_sleep,
        eval_train=False)

    metrics = {
        "run_id": args.run_id, "exp": args.exp, "model": args.model, "steps": state.step, "epochs_completed": state.epoch,
        "stopped_early": stopped_early,
        "val_accuracy": quicktrain.accuracy(model, torch.utils.data.DataLoader(val_set, batch_size=1000), device),
        "test_accuracy": quicktrain.accuracy(model, torch.utils.data.DataLoader(test_set, batch_size=1000), device),
        "seconds": round(time.time() - started, 1), "device": str(device), "torch": torch.__version__,
        "seed": args.seed, "resumed_from": resume_from.name if resume_from else None,
    }
    runlib.write_json_atomic(run_dir / "metrics.json", metrics)
    heartbeat.finish(state, history["step_loss"][-1] if history["step_loss"] else None, "done")
    runlib.append_jsonl(run_dir / "attempts.jsonl", {
        "time": now_iso(), "event": "stopped" if stopped_early else "finish", "step": state.step})
    (run_dir / "DONE").write_text("finished\n")
    say(f"finished {metrics}")
    return metrics


def main(argv=None):
    """Command line entry point: builds the MNIST datasets, then calls run()."""
    args = parse_args(argv)
    root = Path(args.artifact_root) if args.artifact_root else paths.artifact_root()
    run_dir = paths.run_dir(args.exp, args.run_id, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "train.log", "a", buffering=1)

    def say(message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log.write(line + "\n")

    train_full, val_source, test_set = data.load_mnist(paths.data_dir(root), args.augment, mode=args.data_mode)
    train_set, val_set, _, _ = data.make_sets(train_full, val_source, args.train_size, args.val_size, args.seed)
    run(args, train_set, val_set, test_set, say)
    return 0


if __name__ == "__main__":
    sys.exit(main())
