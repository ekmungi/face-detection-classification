"""Command-line evaluation run (the evaluation variant of stage 5): scores a pretrained model and trains nothing.

The scoring is evaluation.score(table, progress), a file of your own: it gets the specification's [evaluate] table and a
progress(done, total) callback to call after each batch, and returns a dict of scores (text names, finite numbers; anything else
is a failed run). The arguments match train.py. Writes
<artifact-root>/<exp>/<run-id>/: config.json, env.json, git.txt, heartbeat.json (step = items scored), run_eval.log, metrics.json, DONE.
"""
import argparse
import importlib
import math
import sys
import time
from pathlib import Path

import paths
import runlib


def parse_args(argv=None):
    """Parses the command line; --spec is required because the model, split and metric come from its [evaluate] table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="experiment specification (TOML) with an [evaluate] table")
    parser.add_argument("--exp", default="exp-001", help="experiment folder, exp-NNN, or scratch for smoke runs")
    parser.add_argument("--run-id", default="run-001", help="run-NNN, or smoke-NNN inside scratch")
    parser.add_argument("--artifact-root", default=None, help="defaults to paths.artifact_root()")
    return parser.parse_args(argv)


def run_folder(args):
    """Returns the run folder for the arguments, after checking the experiment and run names."""
    return paths.run_dir(args.exp, args.run_id, Path(args.artifact_root) if args.artifact_root else paths.artifact_root())


def check_scores(scores):
    """Returns `scores` unchanged when it is a dict with text keys and finite numbers as values; raises otherwise.

    A boolean is not a number here, and NaN or infinity is refused because a metric that was not measured must not be written
    to metrics.json as if it were. Raises TypeError for a result of the wrong shape and ValueError for a non-finite number.
    """
    if not isinstance(scores, dict):
        raise TypeError(f"evaluation.score must return a dict of scores, not {type(scores).__name__}")
    for key, value in scores.items():
        if not isinstance(key, str):
            raise TypeError(f"evaluation.score returned a score named {key!r}; score names must be text")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"evaluation.score returned {key!r} = {value!r}; each score must be a number")
        if not math.isfinite(value):
            raise ValueError(f"evaluation.score returned {key!r} = {value!r}; a score must be a finite number")
    return scores


def run(args, score, say=print):
    """Calls score(table, progress), writes the run folder and returns the metrics dict that went into metrics.json.

    `score` is passed in so tests can use a stand-in; main() passes the project's evaluation.score.
    """
    folder = run_folder(args)
    table = runlib.load_spec(args.spec).get("evaluate")
    if not isinstance(table, dict):
        raise SystemExit(f"{args.spec} has no [evaluate] table; check it with `colabtool speccheck`")
    folder.mkdir(parents=True, exist_ok=True)
    runlib.write_once(folder / "config.json", {**vars(args), "evaluate": table})
    runlib.write_once(folder / "env.json", runlib.environment_info())
    runlib.write_once(folder / "git.txt", runlib.read_code_version(paths.project_root()))
    beat, scored = folder / "heartbeat.json", [0]

    def progress(done, total):
        """Records that `done` of `total` items are scored: moves the heartbeat and logs a line."""
        scored[0] = done
        runlib.write_heartbeat(beat, done, 0, None, None, "running")
        say(f"scored {done} of {total}")

    started = time.time()
    runlib.write_heartbeat(beat, 0, 0, None, None, "running")
    try:
        # Everything from the scoring to the DONE marker is inside the try, so a wrong result or a failed write is a failed run
        # (heartbeat "failed", no DONE), never a crash that leaves the heartbeat at "running".
        scores = check_scores(score(table, progress))
        metrics = {"run_id": args.run_id, "exp": args.exp, "seconds": round(time.time() - started, 1), **scores}
        runlib.write_json_atomic(folder / "metrics.json", metrics)
        runlib.write_heartbeat(beat, scored[0], 0, None, None, "done")
        (folder / "DONE").write_text("finished\n")
    except BaseException:
        runlib.write_heartbeat(beat, scored[0], 0, None, None, "failed")
        raise
    say(f"finished {metrics}")
    return metrics


def main(argv=None):
    """Command line entry point: imports the project's evaluation.score, opens run_eval.log, then calls run()."""
    args = parse_args(argv)
    try:
        score = importlib.import_module("evaluation").score
    except ModuleNotFoundError as exc:
        if exc.name != "evaluation":
            raise
        raise SystemExit("run_eval.py needs evaluation.py with a score(table, progress) function; this project has none yet") from exc
    folder = run_folder(args)
    folder.mkdir(parents=True, exist_ok=True)
    log = open(folder / "run_eval.log", "a", buffering=1)

    def say(message):
        """Prints a message and appends it, with the time, to run_eval.log."""
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log.write(line + "\n")

    run(args, score, say)
    return 0


if __name__ == "__main__":
    sys.exit(main())
