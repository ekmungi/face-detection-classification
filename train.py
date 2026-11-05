"""Trains the identity model from the command line, with the same task and loop that the notebook uses.

The notebook starts this script as a second process, so that identity training shares the GPU with detector training:

    python train.py --task identity --arrays /content/data/arrays --run-dir RUN_DIR --budget-minutes 55

The script first measures its own speed and plans the number of epochs that fit the time budget. It writes the plan to
plan.json and then trains with facekit.engine.fit, which resumes from the newest checkpoint when the run was
interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from facekit import engine
from facekit.arrays import read_array
from facekit.identity import IdentityTrainer
from facekit.tasks import IdentityTask


def identity_split(labels: np.ndarray, num_train_ids: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits the sample indices by identity label and returns the training indices and the validation indices.

    Training identities have labels below num_train_ids, and validation identities have all the labels from
    num_train_ids upwards. Keeping the two groups of identities apart tests the model on identities it never saw.
    """
    return np.flatnonzero(labels < num_train_ids), np.flatnonzero(labels >= num_train_ids)


def build_task(args, images, labels, num_train_ids):
    """Builds the identity training task from the command-line arguments and the loaded arrays.

    Args:
        args: The parsed command-line arguments. The model settings and the data loader settings are read from it.
        images: The uint8 image array of the digiface_train array set, which holds both the training and the
            validation identities.
        labels: The identity label of each image.
        num_train_ids: The number of identities that are used for training. The others are used for validation.
    """
    train_idx, val_idx = identity_split(labels, num_train_ids)
    model = IdentityTrainer(num_train_ids, depth=args.depth, scale=args.scale, margin=args.margin, loss=args.loss)
    return IdentityTask(model, images, labels, train_idx, val_idx, batch_size=args.batch_size,
                        num_workers=args.workers, seed=args.seed)


def main(argv=None) -> int:
    """Plans and runs one identity training and returns the process exit code, which is 0 on success.

    When the run folder holds a plan.json file from an earlier start, that plan is reused. Otherwise the candidate
    depths are measured in turn, and the first one that fits at least min_epochs into the budget is chosen. When none
    fits, the last depth measured is used. The argv argument is a list of command-line arguments for tests. It defaults
    to the real command line.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["identity"], default="identity")
    p.add_argument("--arrays", type=Path, required=True, help="folder with the digiface_train array set")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--depth", default="iresnet50")
    p.add_argument("--loss", default="cosface", choices=["cosface", "arcface"])
    p.add_argument("--scale", type=float, default=64.0)
    p.add_argument("--margin", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-minutes", type=float, default=55.0)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=8, help="with --depth auto, the fewest epochs a depth must fit")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # Use the GPU when there is one. cuDNN (the NVIDIA library of neural-network routines) is told to time its
    # convolution methods and keep the fastest. TF32 (a faster, slightly less precise number format on recent NVIDIA
    # GPUs) is allowed for matrix products and convolutions.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the training arrays and read how many identities are used for training.
    images, labels, description = read_array(args.arrays, "digiface_train")
    num_train_ids = int(description["num_train_ids"])
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the plan of an interrupted run, so that a restart trains the same schedule. A new run makes a plan.
    plan_path = args.run_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(f"[train] using the existing plan: {plan}", flush=True)
    else:
        # With "auto", the networks are tried from the deepest to the shallowest. The first one that still fits
        # min_epochs into the time budget is chosen, and the shallowest is used when none fits.
        candidates = ["iresnet50", "iresnet34", "iresnet18"] if args.depth == "auto" else [args.depth]
        tried = []
        for depth in candidates:
            # Build a temporary task for this depth and measure the average time of one training step. Thirty steps
            # are timed after eight warm-up steps.
            args_depth = argparse.Namespace(**{**vars(args), "depth": depth})
            engine.seed_everything(args.seed)
            probe = build_task(args_depth, images, labels, num_train_ids)
            seconds = engine.measure_step_seconds(probe, device, steps=30, warmup=8)

            # Time one validation pass, because the plan needs the time that validation takes.
            started = time.time()
            probe.model.to(device).eval()
            with torch.no_grad():
                probe.validate(torch.device(device))
            val_seconds = time.time() - started

            # Plan how many epochs fit into 75 percent of the budget, which leaves a safety margin.
            epochs = engine.plan_epochs(seconds, probe.steps_per_epoch, args.budget_minutes, val_seconds,
                                        min_epochs=1, max_epochs=args.max_epochs, safety=0.75)

            # Record the measurement, and keep the plan of the latest depth tried as the plan to save.
            tried.append({"depth": depth, "seconds_per_step": seconds, "epochs": epochs})
            print(f"[train] {depth}: {seconds:.3f} s per step, "
                  f"validation {val_seconds:.0f} s -> {epochs} epochs", flush=True)
            plan = {"depth": depth, "seconds_per_step": seconds, "val_seconds": val_seconds,
                    "steps_per_epoch": probe.steps_per_epoch, "epochs": epochs, "budget_minutes": args.budget_minutes,
                    "device": device, "num_train_ids": num_train_ids, "train_images": len(probe.train_set),
                    "val_images": len(probe.val_set), "tried": tried}

            # Free the temporary task before the next depth is tried, and stop at the first depth that fits.
            del probe
            if device == "cuda":
                torch.cuda.empty_cache()
            if epochs >= args.min_epochs:
                break
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"[train] plan: {plan}", flush=True)
    args.depth = plan.get("depth", args.depth)

    # Build the real task. Seeding first makes the initial weights of the model reproducible.
    engine.seed_everything(args.seed)
    task = build_task(args, images, labels, num_train_ids)

    # The learning-rate schedule is a cosine curve that follows the elapsed time. It therefore reaches its end by the
    # end of the budget even when sharing the GPU slows the training down. The number of epochs is 1.3 times the
    # planned number, and the time limit of the budget stops the training when the time is used up.
    epochs = max(1, int(round(plan["epochs"] * 1.3)))
    cfg = engine.TrainConfig(epochs=epochs, lr=args.lr, weight_decay=args.weight_decay, optimizer="sgd",
                             warmup_epochs=min(1.0, epochs / 4), checkpoint_every=1000, log_every=100,
                             seed=args.seed, max_steps=args.max_steps, max_minutes=args.budget_minutes,
                             time_schedule=True, grad_clip=5.0)

    # Save the arguments and the plan with the run, with paths turned into text so that they can be written as JSON.
    extra = {"identity": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "plan": plan}

    # Train. The fit function resumes from the newest checkpoint in the run folder when there is one.
    result = engine.fit(task, cfg, args.run_dir, device=device, log=lambda s: print(s, flush=True), extra_config=extra)
    print(f"[train] finished with status {result.status}; best {task.monitor}={result.best_value}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
