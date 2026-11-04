"""Train the identity model from the command line, with the same task and loop the notebook uses.

The notebook starts this script as a second process so identity training shares the GPU with detector training:

    python train.py --task identity --arrays /content/data/arrays --run-dir RUN_DIR --budget-minutes 55

It measures its own speed first and plans the epochs to fit the time budget, writes plan.json,
then trains with facekit.engine.fit, which resumes from the newest checkpoint if the run was interrupted.
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
    """Training identities have labels below num_train_ids; validation identities have the labels above."""
    return np.flatnonzero(labels < num_train_ids), np.flatnonzero(labels >= num_train_ids)


def build_task(args, images, labels, num_train_ids):
    train_idx, val_idx = identity_split(labels, num_train_ids)
    model = IdentityTrainer(num_train_ids, depth=args.depth, scale=args.scale, margin=args.margin, loss=args.loss)
    return IdentityTask(model, images, labels, train_idx, val_idx, batch_size=args.batch_size,
                        num_workers=args.workers, seed=args.seed)


def main(argv=None) -> int:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    images, labels, description = read_array(args.arrays, "digiface_train")
    num_train_ids = int(description["num_train_ids"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.run_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(f"[train] using the existing plan: {plan}", flush=True)
    else:
        # "auto" applies the planned descoping order: the deepest network that still fits min_epochs in the budget.
        candidates = ["iresnet50", "iresnet34", "iresnet18"] if args.depth == "auto" else [args.depth]
        tried = []
        for depth in candidates:
            args_depth = argparse.Namespace(**{**vars(args), "depth": depth})
            engine.seed_everything(args.seed)
            probe = build_task(args_depth, images, labels, num_train_ids)
            seconds = engine.measure_step_seconds(probe, device, steps=30, warmup=8)
            started = time.time()
            probe.model.to(device).eval()
            with torch.no_grad():
                probe.validate(torch.device(device))
            val_seconds = time.time() - started
            epochs = engine.plan_epochs(seconds, probe.steps_per_epoch, args.budget_minutes, val_seconds,
                                        min_epochs=1, max_epochs=args.max_epochs, safety=0.75)
            tried.append({"depth": depth, "seconds_per_step": seconds, "epochs": epochs})
            print(f"[train] {depth}: {seconds:.3f} s per step, validation {val_seconds:.0f} s -> {epochs} epochs", flush=True)
            plan = {"depth": depth, "seconds_per_step": seconds, "val_seconds": val_seconds,
                    "steps_per_epoch": probe.steps_per_epoch, "epochs": epochs, "budget_minutes": args.budget_minutes,
                    "device": device, "num_train_ids": num_train_ids, "train_images": len(probe.train_set),
                    "val_images": len(probe.val_set), "tried": tried}
            del probe
            if device == "cuda":
                torch.cuda.empty_cache()
            if epochs >= args.min_epochs:
                break
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"[train] plan: {plan}", flush=True)
    args.depth = plan.get("depth", args.depth)

    engine.seed_everything(args.seed)          # the model's initial weights are reproducible
    task = build_task(args, images, labels, num_train_ids)
    # Time-aware cosine: the schedule anneals by the end of the budget even if sharing the GPU slows training.
    epochs = max(1, int(round(plan["epochs"] * 1.3)))
    cfg = engine.TrainConfig(epochs=epochs, lr=args.lr, weight_decay=args.weight_decay, optimizer="sgd",
                             warmup_epochs=min(1.0, epochs / 4), checkpoint_every=1000, log_every=100,
                             seed=args.seed, max_steps=args.max_steps, max_minutes=args.budget_minutes,
                             time_schedule=True, grad_clip=5.0)
    extra = {"identity": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "plan": plan}
    result = engine.fit(task, cfg, args.run_dir, device=device, log=lambda s: print(s, flush=True), extra_config=extra)
    print(f"[train] finished with status {result.status}; best {task.monitor}={result.best_value}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
