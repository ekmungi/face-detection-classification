"""The one training loop that the notebook and train.py both use.

A task object supplies the model, the batches and two functions (one training step and one validation pass). This
module does everything else in the same way for all three models. That covers mixed precision (running most
calculations in 16-bit numbers to save time and memory), a learning rate that warms up and then follows a cosine
curve, rolling checkpoints with exact resume, a heartbeat file, selection of the best model on the validation data, a
time budget, and the run files (config.json, env.json, git.txt, metrics_history.jsonl, metrics.json and DONE).
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch
from torch import nn

import runlib


@dataclasses.dataclass
class TrainConfig:
    """Holds the hyperparameters (the settings chosen before training) of one run. Every field goes into config.json."""

    # Number of passes over the whole training set.
    epochs: int = 10

    # Highest learning rate, reached at the end of the warm-up.
    lr: float = 1e-3

    # Strength of weight decay, which pulls the weights towards zero. Biases and normalisation weights are excluded.
    weight_decay: float = 0.05

    # The optimiser, either "adamw" or "sgd" (stochastic gradient descent).
    optimizer: str = "adamw"

    # Momentum of the optimiser. Only SGD uses it.
    momentum: float = 0.9

    # Number of epochs over which the learning rate rises from nearly zero to `lr`. It may be a fraction.
    warmup_epochs: float = 1.0

    # The cosine curve ends at `lr` multiplied by this ratio.
    min_lr_ratio: float = 0.01

    # Whether to use mixed precision. It takes effect only on a CUDA (NVIDIA) GPU.
    amp: bool = True

    # Largest allowed norm of the gradients. A value of None or 0 turns clipping off.
    grad_clip: float | None = 5.0

    # Number of steps between two rolling checkpoints.
    checkpoint_every: int = 500

    # Number of rolling checkpoints that are kept on disk.
    keep_checkpoints: int = 2

    # Number of steps between two log lines.
    log_every: int = 50

    # Seed of the random number generators.
    seed: int = 0

    # Total number of steps after which training stops. A value of 0 means no limit. Smoke tests (quick runs that
    # check the code works) use it to stop early.
    max_steps: int = 0

    # Time limit in minutes. A value of 0 means no limit. When the limit is reached, the run stops cleanly and
    # records why.
    max_minutes: float = 0.0

    # Only used together with `max_minutes`. The cosine curve then follows the elapsed time as well as the steps, so
    # the learning rate reaches its minimum no later than the time limit.
    time_schedule: bool = False

    # Minimum number of seconds between two writes of the heartbeat file.
    heartbeat_every_s: float = 10.0


# The fields that define the learning-rate schedule and the optimisation. A resumed run never changes them.
SCHEDULE_FIELDS = ("epochs", "lr", "weight_decay", "optimizer", "momentum", "warmup_epochs", "min_lr_ratio", "amp",
                   "grad_clip", "seed", "max_minutes", "time_schedule")


class Task(Protocol):
    """Lists what fit() needs from a model-specific task."""

    # The network to train.
    model: nn.Module

    # Number of training batches in one epoch.
    steps_per_epoch: int

    # Name of the validation metric that is used to pick the best checkpoint.
    monitor: str

    # Whether a better value of the monitored metric is higher ("max") or lower ("min").
    monitor_mode: str

    def train_batches(self, epoch: int, start_batch: int) -> Iterable[Any]:
        """Yields the training batches of one epoch, starting at batch number `start_batch`."""
        ...

    def training_step(self, batch: Any, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns the loss of one batch and a dict of extra numbers to log."""
        ...

    def validate(self, device: torch.device) -> dict[str, float]:
        """Evaluates the model on the validation data and returns the metrics, including the one named by `monitor`."""
        ...


def seed_everything(seed: int) -> None:
    """Seeds the random number generators of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_settings(device: torch.device, amp: bool) -> tuple[bool, torch.dtype, bool]:
    """Returns (enabled, dtype, use_grad_scaler) for mixed precision.

    Mixed precision is used only on a CUDA GPU. The 16-bit type is bfloat16 where the GPU supports it and float16
    otherwise. Only float16 needs a gradient scaler, which multiplies the loss so that small gradients do not round
    down to zero.
    """
    if not amp or device.type != "cuda":
        return False, torch.float32, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Splits trainable parameters so biases and normalisation weights get no weight decay (standard practice)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # One-dimensional parameters and biases get no weight decay. Normalisation weights are one-dimensional.
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Builds AdamW or SGD (with Nesterov momentum) over the parameter groups."""
    groups = parameter_groups(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    """Returns the factor by which the base learning rate is multiplied at `step`.

    The factor rises linearly to 1 during the warm-up. It then falls along a cosine curve to `min_ratio` at
    `total_steps`.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclasses.dataclass
class FitResult:
    """Holds what fit() returns: the epoch history, the best validation value and epoch, and how the run ended."""

    # One dict of metrics for each finished epoch.
    history: list[dict]

    # Best value of the monitored validation metric, or None if there was none.
    best_value: float | None

    # Epoch (counted from 1) in which the best value was reached, or None if there was none.
    best_epoch: int | None

    # How the run ended: "done", "already-done", "schedule-complete", "stopped-max-steps" or "stopped-time-budget".
    status: str

    # The folder that holds the files of the run.
    run_dir: Path


def _better(value: float, best: float | None, mode: str) -> bool:
    """Returns True if `value` beats `best` in the given mode ("max" or "min").

    When there is no usable best yet, any finite value wins.
    """
    if best is None or not math.isfinite(best):
        return math.isfinite(value)
    return value > best if mode == "max" else value < best


def _save_checkpoint(path: Path, payload: dict) -> None:
    """Saves `payload` to `path` through a temporary file, so an interrupted save never leaves a broken checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def write_run_files(run_dir: Path, config: dict, code_dir: Path | None = None) -> None:
    """Writes config.json, env.json and git.txt once, at the start of a run.

    A file that already exists is left unchanged, so a resumed run keeps the description of how it began. The files
    hold the configuration, the software and GPU in use, and the version of the code (read from `code_dir`, or from
    the folder above this package when it is not given).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    runlib.write_once(run_dir / "config.json", config)
    extra = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        extra["gpu"] = torch.cuda.get_device_name(0)
    runlib.write_once(run_dir / "env.json", runlib.environment_info(extra))
    runlib.write_once(run_dir / "git.txt", runlib.read_code_version(code_dir or Path(__file__).resolve().parent.parent))


def load_history(run_dir: Path) -> list[dict]:
    """Returns the per-epoch history recorded in run_dir."""
    return runlib.read_jsonl(Path(run_dir) / "metrics_history.jsonl")


def fit(task: Task, cfg: TrainConfig, run_dir: str | Path, device: str | torch.device = "cuda",
        log: Callable[[str], None] = print, extra_config: dict | None = None) -> FitResult:
    """Trains task.model, resuming from run_dir when a checkpoint exists, and returns a FitResult.

    run_dir receives checkpoints/ (rolling), best.pt and final.pt (model weights only), heartbeat.json,
    metrics_history.jsonl (one line per epoch), metrics.json and DONE. A run whose DONE file exists is not trained
    again: its history is returned with status "already-done", so re-running the notebook is safe.

    `extra_config` is added to config.json, and `log` receives the progress messages. A run that stops early because
    of `max_steps`, or because of `max_minutes` without `time_schedule`, gets no DONE file, so running it again
    continues it from its last checkpoint. A time-aware schedule that reaches its end counts as finished.
    """
    run_dir = Path(run_dir)
    device = torch.device(device)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # A finished run is not trained again. Its stored result is returned instead.
    if (run_dir / "DONE").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        log(f"[fit] {run_dir.name}: already trained ({metrics.get('status')}); loading its history")
        return FitResult(load_history(run_dir), metrics.get("best_value"), metrics.get("best_epoch"),
                         "already-done", run_dir)

    # Load the newest checkpoint, if there is one.
    latest = runlib.latest_checkpoint(ckpt_dir)
    saved = torch.load(latest, map_location=device, weights_only=True) if latest is not None else None
    if saved is not None and saved.get("cfg"):
        # A resumed run keeps the schedule it started with, so its learning rate never jumps (this was a finding of
        # a code review). Stop controls (max_steps) and logging settings may change between attempts.
        old = {k: saved["cfg"][k] for k in SCHEDULE_FIELDS if k in saved["cfg"]}
        new = {k: getattr(cfg, k) for k in old}
        if old != new:
            log(f"[fit] resuming with the saved configuration {old} (ignoring {new})")
            cfg = dataclasses.replace(cfg, **old)
    write_run_files(run_dir, {"train": dataclasses.asdict(cfg), **(extra_config or {})})
    seed_everything(cfg.seed)

    # Set up the model, the optimiser and the schedule. The warm-up and the total length are counted in steps. The
    # time budget enters the schedule only when `time_schedule` is on.
    model = task.model.to(device)
    optimizer = make_optimizer(model, cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    total_steps = cfg.epochs * task.steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * task.steps_per_epoch)
    budget_seconds = cfg.max_minutes * 60.0 if (cfg.time_schedule and cfg.max_minutes) else 0.0
    use_amp, amp_dtype, use_scaler = autocast_settings(device, cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def progress(at_step: int, elapsed: float) -> float:
        """Returns how far the cosine schedule has advanced, from 0 to 1, after warm-up.

        The value is the larger of the share of steps done and, for a time-aware schedule, the share of the time
        budget used.
        """
        by_steps = (at_step - warmup_steps) / max(1, total_steps - warmup_steps)
        by_time = elapsed / budget_seconds if budget_seconds else 0.0
        return min(1.0, max(by_steps, by_time, 0.0))

    def set_lr(at_step: int, elapsed: float) -> float:
        """Sets the learning rate of every parameter group for `at_step` and returns the rate of the first group."""
        if warmup_steps > 0 and at_step < warmup_steps:
            factor = (at_step + 1) / warmup_steps
        else:
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress(at_step, elapsed)))
            factor = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        return base_lrs[0] * factor

    # Start from the beginning, or restore everything from the checkpoint. That includes the Python and PyTorch random
    # number generator states, so a resumed run continues their random sequences where it stopped.
    step, epoch, start_batch, elapsed_before = 0, 0, 0, 0.0
    best_value, best_epoch = None, None
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, epoch, start_batch = saved["step"], saved["epoch"], saved["batch_in_epoch"]
        best_value, best_epoch = saved.get("best_value"), saved.get("best_epoch")
        elapsed_before = float(saved.get("elapsed", 0.0))
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        log(f"[fit] resumed from {latest.name}: epoch {epoch}, batch {start_batch}, step {step}, "
            f"{elapsed_before / 60:.1f} minutes already trained")
        runlib.append_jsonl(run_dir / "attempts.jsonl", {"time": time.time(), "resumed_from": latest.name})
    else:
        runlib.append_jsonl(run_dir / "attempts.jsonl", {"time": time.time(), "resumed_from": None})

    started = time.time()

    def elapsed_now() -> float:
        """Returns the training time in seconds, including the time spent in earlier attempts of this run."""
        return elapsed_before + (time.time() - started)

    def checkpoint(batch_in_epoch: int, at_epoch: int) -> str:
        """Saves a rolling checkpoint and returns its file name.

        The checkpoint holds the weights, the optimiser and scaler states, the position in training, the best result
        so far, the elapsed time, the configuration and the Python and PyTorch random number generator states. Older
        checkpoints beyond `keep_checkpoints` are deleted.
        """
        path = runlib.checkpoint_path(ckpt_dir, step)
        _save_checkpoint(path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "epoch": at_epoch, "batch_in_epoch": batch_in_epoch,
            "best_value": best_value, "best_epoch": best_epoch, "elapsed": elapsed_now(),
            "cfg": dataclasses.asdict(cfg),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        })
        runlib.prune_checkpoints(ckpt_dir, cfg.keep_checkpoints)
        return path.name

    # State that is carried through the loop. The loss and metric sums in `running` are averaged for each log line.
    current_lr = set_lr(step, elapsed_before)
    last_beat = 0.0
    last_ckpt = latest.name if latest else None
    status = "done"
    loss_value = None
    running: dict[str, float] = {}
    running_n = 0

    while epoch < cfg.epochs:
        model.train()
        epoch_started = time.time()
        batch_index = start_batch
        for batch in task.train_batches(epoch, start_batch):
            # Forward pass, in mixed precision when it is on. A loss that is not finite stops the run at once.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = task.training_step(batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")

            # Backward pass and optimiser step. Gradient clipping needs the unscaled gradients.
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Move on to the next step and set its learning rate.
            step += 1
            current_lr = set_lr(step, elapsed_now())
            batch_index += 1

            # Add this batch to the running sums that the next log line averages.
            loss_value = float(loss.detach())
            running["loss"] = running.get("loss", 0.0) + loss_value
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + float(value)
            running_n += 1

            # Log the averages every `log_every` steps.
            now = time.time()
            if step % cfg.log_every == 0:
                means = {k: v / running_n for k, v in running.items()}
                rate = batch_index / max(1e-6, now - epoch_started)
                log(f"[fit] epoch {epoch + 1}/{cfg.epochs} step {step} "
                    + " ".join(f"{k}={v:.4f}" for k, v in means.items())
                    + f" lr={current_lr:.2e} batches/s={rate:.2f}")
                running, running_n = {}, 0

            # Update the heartbeat file, which tells an observer that the run is alive.
            if now - last_beat >= cfg.heartbeat_every_s:
                runlib.write_heartbeat(
                    run_dir / "heartbeat.json", step, epoch, loss_value, last_ckpt, "running", now=now
                )
                last_beat = now

            # Save a rolling checkpoint every `checkpoint_every` steps.
            if step % cfg.checkpoint_every == 0:
                last_ckpt = checkpoint(batch_index, epoch)

            # Stop early if the step limit or the time budget has been reached.
            if cfg.max_steps and step >= cfg.max_steps:
                status = "stopped-max-steps"
                break
            if budget_seconds and progress(step, elapsed_now()) >= 1.0:
                # The time-aware cosine has reached its end, which counts as a normal end of training.
                status = "schedule-complete"
                break
            if cfg.max_minutes and not budget_seconds and elapsed_now() / 60.0 >= cfg.max_minutes:
                status = "stopped-time-budget"
                break

        # Only the first epoch after a resume starts part-way through. Every later epoch starts at batch 0. If the run
        # stopped inside an epoch, save a checkpoint so that a later attempt can continue from this exact batch.
        start_batch = 0
        if status != "done" and batch_index < task.steps_per_epoch:
            last_ckpt = checkpoint(batch_index, epoch)

        # Validate at every epoch end, and also when a run stops early, so it always has a result.
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            val = task.validate(device)
        model.train()

        # Record and log the result of this epoch.
        record = {"epoch": epoch + 1, "step": step, "seconds": round(time.time() - epoch_started, 1),
                  "lr": current_lr, "train_loss_last": loss_value,
                  **{f"val_{k}": v for k, v in val.items()}}
        runlib.append_jsonl(run_dir / "metrics_history.jsonl", record)
        log("[fit] epoch {} done in {:.0f}s: ".format(epoch + 1, record["seconds"])
            + " ".join(f"val_{k}={v:.4f}" for k, v in val.items()))

        # Keep the weights of the best epoch so far, judged by the monitored validation metric.
        monitor_value = float(val[task.monitor])
        if _better(monitor_value, best_value, task.monitor_mode):
            best_value, best_epoch = monitor_value, epoch + 1
            _save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1,
                                                  task.monitor: monitor_value})
            log(f"[fit] new best {task.monitor}={monitor_value:.4f} at epoch {epoch + 1}")

        # End the run if it stopped early. Otherwise go on to the next epoch, unless the time-aware schedule is
        # complete. The checkpoint at the epoch boundary lets a resumed run begin the next epoch directly.
        if status != "done":
            break
        epoch += 1
        if budget_seconds and progress(step, elapsed_now()) >= 1.0:
            status = "schedule-complete"
            break
        last_ckpt = checkpoint(0, epoch)

    # Save the final weights and the summary of the run. The DONE file is written only for a run that reached the end
    # of its schedule, so a run that stopped early can be continued.
    _save_checkpoint(run_dir / "final.pt", {"model": model.state_dict(), "epoch": epoch, "step": step})
    metrics = {"status": status, "best_value": best_value, "best_epoch": best_epoch, "monitor": task.monitor,
               "steps": step, "minutes": round(elapsed_now() / 60.0, 2), "final_lr": current_lr,
               "annealed": status in ("done", "schedule-complete")}
    runlib.write_json_atomic(run_dir / "metrics.json", metrics)
    runlib.write_heartbeat(run_dir / "heartbeat.json", step, epoch, loss_value, last_ckpt, "done")
    if status in ("done", "schedule-complete"):
        (run_dir / "DONE").write_text(status + "\n", encoding="utf-8")
    log(f"[fit] finished: {metrics}")
    return FitResult(load_history(run_dir), best_value, best_epoch, status, run_dir)


def load_best(model: nn.Module, run_dir: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Loads best.pt (or final.pt when there is no best) from run_dir into model and returns it in eval mode."""
    run_dir = Path(run_dir)
    path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "final.pt"
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def measure_step_seconds(task: Task, device: str | torch.device, steps: int = 30, warmup: int = 5,
                         amp: bool = True) -> float:
    """Times training steps on the task's real batches and returns seconds per step (after a warm-up).

    The first `warmup` steps are run but not timed, and up to `steps` further steps are timed. The steps change the
    model's weights, so call this on a model that will be rebuilt before training. Raises RuntimeError if the task
    supplies no batch beyond the warm-up.
    """
    device = torch.device(device)
    model = task.model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    use_amp, amp_dtype, _ = autocast_settings(device, amp)
    started = None
    done = 0
    for i, batch in enumerate(task.train_batches(epoch=0, start_batch=0)):
        # The clock starts after the warm-up steps. A GPU runs asynchronously, so it is synchronised first to make
        # sure that the earlier work has finished.
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()

        # One ordinary training step.
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, _ = task.training_step(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i >= warmup:
            done += 1
        if done >= steps:
            break

    # Wait for the GPU to finish before the clock is read.
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not started or done == 0:
        raise RuntimeError("not enough batches to time the training step")
    return (time.time() - started) / done


def plan_epochs(seconds_per_step: float, steps_per_epoch: int, budget_minutes: float, val_seconds: float = 0.0,
                min_epochs: int = 1, max_epochs: int = 200, safety: float = 0.85) -> int:
    """Returns how many whole epochs fit the time budget at the measured speed, so the cosine schedule ends on time.

    One epoch takes `steps_per_epoch` steps plus `val_seconds` for validation. Only the fraction `safety` of the budget
    is planned, which leaves a margin. The result is kept between `min_epochs` and `max_epochs`.
    """
    per_epoch = seconds_per_step * steps_per_epoch + val_seconds
    fit = int(budget_minutes * 60.0 * safety // max(per_epoch, 1e-6))
    return max(min_epochs, min(max_epochs, fit))


def overfit_one_batch(model: nn.Module, step_fn: Callable[[nn.Module], torch.Tensor], steps: int = 60,
                      lr: float = 1e-3, device: str | torch.device = "cpu") -> list[float]:
    """Trains `model` itself, without a copy, on one fixed batch and returns the loss of every step.

    `step_fn` takes the model and returns the loss. The function holds the batch itself, so every step sees the same
    data.

    A model that cannot drive the loss down on one batch has a bug in its data, loss or wiring. This is the standard
    sanity check before a long run. The caller passes a freshly built model so the real one is untouched.
    """
    model = model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        loss = step_fn(model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
