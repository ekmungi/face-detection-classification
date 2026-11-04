"""The one training loop shared by the notebook and train.py.

A task object supplies the model, the batches and two functions (one training step, one validation pass); this module
does everything else the same way for all three models: mixed precision, warm-up then cosine learning rate, rolling
checkpoints with exact resume, a heartbeat file, best-on-validation selection, a time budget, and the run files
(config.json, env.json, git.txt, metrics_history.jsonl, metrics.json, DONE).
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
    """Hyperparameters of one training run. Every field is written to config.json."""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw"          # "adamw" or "sgd"
    momentum: float = 0.9             # SGD only
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01        # the cosine ends at lr * min_lr_ratio
    amp: bool = True
    grad_clip: float | None = 5.0
    checkpoint_every: int = 500       # steps
    keep_checkpoints: int = 2
    log_every: int = 50               # steps
    seed: int = 0
    max_steps: int = 0                # 0 = no limit; smoke tests stop early
    max_minutes: float = 0.0          # 0 = no limit; the run stops cleanly and records why
    heartbeat_every_s: float = 10.0


class Task(Protocol):
    """What fit() needs from a model-specific task."""

    model: nn.Module
    steps_per_epoch: int
    monitor: str          # the validation metric used to pick the best checkpoint
    monitor_mode: str     # "max" or "min"

    def train_batches(self, epoch: int, start_batch: int) -> Iterable[Any]: ...
    def training_step(self, batch: Any, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]: ...
    def validate(self, device: torch.device) -> dict[str, float]: ...


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_settings(device: torch.device, amp: bool) -> tuple[bool, torch.dtype, bool]:
    """Returns (enabled, dtype, use_grad_scaler): bfloat16 where the GPU supports it, else float16 with a scaler."""
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
    """Linear warm-up to 1, then a cosine decay to min_ratio at total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclasses.dataclass
class FitResult:
    """What fit() returns: the epoch history, the best validation value and epoch, and how the run ended."""

    history: list[dict]
    best_value: float | None
    best_epoch: int | None
    status: str           # "done", "already-done", "stopped-max-steps" or "stopped-time-budget"
    run_dir: Path


def _better(value: float, best: float | None, mode: str) -> bool:
    if best is None or not math.isfinite(best):
        return math.isfinite(value)
    return value > best if mode == "max" else value < best


def _save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def write_run_files(run_dir: Path, config: dict, code_dir: Path | None = None) -> None:
    """Writes config.json, env.json and git.txt once, at the start of a run."""
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
    """
    run_dir = Path(run_dir)
    device = torch.device(device)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        log(f"[fit] {run_dir.name}: already trained ({metrics.get('status')}); loading its history")
        return FitResult(load_history(run_dir), metrics.get("best_value"), metrics.get("best_epoch"),
                         "already-done", run_dir)

    write_run_files(run_dir, {"train": dataclasses.asdict(cfg), **(extra_config or {})})
    seed_everything(cfg.seed)
    model = task.model.to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = cfg.epochs * task.steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * task.steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps, cfg.min_lr_ratio))
    use_amp, amp_dtype, use_scaler = autocast_settings(device, cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step, epoch, start_batch = 0, 0, 0
    best_value, best_epoch = None, None
    latest = runlib.latest_checkpoint(ckpt_dir)
    if latest is not None:
        saved = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, epoch, start_batch = saved["step"], saved["epoch"], saved["batch_in_epoch"]
        best_value, best_epoch = saved.get("best_value"), saved.get("best_epoch")
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        log(f"[fit] resumed from {latest.name}: epoch {epoch}, batch {start_batch}, step {step}")
        runlib.append_jsonl(run_dir / "attempts.jsonl", {"time": time.time(), "resumed_from": latest.name})
    else:
        runlib.append_jsonl(run_dir / "attempts.jsonl", {"time": time.time(), "resumed_from": None})

    def checkpoint(batch_in_epoch: int, at_epoch: int) -> str:
        path = runlib.checkpoint_path(ckpt_dir, step)
        _save_checkpoint(path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "epoch": at_epoch, "batch_in_epoch": batch_in_epoch,
            "best_value": best_value, "best_epoch": best_epoch,
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        })
        runlib.prune_checkpoints(ckpt_dir, cfg.keep_checkpoints)
        return path.name

    started = time.time()
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
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = task.training_step(batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            batch_index += 1
            loss_value = float(loss.detach())
            running["loss"] = running.get("loss", 0.0) + loss_value
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + float(value)
            running_n += 1
            now = time.time()
            if step % cfg.log_every == 0:
                means = {k: v / running_n for k, v in running.items()}
                rate = batch_index / max(1e-6, now - epoch_started)
                log(f"[fit] epoch {epoch + 1}/{cfg.epochs} step {step} "
                    + " ".join(f"{k}={v:.4f}" for k, v in means.items())
                    + f" lr={scheduler.get_last_lr()[0]:.2e} batches/s={rate:.2f}")
                running, running_n = {}, 0
            if now - last_beat >= cfg.heartbeat_every_s:
                runlib.write_heartbeat(run_dir / "heartbeat.json", step, epoch, loss_value, last_ckpt, "running", now=now)
                last_beat = now
            if step % cfg.checkpoint_every == 0:
                last_ckpt = checkpoint(batch_index, epoch)
            if cfg.max_steps and step >= cfg.max_steps:
                status = "stopped-max-steps"
                break
            if cfg.max_minutes and (now - started) / 60.0 >= cfg.max_minutes:
                status = "stopped-time-budget"
                break
        start_batch = 0
        if status != "done" and batch_index < task.steps_per_epoch:
            last_ckpt = checkpoint(batch_index, epoch)
        # Validate at every epoch end, and also when a run stops early, so it always has a result.
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            val = task.validate(device)
        model.train()
        record = {"epoch": epoch + 1, "step": step, "seconds": round(time.time() - epoch_started, 1),
                  "lr": scheduler.get_last_lr()[0], "train_loss_last": loss_value,
                  **{f"val_{k}": v for k, v in val.items()}}
        runlib.append_jsonl(run_dir / "metrics_history.jsonl", record)
        log("[fit] epoch {} done in {:.0f}s: ".format(epoch + 1, record["seconds"])
            + " ".join(f"val_{k}={v:.4f}" for k, v in val.items()))
        monitor_value = float(val[task.monitor])
        if _better(monitor_value, best_value, task.monitor_mode):
            best_value, best_epoch = monitor_value, epoch + 1
            _save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1,
                                                  task.monitor: monitor_value})
            log(f"[fit] new best {task.monitor}={monitor_value:.4f} at epoch {epoch + 1}")
        if status != "done":
            break
        epoch += 1
        last_ckpt = checkpoint(0, epoch)

    _save_checkpoint(run_dir / "final.pt", {"model": model.state_dict(), "epoch": epoch, "step": step})
    metrics = {"status": status, "best_value": best_value, "best_epoch": best_epoch, "monitor": task.monitor,
               "steps": step, "minutes": round((time.time() - started) / 60.0, 2)}
    runlib.write_json_atomic(run_dir / "metrics.json", metrics)
    runlib.write_heartbeat(run_dir / "heartbeat.json", step, epoch, loss_value, last_ckpt, "done")
    if status in ("done", "stopped-time-budget"):
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

    The steps change the model's weights, so call this on a model that will be rebuilt before training.
    """
    device = torch.device(device)
    model = task.model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    use_amp, amp_dtype, _ = autocast_settings(device, amp)
    started = None
    done = 0
    for i, batch in enumerate(task.train_batches(epoch=0, start_batch=0)):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, _ = task.training_step(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i >= warmup:
            done += 1
        if done >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not started or done == 0:
        raise RuntimeError("not enough batches to time the training step")
    return (time.time() - started) / done


def plan_epochs(seconds_per_step: float, steps_per_epoch: int, budget_minutes: float, val_seconds: float = 0.0,
                min_epochs: int = 1, max_epochs: int = 200, safety: float = 0.85) -> int:
    """Returns how many whole epochs fit the time budget at the measured speed, so the cosine schedule ends on time."""
    per_epoch = seconds_per_step * steps_per_epoch + val_seconds
    fit = int(budget_minutes * 60.0 * safety // max(per_epoch, 1e-6))
    return max(min_epochs, min(max_epochs, fit))


def overfit_one_batch(model: nn.Module, step_fn: Callable[[nn.Module], torch.Tensor], steps: int = 60,
                      lr: float = 1e-3, device: str | torch.device = "cpu") -> list[float]:
    """Trains a copy-free model on one fixed batch (step_fn closes over it) and returns the loss curve.

    A model that cannot drive the loss down on one batch has a bug in its data, loss or wiring; this is the standard
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
