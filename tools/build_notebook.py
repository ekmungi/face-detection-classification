"""Builds notebooks/03_end_to_end.ipynb from the cells below, so the notebook is reviewed as plain Python.

Run: python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text, tags=None):
    CELLS.append(("code", text.strip("\n"), tags or []))


md("""
# Face detection, expression and identity: trained end to end on Colab

This notebook trains and tests the three models of SPEC-001 and saves them for the CPU pipeline:

1. **Detector**: a CenterFace-style network (ResNet-18 ImageNet backbone) trained on WIDER FACE to find faces and five landmarks.
2. **Expression**: ConvNeXt-Tiny (ImageNet backbone) fine-tuned on EmoNet-Face portraits aligned with our own detector, eight classes.
3. **Identity**: IResNet trained from scratch with a CosFace margin on DigiFace-1M synthetic identities.

No model that was already trained for a face task is used. Every dataset's licence is in the manifest below and in the model card.
The notebook runs headless with papermill (`MODE` below); every training writes checkpoints and resumes if the run is interrupted.
""")

code("""
# Parameters (papermill overrides these).
MODE = "smoke"            # "smoke": tiny subsets to test every cell; "full": the real run
EXP = "scratch"           # exp-NNN for a real run, scratch for a smoke run
RUN_ID = "smoke-001"      # run-NNN or smoke-NNN
DET_BUDGET_MIN = 50.0     # minutes for detector training
ID_BUDGET_MIN = 55.0      # minutes for identity training (a second process on the same GPU)
EXPR_BUDGET_MIN = 12.0    # minutes for expression training
""", tags=["parameters"])

md("## 1. Setup")
code("""
import os, sys, json, time, subprocess, shutil, hashlib, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

REPO = Path(os.environ.get("FACEKIT_REPO", "/content/face-detection-classification"))
if not REPO.exists():
    REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(REPO)
sys.path.insert(0, str(REPO))
try:
    import onnx, onnxruntime  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)

import numpy as np, torch, cv2, matplotlib
import matplotlib.pyplot as plt
import paths, runlib
from facekit import (align, arrays, datasets, det_eval, engine, export, metrics, prepare, sources)
from facekit.det_data import parse_retinaface_labels, parse_wider_official, WiderFaceTrain, detection_collate
from facekit.det_model import FaceDetNet, detection_loss, detect_image
from facekit.det_task import DetectorTask, split_records, predict_records, choose_score_threshold
from facekit.detect import FaceDetector
from facekit.expression import CLASSES, ExpressionNet
from facekit.identity import IdentityTrainer, IResNet
from facekit.tasks import ExpressionTask, IdentityTask, embed, predict_logits
from facekit.gallery import Gallery
from facekit.pipeline import FaceAnalyzer, softmax

plt.rcParams.update({"figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
SMOKE = MODE == "smoke"
CPUS = os.cpu_count() or 2
RUN = paths.run_dir(EXP, RUN_ID)
RUN.mkdir(parents=True, exist_ok=True)
FIG = RUN / "figures"; FIG.mkdir(exist_ok=True)
MODELS = RUN / "models"; MODELS.mkdir(exist_ok=True)
DATA = paths.work_data_dir(); RAW = DATA / "raw"; ARR = DATA / "arrays" / MODE   # smoke and full never share arrays
RAW.mkdir(parents=True, exist_ok=True); ARR.mkdir(parents=True, exist_ok=True)
SUMMARY = {"mode": MODE, "run": str(RUN), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
CHECKS = []

def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(("PASS" if ok else "FAIL"), "-", name, "-", detail)

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.show()

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"mode={MODE} device={DEVICE} gpu={gpu} cpus={CPUS} torch={torch.__version__} cv2={cv2.__version__}")
print("run folder:", RUN); print("data folder:", DATA)
print("code version:", runlib.read_code_version(REPO))
SUMMARY.update({"gpu": gpu, "cpus": CPUS, "code_version": runlib.read_code_version(REPO)})
""")

code("""
# The configuration of each mode. The full run's sizes follow SPEC-001; the smoke run uses tiny subsets of the same data.
CFG = {
    "smoke": dict(digiface_train=["0-1999"], digiface_test="8000-9999", id_max_ids=300, id_val_ids=60, id_max_per_id=24,
                  id_test_ids=100, emonet_files=1, emonet_max=600, hq_files=1, hq_max=120, wider_train_max=600,
                  wider_val_max=120, det_max_steps=60, expr_max_steps=40, id_max_steps=80, id_depth="iresnet18"),
    "full": dict(digiface_train=["0-1999", "2000-3999"], digiface_test="8000-9999", id_max_ids=None, id_val_ids=400,
                 id_max_per_id=None, id_test_ids=500, emonet_files=5, emonet_max=None, hq_files=2, hq_max=None,
                 wider_train_max=None, wider_val_max=None, det_max_steps=0, expr_max_steps=0, id_max_steps=0,
                 id_depth="auto"),
}[MODE]
WORKERS_DET = max(2, CPUS - 4)       # the detector's JPEG decoding needs most of the CPU
WORKERS_ID = 2 if CPUS <= 8 else 3   # identity reads pre-decoded arrays, so it needs few workers
print(json.dumps(CFG, indent=1)); print("workers: detector", WORKERS_DET, "identity", WORKERS_ID)
SUMMARY["config"] = CFG
""")

md("## 2. Data: download every dataset and record its licence")
code("""
LONDON = [sources.LONDON_NEUTRAL, sources.LONDON_SMILING]
WIDER_SPLIT = datasets.Source(name="wider_split", url=f"{sources.HF}/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip",
                              filename="wider_face_split.zip", licence=sources.WIDER_LICENCE, licence_url=sources.WIDER_LICENCE_URL)
WANTED = ([sources.WIDER_TRAIN, sources.RETINAFACE_LABELS, sources.WIDER_VAL, WIDER_SPLIT]
          + sources.EMONET_BINARY[:CFG["emonet_files"]] + sources.EMONET_HQ[:CFG["hq_files"]]
          + [sources.digiface_part(p) for p in CFG["digiface_train"] + [CFG["digiface_test"]]] + LONDON)

def get(source):
    started = time.time()
    path = datasets.download(source, RAW, retries=4)
    extracted = 0
    if path.suffix == ".zip":
        extracted = len(datasets.safe_extract(path, RAW / source.name))
    return source.name, path.stat().st_size, extracted, round(time.time() - started, 1)

started = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    for name, size, extracted, seconds in pool.map(get, WANTED):
        print(f"{name:22s} {size/1e9:6.2f} GB  extracted {extracted:7d} files  {seconds:6.1f} s")
print(f"all downloads and extractions: {time.time() - started:.0f} s")
manifest = datasets.write_manifest(RUN / "data_manifest.json", WANTED, {s.name: datasets.count_images(RAW / s.name) for s in WANTED})
print(f"{'dataset':22s} licence"); [print(f"{s['name']:22s} {s['licence']}") for s in manifest["sources"]]
SUMMARY["datasets"] = [{"name": s["name"], "licence": s["licence"], "url": s["url"]} for s in manifest["sources"]]
""")

code("""
# Integrity checks: the third-party mirrors must agree with the official WIDER FACE boxes (SPEC-001, review change 2).
split_dir = next((RAW / "wider_split").rglob("wider_face_train_bbx_gt.txt")).parent
official_train = {Path(r["path"]).name: r for r in parse_wider_official(split_dir / "wider_face_train_bbx_gt.txt")}
official_val = {Path(r["path"]).name: r for r in parse_wider_official(split_dir / "wider_face_val_bbx_gt.txt")}
retina = parse_retinaface_labels(RAW / sources.RETINAFACE_LABELS.filename)

def boxes_match(a, b, tol=1.0):
    a = np.asarray(a).reshape(-1, 4); b = np.asarray(b).reshape(-1, 4)
    b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
    if len(a) != len(b): return False
    return bool(np.all(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)) <= tol))

matched = sum(boxes_match(r["boxes"], official_train[Path(r["path"]).name]["boxes"]) for r in retina if Path(r["path"]).name in official_train)
share = matched / max(1, len(retina))
check("RetinaFace labels agree with official WIDER train boxes", share >= 0.97, f"{matched}/{len(retina)} images ({share:.1%})")
n_val_checked = n_val_ok = 0
for rec in prepare.wider_val_records(RAW / sources.WIDER_VAL.filename, max_items=400):
    key = Path(rec["path"]).name
    if key in official_val:
        n_val_checked += 1; n_val_ok += boxes_match(rec["boxes"], official_val[key]["boxes"])
check("WIDER validation parquet agrees with official boxes", n_val_ok >= 0.97 * max(1, n_val_checked), f"{n_val_ok}/{n_val_checked} images")
""")

md("## 3. Identity data: DigiFace-1M arrays, split by identity")
code("""
id_paths, id_names = [], []
for part in CFG["digiface_train"]:
    p, n = prepare.digiface_records(RAW / f"digiface_{part}"); id_paths += p; id_names += n
train_ids, val_ids = prepare.split_identities(id_names, CFG["id_val_ids"], seed=0, max_ids=CFG["id_max_ids"])
label_of = {name: i for i, name in enumerate(train_ids + val_ids)}       # training identities first
test_paths, test_names = prepare.digiface_records(RAW / f"digiface_{CFG['digiface_test']}")
test_ids = sorted(set(test_names))[: CFG["id_test_ids"]]
test_label = {name: i for i, name in enumerate(test_ids)}
started = time.time()
if not arrays.array_exists(ARR, "digiface_train"):
    prepare.build_digiface_array(id_paths, id_names, train_ids + val_ids, ARR, "digiface_train", label_of,
                                 max_per_id=CFG["id_max_per_id"], workers=CPUS, extra_meta={"num_train_ids": len(train_ids)})
if not arrays.array_exists(ARR, "digiface_test"):
    prepare.build_digiface_array(test_paths, test_names, test_ids, ARR, "digiface_test", test_label, workers=CPUS)
id_images, id_labels, id_desc = arrays.read_array(ARR, "digiface_train")
test_images, test_labels, test_desc = arrays.read_array(ARR, "digiface_test")
print(f"arrays ready in {time.time() - started:.0f} s: train+val {len(id_labels)} images, test {len(test_labels)} images")
print(f"identities: train {len(train_ids)}, validation {len(val_ids)}, test {len(test_ids)}")
check("identity splits share no identity", arrays.split_overlap(train_ids, val_ids, test_ids) == 0, "train / validation / test")
check("test identities are a different DigiFace part", set(test_ids).isdisjoint(set(train_ids) | set(val_ids)), CFG["digiface_test"])
SUMMARY["identity_data"] = {"train_ids": len(train_ids), "val_ids": len(val_ids), "test_ids": len(test_ids),
                            "train_val_images": int(len(id_labels)), "test_images": int(len(test_labels))}

fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for row, identity in enumerate(np.random.default_rng(0).choice(len(train_ids), 3, replace=False)):
    idx = np.flatnonzero(id_labels == identity)[:8]
    for col, i in enumerate(idx):
        axes[row, col].imshow(id_images[i]); axes[row, col].axis("off")
    axes[row, 0].set_title(f"identity {train_ids[identity]}", fontsize=8, loc="left")
plt.suptitle("DigiFace-1M: three synthetic identities, eight images each"); savefig("digiface_samples")
""")

code("""
# Sanity checks before the identity run: with scale 1 and no margin the first loss is ln(classes); one batch overfits.
torch.manual_seed(0)
k = len(train_ids)
probe = IdentityTrainer(k, depth="iresnet18", scale=1.0, margin=0.0).to(DEVICE)
x = torch.from_numpy(np.ascontiguousarray(id_images[:32])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
y = torch.from_numpy(id_labels[:32]).to(DEVICE)
loss0, _ = probe(x, y)
check("identity first-batch loss is ln(classes)", abs(loss0.item() - np.log(k)) < 0.5, f"{loss0.item():.3f} vs {np.log(k):.3f}")
small = np.flatnonzero(id_labels < 4)[:16]
xs = torch.from_numpy(np.ascontiguousarray(id_images[small])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
ys = torch.from_numpy(id_labels[small]).to(DEVICE)
curve = engine.overfit_one_batch(IdentityTrainer(4, depth="iresnet18"), lambda m: m(xs, ys)[0], steps=80, lr=1e-3, device=DEVICE)
check("identity model overfits one batch", curve[-1] < 0.1 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe; torch.cuda.empty_cache() if torch.cuda.is_available() else None
""")

code("""
# Start identity training as a second process on the same GPU; the notebook trains the detector meanwhile.
ID_RUN = RUN / "identity"
ID_LOG = RUN / "identity_train.log"
id_cmd = [sys.executable, "train.py", "--task", "identity", "--arrays", str(ARR), "--run-dir", str(ID_RUN),
          "--depth", CFG["id_depth"], "--workers", str(WORKERS_ID), "--budget-minutes", str(ID_BUDGET_MIN)]
if CFG["id_max_steps"]:
    id_cmd += ["--max-steps", str(CFG["id_max_steps"]), "--budget-minutes", "3"]
if (ID_RUN / "DONE").exists() or (SMOKE and (ID_RUN / "metrics.json").exists()):
    id_proc = None; print("identity training already finished:", ID_RUN)
else:
    id_proc = subprocess.Popen(id_cmd, cwd=REPO, stdout=open(ID_LOG, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    print("identity training started, process", id_proc.pid); print(" ".join(id_cmd))
""")

md("## 4. Detector: train on WIDER FACE")
code("""
records = retina if not CFG["wider_train_max"] else retina[: CFG["wider_train_max"]]
det_train, det_val = split_records(records, val_fraction=0.05, seed=0)
WIDER_IMAGES = next((RAW / "wider_train").rglob("images"))
print(f"WIDER train images: {len(det_train)} for training, {len(det_val)} held out for validation")
check("detector split shares no image", arrays.split_overlap([r["path"] for r in det_train], [r["path"] for r in det_val]) == 0,
      "train / held-out")
n_faces = sum(len(r["boxes"]) for r in det_train)
n_lm = sum(int(np.isfinite(r["landmarks"][:, 0, 0]).sum()) for r in det_train)
print(f"training faces: {n_faces}, with landmarks: {n_lm} ({n_lm / max(1, n_faces):.0%})")

# Look at augmented samples with landmarks (eyes red/green, nose blue, mouth corners yellow/magenta) to confirm the
# landmark order survives flips: red must stay on the image-left eye.
sample_set = WiderFaceTrain(det_train[:64], WIDER_IMAGES, size=640, augment=True, seed=1)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
colors = ["red", "lime", "blue", "yellow", "magenta"]
for ax, i in zip(axes.flat, range(8)):
    img, target = sample_set[i]
    rec = det_train[i]
    ax.imshow(img.permute(1, 2, 0).numpy()); ax.axis("off")
    heat = target["heatmap"][0].numpy()
    ax.imshow(cv2.resize(heat, (640, 640)), alpha=0.35, cmap="jet")
    idx = target["index"][target["mask"] > 0].numpy(); w = heat.shape[1]
    lm = target["landmarks"][target["mask"] > 0].numpy(); sz = np.exp(target["size"][target["mask"] > 0].numpy()) * 4
    for j, ix in enumerate(idx):
        cx, cy = (ix % w + 0.5) * 4, (ix // w + 0.5) * 4
        for p in range(5):
            if target["landmark_mask"][target["mask"] > 0][j, 2 * p] > 0:
                ax.plot(cx + lm[j, 2 * p] * sz[j, 0], cy + lm[j, 2 * p + 1] * sz[j, 1], "o", ms=3, color=colors[p])
plt.suptitle("Augmented detector training samples: heatmap targets and decoded landmark targets"); savefig("detector_train_samples")
""")

code("""
# Sanity checks: the loss is finite on the first batch, and the detector overfits one batch.
torch.manual_seed(0)
probe = FaceDetNet("resnet18", pretrained_backbone=True).to(DEVICE)
images, targets = detection_collate([sample_set[i] for i in range(8)])
xb = images.to(DEVICE).float() / 255; tb = {k: v.to(DEVICE) for k, v in targets.items()}
loss0, parts0 = detection_loss(probe(xb), tb)
check("detector first-batch loss is finite", torch.isfinite(loss0).item(), {k: round(v, 3) for k, v in parts0.items()})
curve = engine.overfit_one_batch(probe, lambda m: detection_loss(m(xb), tb)[0], steps=80, lr=1e-3, device=DEVICE)
check("detector overfits one batch", curve[-1] < 0.5 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe
""")

code("""
DET_RUN = RUN / "detector"; DET_RUN.mkdir(exist_ok=True)
torch.manual_seed(0)
det_model = FaceDetNet("resnet18", pretrained_backbone=True)
det_task = DetectorTask(det_model, det_train, det_val, WIDER_IMAGES, size=640, batch_size=32, num_workers=WORKERS_DET,
                        val_images=60 if SMOKE else 300)
if (DET_RUN / "plan.json").exists():           # a resumed run keeps its plan, so its schedule does not change
    det_epochs = json.loads((DET_RUN / "plan.json").read_text())["epochs"]
else:
    probe_task = DetectorTask(FaceDetNet("resnet18", pretrained_backbone=True), det_train, det_val, WIDER_IMAGES,
                              batch_size=32, num_workers=WORKERS_DET, val_images=60 if SMOKE else 300)
    sec = engine.measure_step_seconds(probe_task, DEVICE, steps=10 if SMOKE else 40, warmup=3 if SMOKE else 8)
    t0 = time.time(); probe_task.model.eval()
    with torch.no_grad(): probe_task.validate(DEVICE)
    val_sec = time.time() - t0
    fit_epochs = engine.plan_epochs(sec, det_task.steps_per_epoch, DET_BUDGET_MIN, val_sec, min_epochs=1, max_epochs=150, safety=1.0)
    det_epochs = max(1, int(round(fit_epochs * 1.3)))   # the time-aware schedule ends the run at the budget, annealed
    print(f"detector: {sec:.3f} s per step of 32 images ({32 / sec:.0f} images/s), validation {val_sec:.0f} s, "
          f"{det_task.steps_per_epoch} steps per epoch -> about {fit_epochs} epochs fit in {DET_BUDGET_MIN:.0f} minutes")
    (DET_RUN / "plan.json").write_text(json.dumps({"seconds_per_step": sec, "val_seconds": val_sec, "epochs": det_epochs,
                                                   "fit_epochs": fit_epochs, "budget_minutes": DET_BUDGET_MIN}))
    SUMMARY["detector_speed"] = {"seconds_per_step": sec, "images_per_second": 32 / sec, "val_seconds": val_sec}
    del probe_task
det_cfg = engine.TrainConfig(epochs=det_epochs, lr=5e-4, weight_decay=1e-4, optimizer="adamw", warmup_epochs=min(1.0, det_epochs / 5),
                             min_lr_ratio=0.002, grad_clip=10.0, checkpoint_every=300, log_every=50,
                             max_steps=CFG["det_max_steps"], max_minutes=DET_BUDGET_MIN, time_schedule=True)
det_result = engine.fit(det_task, det_cfg, DET_RUN, device=DEVICE, extra_config={"model": "FaceDetNet resnet18", "size": 640})
SUMMARY["detector_training"] = {"planned_epochs": det_epochs, "epochs_run": len(engine.load_history(DET_RUN)),
                                "status": det_result.status, "best_val_ap": det_result.best_value,
                                "best_epoch": det_result.best_epoch,
                                "annealed": json.loads((DET_RUN / "metrics.json").read_text()).get("annealed")}
""")

code("""
def plot_history(history, keys, title, name):
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.4))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        ax.plot([h["epoch"] for h in history], [h.get(key) for h in history], marker="o")
        ax.set_xlabel("epoch"); ax.set_title(key)
    plt.suptitle(title); savefig(name)

det_hist = engine.load_history(DET_RUN)
plot_history(det_hist, ["train_loss_last", "val_ap", "lr"], "Detector training", "detector_curves")
step_log = [l for l in (DET_RUN.parent / "detector").glob("*.jsonl")]
print("detector epochs:", len(det_hist), "best held-out AP:", det_result.best_value)
""")

md("## 5. Detector: test on WIDER FACE validation (easy, medium, hard)")
code("""
det_model = engine.load_best(FaceDetNet("resnet18", pretrained_backbone=False), DET_RUN, DEVICE)
# Score threshold for the photo pipeline: chosen on the held-out TRAINING images, never on the test set.
EVAL_SIDE = 1280   # the same resolution for the threshold (validation), the test and the photo pipeline
val_pred, val_true = predict_records(det_model, det_val, WIDER_IMAGES, DEVICE, max_side=EVAL_SIDE)
thr = choose_score_threshold(val_pred, val_true)
print("chosen detection threshold on held-out training images:", thr)
# Landmark error on held-out training images that carry landmarks.
nme_list = []   # one value per matched face that has landmark annotations
for rec in det_val[:300]:
    gt_lm = rec["landmarks"]; ok = np.isfinite(gt_lm[:, 0, 0])
    if not ok.any(): continue
    faces = detect_image(det_model, prepare.read_rgb(WIDER_IMAGES / rec["path"]), device=DEVICE, max_side=EVAL_SIDE, score_threshold=0.3)
    if not faces: continue
    pb = np.array([f["box"] for f in faces]); pl = np.array([f["landmarks"] for f in faces])
    pi, gi = det_eval.match_boxes(pb, rec["boxes"])        # match against every face, then keep those with landmarks
    for a, b in zip(pi, gi):
        if ok[b]: nme_list.append(det_eval.landmark_nme(pl[[a]], gt_lm[[b]], rec["boxes"][[b]]))
nme = float(np.nanmean(nme_list)) if nme_list else float("nan")
print(f"landmark error measured on {len(nme_list)} faces")
print(f"landmark error (NME, normalised by box size) on held-out training images: {nme:.4f}")

# Test: WIDER FACE validation, VOC-style AP with the official easy/medium/hard face lists.
preds, truths, examples, demo_images = [], [], [], []
t0 = time.time()
keep_in_range, keep_has_zero = True, False
for n, rec in enumerate(prepare.wider_val_records(RAW / sources.WIDER_VAL.filename, max_items=CFG["wider_val_max"])):
    for subset in ("easy", "medium", "hard"):
        k = rec["keep"][subset]
        keep_in_range &= bool(np.all((k >= 0) & (k < len(rec["boxes"]))))
        keep_has_zero |= bool(np.any(k == 0))
    faces = detect_image(det_model, rec["image"], device=DEVICE, max_side=EVAL_SIDE, score_threshold=0.02)
    preds.append((np.array([f["box"] for f in faces], dtype=np.float32).reshape(-1, 4), np.array([f["score"] for f in faces], dtype=np.float32)))
    truths.append({"boxes": rec["boxes"], "keep": rec["keep"]})
    if n < 6: examples.append((rec["image"], [f for f in faces if f["score"] >= thr["threshold"]], rec["boxes"]))
    if 6 <= n < 12: demo_images.append(rec["image"])
print(f"{len(truths)} test images in {time.time() - t0:.0f} s")
check("WIDER subset lists are 0-based and in range", keep_in_range and keep_has_zero, "every index < number of faces; index 0 occurs")
det_test = {}
for subset in ("easy", "medium", "hard"):
    gts = []
    for rec in truths:
        ignore = np.ones(len(rec["boxes"]), dtype=bool); ignore[rec["keep"][subset]] = False
        gts.append({"boxes": rec["boxes"], "ignore": ignore})
    det_test[subset] = det_eval.average_precision(preds, gts)
print("WIDER FACE validation AP:", {k: round(v, 4) for k, v in det_test.items()}, "(YuNet reference 0.834 / 0.824 / 0.708)")
SUMMARY["detector_test"] = {"ap": det_test, "threshold": thr, "landmark_nme_heldout": nme, "test_images": len(truths)}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (img, faces, gt) in zip(axes.flat, examples):
    ax.imshow(img); ax.axis("off")
    for b in gt: ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=0.8, ls="--"))
    for f in faces:
        b = f["box"]; ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.2))
        for p, c in zip(f["landmarks"], colors): ax.plot(p[0], p[1], "o", ms=2, color=c)
    ax.set_title(f"{len(faces)} detected, {len(gt)} annotated", fontsize=9)
plt.suptitle("Detector on WIDER FACE validation (green: ours, dashed white: ground truth)"); savefig("detector_test_examples")

# Does our detector put DigiFace's landmarks where the alignment template expects them? (review recommendation)
errs = []
for i in np.random.default_rng(0).choice(len(test_labels), min(500, len(test_labels)), replace=False):
    img = cv2.resize(np.asarray(test_images[i]), (224, 224), interpolation=cv2.INTER_CUBIC)
    faces = detect_image(det_model, img, device=DEVICE, max_side=224, score_threshold=0.3)
    if faces: errs.append(np.linalg.norm(np.array(faces[0]["landmarks"]) / 2 - align.ARCFACE_TEMPLATE_112, axis=1).mean())
print(f"DigiFace: face found in {len(errs)} of {min(500, len(test_labels))} images; mean landmark distance from the template "
      f"{np.mean(errs) if errs else float('nan'):.2f} px of 112")
SUMMARY["digiface_alignment_px"] = float(np.mean(errs)) if errs else None
""")

md("## 6. Expression data: EmoNet-Face portraits aligned with our detector")
code("""
detector = FaceDetector.from_torch(det_model, device=str(DEVICE), score_threshold=max(0.3, thr["threshold"] - 0.1), max_side=640)
detect_fn = detector.as_detect_fn()
binary_paths = [RAW / s.filename for s in sources.EMONET_BINARY[: CFG["emonet_files"]]]
hq_paths = [RAW / s.filename for s in sources.EMONET_HQ[: CFG["hq_files"]]]
t0 = time.time()
ARR_RUN = ARR / RUN_ID   # these crops depend on this run's detector, so they are never shared between runs
if not arrays.array_exists(ARR_RUN, "emonet_binary"):
    sel = prepare.emonet_selection(binary_paths, max_items=CFG["emonet_max"])
    print("usable EmoNet-Face Binary images:", len(sel))
    prepare.build_emonet_array(binary_paths, sel, detect_fn, ARR_RUN, "emonet_binary", 224, fallback_center_crop=SMOKE)
ex_images, ex_labels, ex_desc = arrays.read_array(ARR_RUN, "emonet_binary")
if not arrays.array_exists(ARR_RUN, "emonet_hq"):
    sel_hq = prepare.emonet_selection(hq_paths, hq=True, max_items=CFG["hq_max"])
    prepare.build_emonet_array(hq_paths, sel_hq, detect_fn, ARR_RUN, "emonet_hq", 224, exclude_hashes=set(ex_desc["ids"]),
                               fallback_center_crop=SMOKE, exclude_prompts=set(ex_desc["groups"]))
hq_images, hq_labels, hq_desc = arrays.read_array(ARR_RUN, "emonet_hq")
print(f"aligned in {time.time() - t0:.0f} s. Binary: {len(ex_labels)} kept, {ex_desc['no_face']} without a face, "
      f"{ex_desc['fallback']} centre-cropped (smoke only). HQ: {len(hq_labels)} kept, {hq_desc['duplicates']} duplicates of Binary "
      f"and {hq_desc.get('shared_prompt', 0)} sharing a Binary prompt removed")
check("HQ test shares no image with Binary", arrays.split_overlap(ex_desc["ids"], hq_desc["ids"]) == 0, "by SHA-1 of the image bytes")
tr_idx, va_idx, te_idx = arrays.grouped_split(ex_labels, ex_desc["groups"], (0.7, 0.15, 0.15), seed=0)
groups = np.asarray(ex_desc["groups"])
check("expression splits share no prompt", arrays.split_overlap(groups[tr_idx], groups[va_idx], groups[te_idx]) == 0,
      f"train {len(tr_idx)}, validation {len(va_idx)}, test {len(te_idx)}")
counts = {c: [int((ex_labels[idx] == i).sum()) for idx in (tr_idx, va_idx, te_idx)] for i, c in enumerate(CLASSES)}
print("class counts (train, validation, test):", counts)
SUMMARY["expression_data"] = {"kept": int(len(ex_labels)), "no_face": ex_desc["no_face"], "counts": counts,
                              "hq_kept": int(len(hq_labels)), "hq_duplicates_removed": hq_desc["duplicates"]}

fig, axes = plt.subplots(len(CLASSES), 8, figsize=(12, 1.6 * len(CLASSES)))
for r, c in enumerate(CLASSES):
    idx = tr_idx[ex_labels[tr_idx] == r][:8]
    for col in range(8):
        axes[r, col].axis("off")
        if col < len(idx): axes[r, col].imshow(ex_images[idx[col]])
    axes[r, 0].set_title(c, fontsize=9, loc="left")
plt.suptitle("Aligned EmoNet-Face training crops by class"); savefig("expression_samples")
""")

md("## 7. Expression: train")
code("""
torch.manual_seed(0)
probe = ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True).to(DEVICE)
xb = torch.from_numpy(np.ascontiguousarray(ex_images[tr_idx[:32]])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
yb = torch.from_numpy(ex_labels[tr_idx[:32]]).to(DEVICE)
with torch.no_grad(): l0 = torch.nn.functional.cross_entropy(probe.eval()(xb), yb).item()
check("expression first-batch loss is near ln(8)", abs(l0 - np.log(len(CLASSES))) < 0.7, f"{l0:.3f} vs {np.log(len(CLASSES)):.3f}")
curve = engine.overfit_one_batch(probe, lambda m: torch.nn.functional.cross_entropy(m(xb), yb), steps=60, lr=3e-4, device=DEVICE)
check("expression model overfits one batch", curve[-1] < 0.1 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe

EXPR_RUN = RUN / "expression"; EXPR_RUN.mkdir(exist_ok=True)
torch.manual_seed(0)
expr_model = ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True)
expr_task = ExpressionTask(expr_model, ex_images, ex_labels, tr_idx, va_idx, len(CLASSES), batch_size=64, num_workers=4)
if (EXPR_RUN / "plan.json").exists():
    expr_epochs = json.loads((EXPR_RUN / "plan.json").read_text())["epochs"]
else:
    probe_task = ExpressionTask(ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True), ex_images, ex_labels, tr_idx, va_idx,
                                len(CLASSES), batch_size=64, num_workers=4)
    sec = engine.measure_step_seconds(probe_task, DEVICE, steps=10 if SMOKE else 30, warmup=3 if SMOKE else 5)
    expr_epochs = engine.plan_epochs(sec, expr_task.steps_per_epoch, EXPR_BUDGET_MIN, 20, min_epochs=1, max_epochs=30)
    print(f"expression: {sec:.3f} s per step of 64 -> {expr_epochs} epochs in {EXPR_BUDGET_MIN:.0f} minutes"); del probe_task
    (EXPR_RUN / "plan.json").write_text(json.dumps({"seconds_per_step": sec, "epochs": expr_epochs, "budget_minutes": EXPR_BUDGET_MIN}))
expr_cfg = engine.TrainConfig(epochs=expr_epochs, lr=2e-4, weight_decay=0.05, optimizer="adamw", warmup_epochs=min(1.0, expr_epochs / 5),
                              grad_clip=5.0, checkpoint_every=200, log_every=50, max_steps=CFG["expr_max_steps"],
                              max_minutes=EXPR_BUDGET_MIN * 1.3, time_schedule=True)
expr_result = engine.fit(expr_task, expr_cfg, EXPR_RUN, device=DEVICE, extra_config={"model": "ExpressionNet convnext_tiny", "size": 224})
plot_history(engine.load_history(EXPR_RUN), ["val_loss", "val_accuracy", "val_macro_f1"], "Expression training", "expression_curves")
SUMMARY["expression_training"] = {"epochs": expr_epochs, "status": expr_result.status, "best_val_macro_f1": expr_result.best_value,
                                  "best_epoch": expr_result.best_epoch}
""")

md("## 8. Expression: test")
code("""
expr_model = engine.load_best(ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=False), EXPR_RUN, DEVICE)
val_logits, val_y = predict_logits(expr_model, arrays.ArrayDataset(ex_images, ex_labels, va_idx), DEVICE)
# Temperature scaling on validation: one number that makes the probabilities honest (lower calibration error).
t = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
vl, vy = torch.from_numpy(val_logits), torch.from_numpy(val_y)
def closure():
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(vl / t.exp(), vy); loss.backward(); return loss
opt.step(closure); TEMPERATURE = float(t.exp())
test_logits, test_y = predict_logits(expr_model, arrays.ArrayDataset(ex_images, ex_labels, te_idx), DEVICE)
report = metrics.classification_report(test_y, test_logits.argmax(1), len(CLASSES))
from sklearn.metrics import f1_score
per_class_f1 = f1_score(test_y, test_logits.argmax(1), labels=list(range(len(CLASSES))), average=None, zero_division=0)
ece_raw = metrics.expected_calibration_error(softmax(test_logits), test_y)
ece_cal = metrics.expected_calibration_error(softmax(test_logits, TEMPERATURE), test_y)
print(f"TEST accuracy {report['accuracy']:.4f} (95% CI {report['accuracy_ci95'][0]:.3f}-{report['accuracy_ci95'][1]:.3f}), "
      f"macro-F1 {report['macro_f1']:.4f} (95% CI {report['macro_f1_ci95'][0]:.3f}-{report['macro_f1_ci95'][1]:.3f}), n={report['n']}")
print("per-class F1:", {c: round(float(f), 3) for c, f in zip(CLASSES, per_class_f1)})
print(f"temperature {TEMPERATURE:.3f}; calibration error (ECE) {ece_raw:.4f} before, {ece_cal:.4f} after")
cm = np.array(report["confusion_matrix"]); cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(len(CLASSES)), CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="black" if cmn[i, j] < 0.6 else "white")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False); plt.colorbar(im); plt.title("Expression test confusion matrix"); savefig("expression_confusion")

hq_logits, hq_y = predict_logits(expr_model, arrays.ArrayDataset(hq_images, hq_labels), DEVICE)
hq_report = metrics.classification_report(hq_y, hq_logits.argmax(1), len(CLASSES), n_boot=300) if len(hq_y) else {}
print(f"HQ (expert-rated, separate images) agreement: accuracy {hq_report.get('accuracy', float('nan')):.4f}, macro-F1 {hq_report.get('macro_f1', float('nan')):.4f}, n={len(hq_y)}")

# Real faces: the London set's neutral and smiling front photos (expression only, never identification).
london = prepare.london_records(RAW / "london_neutral", RAW / "london_smiling")
crops, views = [], []
for r in london:
    img = prepare.read_rgb(r["path"])
    crop = None if img is None else prepare.detect_and_align(detect_fn, img, 224)
    if crop is not None: crops.append(crop); views.append(r["view"])
lp = softmax(predict_logits(expr_model, arrays.ArrayDataset(np.stack(crops), np.zeros(len(crops), dtype=np.int64)), DEVICE)[0], TEMPERATURE) if crops else np.zeros((0, 8))
views = np.array(views); happy = CLASSES.index("happy"); neutral = CLASSES.index("neutral")
smile_happy = float((lp[views == "smiling"].argmax(1) == happy).mean()) if (views == "smiling").any() else float("nan")
neutral_not_happy = float((lp[views == "neutral"].argmax(1) != happy).mean()) if (views == "neutral").any() else float("nan")
neutral_neutral = float((lp[views == "neutral"].argmax(1) == neutral).mean()) if (views == "neutral").any() else float("nan")
print(f"London real faces ({len(crops)} of {len(london)} photos with a detected face): smiling -> happy {smile_happy:.3f}; "
      f"neutral -> not happy {neutral_not_happy:.3f}; neutral -> neutral {neutral_neutral:.3f}")
SUMMARY["expression_test"] = {"accuracy": report["accuracy"], "accuracy_ci95": report["accuracy_ci95"], "macro_f1": report["macro_f1"],
    "macro_f1_ci95": report["macro_f1_ci95"], "per_class_f1": dict(zip(CLASSES, map(float, per_class_f1))), "n": report["n"],
    "temperature": TEMPERATURE, "ece_before": ece_raw, "ece_after": ece_cal, "hq_accuracy": hq_report.get("accuracy"),
    "hq_macro_f1": hq_report.get("macro_f1"), "hq_n": int(len(hq_y)), "london_smiling_to_happy": smile_happy,
    "london_neutral_not_happy": neutral_not_happy, "london_neutral_to_neutral": neutral_neutral, "london_faces": len(crops)}

pick = np.random.default_rng(0).choice(len(te_idx), min(24, len(te_idx)), replace=False)
fig, axes = plt.subplots(3, 8, figsize=(14, 6.2))
probs_all = softmax(test_logits, TEMPERATURE)
for ax, i in zip(axes.flat, pick):
    ax.imshow(ex_images[te_idx[i]]); ax.axis("off")
    pred, true = probs_all[i].argmax(), test_y[i]
    ax.set_title(f"{CLASSES[pred]} {probs_all[i].max():.2f}\\n(true {CLASSES[true]})", fontsize=8, color="green" if pred == true else "red")
plt.suptitle("Expression predictions on test portraits (green right, red wrong)"); savefig("expression_test_examples")
""")

md("## 9. Identity: wait for the training process, then test on unseen identities")
code("""
waited = 0
while id_proc is not None and id_proc.poll() is None:
    time.sleep(30); waited += 30
    if waited % 300 == 0:
        print(f"[{waited // 60} min] identity still training; last lines:")
        print("".join(ID_LOG.read_text().splitlines(True)[-2:]))
print("identity process finished" + (f" with exit code {id_proc.returncode}" if id_proc else ""))
print("".join(ID_LOG.read_text().splitlines(True)[-8:]))
if not (ID_RUN / "metrics.json").exists():
    raise RuntimeError("identity training did not finish; the end of its log is printed above (" + str(ID_LOG) + ")")
id_plan = json.loads((ID_RUN / "plan.json").read_text()); id_metrics = json.loads((ID_RUN / "metrics.json").read_text())
print("plan:", id_plan); print("result:", id_metrics)
plot_history(engine.load_history(ID_RUN), ["train_loss_last", "val_auc", "val_tar@far=0.001"], "Identity training", "identity_curves")
SUMMARY["identity_training"] = {"plan": id_plan, "result": id_metrics, "epochs_run": len(engine.load_history(ID_RUN))}
""")

code("""
ID_DEPTH = id_plan.get("depth", CFG["id_depth"])
embedder = IResNet(ID_DEPTH)
state = torch.load(ID_RUN / ("best.pt" if (ID_RUN / "best.pt").exists() else "final.pt"), map_location="cpu", weights_only=True)["model"]
embedder.load_state_dict({k[len("embedder."):]: v for k, v in state.items() if k.startswith("embedder.")})
embedder = embedder.to(DEVICE).eval()
num_train = int(id_desc["num_train_ids"])
val_idx = np.flatnonzero(id_labels >= num_train)
val_emb, val_lab = embed(embedder, arrays.ArrayDataset(id_images, id_labels, val_idx), DEVICE)
vp, vs = metrics.verification_pairs(val_lab, 3000, 30000, seed=1)
val_rep = metrics.verification_report(metrics.cosine_scores(val_emb, vp), vs, fars=(1e-3,))
ID_THRESHOLD = val_rep["threshold@far=0.001"]
print(f"1:1 identity threshold chosen on validation identities at FAR 1e-3: {ID_THRESHOLD:.4f}")

def open_set(emb, lab, threshold=None, seed=7, n_enrol=3, n_probe=10, fpir=0.01):
    # Enrol half the identities (template = mean of n_enrol photos); probe with other photos of them (genuine) and
    # with photos of the other half (strangers). Choose or apply the threshold at which `fpir` of strangers get a name.
    ids = np.unique(lab); r = np.random.default_rng(seed); r.shuffle(ids)
    enrolled, strangers = ids[: len(ids) // 2], ids[len(ids) // 2:]
    tmpl = np.stack([emb[lab == v][:n_enrol].mean(0) for v in enrolled]); tmpl /= np.linalg.norm(tmpl, axis=1, keepdims=True)
    gen = [(emb[lab == v][n_enrol:n_enrol + n_probe], i) for i, v in enumerate(enrolled)]
    gp = np.concatenate([g for g, _ in gen]); gi = np.concatenate([np.full(len(g), i) for g, i in gen])
    sp = np.concatenate([emb[lab == v][:n_probe] for v in strangers])
    s_best = (sp @ tmpl.T).max(1); g_sims = gp @ tmpl.T
    if threshold is None:
        threshold = float(np.quantile(s_best, 1.0 - fpir))
    named_right = (g_sims.argmax(1) == gi) & (g_sims.max(1) >= threshold)
    return {"threshold": threshold, "gallery_size": int(len(enrolled)), "fpir": float((s_best >= threshold).mean()),
            "fnir": float(1.0 - named_right.mean()), "rank1": float((g_sims.argmax(1) == gi).mean())}

val_open = open_set(val_emb, val_lab)
ID_THRESHOLD_1N = val_open["threshold"]
print(f"1:N gallery threshold chosen on validation identities (1% of strangers named, gallery of {val_open['gallery_size']}): "
      f"{ID_THRESHOLD_1N:.4f}")

# The test part is scored once, after every choice above.
te_emb, te_lab = embed(embedder, arrays.ArrayDataset(test_images, test_labels), DEVICE)
tp, ts = metrics.verification_pairs(te_lab, 5000, 200000, seed=2)
te_scores = metrics.cosine_scores(te_emb, tp)
te_rep = metrics.verification_report(te_scores, ts, fars=(1e-2, 1e-3, 1e-4))
first = {}
for i, l in enumerate(te_lab): first.setdefault(int(l), i)
g_idx = np.array(sorted(first.values())); p_idx = np.setdiff1d(np.arange(len(te_lab)), g_idx)
ident = metrics.identification_report(te_emb[g_idx], te_lab[g_idx], te_emb[p_idx], te_lab[p_idx], threshold=ID_THRESHOLD)
fixed_thr_far = float(((te_scores >= ID_THRESHOLD) & (ts == 0)).sum() / max(1, (ts == 0).sum()))
fixed_thr_tar = float(((te_scores >= ID_THRESHOLD) & (ts == 1)).sum() / max(1, (ts == 1).sum()))
print(f"TEST verification: AUC {te_rep['auc']:.4f}, EER {te_rep['eer']:.4f}, TAR@FAR1e-3 {te_rep['tar@far=0.001']:.4f}, "
      f"TAR@FAR1e-4 {te_rep['tar@far=0.0001']:.4f} ({te_rep['n_pairs']} pairs)")
print(f"at the validation threshold: TAR {fixed_thr_tar:.4f}, FAR {fixed_thr_far:.5f}")
print(f"TEST identification (one enrolment photo per identity, {ident['n_gallery']} identities): rank-1 {ident['rank1']:.4f}, "
      f"rank-5 {ident['rank5']:.4f}, correct and above threshold {ident['open_set_correct_at_threshold']:.4f}")
test_open = open_set(te_emb, te_lab, threshold=ID_THRESHOLD_1N, seed=8)
print(f"TEST open-set identification at the 1:N threshold (gallery of {test_open['gallery_size']}): strangers named "
      f"{test_open['fpir']:.4f}, enrolled people missed or misnamed {test_open['fnir']:.4f}, rank-1 {test_open['rank1']:.4f}")

# The same test identities through OUR detector and alignment, as the photo pipeline sees faces (review finding).
sub = np.random.default_rng(4).choice(len(test_labels), min(3000, len(test_labels)), replace=False)
pipe_crops, pipe_idx = [], []
for i in sub:
    img = cv2.resize(np.asarray(test_images[i]), (224, 224), interpolation=cv2.INTER_CUBIC)
    faces = detect_image(det_model, img, device=DEVICE, max_side=224, score_threshold=0.3)
    if faces:
        pipe_crops.append(align.align_face(img, np.asarray(faces[0]["landmarks"], dtype=np.float32), 112)); pipe_idx.append(i)
pipe_idx = np.asarray(pipe_idx)
pipe_report = {"detected": int(len(pipe_idx)), "of": int(len(sub))}
if len(pipe_idx) > 100:
    pe, pl = embed(embedder, arrays.ArrayDataset(np.stack(pipe_crops), test_labels[pipe_idx]), DEVICE)
    pp, ps = metrics.verification_pairs(pl, 3000, 60000, seed=5)
    via_pipe = metrics.verification_report(metrics.cosine_scores(pe, pp), ps, fars=(1e-3,))
    direct = metrics.verification_report(metrics.cosine_scores(te_emb[pipe_idx], pp), ps, fars=(1e-3,))
    pipe_report.update({"tar@far=0.001_pipeline": via_pipe["tar@far=0.001"], "tar@far=0.001_direct": direct["tar@far=0.001"],
                        "auc_pipeline": via_pipe["auc"], "auc_direct": direct["auc"]})
print("test identities through our detector and alignment:", pipe_report)
SUMMARY["identity_test"] = {**{k: v for k, v in te_rep.items() if k != "roc"}, **ident, "threshold_1to1": ID_THRESHOLD,
                            "tar_at_threshold": fixed_thr_tar, "far_at_threshold": fixed_thr_far,
                            "threshold_1toN": ID_THRESHOLD_1N, "open_set_validation": val_open, "open_set_test": test_open,
                            "through_pipeline": pipe_report}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(te_rep["roc"]["fpr"], te_rep["roc"]["tpr"]); axes[0].set_xscale("log"); axes[0].set_xlim(1e-5, 1)
axes[0].set_xlabel("false accept rate"); axes[0].set_ylabel("true accept rate"); axes[0].set_title("Verification ROC, unseen identities")
axes[1].hist(te_scores[ts == 1], bins=60, alpha=0.6, density=True, label="same person")
axes[1].hist(te_scores[ts == 0], bins=60, alpha=0.6, density=True, label="different people")
axes[1].axvline(ID_THRESHOLD, color="k", ls="--", label="threshold"); axes[1].legend(); axes[1].set_title("Cosine similarity")
savefig("identity_roc")
sims = te_emb[p_idx] @ te_emb[g_idx].T
fig, axes = plt.subplots(4, 6, figsize=(10, 7))
for r, pi in enumerate(np.random.default_rng(3).choice(len(p_idx), 4, replace=False)):
    order = np.argsort(-sims[pi])[:5]
    axes[r, 0].imshow(test_images[p_idx[pi]]); axes[r, 0].set_title("probe", fontsize=8)
    for c, gi in enumerate(order, start=1):
        ok = te_lab[g_idx[gi]] == te_lab[p_idx[pi]]
        axes[r, c].imshow(test_images[g_idx[gi]]); axes[r, c].set_title(f"{sims[pi, gi]:.2f}", fontsize=8, color="green" if ok else "red")
for ax in axes.flat: ax.axis("off")
plt.suptitle("Identification: probe and its five best gallery matches (green = same identity)"); savefig("identity_examples")
""")

md("## 10. Export to ONNX, check parity, and write the model card")
code("""
export.export_detector(det_model.cpu(), MODELS / "detector.onnx")
export.export_classifier(expr_model.cpu(), MODELS / "expression.onnx", 224, "logits")
export.export_classifier(embedder.cpu(), MODELS / "embedder.onnx", 112, "embedding")
par = {"detector": export.parity(det_model, MODELS / "detector.onnx", torch.rand(1, 3, 480, 640)),
       "expression": export.parity(expr_model, MODELS / "expression.onnx", torch.rand(4, 3, 224, 224)),
       "embedder": export.parity(embedder, MODELS / "embedder.onnx", torch.rand(4, 3, 112, 112))}
for k, v in par.items(): check(f"ONNX parity: {k}", v < 1e-3, f"max relative difference {v:.2e}")
for name, run in (("detector", DET_RUN), ("expression", EXPR_RUN), ("identity", ID_RUN)):
    shutil.copy2(run / ("best.pt" if (run / "best.pt").exists() else "final.pt"), MODELS / f"{name}.pt")

card = {
    "name": "facekit", "version": "0.1.0", "created": time.strftime("%Y-%m-%d"), "code_version": SUMMARY["code_version"],
    "use": "Non-commercial research. Licence chain: WIDER FACE CC BY-NC-ND 4.0; RetinaFace annotations non-commercial research "
           "(models trained with them included); DigiFace-1M Microsoft Research Use of Data Agreement (trained models are Results); "
           "EmoNet-Face CC BY 4.0. Personal photo-library use is non-commercial but not strictly research: the owner decides.",
    "detector": {"file": "detector.onnx", "architecture": "CenterFace-style, ResNet-18 ImageNet backbone, stride 4",
                 "input": "RGB float [1,3,H,W] in [0,1], H and W multiples of 32", "score_threshold": thr["threshold"],
                 "nms_iou": 0.4, "trained_on": "WIDER FACE train (95%) with RetinaFace landmarks",
                 "test": {"wider_val_ap_easy_medium_hard": det_test, "protocol": "VOC-style AP@0.5 with the official subset lists"},
                 "landmark_nme_heldout": nme},
    "expression": {"file": "expression.onnx", "architecture": "ConvNeXt-Tiny, ImageNet init", "classes": list(CLASSES),
                   "input_size": 224, "temperature": TEMPERATURE, "trained_on": "EmoNet-Face Binary (synthetic), prompt-grouped split",
                   "test": SUMMARY["expression_test"],
                   "limitations": "Neutral is a proxy (Emotional Numbness, Concentration, Contemplation). Trained on synthetic portraits."},
    "identity": {"file": "embedder.onnx", "architecture": f"{ID_DEPTH} from scratch, CosFace s=64 m=0.35, 512-d",
                 "input_size": 112, "threshold": ID_THRESHOLD_1N,
                 "threshold_rule": f"1:N gallery threshold: 1% of strangers named with a gallery of {val_open['gallery_size']} "
                                   "people, set on held-out synthetic validation identities",
                 "threshold_1to1": ID_THRESHOLD, "threshold_1to1_rule": "FAR 1e-3 on held-out synthetic validation identities",
                 "trained_on": f"DigiFace-1M parts {CFG['digiface_train']}", "test": SUMMARY["identity_test"],
                 "limitations": "Trained and calibrated on synthetic faces only. Accuracy on real faces was NOT measured in this run. "
                                "Recalibrate the threshold on your own enrolled photos before trusting names."},
    "alignment": "5-point similarity transform to the ArcFace 112x112 template (scaled for 224)",
}
(MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=float))

# CPU latency of the whole pipeline from the ONNX files, as the laptop will run it.
analyzer = FaceAnalyzer.from_model_dir(MODELS)
sample = examples[0][0]
t0 = time.time(); faces = analyzer.analyze(sample); latency = time.time() - t0
t0 = time.time(); [analyzer.analyze(sample) for _ in range(3)]; latency = (time.time() - t0) / 3
print(f"CPU pipeline latency on a {sample.shape[1]}x{sample.shape[0]} photo with {len(faces)} faces: {latency:.2f} s")
card["cpu_latency_seconds"] = latency; (MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
SUMMARY["onnx_parity"] = par; SUMMARY["cpu_latency_seconds"] = latency
""")

md("## 11. The whole pipeline on photos")
code("""
# Enrol 12 unseen DigiFace identities with three photos each, then identify other photos of them and of 12 strangers.
rng = np.random.default_rng(5)
people = rng.choice(np.unique(test_labels), 24, replace=False); known, strangers = people[:12], people[12:]
gallery = Gallery(ID_THRESHOLD_1N)
for p in known: gallery.enroll(f"person-{p}", te_emb[np.flatnonzero(te_lab == p)[:3]])
right = wrong = unknown_ok = unknown_bad = 0
for p in known:
    for i in np.flatnonzero(te_lab == p)[3:8]:
        name, _ = gallery.identify(te_emb[i]); right += name == f"person-{p}"; wrong += name not in (None, f"person-{p}")
for p in strangers:
    for i in np.flatnonzero(te_lab == p)[:5]:
        name, _ = gallery.identify(te_emb[i]); unknown_ok += name is None; unknown_bad += name is not None
print(f"enrolled people found: {right}/60, named wrongly: {wrong}; strangers rejected: {unknown_ok}/60, wrongly named: {unknown_bad}")
SUMMARY["gallery_demo"] = {"found": right, "wrong_name": wrong, "strangers_rejected": unknown_ok, "strangers_named": unknown_bad}

analyzer = FaceAnalyzer.from_model_dir(MODELS)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, img in zip(axes.flat, demo_images):
    ax.imshow(img); ax.axis("off")
    for f in analyzer.analyze(img):
        b = f.detection.box
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.2))
        ax.text(b[0], b[1] - 2, f"{f.expression} {max(f.expression_probs.values()):.2f}", color="yellow", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=0.5))
plt.suptitle("Full pipeline from the ONNX files: faces and expressions on WIDER FACE validation photos"); savefig("pipeline_wider")
""")

md("## 12. Summary against SPEC-001")
code("""
targets = [("detector AP easy >= 0.85", SUMMARY["detector_test"]["ap"]["easy"], 0.85),
           ("detector AP medium >= 0.80", SUMMARY["detector_test"]["ap"]["medium"], 0.80),
           ("detector AP hard >= 0.55", SUMMARY["detector_test"]["ap"]["hard"], 0.55),
           ("expression macro-F1 >= 0.70", SUMMARY["expression_test"]["macro_f1"], 0.70),
           ("identity TAR@FAR1e-3 >= 0.80", SUMMARY["identity_test"]["tar@far=0.001"], 0.80)]
print(f"{'target':32s} {'result':>8s}  met")
for name, value, goal in targets:
    print(f"{name:32s} {value:8.4f}  {'yes' if value >= goal else 'no'}")
print(); print("checks:"); [print(("PASS" if c["ok"] else "FAIL"), c["check"], "-", c["detail"]) for c in CHECKS]
SUMMARY["targets"] = [{"target": n, "value": v, "met": v >= g} for n, v, g in targets]
SUMMARY["checks"] = CHECKS; SUMMARY["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(RUN / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
print(); print("ALL CHECKS PASSED" if all(c["ok"] for c in CHECKS) else "SOME CHECKS FAILED: read the lines above")
print("saved:"); [print(f"  {p.relative_to(RUN)}  {p.stat().st_size / 1e6:.1f} MB") for p in sorted(MODELS.iterdir())]
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    for cell in CELLS:
        if cell[0] == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(cell[1]))
        else:
            c = nbf.v4.new_code_cell(cell[1])
            if cell[2]:
                c.metadata["tags"] = cell[2]
            nb.cells.append(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "03_end_to_end.ipynb"
    build(out)
    print("wrote", out, len(CELLS), "cells")
