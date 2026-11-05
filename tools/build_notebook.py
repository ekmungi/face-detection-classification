"""Builds the notebook notebooks/03_end_to_end.ipynb from the cells written in this file.

The text and code of every notebook cell are written below as Python strings, so that the notebook can be reviewed as
plain Python and not as notebook JSON. Each call to md() adds a markdown cell and each call to code() adds a code
cell. The build() function at the end turns the collected cells into the notebook file.

Run: python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

# The cells in notebook order. Each entry holds the cell type, the cell text, and the attachments of a markdown cell or
# the tags of a code cell.
CELLS: list[tuple[str, str]] = []


def md(text, attachments=None):
    """Adds a markdown cell to the notebook.

    Newline characters at the start and the end of the text are removed. The attachments are the images that the text
    refers to, as a dict. They default to none.
    """
    CELLS.append(("markdown", text.strip("\n"), attachments or {}))


def code(text, tags=None):
    """Adds a code cell to the notebook.

    Newline characters at the start and the end of the text are removed. The tags are a list of cell tags, such as the
    "parameters" tag that papermill uses to find the cell whose values it replaces. They default to none.
    """
    CELLS.append(("code", text.strip("\n"), tags or []))


# The cells follow in the order in which they appear in the notebook.
md("""
# Face detection, expression and identity: trained end to end on Colab

This notebook trains three neural networks from open data, tests each one, and saves them in a form that runs on an ordinary laptop CPU.

1. **Detector.** It finds every face in a photo and five landmark points on each face: the two eyes, the nose tip and the two corners of the mouth.
   - It follows the design of CenterFace, a published detector that finds each face by predicting the point at its centre.
   - Its backbone, the main part of the network, is a ResNet-18 that starts from general ImageNet weights. The backbone turns the photo into feature maps, grids of numbers that describe each region of the photo.
   - A small head on top reads those maps and predicts where the face centres are.
   - It is trained on WIDER FACE, a set of real web photos of crowds and events.
2. **Expression.** It says which of eight expressions a face shows.
   - It is a ConvNeXt-Tiny classifier, also started from ImageNet weights.
   - It is fine-tuned on EmoNet-Face, a set of AI-generated portraits labelled by experts.
   - Each portrait is first cut out and turned upright using our own detector's landmarks.
3. **Identity.** It turns a face into an embedding, a list of 512 numbers. The embeddings of two photos of the same person point in almost the same direction.
   - The network is an IResNet, a ResNet variant built for face recognition, trained from random weights.
   - It is trained on DigiFace-1M, a set of synthetic faces rendered from 3D models.
   - Its training uses a CosFace margin: the score for the correct person is lowered by a fixed amount, so the network has to separate different people by a wider gap.

**Data and licences.** No network that was already trained on faces is used.

- Every dataset and its licence are recorded in a manifest in section 2.
- The model card in section 10 also records the licences of the four training sources: WIDER FACE, the RetinaFace annotations, DigiFace-1M and EmoNet-Face.

### How to run it

The notebook needs a Colab machine with an NVIDIA GPU. Choose one of three modes with `MODE` in the parameters cell. Section 1 lists the data that each mode uses.

| | Smoke (the default) | Medium | Full |
|---|---|---|---|
| What it is for | testing that every cell runs | meaningful results, with all eight expressions, from one run on an L4 | the real run |
| Training | a few dozen steps per model | 6 minutes for the detector, 8 for identity and 3 for expression | the three budget parameters: 50, 55 and 12 minutes by default |
| Time on the machine | quick once the files are in Drive | about 30 to 45 minutes on an L4 (an estimate, not measured) | 64 minutes on an A100 for the current models, whose budgets were 40, 50 and 12 minutes |
| The models it makes | not useful | expected to be weaker than a full run's, because they train on less data for minutes | the current models |

- Most sections run checks that print PASS or FAIL, and the last cell prints ALL CHECKS PASSED when every check held.
- The default detector and identity budgets, 50 and 55 minutes, are larger than the 40 and 50 that produced the current models, so a full run with the defaults takes longer than 64 minutes.
- Without a Google Drive mount, everything the run writes stays on the Colab machine and is lost when the server is removed, so download what you want to keep first.
""")

md("""
## How the pipeline fits together

![The face pipeline: the detector finds each face and its five landmarks, the alignment step cuts two upright crops, the expression model turns the larger crop into eight probabilities, and the identity model's embedding of the smaller crop is matched against the gallery of enrolled people.](https://raw.githubusercontent.com/ekmungi/face-detection-classification/main/docs/figures/pipeline_flow.png)

The figure shows what happens to one photo once the models are trained, when the finished pipeline runs on a laptop or a GPU.

1. The detector finds each face and its five landmarks.
2. The alignment step uses those landmarks to cut two upright crops of the face.
3. The larger crop goes to the expression model.
4. The smaller crop goes to the identity model, whose embedding is compared with the people enrolled in the gallery.

- Boxes with a white fill are the three trained networks, which this notebook trains, and each one lists what is inside it.
- The grey alignment box is a fixed calculation with no trained weights. So is the amber gallery match, which is marked because its threshold needs recalibrating.
- The figure's source and the PNG are in `docs/figures/`, and this cell shows the PNG from its address on GitHub.
- The picture appears wherever this notebook is opened while the computer is online: on GitHub, in VS Code, and in Colab, whether Colab opens it from GitHub, from Google Drive or as an uploaded file. Without a connection, the sentence that describes the picture appears instead.

| Block | Model | Input | Output |
|---|---|---|---|
| 1. Detector | CenterFace-style network on a ResNet-18 backbone, 11.5 million weights, with four heads (face-centre heatmap, box size, centre offset and landmarks) on one feature map at a quarter of the photo's resolution | the photo, shrunk so its longest side is at most 1,280 pixels | for each face, a box, five landmarks (eyes, nose tip, mouth corners) and a confidence score; faces below the score threshold (0.32 in the current models) or smaller than 16 pixels are dropped |
| 2. Alignment | none: a rotation and scaling onto the five-point template of ArcFace, a published face-recognition method | the photo and each face's five landmarks | an upright crop of each face at 224 by 224 and at 112 by 112 pixels |
| 3. Expression | ConvNeXt-Tiny, 27.8 million weights, whose head gives eight scores, which are divided by the temperature (section 8) before they become probabilities | the 224 by 224 crop | eight probabilities: neutral, happy, sad, surprise, fear, disgust, anger and contempt |
| 4. Identity | IResNet-50, 43.6 million weights. The CosFace margin is used only in training, so the finished model gives the embedding alone | the 112 by 112 crop and its mirror image | an embedding of 512 numbers |
| 5. Gallery match | none: a comparison | the embedding | the closest enrolled person's name when the cosine similarity reaches the threshold (0.52 in the current models), otherwise "unknown" |

Where each part is in this notebook:

- Sections 3 to 9 prepare the data, train the three networks and test them.
- Section 10 exports them to ONNX files, and section 11 runs this whole pipeline on photos.
- **Warning.** The 0.52 naming threshold was set on synthetic faces. On real faces it gave a name to 98% of strangers, so it needs recalibrating before any name is trusted.
""")

md("""
## Parameters

### Choose the settings of this run

**What it does and why**

- The next cell holds the seven settings of a run. To run the notebook by hand, change them here before you start.
- Papermill is a tool that runs a notebook from top to bottom as an ordinary program. When it runs this notebook unattended, it finds this cell by its tag `parameters` and replaces these values.

**What goes in**

| Setting | What it sets |
|---|---|
| `MODE` | smoke, medium or full, which the introduction compares |
| `EXP` | the experiment folder, inside the artifacts folder, that the results go into |
| `RUN_ID` | this run's own folder inside `EXP` |
| `DET_BUDGET_MIN`, `ID_BUDGET_MIN`, `EXPR_BUDGET_MIN` | the minutes that the detector, identity and expression trainings are planned to take in smoke and full mode, except that smoke mode's identity training always uses 3 minutes. Medium mode uses its own budgets of 6, 8 and 3 minutes |
| `USE_DRIVE` | keeps the downloaded files in Google Drive, so that they are downloaded only once (section 2) |

- Each training measures its own speed first and plans its epochs, the full passes over its training data, to fit its budget.
- The detector and identity trainings stop at their budgets, and the expression training may run up to 1.3 times its budget.
- Give `RUN_ID` a name that no earlier run has used, starting with the mode: `smoke-`, `medium-`, or `run-` for a full run. The run then writes into its own folder and does not reuse an earlier run's training plans.
- Smoke and medium runs live in the `scratch` folder, and a full run needs `EXP` set to an experiment folder. The comments in the cell show the form of each name, and `paths.py` lists the names it accepts.

**What comes out**

- The cell only sets the seven values and prints nothing. The cells below read them.

**How to tell it worked**

- The cell runs without an error.
- The setup cell below stops with a message if `MODE` is not smoke, medium or full, if `RUN_ID` does not start with the mode, or if the run folder already holds a run of another mode.
""")
code("""
# Parameters (papermill overrides these).
MODE = "smoke"            # "smoke": tiny subsets to test every cell; "medium": meaningful results on an L4; "full": the real run
EXP = "scratch"           # the results folder: "scratch" for a smoke or medium run (paths.py lists the accepted names)
RUN_ID = "smoke-001"      # this run's own folder inside EXP: smoke-NNN, medium-NNN, or run-NNN for a full run
DET_BUDGET_MIN = 50.0     # minutes for detector training (medium mode uses its own budgets instead)
ID_BUDGET_MIN = 55.0      # minutes for identity training (a second process on the same GPU)
EXPR_BUDGET_MIN = 12.0    # minutes for expression training
USE_DRIVE = True          # keep the downloaded files in Google Drive (Colab asks you to approve it once)
""", tags=["parameters"])

md("""
## 1. Setup

### Get the code onto the machine

**What it does and why**

Colab gives every new server an empty disk, so the project's code has to be copied onto each new server:

1. **GPU check.** The cell looks for the `nvidia-smi` program that NVIDIA's driver installs. If there is none, it stops at once with a message, because a TPU or CPU machine cannot run this notebook.
2. **Copy the code.** It copies this project's public GitHub repository to `/content/face-detection-classification`, unless a copy is already there.

- The cell matters only on a Colab kernel opened from VS Code or the browser, and it does nothing on a laptop.

**What goes in**

- The cell reads only whether the machine is a Colab machine, whether it has an NVIDIA GPU, and whether the code is already there.

**What comes out**

- The cell prints nothing when all is well.
- On a machine without a GPU, it stops with a message that says how to remove the server and start a GPU server instead.

**How to tell it worked**

- The cell finishes in a few seconds without an error.
""")
code('''
# Bootstrap for a Colab kernel opened from VS Code or the browser. It does nothing on this laptop.
# On a Colab machine it stops at once without an NVIDIA GPU, and it clones the code when the clone is missing.
import os, shutil, subprocess
REPO_URL = "https://github.com/ekmungi/face-detection-classification"
COLAB_CLONE = "/content/face-detection-classification"
if os.path.isdir("/content"):              # a Colab machine: the same test that paths.py uses
    # 1. GPU check
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "This notebook needs a Colab GPU machine (T4, L4 or A100), and this machine has no NVIDIA GPU "
            "(it is a TPU or CPU machine). Remove this server (Ctrl+Shift+P, Colab: Remove Server) and start "
            "a new one with GPU.")
    # 2. Copy the code
    if not os.path.isdir(COLAB_CLONE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, COLAB_CLONE], check=True)
''')
md("""
### Load the project and prepare the run

**What it does and why**

The cell lets Python load the copied code, then prepares the run folder:

1. **Install the ONNX libraries.** It installs ONNX, the library that saves models in the ONNX file format, and ONNX Runtime, the library that later runs those files on a CPU, if either is missing.
2. **Load the modules.** It loads the project's modules and picks the GPU.
3. **Check the mode.** It stops with a message if `MODE` is not smoke, medium or full, or if `RUN_ID` does not start with the mode.
4. **Claim the run folder.** It creates the run folder and records the mode in it. If the folder already holds a run of another mode, it stops with a message asking for a new `RUN_ID`, because the trainings reuse the plans they find in their folder.
5. **Make the folders and helpers.** It creates `figures/` and `models/` in the run folder and defines the helpers that later cells use.

- `check()` prints PASS or FAIL together with the number it measured, and it keeps a list for the final summary.
- `savefig()` saves each figure into the run folder and shows it.

**What goes in**

- The cell reads the settings from the parameters cell and the code that the cell above copied.

**What comes out**

- The first line names the mode, the device, the GPU, the number of CPUs and the library versions.
- The next lines give the run folder and the data folder.
- The last line gives the code version, the Git commit the run uses, so that every result can be traced to the exact code.

**How to tell it worked**

- The first line says `device=cuda` and names the GPU, for example `gpu=NVIDIA L4`.
""")
code("""
import os, sys, json, time, subprocess, shutil, hashlib, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

REPO = Path(os.environ.get("FACEKIT_REPO", "/content/face-detection-classification"))
if not REPO.exists():   # on this laptop the notebook runs from the code folder or its notebooks/ folder
    REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(REPO)
sys.path.insert(0, str(REPO))
# 1. Install the ONNX libraries
try:
    import onnx, onnxruntime  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)

# 2. Load the modules
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
# 3. Check the mode
if MODE not in ("smoke", "medium", "full"):
    raise ValueError(f'MODE must be "smoke", "medium" or "full", not {MODE!r}: change it in the parameters cell')
paths.check_run_name_for_mode(MODE, RUN_ID)   # smoke-NNN, medium-NNN, or run-NNN for a full run
SMOKE = MODE == "smoke"
CPUS = os.cpu_count() or 2
# 4. Claim the run folder
RUN = paths.run_dir(EXP, RUN_ID)
RUN.mkdir(parents=True, exist_ok=True)
runlib.claim_run_mode(RUN, MODE)   # stops when the folder holds a run of another mode, whose training plans would be reused
# 5. Make the folders and helpers
FIG = RUN / "figures"; FIG.mkdir(exist_ok=True)
MODELS = RUN / "models"; MODELS.mkdir(exist_ok=True)
DATA = paths.work_data_dir(); RAW = DATA / "raw"; ARR = DATA / "arrays" / MODE   # the three modes never share arrays
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

md("""
### Choose the amount of data for this mode

**What it does and why**

- All three modes run exactly the same cells, and only the amount of data and the length of training differ.
- The cell picks the mode's entry from the table below. In medium mode, its own budgets of 6, 8 and 3 minutes replace the three budget parameters.
- It also decides how many CPU processes load data for each training. The detector gets most of them, because it decodes JPEG photos as it trains, and the identity model gets two or three, because it reads images that are already decoded.

**What goes in**

- The cell reads `MODE`, the three budget parameters and the number of CPUs that the setup cell found.
- Each mode uses this much data:

| | Smoke | Medium | Full |
|---|---|---|---|
| Identity training (DigiFace-1M part 0-1999, and 2000-3999 in full) | 240 people, 24 photos each | 1,000 people, 72 photos each | 3,600 people, 72 photos each |
| Identity validation, which sets the thresholds | 60 people | 200 people | 400 people |
| Identity test (part 8000-9999) | 100 people | 200 people | 500 people |
| Identity network | IResNet-18 | IResNet-18 | the deepest that fits the budget (IResNet-50 for the current models) |
| Detector photos (WIDER FACE training, 5% held out) | 600 | 3,000 | all 12,880 |
| Detector test photos (WIDER FACE validation) | 120 | 500 | all 3,226 |
| Expression portraits (EmoNet-Face Binary) | up to 600 from one file: happy, sad and fear only | all five files: all eight expressions | all five files |
| Expression second test (EmoNet-Face HQ) | one file, up to 120 portraits | one file, which holds 78 usable portraits | two files |
| Portraits in which no face is found | centre-cropped | centre-cropped | left out |
| Length of training | 60, 80 and 40 steps (detector, identity, expression) | 6, 8 and 3 minutes | the planned epochs or the budget parameters, whichever ends first |

- DigiFace-1M comes in parts, archives of 2,000 synthetic identities each.
- Smoke and medium mode take their WIDER photos at even steps through each list, because the lists are sorted by the kind of event (section 4).
- In medium mode the HQ test has fewer than 80 portraits, so its score is only a rough guide.

**What comes out**

- The cell prints the configuration, the number of data workers for the detector and for identity, and the three budgets it will use.

**How to tell it worked**

- The printed configuration and budgets match the mode you chose. Check them before the long cells start.
""")
code("""
# The configuration of each mode. Smoke and medium use subsets of the same data as the full run.
CFG = {
    "smoke": dict(digiface_train=["0-1999"], digiface_test="8000-9999", id_max_ids=300, id_val_ids=60, id_max_per_id=24,
                  id_test_ids=100, emonet_files=1, emonet_max=600, hq_files=1, hq_max=120, wider_train_max=600,
                  wider_val_max=120, det_max_steps=60, expr_max_steps=40, id_max_steps=80, id_depth="iresnet18",
                  budget_min=None),
    # Medium: one run on an L4 with meaningful results. Every EmoNet-Face Binary file, so all eight expressions
    # appear; the one HQ file holds 78 usable portraits, so it needs no cap. Training stops at medium's own budgets.
    "medium": dict(digiface_train=["0-1999"], digiface_test="8000-9999", id_max_ids=1200, id_val_ids=200,
                   id_max_per_id=None, id_test_ids=200, emonet_files=5, emonet_max=None, hq_files=1, hq_max=None,
                   wider_train_max=3000, wider_val_max=500, det_max_steps=0, expr_max_steps=0, id_max_steps=0,
                   id_depth="iresnet18", budget_min=(6.0, 8.0, 3.0)),
    "full": dict(digiface_train=["0-1999", "2000-3999"], digiface_test="8000-9999", id_max_ids=None, id_val_ids=400,
                 id_max_per_id=None, id_test_ids=500, emonet_files=5, emonet_max=None, hq_files=2, hq_max=None,
                 wider_train_max=None, wider_val_max=None, det_max_steps=0, expr_max_steps=0, id_max_steps=0,
                 id_depth="auto", budget_min=None),
}[MODE]
if CFG["budget_min"]:                # medium mode's budgets replace the three budget parameters
    DET_BUDGET_MIN, ID_BUDGET_MIN, EXPR_BUDGET_MIN = CFG["budget_min"]
WORKERS_DET = max(2, CPUS - 4)       # the detector's JPEG decoding needs most of the CPU
WORKERS_ID = 2 if CPUS <= 8 else 3   # identity reads pre-decoded arrays, so it needs few workers
print(json.dumps(CFG, indent=1)); print("workers: detector", WORKERS_DET, "identity", WORKERS_ID)
print(f"budgets in minutes: detector {DET_BUDGET_MIN:g}, identity {ID_BUDGET_MIN:g}, expression {EXPR_BUDGET_MIN:g}")
SUMMARY["config"] = CFG
""")

md("""
## 2. Data: download every dataset and record its licence

### Get and unpack the datasets

**What it does and why**

The cell gets every dataset onto the machine and records where each one came from:

1. **Find the Drive folder.** On Colab with `USE_DRIVE = True`, the cell mounts Google Drive, which shows a Google page that you must approve within two minutes. The downloaded files then live in the artifacts folder's `data/raw` in Drive, so a file downloaded once is not downloaded again.
2. **Download and unpack.** It downloads each missing file straight into that folder, up to six at a time, checks each file's size where the project records it, and tries each download up to four times in total. It unpacks the zip archives onto the machine's own disk, in `/content/data/raw`.
3. **Write the manifest.** It writes `data_manifest.json`, which lists every source with its address, its licence and the number of images unpacked from each archive.

- The archives are unpacked onto the machine because unpacking the DigiFace images into Drive would be slow and would sync them to the laptop. Smoke and medium mode unpack 288,000 of them, and a full run 432,000.
- The Face Research Lab London Set is downloaded onto the machine only and never goes to Drive. If Drive is not mounted, as in an unattended run, every file is downloaded onto the machine.
- If you stop the cell, it ends within a few seconds, keeps every finished download and says how many are complete. Running the cell again continues with the rest.

**What goes in**

- The cell reads `USE_DRIVE` and the list of datasets that the mode needs. Smoke mode needs about 8.7 GB of files.
- Medium mode needs the same files and the other four EmoNet-Face Binary files, which come from Hugging Face. The first Binary file is 0.44 GB.

**What comes out**

- The cell prints where the files live and where they are unpacked, then a line for each dataset as soon as it is done, saying whether it was downloaded or already there.
- Every 30 seconds it prints a progress line that names what is still downloading or unpacking, how many gigabytes have arrived and the current speed.
- It writes `data_manifest.json` into the run folder and prints the licence of each source. The parquet and text files are recorded with a count of zero, because they are not unpacked.

**How to tell it worked**

- There is a line for every dataset with a size above zero.
- The first smoke download takes about 13 minutes, because Microsoft's DigiFace server is slow, about 5 MB/s for each file. Reading the files from Drive is several times faster.
- The London Set is downloaded only for one expression check in section 8, and this notebook never uses it to identify anyone.
""")
code("""
LONDON = [sources.LONDON_NEUTRAL, sources.LONDON_SMILING]
WIDER_SPLIT = datasets.Source(name="wider_split", url=f"{sources.HF}/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip",
                              filename="wider_face_split.zip", licence=sources.WIDER_LICENCE, licence_url=sources.WIDER_LICENCE_URL)
WANTED = ([sources.WIDER_TRAIN, sources.RETINAFACE_LABELS, sources.WIDER_VAL, WIDER_SPLIT]
          + sources.EMONET_BINARY[:CFG["emonet_files"]] + sources.EMONET_HQ[:CFG["hq_files"]]
          + [sources.digiface_part(p) for p in CFG["digiface_train"] + [CFG["digiface_test"]]] + LONDON)

# 1. Find the Drive folder
# Where the downloaded files live. On Colab with USE_DRIVE, that is the artifacts folder's data/raw in Google Drive,
# so a file downloaded once is there for every later machine. Archives are unpacked onto this machine's own disk
# (RAW), and the London photos never leave this machine.
ARCHIVES = RAW
if os.path.isdir("/content") and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")            # approve the Google page within two minutes
        found = paths.drive_artifacts_dir()
        if found is not None:
            ARCHIVES = found / "data" / "raw"
            ARCHIVES.mkdir(parents=True, exist_ok=True)
    except Exception as error:                   # no approval, or the folder is not in Drive: use this machine
        print("Google Drive is not used, so the files are downloaded onto this machine:", error)
        ARCHIVES = RAW
print("downloaded files:", ARCHIVES)
print("unpacked folders:", RAW)
# 2. Download and unpack
# Six downloads at a time. A line appears as each dataset finishes and a progress line every 30 seconds. The stop
# button ends the cell within seconds: finished downloads are kept, and running the cell again continues.
started = time.time()
fetched = datasets.fetch_all(WANTED, ARCHIVES, extract_root=RAW, local_names={s.name for s in LONDON},
                             workers=6, retries=4, report_every=30)
FILES = {record["name"]: Path(record["path"]) for record in fetched}
print(f"all downloads and extractions: {time.time() - started:.0f} s")
# 3. Write the manifest
manifest =datasets.write_manifest(RUN / "data_manifest.json", WANTED, {s.name: datasets.count_images(RAW / s.name) for s in WANTED})
print(f"{'dataset':22s} licence"); [print(f"{s['name']:22s} {s['licence']}") for s in manifest["sources"]]
SUMMARY["datasets"] = [{"name": s["name"], "licence": s["licence"], "url": s["url"]} for s in manifest["sources"]]
""")

md("""
### Check the third-party copies against the official boxes

**What it does and why**

Two files come from copies that other people uploaded to Hugging Face, and a copy can have been altered or shifted. The cell compares their face boxes with the official WIDER FACE box lists, photo by photo, allowing one pixel of difference:

1. **RetinaFace labels.** One copy holds the landmark annotations that the authors of the RetinaFace detector made for the WIDER training photos.
2. **Validation photos.** The other is a parquet file, a compressed table format, of the WIDER validation photos. The cell checks its first 400 photos.

**What goes in**

- The cell reads the official WIDER FACE box lists, the RetinaFace label file and the validation parquet file from the data cell.

**What comes out**

- The cell prints two PASS or FAIL lines, each with the number of photos that agree.

**How to tell it worked**

- Both lines read PASS, with at least 97% of the photos agreeing.
- In the full run the RetinaFace labels matched in 12,880 of 12,880 photos, and the validation copy matched in 398 of the 400 photos checked.
""")
code("""
# Integrity checks: the third-party mirrors must agree with the official WIDER FACE boxes.
split_dir = next((RAW / "wider_split").rglob("wider_face_train_bbx_gt.txt")).parent
official_train = {Path(r["path"]).name: r for r in parse_wider_official(split_dir / "wider_face_train_bbx_gt.txt")}
official_val = {Path(r["path"]).name: r for r in parse_wider_official(split_dir / "wider_face_val_bbx_gt.txt")}
retina = parse_retinaface_labels(FILES["retinaface_labels"])

def boxes_match(a, b, tol=1.0):
    a = np.asarray(a).reshape(-1, 4); b = np.asarray(b).reshape(-1, 4)
    b = b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]
    if len(a) != len(b): return False
    return bool(np.all(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)) <= tol))

# 1. RetinaFace labels
matched = sum(boxes_match(r["boxes"], official_train[Path(r["path"]).name]["boxes"]) for r in retina if Path(r["path"]).name in official_train)
share = matched / max(1, len(retina))
check("RetinaFace labels agree with official WIDER train boxes", share >= 0.97, f"{matched}/{len(retina)} images ({share:.1%})")
# 2. Validation photos (the first 400 of the parquet file)
n_val_checked = n_val_ok = 0
for rec in prepare.wider_val_records(FILES["wider_val"], max_items=400):
    key = Path(rec["path"]).name
    if key in official_val:
        n_val_checked += 1; n_val_ok += boxes_match(rec["boxes"], official_val[key]["boxes"])
check("WIDER validation parquet agrees with official boxes", n_val_ok >= 0.97 * max(1, n_val_checked), f"{n_val_ok}/{n_val_checked} images")
""")

md("""
## 3. Identity data: DigiFace-1M arrays, split by identity

### Prepare the identity images and split them by person

**What it does and why**

DigiFace-1M provides 72 images for each synthetic identity, and this cell decides which identities each step uses:

1. **Split by person.** The cell picks the identities for training and holds some back for validation, which chooses the thresholds. It splits by identity, so that all the images of one person stay in one group.
2. **Separate test part.** It takes the test identities from a separate DigiFace part, identities 8000 to 9999.
3. **Decode once.** It decodes the PNG images once into arrays on disk, so that training does not decode image files on every pass.
4. **Show samples.** It draws eight images each of three training identities.

**What goes in**

- The cell reads the unpacked DigiFace parts that the mode names, and the mode's numbers of training, validation and test identities.

**What comes out**

- The cell prints how long the arrays took and how many images and identities each group holds.
- It draws a figure with eight images each of three identities. The figure shows what the identity model learns from: the same rendered face in different poses and lighting.

**How to tell it worked**

- Two PASS lines confirm that no identity is shared between the groups and that the test identities come from another part.
""")
code("""
# 1. Split by person
id_paths, id_names = [], []
for part in CFG["digiface_train"]:
    p, n = prepare.digiface_records(RAW / f"digiface_{part}"); id_paths += p; id_names += n
train_ids, val_ids = prepare.split_identities(id_names, CFG["id_val_ids"], seed=0, max_ids=CFG["id_max_ids"])
label_of = {name: i for i, name in enumerate(train_ids + val_ids)}       # training identities first
# 2. Separate test part
test_paths, test_names =prepare.digiface_records(RAW / f"digiface_{CFG['digiface_test']}")
test_ids = sorted(set(test_names))[: CFG["id_test_ids"]]
test_label = {name: i for i, name in enumerate(test_ids)}
# 3. Decode once (an array set already on disk is reused)
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

# 4. Show samples
fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for row, identity in enumerate(np.random.default_rng(0).choice(len(train_ids), 3, replace=False)):
    idx = np.flatnonzero(id_labels == identity)[:8]
    for col, i in enumerate(idx):
        axes[row, col].imshow(id_images[i]); axes[row, col].axis("off")
    axes[row, 0].set_title(f"identity {train_ids[identity]}", fontsize=8, loc="left")
plt.suptitle("DigiFace-1M: three synthetic identities, eight images each"); savefig("digiface_samples")
""")

md("""
### Check that the identity model can learn at all

**What it does and why**

Two quick tests, before the long run, catch most mistakes in how the data, the model and the loss are connected:

1. **First loss near ln(identities).** With the two CosFace settings switched off, an untrained model is equally unsure about every identity, so its first-batch loss should be within 0.5 of the natural log of the number of identities.
2. **Memorise a small batch.** A small model trains for 200 steps on 16 images, and its loss must fall below a tenth of its starting value. A model that cannot memorise 16 images has a bug in its data, its model code or its loss.

- The two CosFace settings are the margin, which lowers the correct person's score, and the scale, which multiplies every score.
- The expected first loss is about 6.9 for the 1,000 training identities of medium mode and 8.2 for the 3,600 of the full run. A very different value means that the labels or the input scaling are wrong.
- The 16 images are the first ones with a label below 4. Because the arrays store each person's images together, they all show one person, which makes the second test weaker than it looks.

**What goes in**

- The cell reads the training images and labels from the cell above.

**What comes out**

- The cell prints two PASS or FAIL lines with the measured numbers.

**How to tell it worked**

- Both lines read PASS.
""")
code("""
# Sanity checks before the identity run: with scale 1 and no margin the first loss is ln(classes); one batch overfits.
torch.manual_seed(0)
# 1. First loss near ln(identities)
k = len(train_ids)
probe = IdentityTrainer(k, depth="iresnet18", scale=1.0, margin=0.0).to(DEVICE)   # the two CosFace settings off
x = torch.from_numpy(np.ascontiguousarray(id_images[:32])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
y = torch.from_numpy(id_labels[:32]).to(DEVICE)
loss0, _ = probe(x, y)
check("identity first-batch loss is ln(classes)", abs(loss0.item() - np.log(k)) < 0.5, f"{loss0.item():.3f} vs {np.log(k):.3f}")
# 2. Memorise a small batch
small = np.flatnonzero(id_labels < 4)[:16]
xs = torch.from_numpy(np.ascontiguousarray(id_images[small])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
ys = torch.from_numpy(id_labels[small]).to(DEVICE)
curve = engine.overfit_one_batch(IdentityTrainer(4, depth="iresnet18"), lambda m: m(xs, ys)[0], steps=200, lr=1e-3, device=DEVICE)
check("identity model overfits one batch", curve[-1] < 0.1 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe; torch.cuda.empty_cache() if torch.cuda.is_available() else None
""")

md("""
### Start the identity training in the background

**What it does and why**

- The identity model trains as a separate program, `train.py`, on the same GPU while the notebook trains the detector. Training the two one after the other would take close to two hours in a full run.
- The program measures its own speed and plans the epochs that fit in `ID_BUDGET_MIN` minutes, which is 8 in medium mode.
- It writes checkpoints so that it can resume after an interruption. An identity training that has already finished in this run folder is not started again.

**What goes in**

- The cell reads the identity arrays, the mode's network depth, the number of data workers and the identity budget.

**What comes out**

- The cell prints the process number and the command it started, and returns at once.
- The program writes its log to `identity_train.log` in the run folder.

**How to tell it worked**

- A line shows the process number and the command. Section 9 waits for the program to finish and then tests the model.
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

md("""
## 4. Detector: train on WIDER FACE

### Split the detector's photos and look at the training targets

**What it does and why**

The detector learns from the WIDER training photos with the RetinaFace annotations, which give each face a box:

1. **Hold back photos.** The cell holds back 5% of the training photos, which measure the detector after every epoch and later choose its score threshold, and it checks that no photo is in both groups.
2. **Show augmented samples.** It shows eight training photos after augmentation, the random changes that help the detector cope with different framings and lighting.

- Smoke and medium mode take their photos at even steps through the list. The list is sorted by the kind of event, such as parades or press conferences: its first 3,000 photos cover 13 of the 61 kinds, and 3,000 taken at even steps cover 57.
- The augmentation is a random square crop of 30% to 100% of the photo's short side, resized to 640 pixels, and a horizontal flip half of the time.
- Half of the photos also get colour changes, and each of the four changes (brightness, contrast, saturation and hue) is applied to half of those.

**What goes in**

- The cell reads the RetinaFace labels, the unpacked WIDER training photos and the mode's number of training photos.

**What comes out**

- The cell prints how many photos are used for training and held out, and how many training faces carry landmarks. Just under half of the faces (47% in the full run) have five landmark points.
- It draws eight augmented photos. Over each it draws the heatmap target, the coloured blobs at the face centres that the detector learns to predict.
- It also draws the five landmark targets: the eyes in red and green, the nose in blue and the mouth corners in yellow and magenta.

**How to tell it worked**

- A PASS line confirms that no photo is in both groups.
- In the figure every face carries a blob.
- The red point stays on the eye on the left of the image even in flipped photos, which shows that the landmark order survives the flip.
""")
code("""
records = prepare.evenly_spaced(retina, CFG["wider_train_max"])   # even steps through the list, which is sorted by event
# 1. Hold back photos
det_train, det_val = split_records(records, val_fraction=0.05, seed=0)
WIDER_IMAGES = next((RAW / "wider_train").rglob("images"))
print(f"WIDER train images: {len(det_train)} for training, {len(det_val)} held out for validation")
check("detector split shares no image", arrays.split_overlap([r["path"] for r in det_train], [r["path"] for r in det_val]) == 0,
      "train / held-out")
n_faces = sum(len(r["boxes"]) for r in det_train)
n_lm = sum(int(np.isfinite(r["landmarks"][:, 0, 0]).sum()) for r in det_train)
print(f"training faces: {n_faces}, with landmarks: {n_lm} ({n_lm / max(1, n_faces):.0%})")

# 2. Show augmented samples
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

md("""
### Check that the detector can learn

**What it does and why**

Two quick checks, before the long run, show that the network, the targets and the loss are connected correctly:

1. **First loss is finite.** The detector's loss has four parts: the heatmap, the box size, the centre offset and the landmarks. It has no simple expected starting value, so the check only requires the loss on the first batch to be a finite number.
2. **Memorise a small batch.** The detector trains on eight photos for 80 steps, and the loss must fall below half of its starting value.

**What goes in**

- The cell reads eight augmented training photos from the cell above, and builds a fresh detector whose backbone starts from ImageNet weights.

**What comes out**

- The cell prints two PASS or FAIL lines. The first also prints each of the four parts of the loss.

**How to tell it worked**

- Both lines read PASS.
""")
code("""
# Sanity checks: the loss is finite on the first batch, and the detector overfits one batch.
torch.manual_seed(0)
# 1. First loss is finite
probe = FaceDetNet("resnet18", pretrained_backbone=True).to(DEVICE)
images, targets = detection_collate([sample_set[i] for i in range(8)])
xb = images.to(DEVICE).float() / 255; tb = {k: v.to(DEVICE) for k, v in targets.items()}
loss0, parts0 = detection_loss(probe(xb), tb)
check("detector first-batch loss is finite", torch.isfinite(loss0).item(), {k: round(v, 3) for k, v in parts0.items()})
# 2. Memorise a small batch
curve = engine.overfit_one_batch(probe, lambda m: detection_loss(m(xb), tb)[0], steps=80, lr=1e-3, device=DEVICE)
check("detector overfits one batch", curve[-1] < 0.5 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe
""")

md("""
### Train the detector

**What it does and why**

The cell plans the training to fit `DET_BUDGET_MIN` minutes, which is 6 in medium mode, then trains:

1. **Plan the epochs.** It times a short series of training steps and one validation pass, and works out how many epochs fit in the budget. It plans 30% more epochs than that, so that the clock is meant to end the run, and saves the plan so that a resumed run keeps the same schedule.
2. **Train.** Training uses the AdamW optimiser. Its learning rate rises from near zero during a short warm-up and then falls along a cosine curve. After each epoch the cell measures the held-out AP and keeps the best checkpoint.

- The schedule follows both the step count and the clock, so the learning rate reaches its low end when either the planned epochs or the time budget runs out. In the full run the 21 planned epochs ended after about 37 of the 40 minutes.
- AP, average precision, summarises the trade-off between finding faces and raising false alarms. An AP of 1.0 means that, when the detections are taken in order of confidence, every face is found before the first false alarm.
- The held-out AP is measured on up to 300 held-out photos, shrunk to at most 640 pixels on the long side. Smoke mode holds out only 30 photos, and medium mode 150.

**What goes in**

- The cell reads the training and held-out photos from the split above, and `DET_BUDGET_MIN`.

**What comes out**

- When the cell plans, it first prints the measured speed, including how many images per second the GPU trains on. Then it prints a log line every 50 steps.
- It saves the checkpoints and `plan.json` in the run folder's `detector/` folder.

**How to tell it worked**

- The held-out AP rises over the epochs.
- One reason this AP is lower than the test AP in section 5 is that these photos are shrunk to 640 pixels.
""")
code("""
DET_RUN = RUN / "detector"; DET_RUN.mkdir(exist_ok=True)
torch.manual_seed(0)
det_model = FaceDetNet("resnet18", pretrained_backbone=True)
det_task = DetectorTask(det_model, det_train, det_val, WIDER_IMAGES, size=640, batch_size=32, num_workers=WORKERS_DET,
                        val_images=60 if SMOKE else 300)
# 1. Plan the epochs
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
# 2. Train
det_cfg = engine.TrainConfig(epochs=det_epochs, lr=5e-4, weight_decay=1e-4, optimizer="adamw", warmup_epochs=min(1.0, det_epochs / 5),
                             min_lr_ratio=0.002, grad_clip=10.0, checkpoint_every=300, log_every=50,
                             max_steps=CFG["det_max_steps"], max_minutes=DET_BUDGET_MIN, time_schedule=True)
det_result = engine.fit(det_task, det_cfg, DET_RUN, device=DEVICE, extra_config={"model": "FaceDetNet resnet18", "size": 640})
SUMMARY["detector_training"] = {"planned_epochs": det_epochs, "epochs_run": len(engine.load_history(DET_RUN)),
                                "status": det_result.status, "best_val_ap": det_result.best_value,
                                "best_epoch": det_result.best_epoch,
                                "annealed": json.loads((DET_RUN / "metrics.json").read_text()).get("annealed")}
""")

md("""
### Plot the detector's training curves

**What it does and why**

- The cell draws three curves over the epochs, one point per epoch: the loss of the last training batch of each epoch, the held-out AP, and the learning rate at the end of each epoch.
- The `plot_history` helper defined here draws the same kind of figure for the other two models later.

**What goes in**

- The cell reads the training history that the detector training saved in its run folder.

**What comes out**

- The cell draws the three curves and prints the number of epochs and the best held-out AP.

**How to tell it worked**

- The loss falls, the AP rises and then levels off, and the learning rate falls smoothly towards zero.
- The plot shows no warm-up rise, because the warm-up ends within the first epoch.
- A held-out AP that falls while the training loss keeps falling would mean that the detector is memorising its training photos.
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

md("""
## 5. Detector: test on WIDER FACE validation (easy, medium, hard)

### Test the detector

**What it does and why**

The cell chooses the score threshold, measures the landmark error, and then scores the detector on photos it has never seen:

1. **Score threshold.** It loads the best checkpoint and picks the score threshold, the confidence above which a detection counts as a face, with the best F1 score on the held-out training photos. The test photos play no part in the choice.
2. **Landmark error.** It measures the landmark error, printed as NME, on held-out faces: the average distance from the true points divided by the size of the face box. A value of 0.05 means that the points miss by 5% of the face size on average.
3. **Test.** It runs the detector on the WIDER FACE validation photos, shrinking any photo whose longer side exceeds 1,280 pixels, and scores it with the official easy, medium and hard face lists.
4. **Alignment check.** It measures how far, in pixels of a 112-pixel crop, our detector puts DigiFace's landmarks from the template positions that the alignment step expects.

- Easy faces are large and clear, and hard faces are small, blurred or partly hidden. The medium face list is WIDER FACE's own middle list and has nothing to do with medium mode.
- With `MODE` set to smoke or medium, the test photos are taken at even steps through the file, so that every kind of event appears.
- A small alignment distance means that the faces our pipeline aligns look like the identity model's training crops. Section 9 tests whether the distance matters for identity.

**What goes in**

- The cell reads the best detector checkpoint, the held-out training photos, the WIDER FACE validation file and the DigiFace test images.

**What comes out**

- The cell prints the threshold, the landmark error, the AP on each face list next to the published figures of YuNet, a small open detector, and the alignment distance.
- A figure shows six test photos with our detections in green and the true boxes dashed in white.

**How to tell it worked**

- A PASS line confirms that the official face lists count their faces from zero, as the scoring code assumes. Lists that counted from one would attach the easy, medium and hard labels to the wrong faces.
- The AP on each face list compares with the targets and the full run like this:

| | Easy faces | Medium faces | Hard faces |
|---|---|---|---|
| Target | 0.85 | 0.80 | 0.55 |
| Full run | 0.857 | 0.845 | 0.753 |
| YuNet (published figures, which the cell prints) | 0.834 | 0.824 | 0.708 |

- Our scoring code differs from the official WIDER FACE scoring tool, so the comparison with YuNet is only approximate.
""")
code("""
# 1. Score threshold for the photo pipeline: chosen on the held-out TRAINING images, never on the test set.
det_model = engine.load_best(FaceDetNet("resnet18", pretrained_backbone=False), DET_RUN, DEVICE)
EVAL_SIDE = 1280   # the same resolution for the threshold (validation), the test and the photo pipeline
val_pred, val_true = predict_records(det_model, det_val, WIDER_IMAGES, DEVICE, max_side=EVAL_SIDE)
thr = choose_score_threshold(val_pred, val_true)
print("chosen detection threshold on held-out training images:", thr)
# 2. Landmark error on held-out training images that carry landmarks.
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

# 3. Test: WIDER FACE validation, VOC-style AP with the official easy/medium/hard face lists.
preds, truths, examples, demo_images = [], [], [], []
t0 = time.time()
keep_in_range, keep_has_zero = True, False
for n, rec in enumerate(prepare.wider_val_records(FILES["wider_val"], max_items=CFG["wider_val_max"], spread=True)):
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

# 4. Alignment check: does our detector put DigiFace's landmarks where the alignment template expects them?
errs = []
for i in np.random.default_rng(0).choice(len(test_labels), min(500, len(test_labels)), replace=False):
    img = cv2.resize(np.asarray(test_images[i]), (224, 224), interpolation=cv2.INTER_CUBIC)
    faces = detect_image(det_model, img, device=DEVICE, max_side=224, score_threshold=0.3)
    if faces: errs.append(np.linalg.norm(np.array(faces[0]["landmarks"]) / 2 - align.ARCFACE_TEMPLATE_112, axis=1).mean())
print(f"DigiFace: face found in {len(errs)} of {min(500, len(test_labels))} images; mean landmark distance from the template "
      f"{np.mean(errs) if errs else float('nan'):.2f} px of 112")
SUMMARY["digiface_alignment_px"] = float(np.mean(errs)) if errs else None
""")

md("""
## 6. Expression data: EmoNet-Face portraits aligned with our detector

### Cut out and align the expression portraits

**What it does and why**

The expression model must see faces the way the finished pipeline will show them, so each portrait goes through our own detector:

1. **Align the Binary portraits.** The cell finds the face in each EmoNet-Face Binary portrait, uses its five landmarks to rotate and scale it to a standard position, and keeps a crop of 224 by 224 pixels.
2. **Align the HQ portraits.** It does the same for the HQ portraits, which experts rated in more detail and which are kept apart as a second test. It removes every HQ image or prompt that also appears in the Binary set.
3. **Split by prompt.** It splits the Binary portraits 70%, 15% and 15% into training, validation and test, grouped by the text prompt that generated them. Portraits from one prompt look alike and would otherwise leak between the groups.
4. **Show samples.** It draws eight aligned crops for each class.

- In full mode, portraits in which the detector finds no face are dropped. Even the full run's detector found no face in 2,196 of the 4,467 Binary portraits.
- In smoke and medium mode those portraits are centre-cropped instead, because the briefly trained detector finds few faces.

**What goes in**

- The cell reads the EmoNet-Face files that the mode names, and uses the detector trained above.

**What comes out**

- The cell prints how many portraits were kept, how many had no face, how many were centre-cropped and how many HQ portraits were removed.
- It prints the class counts for training, validation and test, and draws a figure with eight aligned crops for each class.

**How to tell it worked**

- Two PASS lines confirm that the HQ test shares no image with the Binary set and that no prompt is in two groups.
- In medium and full mode the class counts show all eight expressions. Some classes, such as anger, have very few images, and smoke mode has only happy, sad and fear.
""")
code("""
# 1. Align the Binary portraits with our detector, a little below its own threshold so fewer faces are missed
detector = FaceDetector.from_torch(det_model, device=str(DEVICE), score_threshold=max(0.3, thr["threshold"] - 0.1), max_side=640)
detect_fn = detector.as_detect_fn()
binary_paths = [FILES[s.name] for s in sources.EMONET_BINARY[: CFG["emonet_files"]]]
hq_paths = [FILES[s.name] for s in sources.EMONET_HQ[: CFG["hq_files"]]]
t0 = time.time()
ARR_RUN = ARR / RUN_ID   # these crops depend on this run's detector, so they are never shared between runs
if not arrays.array_exists(ARR_RUN, "emonet_binary"):
    sel = prepare.emonet_selection(binary_paths, max_items=CFG["emonet_max"])
    print("usable EmoNet-Face Binary images:", len(sel))
    prepare.build_emonet_array(binary_paths, sel, detect_fn, ARR_RUN, "emonet_binary", 224, fallback_center_crop=MODE != "full")
ex_images, ex_labels, ex_desc = arrays.read_array(ARR_RUN, "emonet_binary")
# 2. Align the HQ portraits, leaving out every image or prompt that is also in the Binary set
if not arrays.array_exists(ARR_RUN, "emonet_hq"):
    sel_hq = prepare.emonet_selection(hq_paths, hq=True, max_items=CFG["hq_max"])
    prepare.build_emonet_array(hq_paths, sel_hq, detect_fn, ARR_RUN, "emonet_hq", 224, exclude_hashes=set(ex_desc["ids"]),
                               fallback_center_crop=MODE != "full", exclude_prompts=set(ex_desc["groups"]))
hq_images, hq_labels, hq_desc = arrays.read_array(ARR_RUN, "emonet_hq")
print(f"aligned in {time.time() - t0:.0f} s. Binary: {len(ex_labels)} kept, {ex_desc['no_face']} without a face, "
      f"{ex_desc['fallback']} centre-cropped (not in full mode). HQ: {len(hq_labels)} kept, {hq_desc['duplicates']} duplicates of Binary "
      f"and {hq_desc.get('shared_prompt', 0)} sharing a Binary prompt removed")
check("HQ test shares no image with Binary", arrays.split_overlap(ex_desc["ids"], hq_desc["ids"]) == 0, "by SHA-1 of the image bytes")
# 3. Split by prompt
tr_idx, va_idx, te_idx = arrays.grouped_split(ex_labels, ex_desc["groups"], (0.7, 0.15, 0.15), seed=0)
groups = np.asarray(ex_desc["groups"])
check("expression splits share no prompt", arrays.split_overlap(groups[tr_idx], groups[va_idx], groups[te_idx]) == 0,
      f"train {len(tr_idx)}, validation {len(va_idx)}, test {len(te_idx)}")
counts = {c: [int((ex_labels[idx] == i).sum()) for idx in (tr_idx, va_idx, te_idx)] for i, c in enumerate(CLASSES)}
print("class counts (train, validation, test):", counts)
SUMMARY["expression_data"] = {"kept": int(len(ex_labels)), "no_face": ex_desc["no_face"], "counts": counts,
                              "hq_kept": int(len(hq_labels)), "hq_duplicates_removed": hq_desc["duplicates"]}

# 4. Show samples
fig, axes = plt.subplots(len(CLASSES), 8, figsize=(12, 1.6 * len(CLASSES)))
for r, c in enumerate(CLASSES):
    idx = tr_idx[ex_labels[tr_idx] == r][:8]
    for col in range(8):
        axes[r, col].axis("off")
        if col < len(idx): axes[r, col].imshow(ex_images[idx[col]])
    axes[r, 0].set_title(c, fontsize=9, loc="left")
plt.suptitle("Aligned EmoNet-Face training crops by class"); savefig("expression_samples")
""")

md("""
## 7. Expression: train

### Check and train the expression model

**What it does and why**

Two quick checks come first, and then the training:

1. **First loss near ln 8.** The classifier starts from ImageNet weights, but its final layer is new and untrained, so its first-batch loss over eight classes should be within 0.7 of the natural log of 8, which is 2.08.
2. **Memorise one batch.** The model must memorise one batch of 32 portraits within 60 steps, with its loss falling below a tenth of its starting value.
3. **Plan the epochs.** The cell measures the training speed and plans the epochs that fit in `EXPR_BUDGET_MIN` minutes, which is 3 in medium mode.
4. **Train.** It trains ConvNeXt-Tiny with the AdamW optimiser. After each epoch it measures the validation loss, accuracy and macro-F1, and it keeps the checkpoint with the best macro-F1.

- The F1 score of a class combines how many of that class's faces the model finds and how many of its answers for that class are right.
- Macro-F1 averages the F1 score of each class, so a rare class such as anger counts as much as a common one.

**What goes in**

- The cell reads the aligned Binary portraits and their training and validation groups from section 6, and `EXPR_BUDGET_MIN`.

**What comes out**

- The cell prints two PASS or FAIL lines. When it plans, it also prints the measured speed and the planned epochs, and then the training log.
- It draws the three validation curves: loss, accuracy and macro-F1.

**How to tell it worked**

- Both checks read PASS, and the validation macro-F1 rises over the epochs.
""")
code("""
# 1. First loss near ln 8
torch.manual_seed(0)
probe = ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True).to(DEVICE)
xb = torch.from_numpy(np.ascontiguousarray(ex_images[tr_idx[:32]])).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
yb = torch.from_numpy(ex_labels[tr_idx[:32]]).to(DEVICE)
with torch.no_grad(): l0 = torch.nn.functional.cross_entropy(probe.eval()(xb), yb).item()
check("expression first-batch loss is near ln(8)", abs(l0 - np.log(len(CLASSES))) < 0.7, f"{l0:.3f} vs {np.log(len(CLASSES)):.3f}")
# 2. Memorise one batch
curve = engine.overfit_one_batch(probe, lambda m: torch.nn.functional.cross_entropy(m(xb), yb), steps=60, lr=3e-4, device=DEVICE)
check("expression model overfits one batch", curve[-1] < 0.1 * curve[0], f"loss {curve[0]:.2f} -> {curve[-1]:.3f}")
del probe

EXPR_RUN = RUN / "expression"; EXPR_RUN.mkdir(exist_ok=True)
torch.manual_seed(0)
expr_model = ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True)
expr_task = ExpressionTask(expr_model, ex_images, ex_labels, tr_idx, va_idx, len(CLASSES), batch_size=64, num_workers=4)
# 3. Plan the epochs (a resumed run keeps its plan)
if (EXPR_RUN / "plan.json").exists():
    expr_epochs = json.loads((EXPR_RUN / "plan.json").read_text())["epochs"]
else:
    probe_task = ExpressionTask(ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=True), ex_images, ex_labels, tr_idx, va_idx,
                                len(CLASSES), batch_size=64, num_workers=4)
    sec = engine.measure_step_seconds(probe_task, DEVICE, steps=10 if SMOKE else 30, warmup=3 if SMOKE else 5)
    expr_epochs = engine.plan_epochs(sec, expr_task.steps_per_epoch, EXPR_BUDGET_MIN, 20, min_epochs=1, max_epochs=30)
    print(f"expression: {sec:.3f} s per step of 64 -> {expr_epochs} epochs in {EXPR_BUDGET_MIN:.0f} minutes"); del probe_task
    (EXPR_RUN / "plan.json").write_text(json.dumps({"seconds_per_step": sec, "epochs": expr_epochs, "budget_minutes": EXPR_BUDGET_MIN}))
# 4. Train; the time-aware schedule may run up to 1.3 times the budget
expr_cfg = engine.TrainConfig(epochs=expr_epochs, lr=2e-4, weight_decay=0.05, optimizer="adamw", warmup_epochs=min(1.0, expr_epochs / 5),
                              grad_clip=5.0, checkpoint_every=200, log_every=50, max_steps=CFG["expr_max_steps"],
                              max_minutes=EXPR_BUDGET_MIN * 1.3, time_schedule=True)
expr_result = engine.fit(expr_task, expr_cfg, EXPR_RUN, device=DEVICE, extra_config={"model": "ExpressionNet convnext_tiny", "size": 224})
plot_history(engine.load_history(EXPR_RUN), ["val_loss", "val_accuracy", "val_macro_f1"], "Expression training", "expression_curves")
SUMMARY["expression_training"] = {"epochs": expr_epochs, "status": expr_result.status, "best_val_macro_f1": expr_result.best_value,
                                  "best_epoch": expr_result.best_epoch}
""")

md("""
## 8. Expression: test

### Test the expression model on synthetic and on real faces

**What it does and why**

The cell makes the model's probabilities honest, scores the test portraits, and then runs two harder tests:

1. **Temperature.** The cell loads the best checkpoint and fits a single number, the temperature, on the validation portraits. Dividing the model's scores by it makes the probabilities match how often the model is right, so that about 80% of the faces labelled "happy 0.80" really are happy.
2. **Test portraits.** It scores the test portraits and draws a confusion matrix, which shows which expressions are mistaken for which.
3. **HQ set.** It scores the HQ portraits, which have expert ratings and share no image or prompt with the Binary set that the model trained on.
4. **London set.** It runs the detector and the model on real photos of 102 people, one neutral and one smiling each. It only counts how often a smiling photo is called happy and a neutral one neutral, and it saves no image of them.
5. **Examples.** It draws 24 test portraits, with the prediction in green when it is right and in red when it is wrong.

**What goes in**

- The cell reads the best expression checkpoint, the validation and test portraits, the HQ portraits and the London photos.

**What comes out**

- The cell prints accuracy and macro-F1 with 95% confidence intervals, and the F1 of each class.
- It prints the calibration error (ECE), the average gap between the stated probability and the share of answers that are right, before and after the temperature is applied.
- It prints the HQ and London results, and draws the confusion matrix and the 24 examples.

**How to tell it worked**

- The calibration error is lower after the temperature is applied.
- A model that has learned expressions in general scores close to its main test result on the HQ set and calls most of the London smiling photos happy.
- The target is a test macro-F1 of 0.70. The full run reached 0.66, and only 15% of its real smiling photos were called happy, so the model does not yet transfer to real faces.
""")
code("""
# 1. Temperature scaling on validation: one number that makes the probabilities honest (lower calibration error).
expr_model = engine.load_best(ExpressionNet(len(CLASSES), "convnext_tiny", pretrained=False), EXPR_RUN, DEVICE)
val_logits, val_y = predict_logits(expr_model, arrays.ArrayDataset(ex_images, ex_labels, va_idx), DEVICE)
t = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
vl, vy = torch.from_numpy(val_logits), torch.from_numpy(val_y)
def closure():
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(vl / t.exp(), vy); loss.backward(); return loss
opt.step(closure); TEMPERATURE = float(t.exp())
# 2. Test portraits
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

# 3. HQ set
hq_logits, hq_y = predict_logits(expr_model, arrays.ArrayDataset(hq_images, hq_labels), DEVICE)
hq_report = metrics.classification_report(hq_y, hq_logits.argmax(1), len(CLASSES), n_boot=300) if len(hq_y) else {}
print(f"HQ (expert-rated, separate images) agreement: accuracy {hq_report.get('accuracy', float('nan')):.4f}, macro-F1 {hq_report.get('macro_f1', float('nan')):.4f}, n={len(hq_y)}")

# 4. London set: real faces, the neutral and smiling front photos (expression only, never identification).
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

# 5. Examples
pick = np.random.default_rng(0).choice(len(te_idx), min(24, len(te_idx)), replace=False)
fig, axes = plt.subplots(3, 8, figsize=(14, 6.2))
probs_all = softmax(test_logits, TEMPERATURE)
for ax, i in zip(axes.flat, pick):
    ax.imshow(ex_images[te_idx[i]]); ax.axis("off")
    pred, true = probs_all[i].argmax(), test_y[i]
    ax.set_title(f"{CLASSES[pred]} {probs_all[i].max():.2f}\\n(true {CLASSES[true]})", fontsize=8, color="green" if pred == true else "red")
plt.suptitle("Expression predictions on test portraits (green right, red wrong)"); savefig("expression_test_examples")
""")

md("""
## 9. Identity: wait for the training process, then test on unseen identities

### Wait for the identity training

**What it does and why**

The identity program started in section 3 has been training while the detector and the expression model trained:

1. **Wait.** The cell waits for the program to finish and prints the last two lines of its log every five minutes.
2. **Report.** It prints the end of the log, the plan and the result, and stops with an error if the program ended without results.
3. **Plot.** It draws the training loss and two validation measures over the epochs.

- The plan gives the network depth and the number of epochs that fit the budget.
- The result gives the best validation true accept rate at one false accept in a thousand, the epoch that reached it, and how the run ended.

**What goes in**

- The cell reads the running identity program, its log and its run folder.

**What comes out**

- The cell prints the log lines, the plan and the result. If the program ended without results, the end of its log appears above the error.
- It draws the training loss, the area under the ROC curve, and the true accept rate at one false accept in a thousand.
- The ROC curve plots true matches accepted against false matches accepted at every threshold. Its area reaches 1.0 when every same-person pair scores above every different-person pair.

**How to tell it worked**

- The loss falls while both validation measures rise.
""")
code("""
# 1. Wait
waited = 0
while id_proc is not None and id_proc.poll() is None:
    time.sleep(30); waited += 30
    if waited % 300 == 0:
        print(f"[{waited // 60} min] identity still training; last lines:")
        print("".join(ID_LOG.read_text().splitlines(True)[-2:]))
# 2. Report
print("identity process finished" +(f" with exit code {id_proc.returncode}" if id_proc else ""))
print("".join(ID_LOG.read_text().splitlines(True)[-8:]))
if not (ID_RUN / "metrics.json").exists():
    raise RuntimeError("identity training did not finish; the end of its log is printed above (" + str(ID_LOG) + ")")
id_plan = json.loads((ID_RUN / "plan.json").read_text()); id_metrics = json.loads((ID_RUN / "metrics.json").read_text())
print("plan:", id_plan); print("result:", id_metrics)
# 3. Plot
plot_history(engine.load_history(ID_RUN),["train_loss_last", "val_auc", "val_tar@far=0.001"], "Identity training", "identity_curves")
SUMMARY["identity_training"] = {"plan": id_plan, "result": id_metrics, "epochs_run": len(engine.load_history(ID_RUN))}
""")

md("""
### Choose the thresholds and test the identity model

**What it does and why**

The model gives two photos a cosine similarity, a number between -1 and 1 that measures how closely their embeddings point in the same direction. A threshold turns that number into a decision:

1. **One-to-one threshold.** On the validation identities only, the cell sets the threshold that decides whether two photos show the same person, so that one in a thousand pairs of photos of different people is wrongly accepted.
2. **One-to-many threshold.** Also on the validation identities, it sets the threshold that decides whether a face gets a name from the gallery of enrolled people, so that 1% of strangers are given a name.
3. **Test.** Only then does it score the test identities, once: 200 in medium mode and 500 in the full run.
4. **Through our pipeline.** It runs a random 3,000 of the test images through our own detector and alignment, as the photo pipeline sees them, and measures the area under the ROC curve and the true accept rate at one false accept in a thousand again.
5. **Figures.** It draws the ROC curve with the similarity histograms, and four probe faces with their closest gallery matches.

**What goes in**

- The cell reads the best identity checkpoint, the validation and test images, and the detector.

**What comes out**

- The cell prints both thresholds, then the area under the ROC curve and the equal error rate, the point where false accepts and false rejects are equally common.
- It prints the true accept rate at one false accept in a thousand and in ten thousand, and rank-1, which is how often the right person is the closest match.
- The first figure shows the ROC curve and the similarity histograms of same-person and different-person pairs.
- The second figure shows four probe faces, the photos to be identified, each with its five closest gallery matches.

**How to tell it worked**

- The two histograms barely overlap, and a green first match means that the right person was found.
- The target is a true accept rate of 0.80 at one false accept in a thousand. The full run reached these rates:

| Faces tested | Full run |
|---|---|
| All test identities, DigiFace's own crops | 0.92 |
| The 2,977 of 3,000 random test images in which our detector found a face, DigiFace's own crops | 0.94 |
| The same 2,977 images through our detector and alignment | 0.79 |
""")
code("""
# Load the embedder alone, for every step below: the CosFace head in the checkpoint is used only in training.
ID_DEPTH = id_plan.get("depth", CFG["id_depth"])
embedder = IResNet(ID_DEPTH)
state = torch.load(ID_RUN / ("best.pt" if (ID_RUN / "best.pt").exists() else "final.pt"), map_location="cpu", weights_only=True)["model"]
embedder.load_state_dict({k[len("embedder."):]: v for k, v in state.items() if k.startswith("embedder.")})
embedder = embedder.to(DEVICE).eval()
# 1. One-to-one threshold, on the validation identities only
num_train = int(id_desc["num_train_ids"])
val_idx = np.flatnonzero(id_labels >= num_train)
val_emb, val_lab = embed(embedder, arrays.ArrayDataset(id_images, id_labels, val_idx), DEVICE)
vp, vs = metrics.verification_pairs(val_lab, 3000, 30000, seed=1)
val_rep = metrics.verification_report(metrics.cosine_scores(val_emb, vp), vs, fars=(1e-3,))
ID_THRESHOLD = val_rep["threshold@far=0.001"]
print(f"1:1 identity threshold chosen on validation identities at FAR 1e-3: {ID_THRESHOLD:.4f}")

# 2. One-to-many threshold, also on the validation identities
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

# 3. Test: the test part is scored once, after every choice above.
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

# 4. Through our pipeline: the same test identities through OUR detector and alignment, as the photo pipeline sees faces.
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

# 5. Figures
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

md("""
## 10. Export to ONNX, check parity, and write the model card

### Save the models for the laptop

**What it does and why**

The three trained networks are exported to ONNX, a model file format that ONNX Runtime runs on a CPU or a GPU without PyTorch:

1. **Export and parity.** The cell exports each network, feeds the same random input to the PyTorch model and to its ONNX file, and checks that the outputs agree within one part in a thousand.
2. **Checkpoints.** It copies the PyTorch checkpoints next to the ONNX files.
3. **Model card.** It writes `model_card.json`, which records each model's architecture, input size, thresholds, test results, training data, licences and known limits, so that the pipeline reads its settings from one place.
4. **Speed.** It times the pipeline on the first test photo by running the ONNX files on the Colab machine's CPU, the way a laptop runs them.

**What goes in**

- The cell reads the three trained models and the thresholds and test results of sections 5 to 9. The licence chain that it writes into the card is part of the cell's own code.

**What comes out**

- The run folder's `models/` holds the three ONNX files, the three checkpoints and `model_card.json`.
- The cell prints three parity lines, then the CPU time of the pipeline and the number of faces in that photo, a rough guide to the time on a laptop.
- In the full run the photo had no face, so its 0.23 seconds covered the detector alone.

**How to tell it worked**

- All three parity lines read PASS. In the full run the outputs agreed to about one part in a million.
- Three of the datasets allow non-commercial research only, and the card says so.
""")
code("""
# 1. Export and parity
export.export_detector(det_model.cpu(), MODELS / "detector.onnx")
export.export_classifier(expr_model.cpu(), MODELS / "expression.onnx", 224, "logits")
export.export_classifier(embedder.cpu(), MODELS / "embedder.onnx", 112, "embedding")
par = {"detector": export.parity(det_model, MODELS / "detector.onnx", torch.rand(1, 3, 480, 640)),
       "expression": export.parity(expr_model, MODELS / "expression.onnx", torch.rand(4, 3, 224, 224)),
       "embedder": export.parity(embedder, MODELS / "embedder.onnx", torch.rand(4, 3, 112, 112))}
for k, v in par.items(): check(f"ONNX parity: {k}", v < 1e-3, f"max relative difference {v:.2e}")
# 2. Checkpoints
for name, run in (("detector", DET_RUN), ("expression", EXPR_RUN), ("identity", ID_RUN)):
    shutil.copy2(run / ("best.pt" if (run / "best.pt").exists() else "final.pt"), MODELS / f"{name}.pt")

# 3. Model card
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

# 4. Speed: CPU latency of the pipeline from the ONNX files on the first test photo, as the laptop will run it.
analyzer = FaceAnalyzer.from_model_dir(MODELS)
sample = examples[0][0]
t0 = time.time(); faces = analyzer.analyze(sample); latency = time.time() - t0
t0 = time.time(); [analyzer.analyze(sample) for _ in range(3)]; latency = (time.time() - t0) / 3
print(f"CPU pipeline latency on a {sample.shape[1]}x{sample.shape[0]} photo with {len(faces)} faces: {latency:.2f} s")
card["cpu_latency_seconds"] = latency; (MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
SUMMARY["onnx_parity"] = par; SUMMARY["cpu_latency_seconds"] = latency
""")

md("""
## 11. The whole pipeline on photos

### Try the gallery, then the finished pipeline

**What it does and why**

The cell tests the gallery on DigiFace faces, then runs the whole pipeline on real photos:

1. **Gallery.** It enrols 12 unseen DigiFace identities with three photos each, in the same kind of gallery that `facekit enroll` builds, and tries to name five more photos of each and five photos each of 12 strangers.
2. **Whole pipeline.** It loads the three ONNX files into the same `FaceAnalyzer` that `facekit scan` uses and runs it on six WIDER validation photos, from the photo onwards.

- The gallery part uses the embeddings that section 9 computed from DigiFace's own crops, so it skips our detector.

**What goes in**

- The cell reads the test embeddings and the one-to-many threshold from section 9, the ONNX files from section 10 and six WIDER validation photos from section 5.

**What comes out**

- The cell prints how many enrolled people were found or named wrongly, and how many strangers were rejected or wrongly named.
- A figure shows each detected face boxed in green, with its expression and that expression's probability.

**How to tell it worked**

- Most enrolled people are found with few wrong names, and most strangers stay unnamed.
""")
code("""
# 1. Gallery: enrol 12 unseen DigiFace identities with three photos each, then identify other photos of them and of 12 strangers.
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

# 2. Whole pipeline
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

md("""
## 12. Summary against the targets

### Compare the results with the targets and save the summary

**What it does and why**

1. **Targets.** The cell prints each model's result next to the target that was set before training.
2. **Checks.** It lists every check of the run with PASS or FAIL.
3. **Summary.** It writes everything into `summary.json` in the run folder and lists the saved model files with their sizes.

**What goes in**

- The cell reads the results and the checks that every cell above collected.

**What comes out**

- The cell prints the five targets with their results, the list of checks and the list of saved model files, and it writes `summary.json`.

**How to tell it worked**

- The line ALL CHECKS PASSED appears just before the list of saved model files.
- The targets are these:

| Model | Measure | Target |
|---|---|---|
| Detector | AP on WIDER FACE's easy, medium and hard face lists | 0.85, 0.80 and 0.55 |
| Expression | test macro-F1 | 0.70 |
| Identity | true accept rate at one false accept in a thousand | 0.80 |

- In smoke mode the targets are not met, because the models train for only a few dozen steps, and only the checks matter.
- Medium mode trains on less data for a few minutes, so its results show whether the pipeline learns rather than what a full run reaches.
""")
code("""
# 1. Targets
targets = [("detector AP easy >= 0.85", SUMMARY["detector_test"]["ap"]["easy"], 0.85),
           ("detector AP medium >= 0.80", SUMMARY["detector_test"]["ap"]["medium"], 0.80),
           ("detector AP hard >= 0.55", SUMMARY["detector_test"]["ap"]["hard"], 0.55),
           ("expression macro-F1 >= 0.70", SUMMARY["expression_test"]["macro_f1"], 0.70),
           ("identity TAR@FAR1e-3 >= 0.80", SUMMARY["identity_test"]["tar@far=0.001"], 0.80)]
print(f"{'target':32s} {'result':>8s}  met")
for name, value, goal in targets:
    print(f"{name:32s} {value:8.4f}  {'yes' if value >= goal else 'no'}")
# 2. Checks
print(); print("checks:"); [print(("PASS" if c["ok"] else "FAIL"), c["check"], "-", c["detail"]) for c in CHECKS]
# 3. Summary
SUMMARY["targets"] = [{"target": n, "value": v, "met": v >= g} for n, v, g in targets]
SUMMARY["checks"] = CHECKS; SUMMARY["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(RUN / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
print(); print("ALL CHECKS PASSED" if all(c["ok"] for c in CHECKS) else "SOME CHECKS FAILED: read the lines above")
print("saved:"); [print(f"  {p.relative_to(RUN)}  {p.stat().st_size / 1e6:.1f} MB") for p in sorted(MODELS.iterdir())]
""")


def build(path: Path) -> None:
    """Writes the notebook made of all the collected cells to the file at path, creating its folder when needed."""
    nb = nbf.v4.new_notebook()

    # The notebook asks for a Python 3 kernel and, on Colab, for a GPU.
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"

    # Turn each collected cell into a notebook cell, and attach its images or tags.
    for cell in CELLS:
        if cell[0] == "markdown":
            markdown = nbf.v4.new_markdown_cell(cell[1])
            if len(cell) > 2 and cell[2]:
                markdown["attachments"] = cell[2]
            nb.cells.append(markdown)
        else:
            c = nbf.v4.new_code_cell(cell[1])
            if cell[2]:
                c.metadata["tags"] = cell[2]
            nb.cells.append(c)

    # Write the notebook file.
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))


if __name__ == "__main__":
    # The notebook is written to the notebooks folder at the top of the repository.
    out = Path(__file__).resolve().parent.parent / "notebooks" / "03_end_to_end.ipynb"
    build(out)
    print("wrote", out, len(CELLS), "cells")
