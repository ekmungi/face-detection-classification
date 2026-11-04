# face-detection-classification

Detect every face in a photo, classify its expression, and identify whose face it is. All three models are trained
in this project; none is a model that was already trained for a face task.

| Model | What it does | Architecture | Trained on |
|---|---|---|---|
| Detector | finds faces and five landmarks (eyes, nose, mouth corners) | CenterFace-style, ResNet-18 ImageNet backbone | WIDER FACE with the RetinaFace landmark annotations |
| Expression | neutral, happy, sad, surprise, fear, disgust, anger, contempt | ConvNeXt-Tiny, ImageNet backbone | EmoNet-Face Binary (synthetic portraits), aligned with our detector |
| Identity | a 512-number face embedding compared by cosine similarity | IResNet from scratch, CosFace margin | DigiFace-1M (synthetic identities rendered from consented scans) |

Each face is aligned to a fixed 112-pixel template from its five landmarks before the expression and identity models
see it, and the same alignment code runs in training and in the pipeline.

## Results of the first full run (2026-11-04, one A100, about an hour)

| Model | Test | Result |
|---|---|---|
| Detector | WIDER FACE validation, AP at IoU 0.5 with the official easy / medium / hard face lists (VOC-style) | 0.857 / 0.845 / 0.753 |
| Expression | EmoNet-Face Binary held-out portraits (318, grouped by prompt) | accuracy 0.849, macro-F1 0.663 |
| Identity | 500 unseen DigiFace identities, 36,000 images | AUC 0.997, TAR 0.920 at FAR 1e-3, rank-1 0.938 |

Known limits: the detector misses many faces that fill the whole frame; the expression model was trained on
synthetic portraits and recognises real smiles poorly; the identity model was trained and calibrated on synthetic
faces only and loses accuracy through the detector's own alignment (TAR 0.793), so recalibrate its threshold on
your own enrolled photos. Real-face identity accuracy was not measured.

## Licences: non-commercial research only

WIDER FACE is CC BY-NC-ND 4.0, the RetinaFace annotations are for non-commercial research (models trained with them
included), DigiFace-1M is under the Microsoft Research Use of Data Agreement (trained models count as results under
the same terms), and EmoNet-Face is CC BY 4.0. The trained models therefore inherit a non-commercial research licence.
No data, face image, label, embedding or model weight is in this repository; a pre-commit hook enforces that
(`git config core.hooksPath hooks`).

## Layout

- `facekit/`: the package. `det_model.py`, `det_data.py`, `det_eval.py`, `det_task.py` (detector); `expression.py`;
  `identity.py`; `align.py`; `engine.py` (the shared training loop: mixed precision, warm-up and cosine schedule,
  checkpoints, exact resume, heartbeat, time budget); `tasks.py`; `arrays.py` (decoded image arrays, resumable
  sampler, grouped splits); `augment.py` (GPU augmentation); `metrics.py`; `datasets.py` (downloads with size and
  checksum checks and safe extraction, written by Antigravity CLI against tests written first); `sources.py` (every
  dataset with its licence); `prepare.py`; `export.py` (ONNX); `detect.py`, `pipeline.py`, `gallery.py`, `cli.py`.
- `notebooks/03_end_to_end.ipynb`: data, checks, the three trainings, the tests, figures, ONNX export and the model
  card, in one run. It is generated from `tools/build_notebook.py`.
- `train.py`: identity training from the command line (the notebook runs it as a second process).
- `tools/colab_launch.py`, `tools/colab_status.py`: start the notebook headless on a Colab machine and read its state.

## Running the training on Colab

The notebook runs on a Colab GPU (an L4 was used) and downloads every dataset itself; nothing needs a login.

```bash
colab new -s fdc-run --gpu L4
colab exec -s fdc-run --env PROJECT=face-detection-classification --env REPO_URL=https://github.com/ekmungi/face-detection-classification -f clone.py
colab exec -s fdc-run --env MODE=full --env EXP=exp-001 --env RUN_ID=run-001 -f tools/colab_launch.py
colab exec -s fdc-run --env EXP=exp-001 --env RUN_ID=run-001 -f tools/colab_status.py
```

`MODE=smoke` runs every cell on tiny subsets in about fifteen minutes; `MODE=full` is the real run (about two
hours). Each training plans its epochs from the speed it measures, so it finishes within its time budget.

## Using the models on a laptop

The run writes `models/` with `detector.onnx`, `expression.onnx`, `embedder.onnx`, the PyTorch checkpoints and
`model_card.json` (thresholds, metrics, datasets, limitations).

```bash
uv sync --extra export
uv run facekit enroll --models C:\Users\<you>\models --people C:\Users\<you>\people --gallery C:\Users\<you>\gallery.npz
uv run facekit scan --models C:\Users\<you>\models --photos C:\Users\<you>\Pictures --out results.jsonl --gallery C:\Users\<you>\gallery.npz
```

`people` holds one folder per person (the folder name is the name) with a few clear photos each. The identity
threshold in the model card was calibrated on synthetic faces only: check it on your own enrolled photos and pass
`--threshold` to `enroll` when it needs changing. The gallery holds biometric data about real people, so keep it on
your computer.

```python
from facekit.pipeline import FaceAnalyzer
from facekit.prepare import read_rgb

analyzer = FaceAnalyzer.from_model_dir("models", gallery="gallery.npz")
for face in analyzer.analyze(read_rgb("photo.jpg")):
    print(face.as_dict())   # box, landmarks, expression and its probabilities, identity and similarity
```

## Tests

```bash
uv sync --extra local
uv run pytest
```

The tests run offline on a CPU and build every fixture in code, so no face image is needed.
