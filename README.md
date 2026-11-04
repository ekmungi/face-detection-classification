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
| Expression | EmoNet-Face Binary held-out portraits (318, grouped by prompt) | accuracy 0.849, macro-F1 0.662 |
| Identity | 500 unseen DigiFace identities, 36,000 images | AUC 0.997, TAR 0.920 at FAR 1e-3, rank-1 0.938 |
| Identity on real people | Face Research Lab London Set: 102 people, a neutral and a smiling photo each, through our detector and alignment on a laptop CPU | AUC 0.972, TAR 0.716 at FAR 1e-3, rank-1 0.863 |

Known limits: the detector misses many faces that fill the whole frame; the expression model was trained on
synthetic portraits and recognises real smiles poorly; the identity model was trained and calibrated on synthetic
faces only and loses accuracy through the detector's own alignment (TAR 0.793). On real faces its thresholds are
wrong: two photos of different real people score 0.46 on average, so at the model card's gallery threshold (0.522)
98% of strangers are given a name. Recalibrate with `facekit evaluate` on your own people before using a gallery.

## Licences: the trained models are for non-commercial use

The code is ours, and the architectures come from published papers, so neither restricts the models. The torchvision
ImageNet backbones carry no licence of their own. The restrictions come from the data each model learned from:

- **Detector.** WIDER FACE is CC BY-NC-ND, and the InsightFace authors state that the RetinaFace annotations "(and
  the models trained with these data) are available for non-commercial research purposes only".
- **Identity.** DigiFace-1M is under Microsoft's Research Use of Data Agreement v1.0. Clause 5.5 makes models trained
  on the data "Results", and clause 2.1 bars Results from any commercial offering.
- **Expression.** EmoNet-Face is CC BY 4.0, which allows commercial use with attribution. LAION's dataset card bars
  the data and models trained on it from high-stakes uses such as emotion recognition at work or school,
  surveillance, law enforcement, credit, insurance and hiring. Its training crops were made with the detector above.

Every face reaches the expression and identity models through the detector, so the pipeline as a whole is for
non-commercial use; private use on your own photos is non-commercial, while selling it, offering it as a service or
publishing the weights is not covered. This is a reading of the published terms, not legal advice.
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

To check the thresholds on your own people, give `evaluate` a folder with at least two photos per person:

```bash
uv run facekit evaluate --models C:\Users\<you>\models --people C:\Users\<you>\people --out C:\Users\<you>\eval
```

It runs every photo through the same detector, alignment and embedder as `scan`, and writes `report.json` (ROC AUC,
TAR at FAR 1e-2 and 1e-3 with bootstrap intervals, the real false-accept rate at the model card's thresholds,
rank-1 identification, open-set rates over random splits, and an indicative threshold), `scores.npz` (similarities
only, never embeddings) and two figures. `tools/eval_london.py` runs the same evaluation on the Face Research Lab
London Set (DeBruine and Jones, 2017, CC BY 4.0), downloaded to a local folder outside any synced folder.

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
