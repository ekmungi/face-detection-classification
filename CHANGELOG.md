# Changelog

## 0.2.3 - 2026-11-04

- The notebook's download cell no longer looks frozen and no longer hangs the kernel when it is stopped.
  `datasets.fetch_all` prints a line for each dataset as soon as it finishes, and every 30 seconds a progress line
  with what is still downloading or unpacking, the gigabytes so far and the current speed. Stopping the cell ends
  every download within seconds, keeps the finished files and says how many are complete; running the cell again
  skips them. A failed download stops the others and reports its error.
- `datasets.download` and `datasets.safe_extract` take an optional stop signal, checked between chunks and between
  archive members, and a stopped download leaves no partial file.

## 0.2.2 - 2026-11-04

- Every code cell of the end-to-end notebook now has a markdown explanation before it: what the code does and why,
  what goes in and comes out, what in its output shows that it worked, and, for the training and test cells, what a
  good result looks like. The introduction says how to run the notebook and what smoke mode is for.
- Internal record numbers from the project's private notes are removed from the code, the notebook, the hook and
  the evaluation report, which now states the reason itself (the London set is a test-only set, because its consent
  does not name face recognition).
- Two tests guard both rules: one fails when a notebook code cell has no explanation before it, and one fails when a
  tracked file carries a record number.

## 0.2.1 - 2026-11-04

- The end-to-end notebook now runs on a Colab kernel opened from VS Code or the browser. A new first setup cell
  clones the repository on a Colab machine when the clone is missing, and it stops at once, with a plain message,
  on a Colab machine without an NVIDIA GPU (a TPU or CPU machine), where the notebook used to fail later on a
  missing `cv2` or train on the CPU. On this laptop and under `tools/colab_launch.py` nothing changes.
- A test now fails when `notebooks/03_end_to_end.ipynb` is out of date with `tools/build_notebook.py`.

## 0.2.0 - 2026-11-04

- `facekit evaluate` measures identity accuracy on any folder with one sub-folder per person, through the same
  detector, alignment and embedder as `scan`, and reports how the model card's thresholds behave on those faces.
- `tools/eval_london.py` runs it on the Face Research Lab London Set, the first identity test on real people.
- The equal error rate now uses every point of the ROC curve, so it is exact rather than read at a nearby corner.
- `count_images` no longer counts macOS `._` resource-fork files as images.
- The licence notes are corrected: the restrictions come from each model's training data, and the torchvision
  ImageNet backbones carry no licence of their own.

## 0.1.0 - 2026-11-04

- First release: a face detector with five landmarks, an eight-class expression classifier and a face-embedding
  model, all trained in this project, with an end-to-end Colab notebook, ONNX exports, a CPU pipeline and the
  `facekit enroll` and `facekit scan` commands.
- `facekit/datasets.py` was written by Antigravity CLI (Gemini) against tests written first by Claude Code.
