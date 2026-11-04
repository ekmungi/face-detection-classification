# Changelog

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
