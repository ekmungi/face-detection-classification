# Changelog

## 0.2.8 - 2026-11-04

- The notebook shows the pipeline figure from the PNG's address on GitHub instead of the relative path
  `../docs/figures/pipeline_flow.png`. The relative path resolves on GitHub, in a checkout opened in VS Code, and in
  Colab when Colab opens the notebook from GitHub, because Colab rewrites it to a GitHub address. A copy opened from
  Google Drive or uploaded, such as the executed copy an unattended run saves in its run folder, has no repository to
  resolve it against. The full address works in all of these whenever the computer is online. The README's sentence
  that Colab shows only the figure's description was wrong and is gone.
- A test finds every picture a markdown cell shows, whether as a markdown image, a reference-style image or an HTML
  `<img>` tag, and fails on any that is embedded in the cell or not shown from a full https address.

## 0.2.7 - 2026-11-04

- The synced-folder guard test builds its OneDrive, Google Drive and Dropbox folders under pytest's `tmp_path`
  instead of spelling out paths under a Windows user folder, so the repository holds no home-folder path. The
  guard itself is unchanged.

## 0.2.6 - 2026-11-04

- A third notebook mode, `MODE = "medium"`, between smoke and full: one run on a Colab L4 that gives meaningful
  results with all eight expressions. It trains identity on 1,000 DigiFace identities from the part already used by
  smoke mode and tests on 200, uses 3,000 WIDER training photos and 500 test photos, reads all five EmoNet-Face Binary
  files (smoke mode's one file holds only happy, sad and fear) and the one HQ file, and trains to its own budgets of
  6, 8 and 3 minutes, which replace the three budget parameters. The notebook estimates 30 to 45 minutes; the
  estimate has not been measured. Medium runs live in `scratch` as `medium-NNN`.
- Smoke and medium mode take their WIDER photos at even steps through each list, because both lists are sorted by the
  kind of event: the first 3,000 training photos covered 13 of the 61 kinds, and 3,000 at even steps cover 57; 500
  validation photos at even steps cover all 61. `prepare.evenly_spaced` and `wider_val_records(spread=True)` do this.
- Portraits in which no face is found are centre-cropped in every mode but full.
- The notebook records its mode in the run folder (`run_mode.txt`, written by `runlib.claim_run_mode`) and stops with
  a plain message when the folder already holds a run of another mode, because the trainings reuse the plans they
  find there. It also stops at once on an unknown `MODE`, and when `RUN_ID` does not start with the mode
  (`paths.check_run_name_for_mode`: `smoke-`, `medium-`, or `run-` for a full run).
- Every explanation in the notebook is rewritten to be scanned. Each cell above a code cell has a `###` step heading
  and four bold labels (what it does and why, what goes in, what comes out, how to tell it worked), each a list of
  full sentences, with tables where values are compared, such as the three modes side by side. The code cells open
  each numbered step with a matching comment; their logic is unchanged apart from the medium-mode changes above.
  Tests hold the four labels, the matching step comments, the list form and the length of each item, and keep run
  folder names out of the text. An independent check against the code corrected eleven statements, among them the
  smoke mode's held-out photos (30, not 60) and the CPU timing photo, which had no face in the full run.
- The pipeline figure's network boxes say a little more about what is inside each network: the detector's four heads
  on its stride-4 map, the expression model's eight scores divided by the temperature, and the identity model's
  CosFace margin, used only in training. The figure keeps its six blocks.
- The notebook links the figure in `docs/figures/` instead of embedding it, because Colab cannot show a notebook
  attachment and the embedded copy more than doubled the notebook's size (about 118 KB without it, 255 KB with it). The test now checks the link,
  the alt text and that no cell carries an attachment.

## 0.2.5 - 2026-11-04

- The notebook's explanations no longer name particular run or experiment folders; the parameters cell says what
  `EXP` and `RUN_ID` are for and where the accepted names are listed. A test keeps run folder names out of the
  notebook's markdown cells.

## 0.2.4 - 2026-11-04

- The downloaded dataset files can live in Google Drive. With the new `USE_DRIVE` parameter (on by default), the
  data cell mounts Drive and keeps the files in the artifacts folder's `data/raw`, so each one is downloaded only
  once; a file already there is not downloaded again. The archives are still unpacked onto the machine's own disk,
  and the Face Research Lab London Set stays on the machine. `datasets.fetch_all` takes `extract_root` and
  `local_names` for this, and reports each file as downloaded or already there. `tools/colab_launch.py` turns
  Drive off, because an unattended run cannot approve the Drive page.
- A figure of how the models feed each other now follows the notebook's introduction, with a table of each block's
  model, input and output. It is kept as a PNG with its source in `docs/figures/`, and a copy is embedded in the
  notebook cell; a test fails when the two copies differ. The pre-commit hook now allows PNGs in `docs/figures/`,
  as `.gitignore` already did.

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
