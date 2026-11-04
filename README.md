# face-detection-classification

A research project built with the Jeeves Colab workflow: write the code here, check it in notebooks on this computer, run it on a
rented Google Colab machine, and keep a record of every run. `specs/exp-001_first-experiment.toml` states the first question.
The starter code trains a small classifier on MNIST so that the plumbing can be checked; replace `data.py`, `augment.py`,
`models.py` and `models_extra.py` with the project's own, and keep the rest.

## Where things live

One project has three places, and each place has one job.

| Place | Location on this computer | Synced with | Holds |
|---|---|---|---|
| Notes | `<vault>\01 PROJECTS\<Project Name>\` | Obsidian | the plan, decisions, tasks and research notes |
| Code (this repository) | `C:\Users\<you>\Dev\face-detection-classification\` | GitHub (public) | everything below except artifacts |
| Artifacts | `C:\Users\<you>\ResearchArtifacts\face-detection-classification\` | Google Drive (two-way) | kept datasets, scratch work, experiments and their runs |

A Colab machine gets the code by cloning this public repository and reads and writes artifacts through the mounted Drive.
Nothing from the artifacts folder is ever committed, so the repository holds no data, checkpoints or patient information. Notebook
outputs are never committed either: run `python tools/strip_notebook_outputs.py` before a commit (`--check` only reports).

### This repository

| Folder or file | What it holds |
|---|---|
| Helper scripts at the root, `train.py` | `data.py`, `datastore.py`, `augment.py`, `models.py`, `models_extra.py`, `trainer.py`, `callbacks.py`, `checks.py`, `quicktrain.py`, `runlib.py`, `paths.py` |
| `run_eval.py` | optional: the entry point for an evaluation-only run (`pipeline = "evaluate"`); it calls `evaluation.score`, a file you write for the pretrained model |
| `specs/` | `exp-NNN_<slug>.toml`, one specification per experiment |
| `notebooks/` | `01_helpers.ipynb` (stage 2), `02_pipeline.ipynb` (stages 3 and 4) |
| `tests/`, `tools/` | unit tests; `run_notebook_local.py` runs a notebook cell by cell on this computer; `strip_notebook_outputs.py` |
| `README.md`, `docs/figures/` | the method in words, and result pictures copied here by hand after review |

### Data: fetch or store

Each dataset has one of two modes, set in the experiment specification (`[data] mode`) and overridable with `--data-mode`.

| Mode | Use it for | What a session does | Drive space |
|---|---|---|---|
| `fetch` (default) | Public data that is easy to download, such as the torchvision datasets | Downloads it onto the machine's own disk (`/content/data` on Colab) | none |
| `store` | Data that needs a login, or that you chose to keep | Copies it from `artifacts/data/` and unpacks a zip onto the machine's own disk | one copy per project |

In `store` mode a missing dataset stops the run with a message, and it never falls back to a download. A dataset is kept with
`datastore.keep_copy` only after the owner says so, as a folder or a zip with a `source.md` note (origin, licence, date, a
checksum per file). A dataset behind a login is downloaded by the person who holds the login; credentials are never stored.

### The artifacts folder

```
ResearchArtifacts/face-detection-classification/
  data/                         only the datasets you chose to keep (store mode), each with a source.md note
  scratch/                      smoke-NNN runs, notebook-out/, figures/: throwaway, may be deleted
  exp-NNN/
    exp.md  findings.md         the question and conclusion; what was learned, for blog posts
    analysis/                   notebooks run on the results, tables and plots for the write-up
    run-NNN/
      config.json env.json git.txt     written once, never changed
      attempts.jsonl metrics_history.jsonl train.log    only appended to
      heartbeat.json                   overwritten (latest state)
      checkpoints/                     rolling: the newest two are kept
      metrics.json DONE                written when the run ends
      record.md                        written afterwards, from the results
      figures/ predictions/ stdout.log optional: made only by your own code or notebooks
```

`exp-NNN` and `run-NNN` numbers are never reused. A resume keeps its run id; a new attempt gets a new one. Smoke tests live in
`scratch/smoke-NNN`. `paths.py` is the only place that knows where the artifacts folder is: `ARTIFACT_ROOT` if set, otherwise
the Drive folder on Colab, otherwise `~/ResearchArtifacts/face-detection-classification` here.

## The five-stage development pipeline

| Stage | What | Files | Gate before moving on |
|---|---|---|---|
| 1 | Helper scripts | `data.py`, `augment.py`, `models.py`, `runlib.py`, `paths.py` | `pytest` passes |
| 2 | Small runs of each helper | `notebooks/01_helpers.ipynb` | The pictures look right and the checks print PASS |
| 3 | The training loop, written once | `trainer.py`, `callbacks.py` | The notebook calls `trainer.fit`; nothing is copied |
| 4 | A few epochs with logic checks | `notebooks/02_pipeline.ipynb`, `checks.py` | Restart and Run All, every check prints PASS |
| 5 | Deploy from the script | `train.py` | 200-step smoke test, checkpoint and resume, parity with the notebook |

## Run it

On this computer (CPU, small settings):

```
uv run --extra local pytest
uv run --extra local python tools/run_notebook_local.py notebooks/02_pipeline.ipynb
```

On Colab: push the code first, because the machine clones what is on GitHub. Then open `notebooks/02_pipeline.ipynb` in VS Code,
choose Select Kernel, Colab, New Colab Server, GPU, T4 (never TPU), and run the first code cell. It clones this repository into
`/content/face-detection-classification` and mounts Drive for the artifacts; approve the Drive prompt within two minutes. A script run:

```
python train.py --spec specs/exp-001_first-experiment.toml --exp exp-001 --run-id run-001
python train.py --exp scratch --run-id smoke-001 --max-steps 200
```

A run that loads a pretrained model and trains nothing uses `python run_eval.py --spec specs/exp-002_<slug>.toml --exp exp-002 --run-id run-001` instead; its log is `run_eval.log` and it writes `metrics.json` and `DONE` like a training run.
