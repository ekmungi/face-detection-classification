"""Identity evaluation on a folder of real people, through the same path a photo takes in production.

Every photo is read, its largest face is detected, aligned and embedded (with the mirrored face added), and the
embeddings are scored as SPEC-002 sets out:

- verification over every pair of photos: ROC AUC, equal error rate, TAR at FAR 1e-2 and 1e-3 with 95% bootstrap
  intervals, and the real FAR and TAR at the model card's 1:1 threshold;
- closed-set identification: rank-1 and rank-5 against a gallery of one photo per person;
- open-set identification at the model card's 1:N threshold over random enrolled/stranger splits: FPIR (strangers
  given a name) and FNIR (enrolled people missed or misnamed), with an indicative threshold for 1% FPIR.

Embeddings are biometric data, so they live only in memory: callers may save the similarity matrix, never the
embeddings themselves.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from facekit import metrics
from facekit.prepare import read_rgb

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FARS = (1e-2, 1e-3)          # 1e-4 needs far more impostor pairs than a folder of people gives


@dataclasses.dataclass
class Embedded:
    """Embeddings of the photos in which a face was found, with their labels; `failed` lists the others."""

    embeddings: np.ndarray            # [N, D], L2-normalised; keep in memory only
    labels: np.ndarray                # [N] person ids (str)
    views: np.ndarray                 # [N] view names (str, "" when the set has none)
    paths: list[str]
    failed: list[dict]                # {"path", "person", "view", "reason"}


def people_folder_records(root: str | Path) -> list[dict]:
    """Returns {"path", "person", "view": ""} for every image in a folder with one sub-folder per person."""
    records = []
    for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for path in sorted(person.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._"):
                records.append({"path": str(path), "person": person.name, "view": ""})
    return records


def embed_records(embed_fn: Callable[[np.ndarray], np.ndarray | None], records: list[dict],
                  read_fn: Callable[[Path], np.ndarray | None] = read_rgb) -> Embedded:
    """Runs `embed_fn` (an RGB image to an embedding, or None when no face is found) over every record."""
    vectors, labels, views, paths, failed = [], [], [], [], []
    for record in records:
        image = read_fn(Path(record["path"]))
        if image is None:
            failed.append({**record, "reason": "unreadable"})
            continue
        vector = embed_fn(image)
        if vector is None:
            failed.append({**record, "reason": "no face"})
            continue
        vectors.append(np.asarray(vector, dtype=np.float32))
        labels.append(record["person"])
        views.append(record.get("view", ""))
        paths.append(record["path"])
    embeddings = np.stack(vectors) if vectors else np.zeros((0, 0), np.float32)
    if len(embeddings):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    return Embedded(embeddings, np.array(labels), np.array(views), paths, failed)


def pair_scores(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cosine scores, same-person flags, index pairs) for every pair i < j."""
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    i, j = np.triu_indices(len(e), k=1)
    labels = np.asarray(labels)
    return np.sum(e[i] * e[j], axis=1), (labels[i] == labels[j]).astype(np.int64), np.stack([i, j], axis=1)


def rates_at_threshold(scores: np.ndarray, same: np.ndarray, threshold: float) -> dict:
    """Returns the TAR and FAR when a pair is accepted at a score of at least `threshold`."""
    gen, imp = scores[same == 1], scores[same == 0]
    return {"threshold": float(threshold), "tar": float((gen >= threshold).mean()),
            "far": float((imp >= threshold).mean()), "n_genuine": int(len(gen)), "n_impostor": int(len(imp))}


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> tuple[float, float]:
    """Returns (TAR, threshold) at the lowest threshold whose FAR does not exceed `far` (accept at >= threshold).

    The threshold sits just above the (k+1)-th highest impostor score, where k = floor(far * impostors), so at most k
    impostor pairs are accepted.
    """
    imp = np.sort(np.asarray(impostor))[::-1]
    k = int(np.floor(far * len(imp)))
    threshold = float(np.nextafter(imp[k], np.inf)) if k < len(imp) else float(imp[-1])
    return float((np.asarray(genuine) >= threshold).mean()), threshold


def bootstrap_tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float, n_boot: int = 1000,
                         seed: int = 0) -> tuple[float, float]:
    """Returns the 95% interval of TAR at `far`, resampling genuine and impostor scores separately."""
    rng = np.random.default_rng(seed)
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    values = [tar_at_far(rng.choice(genuine, len(genuine)), rng.choice(impostor, len(impostor)), far)[0]
              for _ in range(n_boot)]
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def closed_set(embeddings: np.ndarray, labels: np.ndarray, gallery_idx: np.ndarray, probe_idx: np.ndarray) -> dict:
    """Rank-1 and rank-5 of the probes against a gallery; probes whose person has no gallery photo are left out."""
    labels = np.asarray(labels)
    enrolled = set(labels[gallery_idx].tolist())
    usable = np.array([i for i in probe_idx if labels[i] in enrolled], dtype=np.int64)
    report = metrics.identification_report(embeddings[gallery_idx], labels[gallery_idx], embeddings[usable],
                                           labels[usable])
    report["probes_without_gallery_entry"] = int(len(probe_idx) - len(usable))
    return report


def _gallery_and_probes(labels: np.ndarray, views: np.ndarray | None, people: np.ndarray, enrol_view: str
                        ) -> tuple[dict, list[int]]:
    """For each person: the index of the enrolment photo and the indices of the other photos."""
    enrol, rest = {}, []
    for person in people:
        idx = np.flatnonzero(labels == person)
        if views is not None and enrol_view in set(views[idx].tolist()):
            first = int(idx[views[idx] == enrol_view][0])
        else:
            first = int(idx[0])
        enrol[person] = first
        rest += [int(i) for i in idx if i != first]
    return enrol, rest


def open_set_once(embeddings: np.ndarray, labels: np.ndarray, views: np.ndarray | None, enrolled, strangers,
                  threshold: float, enrol_view: str = "neutral", return_scores: bool = False) -> dict:
    """Open-set identification for one split of the people into enrolled people and strangers.

    Each enrolled person is enrolled with one photo (the `enrol_view` photo when the set has views, else the first)
    and probed with every other photo of theirs; every photo of a stranger is a stranger probe. FPIR is the share of
    stranger probes whose best match reaches the threshold; FNIR is the share of enrolled probes that are not both
    matched to the right person and at or above the threshold, so a misname counts as a miss.
    """
    labels = np.asarray(labels)
    enrol, genuine_idx = _gallery_and_probes(labels, views, np.asarray(enrolled), enrol_view)
    names = np.array(list(enrol))
    templates = embeddings[[enrol[p] for p in names]]
    stranger_idx = np.flatnonzero(np.isin(labels, np.asarray(strangers)))
    g_sims = embeddings[genuine_idx] @ templates.T
    s_best = (embeddings[stranger_idx] @ templates.T).max(1)
    right = names[g_sims.argmax(1)] == labels[genuine_idx]
    out = {"fpir": float((s_best >= threshold).mean()),
           "fnir": float(1.0 - (right & (g_sims.max(1) >= threshold)).mean()),
           "rank1": float(right.mean()), "n_genuine_probes": int(len(genuine_idx)),
           "n_stranger_probes": int(len(stranger_idx))}
    if return_scores:
        out["stranger_best"] = s_best
    return out


def open_set_splits(embeddings: np.ndarray, labels: np.ndarray, views: np.ndarray | None, threshold: float,
                    n_splits: int = 20, seed: int = 0, enrol_view: str = "neutral", keep_splits: bool = False
                    ) -> dict:
    """`open_set_once` over random splits of the people into enrolled and stranger halves (fixed seed).

    Only people with at least two photos take part. The ranges reflect the randomness of the splits only, not the
    sampling of the people, who are reused across splits.
    """
    labels = np.asarray(labels)
    views = None if views is None else np.asarray(views)
    counts = {p: int((labels == p).sum()) for p in np.unique(labels)}
    people = np.array(sorted(p for p, c in counts.items() if c >= 2))
    rng = np.random.default_rng(seed)
    fpir, fnir, rank1, stranger_best, splits = [], [], [], [], []
    for _ in range(n_splits):
        shuffled = rng.permutation(people)
        enrolled, strangers = shuffled[: len(people) // 2], shuffled[len(people) // 2:]
        one = open_set_once(embeddings, labels, views, enrolled, strangers, threshold, enrol_view, True)
        fpir.append(one["fpir"])
        fnir.append(one["fnir"])
        rank1.append(one["rank1"])
        stranger_best.append(one["stranger_best"])
        if keep_splits:
            splits.append({"enrolled": enrolled.tolist(), "strangers": strangers.tolist()})
    pooled = np.concatenate(stranger_best)
    out = {"threshold": float(threshold), "n_splits": int(n_splits), "gallery_size": int(len(people) // 2),
           "fpir_mean": float(np.mean(fpir)), "fpir_min": float(np.min(fpir)), "fpir_max": float(np.max(fpir)),
           "fnir_mean": float(np.mean(fnir)), "fnir_min": float(np.min(fnir)), "fnir_max": float(np.max(fnir)),
           "rank1_mean": float(np.mean(rank1)), "n_stranger_probes_pooled": int(len(pooled)),
           "pooled_stranger_best_scores": pooled}
    if keep_splits:
        out["splits"] = splits
    return out


def evaluate_embeddings(embeddings: np.ndarray, labels: np.ndarray, views: np.ndarray | None, threshold_1to1: float,
                        threshold_1toN: float, n_boot: int = 1000, n_splits: int = 20, seed: int = 0) -> dict:
    """Returns the SPEC-002 report (plain JSON) for embeddings of people with at least one photo each."""
    labels = np.asarray(labels)
    views = None if views is None or not any(np.asarray(views)) else np.asarray(views)
    multi = sum(1 for p in np.unique(labels) if (labels == p).sum() >= 2)
    if multi < 2:
        raise ValueError(f"need at least two people with two photos each in which a face is found (found {multi})")
    scores, same, _ = pair_scores(embeddings, labels)
    gen, imp = scores[same == 1], scores[same == 0]
    roc = metrics.verification_report(scores, same, fars=FARS)
    verification = {"auc": roc["auc"], "eer": roc["eer"], "n_genuine": int(len(gen)), "n_impostor": int(len(imp)),
                    "genuine_mean": float(gen.mean()), "impostor_mean": float(imp.mean())}
    for k, far in enumerate(FARS):
        tar, threshold = tar_at_far(gen, imp, far)
        verification[f"tar@far={far:g}"] = tar
        verification[f"threshold@far={far:g}"] = threshold
        verification[f"tar@far={far:g}_ci95"] = list(bootstrap_tar_at_far(gen, imp, far, n_boot, seed + k))
    if views is not None and {"neutral", "smiling"} <= set(views.tolist()):
        neutral, smiling = np.flatnonzero(views == "neutral"), np.flatnonzero(views == "smiling")
        closed = {"neutral_to_smiling": closed_set(embeddings, labels, neutral, smiling),
                  "smiling_to_neutral": closed_set(embeddings, labels, smiling, neutral)}
    else:
        first: dict = {}
        for i, person in enumerate(labels):
            first.setdefault(person, i)
        gallery = np.array(sorted(first.values()))
        closed = {"first_photo_gallery": closed_set(embeddings, labels, gallery,
                                                    np.setdiff1d(np.arange(len(labels)), gallery))}
    open_set = open_set_splits(embeddings, labels, views, threshold_1toN, n_splits, seed)
    pooled = open_set.pop("pooled_stranger_best_scores")
    indicative = float(np.nextafter(np.quantile(pooled, 0.99), np.inf))
    return {
        "n_photos": int(len(labels)), "n_people": int(len(np.unique(labels))),
        "verification": verification,
        "at_card_threshold_1to1": rates_at_threshold(scores, same, threshold_1to1),
        "closed_set": closed,
        "open_set_at_card_threshold_1toN": open_set,
        "indicative_threshold_1toN_fpir_1pct": {
            "threshold": indicative, "gallery_size": open_set["gallery_size"],
            "note": "99th percentile of the pooled stranger best-match scores, measured on the test photos "
                    "themselves; a guide for recalibration, not a calibrated value. FPIR grows with gallery size."},
    }


SYNCED_PREFIXES = ("onedrive", "my drive", "google drive", "googledrive", "dropbox", "icloud", "box sync")
REPO = Path(__file__).resolve().parents[1]


def refuse_synced_folder(path: str | Path) -> None:
    """Stops when `path` lies in a folder that a cloud service usually syncs (by its standard folder names).

    The check matches the folder names OneDrive (including "OneDrive - <organisation>"), Google Drive's "My Drive",
    Dropbox, iCloud and Box use; a sync root with a custom name is not detected.
    """
    for part in Path(path).parts:
        if part.lower().startswith(SYNCED_PREFIXES):
            raise SystemExit(f"refusing to use a folder that a cloud service syncs: {path} (part {part!r})")


def refuse_inside_repo(path: str | Path) -> None:
    """Stops when `path` lies inside this repository, which is public and must hold no face data or results."""
    resolved = Path(path).resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise SystemExit(f"refusing to write face data or results inside the repository: {resolved}")


def code_version() -> dict:
    """The commit of this code and whether the working tree had uncommitted changes (None outside a checkout)."""
    import subprocess

    def git(*args):
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


def run_evaluation(embed_fn: Callable[[np.ndarray], np.ndarray | None], records: list[dict], threshold_1to1: float,
                   threshold_1toN: float, out_dir: str | Path, title: str = "", extra: dict | None = None,
                   n_boot: int = 1000, n_splits: int = 20, seed: int = 0, save_matrix: bool = False,
                   read_fn: Callable[[Path], np.ndarray | None] = read_rgb) -> dict:
    """Embeds every record, scores it, and writes report.json, scores.npz and the figures into `out_dir`.

    scores.npz holds the genuine and impostor score vectors. The full similarity matrix is written only with
    `save_matrix`, because a full-rank matrix lets anyone rebuild the embeddings up to a rotation: keep it for public
    test sets, never for private photos.
    """
    import json
    import time

    refuse_inside_repo(out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds: list[float] = []

    def timed(image):
        start = time.perf_counter()
        vector = embed_fn(image)
        seconds.append(time.perf_counter() - start)
        return vector

    embedded = embed_records(timed, records, read_fn)
    report = evaluate_embeddings(embedded.embeddings, embedded.labels, embedded.views, threshold_1to1,
                                 threshold_1toN, n_boot, n_splits, seed)
    report["failed_to_acquire"] = {
        "count": len(embedded.failed), "of": len(records), "rate": len(embedded.failed) / max(1, len(records)),
        "people_affected": len({f["person"] for f in embedded.failed}),
        "note": "photos without a face are left out of every rate above, so those rates do not count them as misses",
        "photos": [{"photo": Path(f["path"]).name, "person": f["person"], "view": f.get("view", ""),
                    "reason": f["reason"]} for f in embedded.failed]}
    report["seconds_per_photo_detect_align_embed"] = {
        "median": float(np.median(seconds)), "mean": float(np.mean(seconds)), "p90": float(np.quantile(seconds, 0.9)),
        "n": len(seconds)}
    report["settings"] = {"n_boot": n_boot, "n_splits": n_splits, "seed": seed,
                          "bootstrap": "genuine and impostor scores resampled separately; pairs share photos, so "
                                       "the intervals understate the uncertainty somewhat",
                          "evaluation_code": code_version()}
    if extra:
        report.update(extra)
    scores, same, _ = pair_scores(embedded.embeddings, embedded.labels)
    report["figures"] = [p.name for p in plot_report(scores, same, report, out_dir, title)]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if save_matrix:
        np.savez_compressed(out_dir / "scores.npz", similarity=similarity_matrix(embedded.embeddings),
                            labels=embedded.labels, views=embedded.views,
                            photos=np.array([Path(p).name for p in embedded.paths]))
    else:
        np.savez_compressed(out_dir / "scores.npz", genuine=scores[same == 1], impostor=scores[same == 0])
    return report


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Returns the cosine similarity of every pair of photos, the only derived data a run may keep."""
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    return (e @ e.T).astype(np.float32)


def plot_report(scores: np.ndarray, same: np.ndarray, report: dict, out_dir: str | Path, title: str = "") -> list[Path]:
    """Writes the ROC curve (log FAR) and the genuine/impostor histograms with the card thresholds marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(same, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.clip(fpr, 1e-5, 1), tpr)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.01)
    for far in FARS:
        ax.axvline(far, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("false accept rate (impostor pairs accepted)")
    ax.set_ylabel("true accept rate (genuine pairs accepted)")
    ax.set_title(f"ROC {title}".strip())
    roc_path = out_dir / "roc.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.3, 1.0, 66)
    ax.hist(scores[same == 0], bins=bins, density=True, alpha=0.6, label=f"different people ({int((same == 0).sum())})")
    ax.hist(scores[same == 1], bins=bins, density=True, alpha=0.6, label=f"same person ({int((same == 1).sum())})")
    t11 = report["at_card_threshold_1to1"]["threshold"]
    t1n = report["open_set_at_card_threshold_1toN"]["threshold"]
    ax.axvline(t11, color="k", ls="--", lw=1, label=f"card 1:1 threshold {t11:.3f}")
    ax.axvline(t1n, color="r", ls="--", lw=1, label=f"card 1:N threshold {t1n:.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"Score distributions {title}".strip())
    ax.legend(fontsize=8)
    hist_path = out_dir / "score_histograms.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    return [roc_path, hist_path]
