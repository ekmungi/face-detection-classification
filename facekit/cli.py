"""Command line for the photo library: enrol people, then scan a folder of photos.

    facekit enroll --models MODEL_DIR --people PEOPLE_DIR --gallery gallery.npz
        PEOPLE_DIR holds one folder per person (the folder name is the person's name) with a few clear photos each.
    facekit scan --models MODEL_DIR --photos PHOTOS_DIR --out results.jsonl [--gallery gallery.npz]
        Writes one JSON line per photo with every face's box, landmarks, expression and identity.
    facekit evaluate --models MODEL_DIR --people PEOPLE_DIR --out OUT_DIR
        Measures identity accuracy on one folder per person (at least two photos each) and reports how the model
        card's thresholds behave on those faces. Use it to recalibrate before trusting names.

The gallery and any saved embeddings are biometric data about real people, so keep them on this computer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from facekit.gallery import Gallery
from facekit.pipeline import FaceAnalyzer
from facekit.prepare import read_rgb

# The file types that count as photos. The comparison ignores upper and lower case.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(folder: Path):
    """Returns the sorted paths of all photos below `folder`, skipping the hidden "._" files that macOS creates."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._"))


def enroll(models: Path, people: Path, gallery_path: Path, threshold: float | None = None) -> dict:
    """Enrols each person folder's largest face per photo and returns how many photos were used per person.

    A photo is used when it can be read and contains a face. The people are added to the gallery at `gallery_path` when
    that file already exists, or to a new gallery that starts with the identity threshold of the model card. A
    `threshold` replaces the gallery's threshold. The gallery is saved at the end.
    """
    analyzer = FaceAnalyzer.from_model_dir(models)
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))
    gallery = Gallery.load(gallery_path) if gallery_path.exists() else Gallery(card["identity"]["threshold"])
    if threshold is not None:
        gallery.threshold = float(threshold)

    # Embed the largest face of every readable photo of each person, and enrol the embeddings under the folder name.
    used: dict[str, int] = {}
    for person in sorted(p for p in people.iterdir() if p.is_dir()):
        vectors = []
        for photo in _images(person):
            image = read_rgb(photo)
            if image is None:
                continue
            embedding = analyzer.embed_largest_face(image)
            if embedding is not None:
                vectors.append(embedding)
        if vectors:
            gallery.enroll(person.name, np.stack(vectors))
        used[person.name] = len(vectors)
    gallery.save(gallery_path)
    return used


def scan(models: Path, photos: Path, out: Path, gallery: Path | None = None, max_side: int = 1280) -> int:
    """Analyses every photo under `photos`, writes one JSON line per photo to `out`, and returns the photos read.

    A photo that cannot be read gets a line with an "error" entry and is not counted.
    """
    analyzer = FaceAnalyzer.from_model_dir(models, gallery=gallery, max_side=max_side)
    count = 0
    with open(out, "w", encoding="utf-8") as handle:
        for photo in _images(photos):
            image = read_rgb(photo)
            if image is None:
                handle.write(json.dumps({"photo": str(photo), "error": "unreadable"}) + "\n")
                continue
            faces = [f.as_dict() for f in analyzer.analyze(image)]
            handle.write(json.dumps({"photo": str(photo), "faces": faces}) + "\n")
            count += 1
    return count


def evaluate(models: Path, people: Path, out_dir: Path, n_boot: int = 1000, n_splits: int = 20) -> dict:
    """Scores identity on a folder with one sub-folder per person and returns the report.

    Also writes report.json, scores.npz and figures to `out_dir`. Raises SystemExit when the model card is too old to
    hold the identity threshold that this evaluation needs.
    """
    from facekit import evaluate as ev

    analyzer = FaceAnalyzer.from_model_dir(models)
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))["identity"]
    if "threshold_1to1" not in card:
        raise SystemExit("model_card.json has no identity threshold_1to1; this model card is too old for evaluate")
    t11 = card["threshold_1to1"]
    return ev.run_evaluation(analyzer.embed_largest_face, ev.people_folder_records(people), t11, card["threshold"],
                             out_dir, title=people.name, n_boot=n_boot, n_splits=n_splits,
                             extra={"models": str(models), "people": str(people)})


def main(argv: list[str] | None = None) -> int:
    """Runs the command line on `argv` (the process arguments by default) and returns the exit code."""
    # Describe the three commands and their options. The module docstring is the help text of the whole program, and
    # it is printed exactly as written, so its usage lines keep their layout.
    parser = argparse.ArgumentParser(prog="facekit", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_enroll = sub.add_parser("enroll", help="enrol people from one folder per person")
    p_enroll.add_argument("--models", type=Path, required=True)
    p_enroll.add_argument("--people", type=Path, required=True)
    p_enroll.add_argument("--gallery", type=Path, required=True)
    p_enroll.add_argument(
        "--threshold", type=float, default=None,
        help="identity threshold; recalibrate on your own photos (the default comes from synthetic faces)")
    p_scan = sub.add_parser("scan", help="analyse every photo in a folder")
    p_scan.add_argument("--models", type=Path, required=True)
    p_scan.add_argument("--photos", type=Path, required=True)
    p_scan.add_argument("--out", type=Path, required=True)
    p_scan.add_argument("--gallery", type=Path, default=None)
    p_scan.add_argument("--max-side", type=int, default=1280)
    p_eval = sub.add_parser(
        "evaluate", help="measure identity accuracy on one folder per person, to check thresholds")
    p_eval.add_argument("--models", type=Path, required=True)
    p_eval.add_argument("--people", type=Path, required=True)
    p_eval.add_argument("--out", type=Path, required=True, help="folder for report.json, scores.npz and figures")
    args = parser.parse_args(argv)

    # Run the chosen command and print a short summary of its result.
    if args.command == "enroll":
        print(json.dumps(enroll(args.models, args.people, args.gallery, args.threshold), indent=2))
    elif args.command == "evaluate":
        report = evaluate(args.models, args.people, args.out)
        v = report["verification"]
        print(f"{report['n_people']} people, {report['n_photos']} photos with a face "
              f"({report['failed_to_acquire']['count']} without); AUC {v['auc']:.4f}, "
              f"TAR at FAR 1e-3 {v['tar@far=0.001']:.3f}; report in {args.out / 'report.json'}")
    else:
        print(f"{scan(args.models, args.photos, args.out, args.gallery, args.max_side)} photos written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
