"""Command line for the photo library: enrol people, then scan a folder of photos.

    facekit enroll --models MODEL_DIR --people PEOPLE_DIR --gallery gallery.npz
        PEOPLE_DIR holds one folder per person (the folder name is the person's name) with a few clear photos each.
    facekit scan --models MODEL_DIR --photos PHOTOS_DIR --out results.jsonl [--gallery gallery.npz]
        Writes one JSON line per photo with every face's box, landmarks, expression and identity.

The gallery and any saved embeddings are biometric data about real people: keep them on this computer.
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def enroll(models: Path, people: Path, gallery_path: Path, threshold: float | None = None) -> dict:
    """Enrols each person folder's largest face per photo; returns how many photos were used per person."""
    analyzer = FaceAnalyzer.from_model_dir(models)
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))
    gallery = Gallery.load(gallery_path) if gallery_path.exists() else Gallery(threshold or card["identity"]["threshold"])
    used: dict[str, int] = {}
    for person in sorted(p for p in people.iterdir() if p.is_dir()):
        vectors = []
        for photo in _images(person):
            image = read_rgb(photo)
            if image is None:
                continue
            faces = analyzer.analyze(image)
            if faces:
                largest = max(faces, key=lambda f: (f.detection.box[2] - f.detection.box[0]) *
                              (f.detection.box[3] - f.detection.box[1]))
                vectors.append(largest.embedding)
        if vectors:
            gallery.enroll(person.name, np.stack(vectors))
        used[person.name] = len(vectors)
    gallery.save(gallery_path)
    return used


def scan(models: Path, photos: Path, out: Path, gallery: Path | None = None, max_side: int = 1280) -> int:
    """Analyses every photo under `photos` and writes JSON lines to `out`; returns the number of photos read."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facekit", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_enroll = sub.add_parser("enroll", help="enrol people from one folder per person")
    p_enroll.add_argument("--models", type=Path, required=True)
    p_enroll.add_argument("--people", type=Path, required=True)
    p_enroll.add_argument("--gallery", type=Path, required=True)
    p_enroll.add_argument("--threshold", type=float, default=None,
                          help="identity threshold; recalibrate on your own photos (the default comes from synthetic faces)")
    p_scan = sub.add_parser("scan", help="analyse every photo in a folder")
    p_scan.add_argument("--models", type=Path, required=True)
    p_scan.add_argument("--photos", type=Path, required=True)
    p_scan.add_argument("--out", type=Path, required=True)
    p_scan.add_argument("--gallery", type=Path, default=None)
    p_scan.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args(argv)
    if args.command == "enroll":
        print(json.dumps(enroll(args.models, args.people, args.gallery, args.threshold), indent=2))
    else:
        print(f"{scan(args.models, args.photos, args.out, args.gallery, args.max_side)} photos written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
