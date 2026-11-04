"""Identity on real faces: the Face Research Lab London Set, as SPEC-002 and DEC-001 in the vault set out.

Downloads the neutral and smiling front zips (size-checked) into a local folder that must be outside the repository
and outside any synced folder, runs every photo through the production pipeline on this computer's CPU (our ONNX
detector, five-point alignment, the ONNX embedder with the mirrored face), and writes the SPEC-002 report, the
similarity matrix (no embeddings) and the figures. It also times each stage, for choosing the machine that will
index a photo library.

Credit: DeBruine, L. and Jones, B. (2017). Face Research Lab London Set. figshare. Licence CC BY 4.0. Test use only.

    python tools/eval_london.py --models RUN/models --data C:/tmp/fdc-london --out EXP/eval-london-001
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from facekit import datasets, evaluate, prepare, sources  # noqa: E402
from facekit.pipeline import FaceAnalyzer  # noqa: E402

CREDIT = "DeBruine, L. and Jones, B. (2017). Face Research Lab London Set. figshare. CC BY 4.0."
LIMITS = ("102 adults photographed in one session, front views, with only an expression change between the two "
          "photos of each person. Real photo libraries add age, pose, lighting, blur and occlusion, so expect lower "
          "accuracy there; recalibrate the thresholds on your own enrolled photos.")


def check_data_folder(data: Path) -> None:
    """Refuses a download folder inside the repository or inside a folder that syncs to the cloud (DEC-001).

    ResearchArtifacts is refused by name too, because Google Drive for desktop syncs it on this computer.
    """
    resolved = data.resolve()
    evaluate.refuse_inside_repo(resolved)
    evaluate.refuse_synced_folder(resolved)
    if any(part.lower() == "researchartifacts" for part in resolved.parts):
        raise SystemExit(f"refusing to put face photos in the synced ResearchArtifacts folder: {resolved}")


def time_stages(analyzer: FaceAnalyzer, records: list[dict], n: int = 30) -> dict:
    """Median seconds per photo for detection alone and for the whole analysis (detection, expression, identity)."""
    detect, whole, faces = [], [], []
    for record in records[:n]:
        image = prepare.read_rgb(Path(record["path"]))
        if image is None:
            continue
        start = time.perf_counter()
        analyzer.detector.detect(image)
        detect.append(time.perf_counter() - start)
        start = time.perf_counter()
        faces.append(len(analyzer.analyze(image)))
        whole.append(time.perf_counter() - start)
    import onnxruntime
    return {"n_photos": len(whole), "photo_size": "1350 x 1350 (resized to 1280 on the long side)",
            "detect_median_s": float(np.median(detect)), "analyze_median_s": float(np.median(whole)),
            "faces_per_photo_mean": float(np.mean(faces)), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "onnxruntime": onnxruntime.__version__,
            "providers": onnxruntime.get_available_providers()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="local download folder, not synced, not in the repo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    check_data_folder(args.data)
    manifest = datasets.fetch([sources.LONDON_NEUTRAL, sources.LONDON_SMILING], args.data)
    records = prepare.london_records(args.data / sources.LONDON_NEUTRAL.name, args.data / sources.LONDON_SMILING.name)
    people = {r["person"] for r in records}
    print(f"London set: {len(records)} photos of {len(people)} people")

    card = json.loads((args.models / "model_card.json").read_text(encoding="utf-8"))
    identity = card["identity"]
    analyzer = FaceAnalyzer.from_model_dir(args.models)
    timing = time_stages(analyzer, records)
    print("timing on this computer:", json.dumps(timing))

    extra = {
        "dataset": "Face Research Lab London Set, neutral_front and smiling_front", "credit": CREDIT,
        "use": "test only (DEC-001_FDC)",
        "photos_per_view": {v: sum(1 for x in records if x["view"] == v) for v in ("neutral", "smiling")},
        "model_card": {"code_version": card.get("code_version"), "trained_on": identity.get("trained_on"),
                       "threshold_1to1": identity["threshold_1to1"], "threshold_1toN": identity["threshold"]},
        "synthetic_reference": {"tar@far=0.001_digiface_test": identity["test"]["tar@far=0.001"],
                                "tar@far=0.001_through_pipeline": identity["test"]["through_pipeline"][
                                    "tar@far=0.001_pipeline"]},
        "pipeline_timing": timing, "limits": LIMITS,
    }
    report = evaluate.run_evaluation(analyzer.embed_largest_face, records, identity["threshold_1to1"],
                                     identity["threshold"], args.out, title="(London set)", extra=extra,
                                     n_boot=args.boot, n_splits=args.splits, save_matrix=True)
    v, a, o = report["verification"], report["at_card_threshold_1to1"], report["open_set_at_card_threshold_1toN"]
    print(f"faces found in {report['n_photos']} of {len(records)} photos ({report['n_people']} people)")
    print(f"verification: AUC {v['auc']:.4f}, EER {v['eer']:.4f}, TAR at FAR 1e-2 {v['tar@far=0.01']:.3f} "
          f"{v['tar@far=0.01_ci95']}, TAR at FAR 1e-3 {v['tar@far=0.001']:.3f} {v['tar@far=0.001_ci95']} "
          f"({v['n_genuine']} genuine, {v['n_impostor']} impostor pairs)")
    print(f"at the card's 1:1 threshold {a['threshold']:.3f}: TAR {a['tar']:.3f}, FAR {a['far']:.5f}")
    for name, c in report["closed_set"].items():
        print(f"closed set {name}: rank-1 {c['rank1']:.3f}, rank-5 {c['rank5']:.3f} ({c['n_probes']} probes)")
    print(f"open set at the card's 1:N threshold {o['threshold']:.3f}, gallery {o['gallery_size']}, {o['n_splits']} "
          f"splits: FPIR {o['fpir_mean']:.3f} ({o['fpir_min']:.3f}-{o['fpir_max']:.3f}), FNIR {o['fnir_mean']:.3f} "
          f"({o['fnir_min']:.3f}-{o['fnir_max']:.3f})")
    print("indicative 1:N threshold for 1% FPIR:", round(report["indicative_threshold_1toN_fpir_1pct"]["threshold"], 4))
    print("written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
