"""Tests identity recognition on real faces with the Face Research Lab London Set.

The London Set is used only as a test set, because its consent covers research use and public sharing but does not
name face recognition.

The script downloads the zip files of the neutral and the smiling front photos and checks their sizes. It saves them in
a local folder that must be outside the repository and outside any synced folder. It then runs every photo through the
production pipeline on the CPU of this computer. The pipeline uses our ONNX detector (ONNX is an open file format for
trained models), five-point alignment of the face (on the two eyes, the nose tip and the two mouth corners), and the
ONNX embedder (the model that turns a face into a list of numbers), which is also given the mirrored face. The script
writes the evaluation report, the similarity matrix (without the embeddings) and the figures. It also times each
stage, to help choose the machine that will index a photo library.

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

# The repository folder is put on the import path, so that the script runs from any folder and can import facekit.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from facekit import datasets, evaluate, prepare, sources  # noqa: E402
from facekit.pipeline import FaceAnalyzer  # noqa: E402

# The credit line for the dataset, which is written into the evaluation report.
CREDIT = "DeBruine, L. and Jones, B. (2017). Face Research Lab London Set. figshare. CC BY 4.0."

# The warning about what the result does and does not show, which is written into the evaluation report.
LIMITS = ("102 adults photographed in one session, front views, with only an expression change between the two "
          "photos of each person. Real photo libraries add age, pose, lighting, blur and occlusion, so expect lower "
          "accuracy there; recalibrate the thresholds on your own enrolled photos.")


def check_data_folder(data: Path) -> None:
    """Refuses a download folder inside the repository or inside a folder that syncs to the cloud.

    The London photos stay on this computer and never go to a synced or shared folder. A folder named
    ResearchArtifacts is refused by name too, because Google Drive for desktop syncs it on this computer.
    """
    resolved = data.resolve()
    evaluate.refuse_inside_repo(resolved)
    evaluate.refuse_synced_folder(resolved)
    if any(part.lower() == "researchartifacts" for part in resolved.parts):
        raise SystemExit(f"refusing to put face photos in the synced ResearchArtifacts folder: {resolved}")


def time_stages(analyzer: FaceAnalyzer, records: list[dict], n: int = 30) -> dict:
    """Returns the median time per photo, in seconds, together with facts about the machine.

    The times are measured for detection alone and for the whole analysis, which is detection, expression and
    identity. Only the first n photos are timed.
    """
    detect, whole, faces = [], [], []

    # Time both stages on each photo. A photo that cannot be read is skipped.
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

    # The runtime is imported here, where its version and its available providers (the hardware it can use) are read.
    import onnxruntime
    return {"n_photos": len(whole), "photo_size": "1350 x 1350 (resized to 1280 on the long side)",
            "detect_median_s": float(np.median(detect)), "analyze_median_s": float(np.median(whole)),
            "faces_per_photo_mean": float(np.mean(faces)), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "onnxruntime": onnxruntime.__version__,
            "providers": onnxruntime.get_available_providers()}


def main() -> int:
    """Runs the London Set evaluation, prints the results, and returns the exit code, which is 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="local download folder, not synced, not in the repo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    # Check the folder first, so that no photo is downloaded into a place where it must not be.
    check_data_folder(args.data)

    # Download the two photo sets and list the photos, with the person and the view (neutral or smiling) of each.
    manifest = datasets.fetch([sources.LONDON_NEUTRAL, sources.LONDON_SMILING], args.data)
    records = prepare.london_records(args.data / sources.LONDON_NEUTRAL.name, args.data / sources.LONDON_SMILING.name)
    people = {r["person"] for r in records}
    print(f"London set: {len(records)} photos of {len(people)} people")

    # Load the models, read the thresholds that the model card recorded after training, and time the pipeline.
    card = json.loads((args.models / "model_card.json").read_text(encoding="utf-8"))
    identity = card["identity"]
    analyzer = FaceAnalyzer.from_model_dir(args.models)
    timing = time_stages(analyzer, records)
    print("timing on this computer:", json.dumps(timing))

    # The extra entries are written into the report. They record the data, its permitted use, the model card values
    # and the synthetic-data results that the real-face results can be compared with.
    extra = {
        "dataset": "Face Research Lab London Set, neutral_front and smiling_front", "credit": CREDIT,
        "use": "test only: the subjects consented to research use and public sharing, "
               "and the consent does not name face recognition",
        "photos_per_view": {v: sum(1 for x in records if x["view"] == v) for v in ("neutral", "smiling")},
        "model_card": {"code_version": card.get("code_version"), "trained_on": identity.get("trained_on"),
                       "threshold_1to1": identity["threshold_1to1"], "threshold_1toN": identity["threshold"]},
        "synthetic_reference": {"tar@far=0.001_digiface_test": identity["test"]["tar@far=0.001"],
                                "tar@far=0.001_through_pipeline": identity["test"]["through_pipeline"][
                                    "tar@far=0.001_pipeline"]},
        "pipeline_timing": timing, "limits": LIMITS,
    }

    # Run the evaluation and save the report, the similarity matrix and the figures.
    report = evaluate.run_evaluation(analyzer.embed_largest_face, records, identity["threshold_1to1"],
                                     identity["threshold"], args.out, title="(London set)", extra=extra,
                                     n_boot=args.boot, n_splits=args.splits, save_matrix=True)

    # Print the main results. Verification asks whether two photos show the same person. Its metrics are the AUC (the
    # area under the curve of true accepts against false accepts), the EER (the equal error rate, where false accepts
    # and false rejects are equally frequent) and the TAR (true accept rate) at a fixed FAR (false accept rate). The
    # closed-set results ask whether the correct person is the best match (rank 1) or among the five best matches
    # (rank 5). The open-set results use the FPIR (false-positive identification rate, the share of people outside the
    # gallery who are wrongly matched) and the FNIR (false-negative identification rate, the share of people in the
    # gallery who are not matched correctly).
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
