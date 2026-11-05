"""Tests the identity evaluator on synthetic embeddings and a synthetic people folder.

The tests run offline on the CPU and use no real face image. The evaluator reports several standard measures.
TAR (true accept rate) is the share of same-person pairs that are accepted. FAR (false accept rate) is the share of
different-person pairs that are wrongly accepted. Closed-set identification looks a probe photo up in a gallery of
enrolled people, and open-set identification also includes strangers who are not enrolled. FPIR and FNIR are the
open-set false-positive and false-negative identification rates. Rank-1 is the share of probes whose top match is
the right person.
"""
import json

import cv2
import numpy as np
import pytest

from facekit import evaluate, metrics


def _clustered(n_people=20, per_person=2, dim=256, noise=0.05, seed=0):
    """Returns (embeddings, labels, views) for a made-up set of people.

    Each person is a random direction, and each photo of that person adds a little noise to it. In 256 dimensions two
    random directions have a cosine similarity near 0 (spread about 0.06). Different people therefore sit well below
    a threshold of 0.5 and the same person sits well above it. The first photo of each person has the view
    "neutral" and every later photo has the view "smiling".
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_people, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    emb, lab, views = [], [], []
    for p in range(n_people):
        for k in range(per_person):
            v = centres[p] + noise * rng.normal(size=dim)
            emb.append(v / np.linalg.norm(v))
            lab.append(f"p{p:03d}")
            views.append("neutral" if k == 0 else "smiling")
    return np.array(emb, dtype=np.float32), np.array(lab), np.array(views)


def test_pair_scores_uses_every_pair_once():
    emb, lab, _ = _clustered(n_people=5, per_person=2)
    scores, same, pairs = evaluate.pair_scores(emb, lab)
    n = len(lab)
    assert len(scores) == n * (n - 1) // 2

    # Each of the five people has two photos, so there are five same-person pairs.
    assert same.sum() == 5
    assert (pairs[:, 0] < pairs[:, 1]).all()

    # The score of one pair equals the dot product of its two unit-length embeddings.
    i, j = pairs[3]
    assert scores[3] == pytest.approx(float(emb[i] @ emb[j]), abs=1e-6)


def test_rates_at_threshold_counts_inclusively():
    """A score exactly equal to the threshold counts as accepted."""
    scores = np.array([0.9, 0.5, 0.2, 0.6, 0.1])
    same = np.array([1, 1, 1, 0, 0])
    r = evaluate.rates_at_threshold(scores, same, 0.5)
    assert r["tar"] == pytest.approx(2 / 3)
    assert r["far"] == pytest.approx(1 / 2)
    assert r["n_genuine"] == 3 and r["n_impostor"] == 2


def test_tar_at_far_agrees_with_the_roc_based_report():
    """The direct TAR-at-FAR calculation agrees with the ROC (receiver operating characteristic) report.

    The two TARs may differ by at most two of the 400 same-person scores, and the direct threshold must keep the FAR
    within the requested limit.
    """
    rng = np.random.default_rng(1)
    gen = rng.normal(0.6, 0.15, 400)
    imp = rng.normal(0.1, 0.1, 20000)
    scores = np.concatenate([gen, imp])
    same = np.concatenate([np.ones(400, int), np.zeros(20000, int)])
    roc = metrics.verification_report(scores, same, fars=(1e-2, 1e-3))
    for far in (1e-2, 1e-3):
        tar, threshold = evaluate.tar_at_far(gen, imp, far)
        assert tar == pytest.approx(roc[f"tar@far={far:g}"], abs=2 / 400)
        assert (imp >= threshold).mean() <= far + 1e-12


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    gen = rng.normal(0.5, 0.2, 100)
    imp = rng.normal(0.0, 0.1, 5000)
    point, _ = evaluate.tar_at_far(gen, imp, 1e-2)
    low, high = evaluate.bootstrap_tar_at_far(gen, imp, 1e-2, n_boot=300, seed=0)
    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0

    # The same seed gives the same interval.
    assert evaluate.bootstrap_tar_at_far(gen, imp, 1e-2, n_boot=300, seed=0) == (low, high)


def test_closed_set_is_perfect_on_well_separated_people():
    emb, lab, views = _clustered(noise=0.02)
    out = evaluate.closed_set(emb, lab, np.flatnonzero(views == "neutral"), np.flatnonzero(views == "smiling"))
    assert out["rank1"] == 1.0 and out["rank5"] == 1.0
    assert out["n_gallery"] == 20 and out["n_probes"] == 20 and out["probes_without_gallery_entry"] == 0


def test_closed_set_skips_probes_whose_person_is_not_enrolled():
    emb, lab, views = _clustered(n_people=4, noise=0.02)

    # The gallery holds the first three people only, so the fourth person has no gallery photo.
    gallery = np.flatnonzero(views == "neutral")[:3]
    out = evaluate.closed_set(emb, lab, gallery, np.flatnonzero(views == "smiling"))
    assert out["n_probes"] == 3 and out["probes_without_gallery_entry"] == 1


def test_open_set_extremes():
    """Open-set results are perfect at a good threshold and take the two extreme values at impossible thresholds."""
    emb, lab, views = _clustered(noise=0.02)

    # A threshold of 0.5 separates the well-spread people perfectly.
    good = evaluate.open_set_splits(emb, lab, views, threshold=0.5, n_splits=5, seed=0)
    assert good["fpir_mean"] == 0.0 and good["fnir_mean"] == 0.0 and good["rank1_mean"] == 1.0
    assert good["gallery_size"] == 10 and good["n_splits"] == 5

    # No cosine similarity can reach 1.01, so nobody is ever named and every enrolled person is missed.
    never = evaluate.open_set_splits(emb, lab, views, threshold=1.01, n_splits=5, seed=0)
    assert never["fpir_mean"] == 0.0 and never["fnir_mean"] == 1.0

    # Every cosine similarity is above -1.01, so every stranger is named as somebody.
    always = evaluate.open_set_splits(emb, lab, views, threshold=-1.01, n_splits=5, seed=0)
    assert always["fpir_mean"] == 1.0 and always["fnir_mean"] == 0.0


def test_open_set_strangers_are_never_enrolled():
    emb, lab, views = _clustered(n_people=10)
    out = evaluate.open_set_splits(emb, lab, views, threshold=0.5, n_splits=3, seed=4, keep_splits=True)
    for split in out["splits"]:
        assert not set(split["enrolled"]) & set(split["strangers"])
        assert len(split["enrolled"]) == 5 and len(split["strangers"]) == 5


def test_evaluate_embeddings_report_shape():
    emb, lab, views = _clustered(n_people=30)
    report = evaluate.evaluate_embeddings(emb, lab, views, threshold_1to1=0.37, threshold_1toN=0.52,
                                          n_boot=50, n_splits=4, seed=0)
    v = report["verification"]
    assert v["n_genuine"] == 30 and v["n_impostor"] == 60 * 59 // 2 - 30

    # The report covers the two fixed false accept rates of 0.01 and 0.001 only, so it has no entry for 0.0001.
    assert "tar@far=0.01" in v and "tar@far=0.01_ci95" in v and "tar@far=0.0001" not in v

    # The thresholds given to the evaluator are echoed back in the report.
    assert report["at_card_threshold_1to1"]["threshold"] == 0.37
    assert set(report["closed_set"]) == {"neutral_to_smiling", "smiling_to_neutral"}
    assert report["open_set_at_card_threshold_1toN"]["threshold"] == 0.52
    assert "indicative_threshold_1toN_fpir_1pct" in report

    # The report must be plain JSON, so that it can be saved to a file.
    json.dumps(report)


def test_evaluate_embeddings_without_views_uses_first_photo_as_gallery():
    emb, lab, _ = _clustered(n_people=12, per_person=3)
    report = evaluate.evaluate_embeddings(emb, lab, None, threshold_1to1=0.37, threshold_1toN=0.52,
                                          n_boot=20, n_splits=2, seed=0)
    assert set(report["closed_set"]) == {"first_photo_gallery"}
    assert report["closed_set"]["first_photo_gallery"]["n_probes"] == 24


def test_people_folder_records(tmp_path):
    """Only image files inside a person's own folder become records."""
    for person in ("alice", "bob"):
        (tmp_path / person).mkdir()
        for k in range(2):
            cv2.imwrite(str(tmp_path / person / f"{k}.jpg"), np.zeros((8, 8, 3), np.uint8))

    # Three kinds of file must be ignored: a macOS resource-fork file, a text file and an image outside any person's
    # folder.
    (tmp_path / "alice" / "._0.jpg").write_bytes(b"mac resource fork")
    (tmp_path / "alice" / "notes.txt").write_text("not an image")
    (tmp_path / "loose.jpg").write_bytes(b"not in a person folder")
    records = evaluate.people_folder_records(tmp_path)
    assert [(r["person"], r["view"]) for r in records] == [("alice", ""), ("alice", ""), ("bob", ""), ("bob", "")]


def test_embed_records_counts_failures(tmp_path):
    """Photos with no face and unreadable files are listed as failures and left out of the embeddings."""
    good, blank = tmp_path / "good.jpg", tmp_path / "blank.jpg"
    cv2.imwrite(str(good), np.full((8, 8, 3), 200, np.uint8))
    cv2.imwrite(str(blank), np.zeros((8, 8, 3), np.uint8))
    records = [{"path": str(good), "person": "a", "view": "neutral"},
               {"path": str(blank), "person": "a", "view": "smiling"},
               {"path": str(tmp_path / "missing.jpg"), "person": "b", "view": "neutral"}]

    def embed(image):
        """Pretends to find a face only in bright images, and finds none in dark ones."""
        return np.ones(4, np.float32) if image.mean() > 100 else None

    out = evaluate.embed_records(embed, records)
    assert out.embeddings.shape == (1, 4) and list(out.labels) == ["a"] and list(out.views) == ["neutral"]
    assert sorted(f["reason"] for f in out.failed) == ["no face", "unreadable"]


# Tests added after an independent review. Each one fails for a specific plausible bug.


def test_tar_at_far_exact_when_far_times_n_is_not_an_integer():
    # All impostor scores are different, and 1e-2 * 2050 = 20.5, so the 20 highest impostors may be accepted.
    imp = np.linspace(0.0, 1.0, 2050)

    # The 20th highest impostor score is accepted and the 21st is not.
    gen = np.array([imp[-20], imp[-21], imp[-22], 1.0])
    tar, threshold = evaluate.tar_at_far(gen, imp, 1e-2)

    # The count is 20.5 rounded down to 20. Rounding up would give 21, and an off-by-one error would give 19.
    assert int((imp >= threshold).sum()) == 20
    assert tar == pytest.approx(2 / 4)


def test_tar_at_far_rejects_ties_at_the_boundary():
    # Two impostors may be accepted, but three impostors tie at 0.9.
    imp = np.array([0.9] * 3 + [0.1] * 97)
    tar, threshold = evaluate.tar_at_far(np.array([0.95, 0.9, 0.5]), imp, 0.02)

    # Accepting the tie would give a FAR of 0.03, so the tied scores must all be rejected.
    assert (imp >= threshold).mean() <= 0.02
    assert tar == pytest.approx(1 / 3)


def _confusable():
    """Returns (embeddings, labels, views) for three people in a small made-up case.

    People A and B are enrolled, and B's smiling photo looks like A, which causes a misnaming. Person C is a stranger
    whose smiling photo looks like A, which causes a false name, and whose neutral photo looks like nobody.
    """
    def unit(v):
        """Returns the vector v scaled to unit length."""
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    # The rows come in pairs: neutral then smiling for A, then for B, then for C. The smiling photo of B is closer to
    # A than to B, and the smiling photo of C has a cosine similarity of 0.62 to A.
    emb = np.stack([unit([1, 0, 0]), unit([1, 0, 0]),
                    unit([0, 1, 0]), unit([1, 0.1, 0]),
                    unit([0, 0, 1]), unit([0.8, 0, 1])])
    labels = np.array(["A", "A", "B", "B", "C", "C"])
    views = np.array(["neutral", "smiling"] * 3)
    return emb, labels, views


def test_open_set_once_counts_misnames_strangers_and_probes_exactly():
    emb, labels, views = _confusable()
    out = evaluate.open_set_once(emb, labels, views, ["A", "B"], ["C"], threshold=0.5)

    # The enrolment photos are never used as probes, so only the two smiling photos of A and B are genuine probes.
    assert out["n_genuine_probes"] == 2

    # Both photos of the stranger C are probes.
    assert out["n_stranger_probes"] == 2

    # B is misnamed as A, which counts as a miss.
    assert out["fnir"] == pytest.approx(0.5)
    assert out["rank1"] == pytest.approx(0.5)

    # The smiling photo of C is named A, but the neutral photo of C is not named at all.
    assert out["fpir"] == pytest.approx(0.5)


def test_bootstrap_interval_is_a_95_percent_interval():
    rng = np.random.default_rng(3)

    # The genuine scores are shifted so that the TAR is near 0.5. A 95% interval is then about 0.1 either side.
    gen = rng.uniform(0, 1, 100)
    imp = rng.uniform(-1, 0, 5000)
    low, high = evaluate.bootstrap_tar_at_far(gen - 0.5, imp, 1e-2, n_boot=500, seed=0)

    # A 50% interval would be only about 0.07 wide, so a width above 0.12 shows that the interval is the 95% one.
    assert high - low > 0.12


def test_bootstrap_resamples_impostors_too():
    # Every genuine score has the same value, and the threshold for a FAR of 0.5 lands near that value.
    gen = np.full(50, 0.5)
    imp = np.random.default_rng(4).uniform(0, 1, 200)
    low, high = evaluate.bootstrap_tar_at_far(gen, imp, 0.5, n_boot=200, seed=0)

    # If the impostors were not resampled, every repetition would give the same TAR and the interval would have zero
    # width.
    assert low < high


def test_evaluate_needs_two_people_with_two_photos():
    emb, lab, _ = _clustered(n_people=6, per_person=1)
    with pytest.raises(ValueError, match="two photos"):
        evaluate.evaluate_embeddings(emb, lab, None, 0.37, 0.52, n_boot=10, n_splits=2)


def test_private_folder_guard(tmp_path):
    """The evaluator refuses to write into folders that a cloud service synchronises, and allows plain local ones."""
    # The synchronised folders are built under tmp_path, so that no home folder is written into the repository.
    for bad in (tmp_path / "OneDrive" / "photos", tmp_path / "OneDrive - Contoso" / "photos", "G:/My Drive/photos",
                tmp_path / "Google Drive" / "photos", tmp_path / "Dropbox" / "photos"):
        with pytest.raises(SystemExit):
            evaluate.refuse_synced_folder(bad)

    # A plain local folder is allowed.
    evaluate.refuse_synced_folder("C:/tmp/fdc-london")


def test_run_evaluation_keeps_only_score_vectors_unless_asked(tmp_path):
    emb, lab, views = _clustered(n_people=8)

    # Each fake photo is an image filled with its own index, so the fake embedding function can look up which photo it
    # was given and return the matching embedding.
    records = [{"path": f"p{i}.jpg", "person": l, "view": v} for i, (l, v) in enumerate(zip(lab, views))]
    lookup = {r["path"]: e for r, e in zip(records, emb)}
    images = {r["path"]: np.full((4, 4, 3), i, np.uint8) for i, r in enumerate(records)}
    inverse = {images[p].tobytes(): p for p in images}

    def read(path):
        """Returns the fake image that belongs to the given path."""
        return images[str(path)]

    def embed(image):
        """Returns the prepared embedding of the fake image."""
        return lookup[inverse[image.tobytes()]]

    report = evaluate.run_evaluation(embed, records, 0.37, 0.52, tmp_path / "a", n_boot=20, n_splits=2,
                                     read_fn=read)

    # The full similarity matrix is not saved by default, because it would preserve the embeddings.
    with np.load(tmp_path / "a" / "scores.npz") as saved:
        assert set(saved.files) == {"genuine", "impostor"}
    assert report["settings"]["n_boot"] == 20 and report["settings"]["n_splits"] == 2

    # With save_matrix switched on, the similarity matrix is saved as well.
    evaluate.run_evaluation(embed, records, 0.37, 0.52, tmp_path / "b", n_boot=20, n_splits=2, read_fn=read,
                            save_matrix=True)
    with np.load(tmp_path / "b" / "scores.npz") as saved:
        assert "similarity" in saved.files
