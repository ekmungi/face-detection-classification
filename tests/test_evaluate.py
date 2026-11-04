"""The identity evaluator on synthetic embeddings and a synthetic people folder (offline, CPU, no face image)."""
import json

import cv2
import numpy as np
import pytest

from facekit import evaluate, metrics


def _clustered(n_people=20, per_person=2, dim=256, noise=0.05, seed=0):
    """Returns (embeddings, labels, views): each person is a random direction plus small noise per photo.

    In 256 dimensions random directions have cosine similarity near 0 (spread about 0.06), so different people sit
    well below a 0.5 threshold and the same person well above it.
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
    assert same.sum() == 5                                   # one genuine pair per person with two photos
    assert (pairs[:, 0] < pairs[:, 1]).all()
    i, j = pairs[3]
    assert scores[3] == pytest.approx(float(emb[i] @ emb[j]), abs=1e-6)


def test_rates_at_threshold_counts_inclusively():
    scores = np.array([0.9, 0.5, 0.2, 0.6, 0.1])
    same = np.array([1, 1, 1, 0, 0])
    r = evaluate.rates_at_threshold(scores, same, 0.5)
    assert r["tar"] == pytest.approx(2 / 3)
    assert r["far"] == pytest.approx(1 / 2)
    assert r["n_genuine"] == 3 and r["n_impostor"] == 2


def test_tar_at_far_agrees_with_the_roc_based_report():
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
    assert evaluate.bootstrap_tar_at_far(gen, imp, 1e-2, n_boot=300, seed=0) == (low, high)   # fixed seed repeats


def test_closed_set_is_perfect_on_well_separated_people():
    emb, lab, views = _clustered(noise=0.02)
    out = evaluate.closed_set(emb, lab, np.flatnonzero(views == "neutral"), np.flatnonzero(views == "smiling"))
    assert out["rank1"] == 1.0 and out["rank5"] == 1.0
    assert out["n_gallery"] == 20 and out["n_probes"] == 20 and out["probes_without_gallery_entry"] == 0


def test_closed_set_skips_probes_whose_person_is_not_enrolled():
    emb, lab, views = _clustered(n_people=4, noise=0.02)
    gallery = np.flatnonzero(views == "neutral")[:3]         # person 3 has no gallery photo
    out = evaluate.closed_set(emb, lab, gallery, np.flatnonzero(views == "smiling"))
    assert out["n_probes"] == 3 and out["probes_without_gallery_entry"] == 1


def test_open_set_extremes():
    emb, lab, views = _clustered(noise=0.02)
    good = evaluate.open_set_splits(emb, lab, views, threshold=0.5, n_splits=5, seed=0)
    assert good["fpir_mean"] == 0.0 and good["fnir_mean"] == 0.0 and good["rank1_mean"] == 1.0
    assert good["gallery_size"] == 10 and good["n_splits"] == 5
    never = evaluate.open_set_splits(emb, lab, views, threshold=1.01, n_splits=5, seed=0)
    assert never["fpir_mean"] == 0.0 and never["fnir_mean"] == 1.0
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
    assert "tar@far=0.01" in v and "tar@far=0.01_ci95" in v and "tar@far=0.0001" not in v
    assert report["at_card_threshold_1to1"]["threshold"] == 0.37
    assert set(report["closed_set"]) == {"neutral_to_smiling", "smiling_to_neutral"}
    assert report["open_set_at_card_threshold_1toN"]["threshold"] == 0.52
    assert "indicative_threshold_1toN_fpir_1pct" in report
    json.dumps(report)                                       # the report must be plain JSON


def test_evaluate_embeddings_without_views_uses_first_photo_as_gallery():
    emb, lab, _ = _clustered(n_people=12, per_person=3)
    report = evaluate.evaluate_embeddings(emb, lab, None, threshold_1to1=0.37, threshold_1toN=0.52,
                                          n_boot=20, n_splits=2, seed=0)
    assert set(report["closed_set"]) == {"first_photo_gallery"}
    assert report["closed_set"]["first_photo_gallery"]["n_probes"] == 24


def test_people_folder_records(tmp_path):
    for person in ("alice", "bob"):
        (tmp_path / person).mkdir()
        for k in range(2):
            cv2.imwrite(str(tmp_path / person / f"{k}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "alice" / "._0.jpg").write_bytes(b"mac resource fork")
    (tmp_path / "alice" / "notes.txt").write_text("not an image")
    (tmp_path / "loose.jpg").write_bytes(b"not in a person folder")
    records = evaluate.people_folder_records(tmp_path)
    assert [(r["person"], r["view"]) for r in records] == [("alice", ""), ("alice", ""), ("bob", ""), ("bob", "")]


def test_embed_records_counts_failures(tmp_path):
    good, blank = tmp_path / "good.jpg", tmp_path / "blank.jpg"
    cv2.imwrite(str(good), np.full((8, 8, 3), 200, np.uint8))
    cv2.imwrite(str(blank), np.zeros((8, 8, 3), np.uint8))
    records = [{"path": str(good), "person": "a", "view": "neutral"},
               {"path": str(blank), "person": "a", "view": "smiling"},
               {"path": str(tmp_path / "missing.jpg"), "person": "b", "view": "neutral"}]

    def embed(image):                                        # "detects" a face only in bright images
        return np.ones(4, np.float32) if image.mean() > 100 else None

    out = evaluate.embed_records(embed, records)
    assert out.embeddings.shape == (1, 4) and list(out.labels) == ["a"] and list(out.views) == ["neutral"]
    assert sorted(f["reason"] for f in out.failed) == ["no face", "unreadable"]


# --- Tests added after the independent review: each one fails for a specific plausible bug. ---------------------


def test_tar_at_far_exact_when_far_times_n_is_not_an_integer():
    imp = np.linspace(0.0, 1.0, 2050)                        # distinct scores; 1e-2 * 2050 = 20.5, so k = 20
    gen = np.array([imp[-20], imp[-21], imp[-22], 1.0])      # the 20th highest impostor is accepted, the 21st not
    tar, threshold = evaluate.tar_at_far(gen, imp, 1e-2)
    assert int((imp >= threshold).sum()) == 20               # floor, not ceil (21) and not k - 1 (19)
    assert tar == pytest.approx(2 / 4)


def test_tar_at_far_rejects_ties_at_the_boundary():
    imp = np.array([0.9] * 3 + [0.1] * 97)                   # k = 2, but three impostors tie at 0.9
    tar, threshold = evaluate.tar_at_far(np.array([0.95, 0.9, 0.5]), imp, 0.02)
    assert (imp >= threshold).mean() <= 0.02                 # accepting the tie would give FAR 0.03
    assert tar == pytest.approx(1 / 3)


def _confusable():
    """A and B are enrolled; B's smiling photo looks like A (a misname). C is a stranger whose smiling photo looks
    like A (a false name) and whose neutral photo looks like nobody."""
    def unit(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)
    emb = np.stack([unit([1, 0, 0]), unit([1, 0, 0]),              # A neutral, A smiling
                    unit([0, 1, 0]), unit([1, 0.1, 0]),            # B neutral, B smiling (closer to A)
                    unit([0, 0, 1]), unit([0.8, 0, 1])])           # C neutral, C smiling (0.62 to A)
    labels = np.array(["A", "A", "B", "B", "C", "C"])
    views = np.array(["neutral", "smiling"] * 3)
    return emb, labels, views


def test_open_set_once_counts_misnames_strangers_and_probes_exactly():
    emb, labels, views = _confusable()
    out = evaluate.open_set_once(emb, labels, views, ["A", "B"], ["C"], threshold=0.5)
    assert out["n_genuine_probes"] == 2                      # the enrolment photos are never probes
    assert out["n_stranger_probes"] == 2                     # both photos of every stranger
    assert out["fnir"] == pytest.approx(0.5)                 # B is misnamed as A, which counts as a miss
    assert out["rank1"] == pytest.approx(0.5)
    assert out["fpir"] == pytest.approx(0.5)                 # C's smiling photo is named A, the neutral one is not


def test_bootstrap_interval_is_a_95_percent_interval():
    rng = np.random.default_rng(3)
    gen = rng.uniform(0, 1, 100)                             # TAR near 0.5: a 95% interval is about +-0.1
    imp = rng.uniform(-1, 0, 5000)
    low, high = evaluate.bootstrap_tar_at_far(gen - 0.5, imp, 1e-2, n_boot=500, seed=0)
    assert high - low > 0.12                                 # a 50% interval would be about 0.07 wide


def test_bootstrap_resamples_impostors_too():
    gen = np.full(50, 0.5)                                   # every genuine score sits at the same value
    imp = np.random.default_rng(4).uniform(0, 1, 200)        # the FAR 0.5 threshold lands near 0.5
    low, high = evaluate.bootstrap_tar_at_far(gen, imp, 0.5, n_boot=200, seed=0)
    assert low < high                                        # fixed impostors would give one TAR every time


def test_evaluate_needs_two_people_with_two_photos():
    emb, lab, _ = _clustered(n_people=6, per_person=1)
    with pytest.raises(ValueError, match="two photos"):
        evaluate.evaluate_embeddings(emb, lab, None, 0.37, 0.52, n_boot=10, n_splits=2)


def test_private_folder_guard():
    for bad in ("C:/Users/x/OneDrive/photos", "C:/Users/x/OneDrive - Contoso/photos", "G:/My Drive/photos",
                "C:/Users/x/Google Drive/photos", "C:/Users/x/Dropbox/photos"):
        with pytest.raises(SystemExit):
            evaluate.refuse_synced_folder(bad)
    evaluate.refuse_synced_folder("C:/tmp/fdc-london")       # a plain local folder is fine


def test_run_evaluation_keeps_only_score_vectors_unless_asked(tmp_path):
    emb, lab, views = _clustered(n_people=8)
    records = [{"path": f"p{i}.jpg", "person": l, "view": v} for i, (l, v) in enumerate(zip(lab, views))]
    lookup = {r["path"]: e for r, e in zip(records, emb)}
    images = {r["path"]: np.full((4, 4, 3), i, np.uint8) for i, r in enumerate(records)}
    inverse = {images[p].tobytes(): p for p in images}

    def read(path):
        return images[str(path)]

    def embed(image):
        return lookup[inverse[image.tobytes()]]

    report = evaluate.run_evaluation(embed, records, 0.37, 0.52, tmp_path / "a", n_boot=20, n_splits=2,
                                     read_fn=read)
    with np.load(tmp_path / "a" / "scores.npz") as saved:
        assert set(saved.files) == {"genuine", "impostor"}  # no matrix: it would preserve the embeddings
    assert report["settings"]["n_boot"] == 20 and report["settings"]["n_splits"] == 2
    evaluate.run_evaluation(embed, records, 0.37, 0.52, tmp_path / "b", n_boot=20, n_splits=2, read_fn=read,
                            save_matrix=True)
    with np.load(tmp_path / "b" / "scores.npz") as saved:
        assert "similarity" in saved.files
