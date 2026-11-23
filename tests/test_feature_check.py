"""Checking the features downloaded from the GPU job: what the real extraction job writes (in two
parts, as on Kaggle's two GPUs) passes, and each kind of damage is caught."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fakes import make_noise_folder
from training.archives import sha256_of
from training.augment import NoiseBank, load_augment_spec
from training.extract_job import run
from training.feature_check import FeatureCheckError, check_output
from training.feature_items import SETS, Job, make_items

REPO = Path(__file__).resolve().parents[1]
SHAPE = (3, 4)
BUNDLE = "b" * 64


def fake_extractor(audio):
    """Stands in for the base model: depends on the audio and is never all zeros."""
    level = np.float32(np.abs(audio).mean() + 0.5)
    return level * np.arange(1, SHAPE[0] * SHAPE[1] + 1, dtype=np.float32).reshape(SHAPE)


def write_summary(out, bundle=BUNDLE):
    files = {p.name: sha256_of(p) for p in sorted(out.iterdir()) if p.suffix in (".npy", ".ids")}
    (out / "summary.json").write_text(json.dumps({
        "manifest": {"bundle_sha256": bundle}, "environment": {"gpus": ["fake"]},
        "reference_check": {}, "run_minutes": 1.0, "total_minutes": 2.0, "files_sha256": files}))


def read_rows(out):
    ids, rows = [], []
    for s in SETS:
        if (out / f"{s}.part0of1.ids").exists():
            ids += (out / f"{s}.part0of1.ids").read_text().split()
            rows.append(np.load(out / f"{s}.part0of1.npy"))
    return ids, np.concatenate(rows)


@pytest.fixture
def job_output(tmp_path):
    rng = np.random.default_rng(3)
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = {"train": [], "dev": [], "test": []}
    for name, n in (("train", 5), ("dev", 4), ("test", 5)):
        for c in range(n):
            path = f"{name}{c}.wav"
            sf.write(clips / path, (0.05 * (c + 1) * rng.standard_normal(8000)).astype(np.float32), 16000)
            rows[name].append({"path": path, "speaker": f"{name}s{c % 2}", "label": "a", "seconds": "0.5",
                               "gender": "f"})
    items = make_items(rows, "seed")
    spec = load_augment_spec(REPO / "training/cv27_es10/augment.yaml")
    make_noise_folder(tmp_path / "noise")
    bank = NoiseBank(tmp_path / "noise", spec.reverb.simulated_rooms)
    jobs = [Job(it.id, it.set, it.clips, it.augment) for it in items]
    out = tmp_path / "out"
    for part in range(2):
        run(jobs, clips, out, fake_extractor, SHAPE, spec, bank, part=part, parts=2, log=lambda m: None)
    write_summary(out)
    ref_jobs = [next(j for j in jobs if j.set == s) for s in SETS] + [jobs[3]]   # one of each set, one more
    run(ref_jobs, clips, tmp_path / "ref", fake_extractor, SHAPE, spec, bank, log=lambda m: None)
    expected = {s: [it.id for it in items if it.set == s] for s in SETS}
    return out, expected, read_rows(tmp_path / "ref")


def check(out, expected, reference, bundle=BUNDLE):
    return check_output(out, expected, bundle, SHAPE, reference, chunk=2, log=lambda m: None)


def rewrite(out, name, change):
    """Change an output array and update summary.json, as if the job itself had written it so."""
    rows = np.load(out / name)
    np.save(out / name, change(rows))
    write_summary(out)


def test_what_the_job_writes_passes(job_output):
    out, expected, reference = job_output
    report = check(out, expected, reference)
    assert report["sets"] == {"train": 5, "train_aug": 5, "dev": 4, "test": 5, "dev_pairs": 2, "test_pairs": 2}
    assert report["items"] == 23 and report["parts"] == {s: 2 for s in SETS}
    assert report["reference"]["min_cosine"] > 0.9999 and report["largest_stored_value"]["layer"] == 2
    assert report["kaggle"]["environment"] == {"gpus": ["fake"]}


def test_results_of_another_bundle_are_refused(job_output):
    with pytest.raises(FeatureCheckError, match="another bundle"):
        check(*job_output, bundle="c" * 64)


def test_a_damaged_download_is_refused(job_output):
    out, expected, reference = job_output
    data = bytearray((out / "dev.part1of2.npy").read_bytes())
    data[-1] ^= 1
    (out / "dev.part1of2.npy").write_bytes(bytes(data))
    with pytest.raises(FeatureCheckError, match="dev.part1of2.npy has the wrong checksum"):
        check(out, expected, reference)


@pytest.mark.parametrize("name, message", [("test.part1of2.npy", "files differ from summary.json"),
                                           ("test.part1of2.done", "missing")])
def test_a_missing_file_is_refused(job_output, name, message):
    out, expected, reference = job_output
    (out / name).unlink()
    with pytest.raises(FeatureCheckError, match=message):
        check(out, expected, reference)


def test_items_in_the_wrong_part_or_order_are_refused(job_output):
    out, expected, reference = job_output
    ids = (out / "train.part0of2.ids").read_text().split()
    (out / "train.part0of2.ids").write_text("".join(f"{i}\n" for i in reversed(ids)))
    write_summary(out)
    with pytest.raises(FeatureCheckError, match="in order"):
        check(out, expected, reference)


def test_an_item_the_table_lacks_is_refused(job_output):
    out, expected, reference = job_output
    expected["test"] = expected["test"][:-1]
    with pytest.raises(FeatureCheckError, match="in order"):
        check(out, expected, reference)


def test_unfinished_parts_are_refused(job_output):
    out, expected, reference = job_output
    (out / "test_pairs.part0of2.done").write_text("0\n")
    with pytest.raises(FeatureCheckError, match="0 of 1 rows written"):
        check(out, expected, reference)


@pytest.mark.parametrize("change, message", [
    (lambda r: np.concatenate([r[:1], np.zeros_like(r[1:2]), r[2:]]), "row 1 is all zeros"),
    (lambda r: np.where(np.arange(r.size).reshape(r.shape) == 5, np.float16(np.inf), r), "not finite"),
    (lambda r: r.astype(np.float32), "expected float16"),
    (lambda r: r[:, :, :2], "expected float16"),
])
def test_bad_rows_are_refused(job_output, change, message):
    out, expected, reference = job_output
    rewrite(out, "train_aug.part1of2.npy", change)
    with pytest.raises(FeatureCheckError, match=message):
        check(out, expected, reference)


def test_rows_that_differ_from_the_laptop_are_refused(job_output):
    out, expected, (ref_ids, ref_rows) = job_output
    swapped = ref_rows.copy()
    swapped[[0, 1]] = ref_rows[[1, 0]]            # results of two items swapped
    swapped[0] *= np.linspace(0.5, 1.5, SHAPE[1], dtype=np.float16)
    with pytest.raises(FeatureCheckError, match="differs from the laptop"):
        check(out, expected, (ref_ids, swapped))
    with pytest.raises(FeatureCheckError, match="missing from the output"):
        check(out, expected, (ref_ids + ["train/999999"], np.concatenate([ref_rows, ref_rows[:1]])))
