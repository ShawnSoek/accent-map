"""Feature extraction: normalization as transformers does it, pooling, strict loading, items,
and a job that resumes and splits across GPUs. Uses a tiny random wav2vec2, not the real one."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import training.extract_job as extract_job
from fakes import make_noise_folder, tiny_base_model
from training.augment import NoiseBank, load_augment_spec
from training.extract_job import compare, run
from training.feature_items import (SETS, make_items, read_split, read_upload_table, write_local_table,
                                    write_upload_table)
from training.features import FeatureExtractor, load_features_spec, normalize, pool, scale_vector

REPO = Path(__file__).resolve().parents[1]
SR = 16000


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return tiny_base_model(tmp_path_factory.mktemp("tiny_model"))


def test_normalize_matches_transformers():
    from transformers import Wav2Vec2FeatureExtractor
    x = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1 + 0.02
    expected = Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm([x], attention_mask=None)[0]
    assert np.allclose(normalize(x), expected, atol=1e-6)


def test_pool_is_mean_then_std_per_layer():
    h = tuple(torch.randn(1, 7, 4) for _ in range(3))
    out = pool(h)
    assert out.shape == (3, 8) and out.dtype == np.float16
    stacked = np.stack([t[0].numpy() for t in h])
    assert np.allclose(out[:, :4], stacked.mean(1), atol=2e-3)
    assert np.allclose(out[:, 4:], stacked.std(1), atol=2e-3)


def test_storage_scale_and_overflow():
    h = (torch.ones(1, 5, 4), torch.full((1, 5, 4), 40000.0), torch.full((1, 5, 4), 70000.0))
    with pytest.raises(OverflowError, match="layer 2 reaches 70000"):
        pool(h)
    out = pool(h, scale_vector(3, {1: 256, 2: 256}))
    assert out[1, 0] == 40000 / 256 and out[2, 0] == np.float16(70000 / 256) and out[0, 0] == 1
    with pytest.raises(ValueError, match="layer 5"):
        scale_vector(3, {5: 2})
    spec = load_features_spec(REPO / "training/cv27_es10/features.yaml")
    assert spec.storage_scale == {22: 256, 23: 256} and len(spec.files_sha256) == 3


def test_extractor_shape_and_determinism(model_dir):
    ex = FeatureExtractor(model_dir)
    audio = np.random.default_rng(1).standard_normal(2 * SR).astype(np.float32) * 0.1
    a, b = ex(audio), ex(audio)
    assert a.shape == (3, 32) and a.dtype == np.float16 and np.array_equal(a, b)
    assert not np.array_equal(a, ex(audio[::-1].copy()))
    assert np.array_equal(ex(audio), ex(audio * 3.0))          # normalized: loudness does not matter



def test_stopping_after_a_layer_keeps_its_values(model_dir):
    audio = np.random.default_rng(4).standard_normal(2 * SR).astype(np.float32) * 0.1
    full = FeatureExtractor(model_dir)(audio)
    short = FeatureExtractor(model_dir, storage_scale={0: 2, 2: 4}, last_layer=1)
    assert short.layers == 2 and len(short.model.encoder.layers) == 1
    got = short(audio)
    assert got.shape == (2, 32)
    np.testing.assert_array_equal(got[0], np.float16(full[0].astype(np.float32) / 2))     # scale kept for layer 0
    np.testing.assert_array_equal(got[1], full[1])
    naive = FeatureExtractor(model_dir).model                   # cutting without removing the final norm
    naive.encoder.layers = naive.encoder.layers[:1]
    with torch.inference_mode():
        states = naive(torch.from_numpy(normalize(audio))[None], output_hidden_states=True).hidden_states
    assert not np.allclose(pool(states)[1].astype(np.float32), full[1].astype(np.float32), atol=1e-2)
    assert FeatureExtractor(model_dir, last_layer=5).layers == 3            # beyond the model: nothing cut

def test_missing_weights_fail_loudly(model_dir, tmp_path):
    from safetensors.torch import load_file, save_file
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((model_dir / "config.json").read_text())
    weights = load_file(model_dir / "model.safetensors")
    weights.pop(next(k for k in weights if "encoder.layers.1" in k))
    save_file(weights, broken / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing"):
        FeatureExtractor(broken)


def split_rows():
    rows = {"train": [], "dev": [], "test": []}
    for name, speakers, per in (("train", 3, 2), ("dev", 2, 3), ("test", 2, 5)):
        for s in range(speakers):
            for c in range(per):
                rows[name].append({"path": f"{name}_{s}_{c}.wav", "speaker": f"{name}spk{s}", "label": f"L{s}",
                                   "sentence_id": f"x{c}", "seconds": "2.0", "gender": "f"})
    return rows


def test_items_and_tables(tmp_path):
    items = make_items(split_rows(), seed="s")
    by_set = {s: [it for it in items if it.set == s] for s in SETS}
    assert {s: len(v) for s, v in by_set.items()} == \
        {"train": 6, "train_aug": 6, "dev": 6, "test": 10, "dev_pairs": 2, "test_pairs": 4}   # odd clip left over
    assert [it.clips for it in by_set["train"]] == [it.clips for it in by_set["train_aug"]]
    assert all(it.augment for it in by_set["train_aug"]) and not any(it.augment for it in by_set["train"])
    for it in by_set["test_pairs"] + by_set["dev_pairs"]:
        assert len(it.clips) == 2 and len({c.split("_")[1] for c in it.clips}) == 1 and it.seconds == 4.0
    assert len({it.id for it in items}) == len(items)
    write_upload_table(items, tmp_path / "items.tsv")
    text = (tmp_path / "items.tsv").read_text()
    assert "spk" not in text and "\tL0" not in text and "\tf" not in text      # no speakers, labels, genders
    jobs = read_upload_table(tmp_path / "items.tsv")
    assert [(j.id, j.clips, j.augment) for j in jobs] == [(it.id, it.clips, it.augment) for it in items]
    write_local_table(items, tmp_path / "local.tsv")
    assert read_split(tmp_path / "local.tsv")[0]["speaker"] == "trainspk0"


@pytest.fixture
def small_bundle(tmp_path):
    """Clips as WAV files, the items table and a fake noise folder."""
    rng = np.random.default_rng(2)
    clips = tmp_path / "bundle" / "clips"
    clips.mkdir(parents=True)
    rows = split_rows()
    for name in rows:
        for r in rows[name]:
            sf.write(clips / r["path"], (0.1 * rng.standard_normal(int(0.6 * SR))).astype(np.float32), 48000)
    items = make_items(rows, seed="s")
    write_upload_table(items, tmp_path / "bundle" / "items.tsv")
    make_noise_folder(tmp_path / "bundle" / "noise")
    return tmp_path / "bundle"


def outputs(out_dir, set_name, parts=1):
    ids, rows = [], []
    for part in range(parts):
        stem = f"{set_name}.part{part}of{parts}"
        ids += (out_dir / f"{stem}.ids").read_text().split()
        rows.append(np.load(out_dir / f"{stem}.npy"))
    return ids, np.concatenate(rows)


EXTS = (".ids", ".npy", ".done")


def split_files(folder):
    return [p for p in folder.iterdir() if p.suffix in EXTS]


def test_job_resumes_and_splits_across_parts(model_dir, small_bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(extract_job, "SAVE_EVERY", 2)
    spec = load_augment_spec(REPO / "training/cv27_es10/augment.yaml")
    bank = NoiseBank(small_bundle / "noise", spec.reverb.simulated_rooms)
    jobs = read_upload_table(small_bundle / "items.tsv")
    ex = FeatureExtractor(model_dir)
    shape = (ex.layers, 2 * ex.width)
    clips = small_bundle / "clips"

    whole = tmp_path / "whole"
    filled = run(jobs, clips, whole, ex, shape, spec, bank, log=lambda m: None)
    assert filled == {"train": 6, "train_aug": 6, "dev": 6, "test": 10, "dev_pairs": 2, "test_pairs": 4}
    ids, rows = outputs(whole, "train_aug")
    clean_ids, clean = outputs(whole, "train")
    assert len(ids) == 6 and not np.array_equal(rows, clean)              # the copies really differ

    calls = {"n": 0}

    def flaky(audio):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt                                        # stops after 3 rows, 2 saved
        return ex(audio)

    resumed = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        run(jobs, clips, resumed, flaky, shape, spec, bank, log=lambda m: None)
    assert (resumed / "train.part0of1.done").read_text().strip() == "2"
    run(jobs, clips, resumed, ex, shape, spec, bank, log=lambda m: None)
    for s in SETS:
        (ids_r, rows_r), (ids_w, rows_w) = outputs(resumed, s), outputs(whole, s)
        assert ids_r == ids_w and np.array_equal(rows_r, rows_w)
    assert sorted(p.name for p in split_files(whole)) == sorted(f"{s}.part0of1{e}" for s in SETS for e in EXTS)

    split = tmp_path / "split"
    for part in (0, 1):
        run(jobs, clips, split, ex, shape, spec, bank, part=part, parts=2, log=lambda m: None)
    assert len(split_files(split)) == 2 * len(SETS) * len(EXTS)               # each part has its own files
    for s in SETS:
        ids2, rows2 = outputs(split, s, parts=2)
        ids1, rows1 = outputs(whole, s)
        order = [ids2.index(i) for i in ids1]
        assert sorted(ids2) == sorted(ids1) and np.array_equal(rows2[order], rows1)


def test_job_refuses_other_items_in_the_same_folder(model_dir, small_bundle, tmp_path):
    jobs = read_upload_table(small_bundle / "items.tsv")
    ex = FeatureExtractor(model_dir)
    out = tmp_path / "out"
    dev = [j for j in jobs if j.set == "dev"]
    run(dev, small_bundle / "clips", out, ex, (ex.layers, 2 * ex.width), log=lambda m: None)
    with pytest.raises(RuntimeError, match="other items"):
        run(dev[:2], small_bundle / "clips", out, ex, (ex.layers, 2 * ex.width), log=lambda m: None)


def test_compare():
    a = np.random.default_rng(3).standard_normal((4, 3, 8)).astype(np.float16)
    same = compare(a, a)
    assert same["min_cosine"] == pytest.approx(1.0, abs=1e-6) and same["max_relative_difference"] == 0.0
    b = a.astype(np.float32).copy()
    b[0, 0, 0] += 0.5
    assert compare(a, b)["max_relative_difference"] > 0.05
    with pytest.raises(ValueError, match="shapes"):
        compare(a, a[:2])
