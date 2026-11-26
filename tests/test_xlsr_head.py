"""The stage-2 backend: the frozen base model up to the classifier's layer, then the exported
classifier. Fast tests use a tiny random wav2vec2 and classifier; `pytest -m model` checks the
real model against the evaluation's own results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fakes import tiny_base_model
from server.model_api import ModelFilesMissing, load_model
from server.paths import DATA_DIR, MODELS_DIR
from training.features import FeatureExtractor
from training.head import AccentHead, HeadConfig, TrainedHead, export_head, save_head

REV = "0123456789abcdef0123456789abcdef01234567"
HEAD_ORDER = ["c2", "c0", "c1"]                  # the classifier's output order; the config lists c0, c1, c2


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("xlsr_head")
    snap = root / "cache" / "models--test--tiny-xlsr" / "snapshots" / REV
    tiny_base_model(snap)
    (snap / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000, "do_normalize": True}))
    torch.manual_seed(3)
    head = AccentHead(32, 3)
    rng = np.random.default_rng(3)
    mean, std = rng.normal(size=(3, 32)).astype(np.float32), rng.uniform(0.5, 2, size=(3, 32)).astype(np.float32)
    save_head(root / "head.pt", head, HEAD_ORDER, [1], mean, std, HeadConfig(), temperature=1.5)
    model_dir = root / "model"
    model_dir.mkdir()
    export_head(TrainedHead(root / "head.pt"), model_dir / "head.safetensors", {}, {"trained_on": "test"})
    config = {
        "name": "tiny", "display_name": "Tiny", "backend": "xlsr_pooled_head",
        "backend_options": {"base_model": {"repo": "test/tiny-xlsr", "revision": REV,
                                           "files": ["config.json", "preprocessor_config.json", "model.safetensors"]},
                            "head_file": "head.safetensors",
                            "head_sha256": hashlib.sha256((model_dir / "head.safetensors").read_bytes()).hexdigest()},
        "classes": [{"id": c, "model_label": c, "label": c.upper(), "zone": c} for c in ("c0", "c1", "c2")]}
    (model_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return root, snap, model_dir, config


def test_contract_and_same_answers_as_the_pieces(setup):
    root, snap, model_dir, _ = setup
    model = load_model(model_dir, cache_dir=root / "cache")
    assert model.model_labels == HEAD_ORDER and len(model.extractor.model.encoder.layers) == 1   # only up to layer 1
    rng = np.random.default_rng(0)
    for seconds in (0.5, 1.0, 3.0):
        audio = (0.1 * rng.standard_normal(int(seconds * 16000))).astype(np.float32)
        probs = model.predict(audio)
        assert list(probs) == ["c0", "c1", "c2"] and sum(probs.values()) == pytest.approx(1, abs=1e-6)
        full = FeatureExtractor(snap)(audio).astype(np.float32)                   # all layers, nothing cut
        by_label = dict(zip(HEAD_ORDER, TrainedHead(model_dir / "head.safetensors").probabilities(full[None])[0]))
        np.testing.assert_allclose([probs[c] for c in ("c0", "c1", "c2")], [by_label[c] for c in ("c0", "c1", "c2")],
                                   atol=1e-6)


def test_a_changed_head_file_is_refused(setup, tmp_path):
    root, _, model_dir, config = setup
    bad = dict(config, backend_options=dict(config["backend_options"], head_sha256="0" * 64))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(bad))
    (tmp_path / "head.safetensors").write_bytes((model_dir / "head.safetensors").read_bytes())
    with pytest.raises(ValueError, match="config.yaml pins"):
        load_model(tmp_path, cache_dir=root / "cache")


def test_a_missing_base_model_says_how_to_fetch(setup, tmp_path):
    root, _, model_dir, config = setup
    other = dict(config, backend_options=dict(config["backend_options"],
                                              base_model=dict(config["backend_options"]["base_model"], revision="f" * 40)))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(other))
    (tmp_path / "head.safetensors").write_bytes((model_dir / "head.safetensors").read_bytes())
    with pytest.raises(ModelFilesMissing, match="fetch_model.py"):
        load_model(tmp_path, cache_dir=root / "cache")


# --- The real stage-2 model (pytest -m model) -----------------------------------------

@pytest.mark.model
def test_real_model_gives_the_evaluations_answers():
    """From raw audio, the app's backend must give what the evaluation computed from the GPU's
    stored features for the same dev clips (they differ only by CPU vs GPU arithmetic)."""
    from server.audio import decode_file
    from training.cv_split import load_spec
    from training.feature_store import open_set

    features = DATA_DIR / "features" / "cv27_es10"
    table = DATA_DIR / "kaggle" / "cv27_es10" / "local" / "items_local.tsv"
    release = load_spec(Path("training/cv27_es10/dataset.yaml")).release
    if not features.exists() or not table.exists():
        pytest.skip("stage-2 features or item table not on this computer")
    try:
        model = load_model(MODELS_DIR / "cv27_es10")
    except ModelFilesMissing as e:
        pytest.skip(str(e))
    clips = next((DATA_DIR / "commonvoice" / release).rglob("validated.tsv")).parent / "clips"
    dev = open_set(features, "dev", {22: 256, 23: 256})
    ids, rows = dev.read(0, 0, 12)                                # the first 12 dev clips of part 0
    clip_of = {}
    with open(table, encoding="utf-8") as f:
        for line in f:
            item, _, _, clip, *_ = line.split("\t")
            clip_of[item] = clip
    stored = model.head.probabilities(rows)                        # in the head's order
    for item, want in zip(ids, stored):
        got = model.predict(decode_file(clips / clip_of[item], 16000))
        got = np.array([got[label] for label in model.model_labels])
        np.testing.assert_allclose(got, want, atol=2e-3, err_msg=item)
