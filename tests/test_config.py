import copy

import pytest
import yaml

from conftest import REAL_MODEL_DIR
from server.config import load_config


@pytest.fixture
def raw_config() -> dict:
    with open(REAL_MODEL_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write(tmp_path, raw):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


def test_stage1_config_is_valid():
    config = load_config(REAL_MODEL_DIR)
    assert config.name == "commonaccent_es6"
    assert len(config.classes) == 6
    assert config.source.revision == "0eff6f06bdbbcd75cafe3260c00a93985ca36d2d"
    assert config.backend_options["base_model"]["revision"] == "c3f9d884181a224a6ac87bf8885c84d1cff3384f"
    assert config.audio.sample_rate == 16000
    assert config.model_dir == REAL_MODEL_DIR.resolve()
    assert (config.model_dir / config.backend_options["hparams_file"]).is_file()


def test_accepts_file_or_directory():
    assert load_config(REAL_MODEL_DIR) == load_config(REAL_MODEL_DIR / "config.yaml")


def _dup_id(c):
    c["classes"][1]["id"] = c["classes"][0]["id"]


def _dup_label(c):
    c["classes"][1]["model_label"] = c["classes"][0]["model_label"]


def _one_class(c):
    c["classes"] = c["classes"][:1]


@pytest.mark.parametrize("mutate, message", [
    (_dup_id, "duplicate class id"),
    (_dup_label, "duplicate class model_label"),
    (_one_class, "at least 2"),
    (lambda c: c.update(clases=c.pop("classes")), "clases"),               # typo in a key
    (lambda c: c["source"].update(revision="main"), "revision"),           # branch, not a pinned SHA
    (lambda c: c["audio"].update(hop_seconds=6.0), "hop_seconds"),
    (lambda c: c["classes"][0].update(id="Southern Spain"), "id"),
    (lambda c: c.update(model_dir="/tmp"), "model_dir"),
])
def test_rejects_invalid_config(tmp_path, raw_config, mutate, message):
    raw = copy.deepcopy(raw_config)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, raw))
