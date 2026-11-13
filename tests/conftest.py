from pathlib import Path

import pytest

from server.paths import MODELS_DIR, hf_cache_dir

REAL_MODEL_DIR = MODELS_DIR / "commonaccent_es6"


@pytest.fixture(scope="session")
def real_model():
    """The stage-1 model, loaded once per test session (~8 s, ~2.7 GB RAM)."""
    from server.model_api import ModelFilesMissing, load_model

    try:
        return load_model(REAL_MODEL_DIR)
    except ModelFilesMissing as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def sample_clips(real_model) -> dict[str, Path]:
    """The 4 example clips shipped in the model repo at the pinned revision."""
    src = real_model.config.source
    data = hf_cache_dir() / f"models--{src.repo.replace('/', '--')}" / "snapshots" / src.revision / "data"
    return {p.name: p for p in sorted(data.glob("*.wav"))}
