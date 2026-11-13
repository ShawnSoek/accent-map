"""Well-known locations. Everything downloaded lives under data/ (not committed)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"


def hf_cache_dir() -> Path:
    """Hugging Face cache for model snapshots. Override with ACCENT_MAP_HF_CACHE."""
    return Path(os.environ.get("ACCENT_MAP_HF_CACHE", DATA_DIR / "hf_cache"))
