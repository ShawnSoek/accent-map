"""Pinned Hugging Face files in the local cache, shared by the backends. Nothing is downloaded
here; scripts/fetch_model.py does that."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import Field

from server.config import SourceInfo
from server.model_api import ModelFilesMissing


class BaseModelSpec(SourceInfo):
    files: list[str] = Field(min_length=1)


def local_snapshot(repo: str, revision: str, files: list[str], cache_dir: Path, model_dir) -> Path:
    try:
        path = Path(snapshot_download(repo, revision=revision, cache_dir=cache_dir, local_files_only=True))
    except LocalEntryNotFoundError:
        path = None
    missing = [f for f in files if path is None or not (path / f).is_file()]
    if missing:
        raise ModelFilesMissing(
            f"{repo}@{revision[:8]} is not fully downloaded to {cache_dir} (missing: {missing}). "
            f"Run: python scripts/fetch_model.py {model_dir}"
        )
    return path
