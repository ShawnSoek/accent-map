"""Schema and loader for models/<name>/config.yaml.

The app learns everything about a model from this file: classes, map zones,
audio windowing and which backend runs it. Nothing about the class list is
hardcoded anywhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SLUG = r"^[a-z][a-z0-9_]*$"
COMMIT_SHA = r"^[0-9a-f]{40}$"


class _Strict(BaseModel):
    # Unknown keys are errors, so a typo in config.yaml fails loudly.
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceInfo(_Strict):
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=COMMIT_SHA, description="Pinned commit SHA, never a branch.")
    license: str | None = None


class AudioSpec(_Strict):
    sample_rate: int = Field(16000, gt=0)
    window_seconds: float = Field(5.0, gt=0)
    hop_seconds: float = Field(0.5, gt=0)

    @model_validator(mode="after")
    def _hop_fits_window(self) -> AudioSpec:
        if self.hop_seconds > self.window_seconds:
            raise ValueError("hop_seconds must not exceed window_seconds")
        return self


class ClassInfo(_Strict):
    id: str = Field(pattern=SLUG)
    model_label: str = Field(min_length=1, description="Label exactly as the model files name it.")
    label: str = Field(min_length=1)
    description: str = ""
    zone: str = Field(pattern=SLUG)


class MapSpec(_Strict):
    zones_file: str | None = None


class ModelConfig(_Strict):
    name: str = Field(pattern=SLUG)
    display_name: str = Field(min_length=1)
    description: str = ""
    source: SourceInfo | None = None
    backend: str = Field(min_length=1)
    backend_options: dict[str, Any] = Field(default_factory=dict)
    audio: AudioSpec = AudioSpec()
    caveats: list[str] = Field(default_factory=list)
    classes: list[ClassInfo] = Field(min_length=2)
    map: MapSpec = MapSpec()
    # Directory holding config.yaml; set by load_config, not read from YAML.
    model_dir: Path | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def _unique_classes(self) -> ModelConfig:
        for attr in ("id", "model_label"):
            values = [getattr(c, attr) for c in self.classes]
            dupes = sorted({v for v in values if values.count(v) > 1})
            if dupes:
                raise ValueError(f"duplicate class {attr}: {dupes}")
        return self

    @property
    def class_ids(self) -> list[str]:
        return [c.id for c in self.classes]


def load_config(path: str | Path) -> ModelConfig:
    """Load and validate a model config. `path` is config.yaml or its directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.yaml"
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    if "model_dir" in raw:
        raise ValueError(f"{path}: 'model_dir' is set automatically and must not appear in the file")
    return ModelConfig.model_validate({**raw, "model_dir": path.parent.resolve()})
