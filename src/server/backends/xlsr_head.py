"""Backend for this repo's own models (stage 2): the frozen base model wav2vec2 XLSR-53, then a
small classifier trained here (src/training/head.py) on the pooled features of some of its layers.

config.yaml backend_options:
  base_model: {repo, revision, license, files}   pinned like the stage-1 backend
  head_file: head.safetensors                     next to config.yaml
  head_sha256: <64 hex>                           the head file is refused if it differs

The class order is read from the head file and mapped to the config by model_label. The base
model runs only up to the highest layer the classifier uses (the same values as the full model;
the rest would cost time). Features are computed exactly as for training: per layer the mean and
spread over time, stored as float16 and scaled back.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

from server.backends.hf_files import BaseModelSpec, local_snapshot
from server.config import ModelConfig
from server.model_api import AccentModel, label_permutation
from server.paths import hf_cache_dir


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: BaseModelSpec
    head_file: str
    head_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _options(config: ModelConfig) -> Options:
    return Options.model_validate(config.backend_options)


class XlsrPooledHead(AccentModel):
    def __init__(self, config: ModelConfig, cache_dir: Path | None = None, device: str = "cpu"):
        from training.features import FeatureExtractor
        from training.head import TrainedHead

        super().__init__(config)
        opts = _options(config)
        head_path = config.model_dir / opts.head_file
        digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        if digest != opts.head_sha256:
            raise ValueError(f"{head_path} has SHA-256 {digest[:16]}..., config.yaml pins {opts.head_sha256[:16]}...")
        self.head = TrainedHead(head_path)
        self.model_labels = self.head.class_ids          # model output order, as read from the file
        self._perm = label_permutation(self.model_labels, config)

        cache_dir = Path(cache_dir) if cache_dir else hf_cache_dir()
        base = local_snapshot(opts.base_model.repo, opts.base_model.revision, opts.base_model.files, cache_dir,
                              config.model_dir)
        rate = json.loads((base / "preprocessor_config.json").read_text())["sampling_rate"]
        if rate != config.audio.sample_rate:
            raise ValueError(f"config audio.sample_rate {config.audio.sample_rate} != the base model's {rate}")
        self.extractor = FeatureExtractor(base, device, self.head.saved.get("storage_scale", {}),
                                          last_layer=max(self.head.layers))

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        pooled = self.extractor(audio).astype(np.float32) * self.extractor.scale[:, None]
        return self.head.probabilities(pooled[None])[0][self._perm]

    @classmethod
    def fetch(cls, config: ModelConfig, cache_dir: Path) -> None:
        opts = _options(config)
        snapshot_download(opts.base_model.repo, revision=opts.base_model.revision, cache_dir=cache_dir,
                          allow_patterns=opts.base_model.files)
