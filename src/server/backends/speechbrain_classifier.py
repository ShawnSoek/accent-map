"""Backend for SpeechBrain models published with a custom interface class
(foreign_class style), such as Jzuluaga/accent-id-commonaccent_xlsr-es-spanish.

Differences from the model card's foreign_class() call, all deliberate (DECISIONS.md):
- hyperparams come from the local model dir (fixed copy), everything else from the
  pinned source revision in the local Hugging Face cache. Nothing is downloaded here;
  run scripts/fetch_model.py first.
- Checkpoints are loaded with strict=True instead of SpeechBrain's Pretrainer,
  which loads non-strictly and only logs a warning on mismatched keys.
- Inference runs encode_batch -> head modules -> softmax ourselves, because the
  published classify_* methods skip part of the trained head.
- Class order is read from the model's label encoder and mapped to config ids
  by label string.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from hyperpyyaml import load_hyperpyyaml
from pydantic import BaseModel, ConfigDict, Field
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.utils.superpowers import import_from_path

from server.config import ModelConfig, SourceInfo
from server.model_api import AccentModel, ModelFilesMissing, label_permutation
from server.paths import hf_cache_dir


class BaseModelSpec(SourceInfo):
    files: list[str] = Field(min_length=1)


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hparams_file: str
    pymodule_file: str
    classname: str
    label_encoder_file: str
    checkpoints: dict[str, str] = Field(min_length=1)
    head: list[str] = Field(min_length=1)
    base_model: BaseModelSpec
    overrides: dict[str, str] = Field(default_factory=dict)


def _options(config: ModelConfig) -> Options:
    if config.source is None:
        raise ValueError(f"{config.name}: this backend needs a 'source' with a pinned revision")
    return Options.model_validate(config.backend_options)


def _local_snapshot(repo: str, revision: str, files: list[str], cache_dir: Path, model_dir) -> Path:
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


class SpeechBrainCustomClassifier(AccentModel):
    def __init__(self, config: ModelConfig, cache_dir: Path | None = None, device: str = "cpu"):
        super().__init__(config)
        opts = _options(config)
        cache_dir = Path(cache_dir) if cache_dir else hf_cache_dir()
        source_files = [opts.pymodule_file, opts.label_encoder_file, *opts.checkpoints.values()]
        src = _local_snapshot(config.source.repo, config.source.revision, source_files, cache_dir, config.model_dir)
        base = _local_snapshot(opts.base_model.repo, opts.base_model.revision, opts.base_model.files,
                               cache_dir, config.model_dir)

        # The pinned custom_interface.py imports the pre-1.0 path speechbrain.pretrained,
        # which SpeechBrain redirects with a DeprecationWarning-style UserWarning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Module 'speechbrain\.pretrained' was deprecated")
            interface_cls = getattr(import_from_path(src / opts.pymodule_file), opts.classname)

        overrides = {k: v.format(source_dir=src, base_dir=base) for k, v in opts.overrides.items()}
        with open(config.model_dir / opts.hparams_file, encoding="utf-8") as f:
            hparams = load_hyperpyyaml(f, overrides)

        for key, filename in opts.checkpoints.items():
            state = torch.load(src / filename, map_location="cpu", weights_only=True)
            hparams[key].load_state_dict(state, strict=True)  # raises on any missing/unexpected key
            del state

        encoder = CategoricalEncoder()
        encoder.load(src / opts.label_encoder_file)
        model_labels = [encoder.ind2lab[i] for i in range(len(encoder.ind2lab))]
        self.model_labels = model_labels  # model output order, as read from the files
        self._perm = label_permutation(model_labels, config)

        if hparams["wav2vec2"].feature_extractor.sampling_rate != config.audio.sample_rate:
            raise ValueError("config audio.sample_rate does not match the model's feature extractor")

        # freeze_params=True puts every module in eval mode and disables gradients.
        self._interface = interface_cls(modules=hparams["modules"], hparams=hparams,
                                        run_opts={"device": device}, freeze_params=True)
        self._head = [getattr(self._interface.mods, name) for name in opts.head]
        self._softmax = hparams["softmax"]
        self._device = device

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(audio).unsqueeze(0)
        with torch.inference_mode():
            x = self._interface.encode_batch(wav, torch.ones(1, device=self._device))
            for module in self._head:
                x = module(x)
            probs = self._softmax(x).reshape(-1).cpu().numpy()
        return probs[self._perm]

    @classmethod
    def fetch(cls, config: ModelConfig, cache_dir: Path) -> None:
        opts = _options(config)
        snapshot_download(config.source.repo, revision=config.source.revision, cache_dir=cache_dir)
        snapshot_download(opts.base_model.repo, revision=opts.base_model.revision, cache_dir=cache_dir,
                          allow_patterns=opts.base_model.files)
