"""What the frozen base model hears in a piece of audio: per-layer mean and spread over time.

The base model is wav2vec2 XLSR-53 (facebook/wav2vec2-large-xlsr-53), which was never trained on
accents. The stage-1 model's fine-tuned encoder is not used: it was trained on Common Voice
speakers who may be in our test set.

Input: 16 kHz mono float32. It is normalized to zero mean and unit variance first, as the base
model's preprocessor_config.json asks (do_normalize). Output: float16, shape (25, 2048). For
each of the 25 hidden states (the CNN projection and the 24 transformer layers), the first
1024 values are the mean over time and the last 1024 the standard deviation. Layers named in
`storage_scale` (training/<name>/features.yaml) are stored divided by their factor, so that
float16 cannot overflow; a value that would still overflow is an error, never stored as inf.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

LAYERS = 25
WIDTH = 1024
FLOAT16_MAX = 65504.0


class FeaturesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files_sha256: dict[str, str]
    storage_scale: dict[int, float] = {}


def load_features_spec(path: Path) -> FeaturesSpec:
    return FeaturesSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def scale_vector(layers: int, storage_scale: dict[int, float] | None) -> np.ndarray:
    scale = np.ones(layers, dtype=np.float32)
    for layer, factor in (storage_scale or {}).items():
        if not 0 <= layer < layers:
            raise ValueError(f"storage_scale names layer {layer}; the model has {layers}")
        scale[layer] = factor
    return scale


def normalize(audio: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance, exactly as transformers' Wav2Vec2FeatureExtractor does it."""
    audio = np.asarray(audio, dtype=np.float32)
    return ((audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)).astype(np.float32)


def pool(hidden_states: tuple[torch.Tensor, ...], scale: np.ndarray | None = None) -> np.ndarray:
    """(layers) x (1, frames, width) -> float16 (layers, 2 * width): mean, then std over frames,
    each layer divided by its storage scale."""
    h = torch.stack(hidden_states)[:, 0].float()
    pooled = torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False)], dim=1).cpu().numpy()
    if scale is not None:
        pooled = pooled / scale[:, None]
    if not np.isfinite(pooled).all() or np.abs(pooled).max() > FLOAT16_MAX:
        worst = int(np.nanargmax(np.abs(pooled).max(axis=1)))
        raise OverflowError(f"layer {worst} reaches {np.abs(pooled[worst]).max():.0f}, too large for float16; "
                            f"raise its storage_scale")
    return pooled.astype(np.float16)


def _truncate(model, last_layer: int) -> None:
    """Keep the first `last_layer` transformer layers. In XLSR-53's layout (stable layer norm),
    hidden state k < 24 is the un-normalized input of layer k, and only the very last one passes
    through the encoder's final layer norm. So after cutting, that final norm must go too, or
    hidden state `last_layer` would differ from the full model's."""
    import torch

    if not model.config.do_stable_layer_norm:
        raise ValueError("last_layer needs a stable-layer-norm wav2vec2 (as XLSR-53)")
    if not 1 <= last_layer:
        raise ValueError(f"last_layer must be at least 1, got {last_layer}")
    model.encoder.layers = model.encoder.layers[:last_layer]
    model.encoder.layer_norm = torch.nn.Identity()
    model.config.num_hidden_layers = last_layer


class FeatureExtractor:
    """The frozen base model on one device. Loading fails if any weight it needs is missing."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", storage_scale: dict[int, float] | None = None,
                 last_layer: int | None = None):
        """`last_layer`: compute only hidden states 0..last_layer, with the same values as the
        full model (the app needs only the layers its classifier uses; later ones cost time)."""
        from transformers import Wav2Vec2Model

        model, info = Wav2Vec2Model.from_pretrained(str(model_dir), output_loading_info=True)
        if info["missing_keys"] or info["mismatched_keys"]:
            raise RuntimeError(f"base model weights missing: {info['missing_keys'][:5]} "
                               f"mismatched: {info['mismatched_keys'][:5]}")
        self.unused_keys = len(info["unexpected_keys"])   # pre-training heads (quantizer etc.)
        if last_layer is not None and last_layer < model.config.num_hidden_layers:
            _truncate(model, last_layer)
        self.model = model.eval().to(device)
        self.device = device
        self.layers = model.config.num_hidden_layers + 1
        self.width = model.config.hidden_size
        self.scale = scale_vector(self.layers, {k: v for k, v in (storage_scale or {}).items() if k < self.layers})

    @torch.inference_mode()
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(normalize(audio))[None].to(self.device)
        out = self.model(x, output_hidden_states=True)
        return pool(out.hidden_states, self.scale)
