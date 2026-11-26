"""The model interface the app relies on, and the backend registry.

Contract (checked on every call, for every backend):
  predict(audio) takes a 1-D float array of mono audio at config.audio.sample_rate,
  at least MIN_AUDIO_SECONDS long, and returns {class_id: probability} with exactly
  the config's class ids, in config order, every value in [0, 1], summing to 1.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from server.config import ModelConfig, load_config

MIN_AUDIO_SECONDS = 0.5
SUM_TOLERANCE = 1e-4


class ModelContractError(RuntimeError):
    """A backend returned output that breaks the contract above."""


class ModelFilesMissing(FileNotFoundError):
    """Model files are not downloaded yet."""


def label_permutation(model_labels: list[str], config: ModelConfig) -> list[int]:
    """Map the model's output order to config order.

    model_labels[i] is the label of the model's i-th output, as read from the model
    files. Returns perm so that probs_in_config_order = probs_in_model_order[perm].
    Fails unless the model's labels and the config's model_label values are the same set.
    """
    wanted = [c.model_label for c in config.classes]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError(f"model files contain duplicate labels: {model_labels}")
    missing = sorted(set(wanted) - set(model_labels))
    extra = sorted(set(model_labels) - set(wanted))
    if missing or extra:
        raise ValueError(
            "config.yaml classes do not match the model's labels. "
            f"In config but not in model: {missing}. In model but not in config: {extra}."
        )
    return [model_labels.index(label) for label in wanted]


class AccentModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def class_ids(self) -> list[str]:
        return self.config.class_ids

    @property
    def sample_rate(self) -> int:
        return self.config.audio.sample_rate

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        audio = self._validate_audio(audio)
        probs = np.asarray(self._predict(audio), dtype=np.float64)
        n = len(self.class_ids)
        if probs.shape != (n,):
            raise ModelContractError(f"expected {n} probabilities, got shape {probs.shape}")
        if not np.all(np.isfinite(probs)) or probs.min() < 0 or probs.max() > 1:
            raise ModelContractError(f"probabilities out of range: {probs}")
        if abs(probs.sum() - 1.0) > SUM_TOLERANCE:
            raise ModelContractError(f"probabilities sum to {probs.sum():.6f}, not 1")
        return dict(zip(self.class_ids, probs.tolist()))

    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError(f"audio must be floating point in [-1, 1], got {audio.dtype}")
        min_samples = int(MIN_AUDIO_SECONDS * self.sample_rate)
        if audio.shape[0] < min_samples:
            raise ValueError(f"audio too short: {audio.shape[0]} samples < {min_samples}")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains NaN or infinity")
        return np.ascontiguousarray(audio, dtype=np.float32)

    @abstractmethod
    def _predict(self, audio: np.ndarray) -> np.ndarray:
        """Return probabilities as an array in config class order."""

    @classmethod
    def fetch(cls, config: ModelConfig, cache_dir: Path) -> None:
        """Download whatever the backend needs. Default: nothing."""


# Backend name -> "module:Class". Imported lazily so tests with the fake
# backend never import torch.
BACKENDS = {
    "speechbrain_custom_classifier": "server.backends.speechbrain_classifier:SpeechBrainCustomClassifier",
    "xlsr_pooled_head": "server.backends.xlsr_head:XlsrPooledHead",
    "fake": "server.backends.fake:FakeAccentModel",
}


def backend_class(name: str) -> type[AccentModel]:
    try:
        target = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}") from None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def load_model(config: ModelConfig | str | Path, **kwargs) -> AccentModel:
    """Build the model described by a config (object, config.yaml path or model dir)."""
    if not isinstance(config, ModelConfig):
        config = load_config(config)
    return backend_class(config.backend)(config, **kwargs)
