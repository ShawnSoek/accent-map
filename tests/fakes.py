"""Test helpers: fake model configs with any class count, and synthetic audio."""

import numpy as np

from server.config import ModelConfig


def fake_config(n_classes: int = 6, **backend_options) -> ModelConfig:
    classes = [
        {"id": f"c{i}", "model_label": f"LABEL {i}", "label": f"Class {i}", "zone": f"z{i}"}
        for i in range(n_classes)
    ]
    return ModelConfig.model_validate({
        "name": f"fake{n_classes}",
        "display_name": f"Fake model ({n_classes} classes)",
        "backend": "fake",
        "backend_options": backend_options,
        "classes": classes,
    })


def tone(freq_hz: float, seconds: float, sample_rate: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
