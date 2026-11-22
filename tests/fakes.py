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


def make_noise_folder(root, sample_rate: int = 16000):
    """A tiny RIRS_NOISES folder (OpenSLR 28 layout and list-file format) under `root`.
    Returns the paths of the noise files that are too short to use."""
    import soundfile as sf

    base = root / "RIRS_NOISES"
    rng = np.random.default_rng(0)

    def wav(rel, data, channels=1):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.repeat(data[:, None], channels, axis=1) if channels > 1 else data, sample_rate)
        return rel

    real = [wav(f"RIRS_NOISES/real_rirs_isotropic_noises/r{i}.wav", np.r_[np.zeros(50), 1, 0.3 * rng.random(400)],
                channels=8) for i in range(3)]
    (base / "real_rirs_isotropic_noises/rir_list").write_text(
        "".join(f"--rir-id {i} --room-id room1 {p}\n" for i, p in enumerate(real)))
    iso = wav("RIRS_NOISES/real_rirs_isotropic_noises/n0.wav", 0.05 * rng.standard_normal(3 * sample_rate), channels=8)
    (base / "real_rirs_isotropic_noises/noise_list").write_text(
        f"--noise-id 1 --noise-type isotropic --room-linkage room1 {iso}\n")
    point = []
    for i, kind in enumerate(["background", "foreground", "foreground"]):
        point.append((kind, wav(f"RIRS_NOISES/pointsource_noises/p{i}.wav", 0.1 * rng.standard_normal(sample_rate // 2))))
    point.append(("foreground", wav("RIRS_NOISES/pointsource_noises/click.wav", 0.1 * rng.standard_normal(400))))
    (base / "pointsource_noises/noise_list").write_text(
        "".join(f"--noise-id p{i} --noise-type point-source --bg-fg-type {k} {p}\n" for i, (k, p) in enumerate(point)))
    for room in ("smallroom", "mediumroom"):
        sims = [wav(f"RIRS_NOISES/simulated_rirs/{room}/Room001/R-{i}.wav", np.r_[np.zeros(10), 1, 0.2 * rng.random(200)])
                for i in range(4)]
        (base / f"simulated_rirs/{room}/rir_list").write_text(
            "".join(f"--rir-id {room}{i} --room-id {room}1 {p}\n" for i, p in enumerate(sims)))


def tiny_base_model(folder, seed: int = 0):
    """A randomly initialized wav2vec2 with the XLSR-53 layout (layer norm, stable layer norm)
    but tiny: 2 transformer layers of width 16. Saved like a Hugging Face model folder."""
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    torch.manual_seed(seed)
    config = Wav2Vec2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                            conv_dim=(8, 8, 8), conv_stride=(5, 4, 4), conv_kernel=(10, 4, 4),
                            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
                            feat_extract_norm="layer", do_stable_layer_norm=True)
    Wav2Vec2Model(config).save_pretrained(folder)
    return folder
