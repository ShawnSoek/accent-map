"""Training copies with room echo, noise and laptop-microphone sound (spec: training/<name>/augment.yaml).

`plan_for(clip, spec, bank)` decides what happens to one clip, from a SHA-256 of the seed and
the clip name, so the same clip always gets the same treatment. `apply(audio, plan, bank)` does it.
The noise collection is OpenSLR 28 (RIRS_NOISES); `NoiseBank` reads its list files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import butter, fftconvolve, sosfilt

from training.archives import safe_relative
from training.cv_split import rank

SAMPLE_RATE = 16000
LIMIT = 0.99
MIN_NOISE_SECONDS = 0.5      # shorter sounds (clicks down to 26 ms) would buzz when repeated


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Chance(_Strict):
    probability: float = Field(ge=0, le=1)
    hz: tuple[float, float]


class ReverbSpec(_Strict):
    probability: float = Field(ge=0, le=1)
    real_share: float = Field(ge=0, le=1)
    simulated_rooms: list[str]


class NoiseSpec(_Strict):
    probability: float = Field(ge=0, le=1)
    snr_db: tuple[float, float]
    kinds: list[str] = Field(min_length=1)


class AugmentSpec(_Strict):
    noise_archive: str
    noise_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: str
    reverb: ReverbSpec
    noise: NoiseSpec
    highpass: Chance
    lowpass: Chance
    gain_db: tuple[float, float]


def load_augment_spec(path: Path) -> AugmentSpec:
    return AugmentSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _list_entries(list_file: Path, root: Path) -> list[tuple[dict[str, str], Path]]:
    """Kaldi-style list lines: --key value ... path. Paths are relative to `root`."""
    out = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        options = dict(zip(tokens[0:-1:2], tokens[1:-1:2]))
        rel = safe_relative(tokens[-1])
        if rel is None:
            raise ValueError(f"{list_file}: unsafe path {tokens[-1]!r}")
        out.append(({k.lstrip("-"): v for k, v in options.items()}, root.joinpath(*rel.parts)))
    return out


class NoiseBank:
    """Room impulse responses and noises of an extracted RIRS_NOISES folder. Noises shorter than
    MIN_NOISE_SECONDS are left out (`too_short` lists them)."""

    def __init__(self, root: Path, simulated_rooms: list[str]):
        self.root = Path(root)
        base = self.root / "RIRS_NOISES"
        self.real_rirs = [p for _, p in _list_entries(base / "real_rirs_isotropic_noises/rir_list", self.root)]
        self.simulated_rirs = [p for room in simulated_rooms
                               for _, p in _list_entries(base / "simulated_rirs" / room / "rir_list", self.root)]
        self.noises: dict[str, list[Path]] = {"isotropic": [], "background": [], "foreground": []}
        self.too_short: list[Path] = []
        entries = [("isotropic", p) for _, p in _list_entries(base / "real_rirs_isotropic_noises/noise_list", self.root)]
        entries += [(o.get("bg-fg-type", "foreground"), p)
                    for o, p in _list_entries(base / "pointsource_noises/noise_list", self.root)]
        for kind, p in entries:
            if sf.info(p).frames < MIN_NOISE_SECONDS * SAMPLE_RATE:
                self.too_short.append(p)
            else:
                self.noises[kind].append(p)
        for name, paths in [("real RIRs", self.real_rirs), ("simulated RIRs", self.simulated_rirs),
                            *self.noises.items()]:
            if not paths:
                raise ValueError(f"no {name} under {self.root}")

    @staticmethod
    def load(path: Path) -> np.ndarray:
        """First channel, float32. The collection is all 16 kHz; anything else is an error."""
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{path}: {rate} Hz, expected {SAMPLE_RATE}")
        return np.ascontiguousarray(audio[:, 0])


@dataclass(frozen=True)
class Plan:
    rir: Path | None = None
    noise: Path | None = None
    noise_kind: str | None = None
    snr_db: float | None = None
    noise_offset: float = 0.0          # where in the noise to start, as a fraction of its length
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    gain_db: float = 0.0

    def describe(self) -> str:
        parts = []
        if self.rir:
            parts.append(f"echo {self.rir.parent.name}/{self.rir.stem}")
        if self.noise:
            parts.append(f"{self.noise_kind} noise {self.noise.stem} at {self.snr_db:.1f} dB SNR")
        if self.highpass_hz:
            parts.append(f"high-pass {self.highpass_hz:.0f} Hz")
        if self.lowpass_hz:
            parts.append(f"low-pass {self.lowpass_hz:.0f} Hz")
        parts.append(f"gain {self.gain_db:+.1f} dB")
        return ", ".join(parts)


def plan_for(clip: str, spec: AugmentSpec, bank: NoiseBank) -> Plan:
    rng = np.random.default_rng(int(rank(spec.seed, clip)[:16], 16))
    rir = noise = kind = snr = None
    offset = 0.0
    if rng.random() < spec.reverb.probability:
        pool = bank.real_rirs if rng.random() < spec.reverb.real_share else bank.simulated_rirs
        rir = pool[rng.integers(len(pool))]
    if rng.random() < spec.noise.probability:
        kind = spec.noise.kinds[rng.integers(len(spec.noise.kinds))]
        pool = bank.noises[kind]
        noise = pool[rng.integers(len(pool))]
        snr = float(rng.uniform(*spec.noise.snr_db))
        offset = float(rng.random())
    highpass = float(rng.uniform(*spec.highpass.hz)) if rng.random() < spec.highpass.probability else None
    lowpass = float(rng.uniform(*spec.lowpass.hz)) if rng.random() < spec.lowpass.probability else None
    return Plan(rir, noise, kind, snr, offset, highpass, lowpass, float(rng.uniform(*spec.gain_db)))


def speech_rms(audio: np.ndarray, frame: int = 320, range_db: float = 30.0) -> float:
    """RMS over 20 ms frames within `range_db` of the loudest one, so leading and trailing
    silence does not make the speech seem quieter than it is."""
    n = len(audio) // frame
    if n == 0:
        return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    energy = np.mean(audio[:n * frame].reshape(n, frame) ** 2, axis=1)
    loud = energy[energy >= energy.max() * 10 ** (-range_db / 10)]
    return float(np.sqrt(np.mean(loud)))


def add_reverb(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve, aligned on the RIR's strongest peak (the direct sound), same length and RMS."""
    peak = int(np.argmax(np.abs(rir)))
    out = fftconvolve(audio, rir)[peak:peak + len(audio)]
    rms_in, rms_out = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(out ** 2))
    return (out * (rms_in / rms_out if rms_out > 0 else 1.0)).astype(np.float32)


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float, offset: float) -> np.ndarray:
    """Add a stretch of `noise` (repeated if too short) at `snr_db` below the speech level."""
    if len(noise) == 0:
        return audio
    start = int(offset * len(noise))
    reps = -(-(start + len(audio)) // len(noise))
    piece = np.tile(noise, reps)[start:start + len(audio)]
    level = np.sqrt(np.mean(piece ** 2))
    speech = speech_rms(audio)
    if level == 0 or speech == 0:
        return audio
    return (audio + piece * (speech / level) * 10 ** (-snr_db / 20)).astype(np.float32)


def apply(audio: np.ndarray, plan: Plan, bank: NoiseBank | None = None) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32)
    if plan.rir is not None:
        out = add_reverb(out, bank.load(plan.rir))
    if plan.noise is not None:
        out = add_noise(out, bank.load(plan.noise), plan.snr_db, plan.noise_offset)
    if plan.highpass_hz:
        out = sosfilt(butter(4, plan.highpass_hz, "highpass", fs=SAMPLE_RATE, output="sos"), out)
    if plan.lowpass_hz:
        out = sosfilt(butter(4, plan.lowpass_hz, "lowpass", fs=SAMPLE_RATE, output="sos"), out)
    out = out * 10 ** (plan.gain_db / 20)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > LIMIT:
        out = out * (LIMIT / peak)
    return out.astype(np.float32)
