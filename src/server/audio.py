"""Audio decoding and resampling. Everything the model sees is mono float32 at its sample rate."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
import soxr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(samples,) or (samples, channels) -> (samples,), averaging channels."""
    audio = np.asarray(audio, dtype=np.float32)
    return audio if audio.ndim == 1 else audio.mean(axis=1, dtype=np.float32)


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a whole signal with soxr (high quality)."""
    if from_rate == to_rate:
        return np.asarray(audio, dtype=np.float32)
    return soxr.resample(audio, from_rate, to_rate).astype(np.float32, copy=False)


def decode_file(file: str | Path | BinaryIO, target_rate: int) -> np.ndarray:
    """Decode an audio file (WAV, FLAC, OGG, MP3) to mono float32 at target_rate."""
    audio, rate = sf.read(file, dtype="float32", always_2d=True)
    return resample(to_mono(audio), rate, target_rate)


class StreamingResampler:
    """Resample audio that arrives in chunks of any size (e.g. from a microphone).

    Keeps filter state between chunks, so the joined output matches resampling the
    whole signal at once (up to tiny edge effects). Call flush() at the end.
    """

    def __init__(self, from_rate: int, to_rate: int):
        self.from_rate, self.to_rate = from_rate, to_rate
        self._stream = None if from_rate == to_rate else soxr.ResampleStream(
            from_rate, to_rate, 1, dtype="float32", quality="HQ")

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"expected mono 1-D chunk, got shape {chunk.shape}")
        if self._stream is None:
            return chunk
        return self._stream.resample_chunk(chunk)

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        return self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)


def voiced_seconds(audio: np.ndarray, sample_rate: int, gate_dbfs: float, frame_seconds: float = 0.02) -> float:
    """Seconds of audio whose 20 ms frames are louder than gate_dbfs (RMS, full scale = 0 dB).

    A simple energy gate, not a speech detector: it tells silence from sound.
    A trailing partial frame is ignored.
    """
    frame = int(frame_seconds * sample_rate)
    n_frames = len(audio) // frame
    if n_frames == 0:
        return 0.0
    frames = np.asarray(audio[: n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    dbfs = 20 * np.log10(np.maximum(rms, 1e-12))
    return int(np.count_nonzero(dbfs > gate_dbfs)) * frame / sample_rate
