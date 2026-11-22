"""Training copies: the same plan for the same clip, the intended SNR, aligned echo, filters, no clipping."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.augment import (LIMIT, NoiseBank, Plan, add_noise, add_reverb, apply, load_augment_spec, plan_for,
                              speech_rms)

REPO = Path(__file__).resolve().parents[1]
SR = 16000


def tone(freq, seconds=1.0, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def bank(tmp_path):
    """A tiny RIRS_NOISES folder with the real layout and list-file format."""
    base = tmp_path / "RIRS_NOISES"
    rng = np.random.default_rng(0)

    def wav(rel, data, channels=1):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.repeat(data[:, None], channels, axis=1) if channels > 1 else data, SR)
        return rel

    real = [wav(f"RIRS_NOISES/real_rirs_isotropic_noises/r{i}.wav", np.r_[np.zeros(50), 1, 0.3 * rng.random(400)],
                channels=8) for i in range(3)]
    (base / "real_rirs_isotropic_noises/rir_list").write_text(
        "".join(f"--rir-id {i} --room-id room1 {p}\n" for i, p in enumerate(real)))
    iso = wav("RIRS_NOISES/real_rirs_isotropic_noises/n0.wav", 0.05 * rng.standard_normal(3 * SR), channels=8)
    (base / "real_rirs_isotropic_noises/noise_list").write_text(
        f"--noise-id 1 --noise-type isotropic --room-linkage room1 {iso}\n")
    point = []
    for i, kind in enumerate(["background", "foreground", "foreground"]):
        point.append((kind, wav(f"RIRS_NOISES/pointsource_noises/p{i}.wav", 0.1 * rng.standard_normal(SR // 2))))
    point.append(("foreground", wav("RIRS_NOISES/pointsource_noises/click.wav", 0.1 * rng.standard_normal(400))))
    (base / "pointsource_noises/noise_list").write_text(
        "".join(f"--noise-id p{i} --noise-type point-source --bg-fg-type {k} {p}\n" for i, (k, p) in enumerate(point)))
    for room in ("smallroom", "mediumroom"):
        sims = [wav(f"RIRS_NOISES/simulated_rirs/{room}/Room001/R-{i}.wav", np.r_[np.zeros(10), 1, 0.2 * rng.random(200)])
                for i in range(4)]
        (base / f"simulated_rirs/{room}/rir_list").write_text(
            "".join(f"--rir-id {room}{i} --room-id {room}1 {p}\n" for i, p in enumerate(sims)))
    return NoiseBank(tmp_path, ["smallroom", "mediumroom"])


def spec():
    return load_augment_spec(REPO / "training/cv27_es10/augment.yaml")


def test_bank_reads_the_list_files(bank):
    assert len(bank.real_rirs) == 3 and len(bank.simulated_rirs) == 8
    assert {k: len(v) for k, v in bank.noises.items()} == {"isotropic": 1, "background": 1, "foreground": 2}
    assert [p.name for p in bank.too_short] == ["click.wav"]                 # 25 ms: left out
    assert NoiseBank.load(bank.real_rirs[0]).ndim == 1                 # first of 8 channels


def test_unsafe_list_paths_are_refused(bank):
    (bank.root / "RIRS_NOISES/pointsource_noises/noise_list").write_text(
        "--noise-id x --bg-fg-type background ../../etc/passwd\n")
    with pytest.raises(ValueError, match="unsafe path"):
        NoiseBank(bank.root, ["smallroom"])


def test_plans_are_fixed_per_clip_and_follow_the_probabilities(bank):
    s = spec()
    assert plan_for("a.mp3", s, bank) == plan_for("a.mp3", s, bank)
    plans = [plan_for(f"clip{i}.mp3", s, bank) for i in range(3000)]
    assert len(set(plans)) > 2900
    share = lambda f: sum(map(f, plans)) / len(plans)          # noqa: E731
    assert share(lambda p: p.rir is not None) == pytest.approx(0.5, abs=0.04)
    assert share(lambda p: p.noise is not None) == pytest.approx(0.6, abs=0.04)
    assert share(lambda p: p.highpass_hz is not None) == pytest.approx(0.5, abs=0.04)
    assert share(lambda p: p.lowpass_hz is not None) == pytest.approx(0.3, abs=0.04)
    rirs = [p.rir for p in plans if p.rir]
    assert sum(r in bank.real_rirs for r in rirs) / len(rirs) == pytest.approx(0.5, abs=0.05)
    assert all(5 <= p.snr_db <= 20 for p in plans if p.noise) and all(-6 <= p.gain_db <= 6 for p in plans)
    assert {p.noise_kind for p in plans if p.noise} == {"isotropic", "background", "foreground"}


def test_noise_is_added_at_the_requested_snr():
    speech = np.r_[np.zeros(SR // 2, dtype=np.float32), tone(440, 2.0)]        # silence must not count
    noise = np.random.default_rng(1).standard_normal(SR // 3).astype(np.float32)  # shorter: repeated
    for snr in (5.0, 12.0, 20.0):
        out = add_noise(speech, noise, snr, offset=0.7)
        added = out - speech
        measured = 20 * np.log10(speech_rms(speech) / np.sqrt(np.mean(added ** 2)))
        assert measured == pytest.approx(snr, abs=0.3) and len(out) == len(speech)
    assert np.array_equal(add_noise(speech, np.zeros(0, dtype=np.float32), 10.0, 0.5), speech)


def test_reverb_keeps_timing_and_level():
    speech = tone(300, 0.5)
    delta = np.r_[np.zeros(120), 1.0, np.zeros(50)].astype(np.float32)          # direct sound after 120 samples
    assert np.allclose(add_reverb(speech, delta), speech, atol=1e-6)
    echo = np.r_[np.zeros(30), 1.0, 0.5 * np.exp(-np.arange(3000) / 500)].astype(np.float32)
    out = add_reverb(speech, echo)
    assert len(out) == len(speech)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(np.sqrt(np.mean(speech ** 2)), rel=1e-4)


def test_microphone_filters():
    def level(x):
        return np.sqrt(np.mean(x[SR // 4:] ** 2))                                # skip the filter's start-up
    low, mid, high = tone(50), tone(1000), tone(7500)
    hp = Plan(highpass_hz=300)
    assert level(apply(low, hp)) < 0.05 * level(low) and level(apply(mid, hp)) > 0.95 * level(mid)
    lp = Plan(lowpass_hz=4000)
    assert level(apply(high, lp)) < 0.05 * level(high) and level(apply(mid, lp)) > 0.95 * level(mid)


def test_gain_and_limiting():
    x = tone(440, amp=0.5)
    assert np.allclose(apply(x, Plan()), x)
    assert np.max(np.abs(apply(x, Plan(gain_db=6.0)))) == pytest.approx(LIMIT, abs=1e-6)
    assert np.max(np.abs(apply(x, Plan(gain_db=-6.0)))) == pytest.approx(0.5 * 10 ** (-6 / 20), rel=1e-3)


def test_a_full_plan_runs(bank):
    s = spec()
    x = tone(220, 1.5)
    for i in range(30):
        out = apply(x, plan_for(f"c{i}.mp3", s, bank), bank)
        assert out.dtype == np.float32 and len(out) == len(x)
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= LIMIT + 1e-6
