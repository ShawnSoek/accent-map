"""Measure how long one model update takes: model.predict() on one window of audio.

Usage:
  python scripts/bench_latency.py [model_dir] [--threads 4 8 16] [--windows 1 2 3 4 5]
                                  [--warmup 3] [--runs 30] [--out data/bench/latest.json]

This is the inference cost per update only. Resampling, windowing and the network add a
little on top; the server reports that end to end per event (step 6).
At the end, the first setting is measured again to show whether the CPU slowed down
(thermal throttling) during the run.
"""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server.audio import decode_file  # noqa: E402
from server.model_api import load_model  # noqa: E402
from server.paths import DATA_DIR, MODELS_DIR, hf_cache_dir  # noqa: E402


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def speech_audio(model) -> np.ndarray:
    """All sample clips of the model repo joined (~26 s), so every window is real speech."""
    src = model.config.source
    clips = sorted((hf_cache_dir() / f"models--{src.repo.replace('/', '--')}" / "snapshots"
                    / src.revision / "data").glob("*.wav"))
    return np.concatenate([decode_file(c, model.sample_rate) for c in clips])


def measure(model, audio, window_s, warmup, runs):
    n = int(window_s * model.sample_rate)
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(audio) - n + 1, size=warmup + runs)
    times = []
    for i, start in enumerate(starts):
        t0 = time.perf_counter()
        model.predict(audio[start:start + n])
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "p90_ms": times[int(0.9 * (len(times) - 1))],
            "min_ms": times[0], "max_ms": times[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", nargs="?", default=str(MODELS_DIR / "commonaccent_es6"))
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--windows", type=float, nargs="+", default=[1, 2.5, 5, 7.5, 10])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=str(DATA_DIR / "bench" / "latest.json"))
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    t0 = time.perf_counter()
    model = load_model(args.model_dir)
    load_s = time.perf_counter() - t0
    audio = speech_audio(model)
    hop_ms = model.config.audio.hop_seconds * 1000

    print(f"CPU: {cpu_name()} | torch {torch.__version__} | default threads {default_threads}")
    print(f"Model: {model.config.display_name} | load {load_s:.1f} s | "
          f"{args.warmup} warm-up + {args.runs} timed runs per row | update budget {hop_ms:.0f} ms\n")
    header = f"{'threads':>7} {'window':>7} {'median':>8} {'p90':>8} {'min':>8} {'max':>8} {'budget':>7} {'RTF':>6}"
    print(header + "\n" + "-" * len(header))

    settings = [(t, w) for t in args.threads for w in args.windows]
    settings.append(settings[0])  # repeat the first one to detect throttling
    rows = []
    for i, (threads, window) in enumerate(settings):
        torch.set_num_threads(threads)
        r = measure(model, audio, window, args.warmup, args.runs)
        r.update(threads=threads, window_s=window, repeat=(i == len(settings) - 1),
                 budget=r["median_ms"] / hop_ms, rtf=r["median_ms"] / (window * 1000))
        rows.append(r)
        print(f"{threads:>7} {window:>6.1f}s {r['median_ms']:>6.0f}ms {r['p90_ms']:>6.0f}ms "
              f"{r['min_ms']:>6.0f}ms {r['max_ms']:>6.0f}ms {r['budget']:>6.0%} {r['rtf']:>6.2f}"
              + ("   <- repeat of first row" if r["repeat"] else ""))

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"\nbudget = median / {hop_ms:.0f} ms update interval (over 100% = cannot keep up). "
          f"RTF = median / window length.\npeak RSS {peak_gb:.2f} GB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cpu": cpu_name(), "torch": torch.__version__, "load_s": load_s,
                               "peak_rss_gb": peak_gb, "hop_ms": hop_ms, "rows": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
