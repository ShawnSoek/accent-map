"""Run the base model over the items of a bundle and store the results. Resumable; the same code
runs on the laptop (CPU) and on Kaggle (GPU).

Bundle folder (made by scripts/kaggle_bundle.py):
  items.tsv  augment.yaml  features.yaml  clips/*.mp3  noise/RIRS_NOISES/...  (and code/, ...)

Output, per set and part, in --out:
  <set>.part<i>of<n>.npy   float16 (items, layers, 2 * width), rows in the order of .ids
  <set>.part<i>of<n>.ids   item ids, one per line
  <set>.part<i>of<n>.done  number of rows filled so far; a new run continues after it

Usage:
  python -m training.extract_job --bundle B --model M --out O [--device cuda] [--part 0 --parts 2]
                                 [--sets train,dev] [--ids FILE] [--workers 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from server.audio import decode_file
from training.augment import SAMPLE_RATE, NoiseBank, apply, load_augment_spec, plan_for
from training.feature_items import SETS, Job, read_upload_table

SAVE_EVERY = 200


class AudioSource(torch.utils.data.Dataset):
    """The audio of each job: its clips decoded to 16 kHz and joined, augmented if asked."""

    def __init__(self, jobs: list[Job], clips_dir: Path, augment_spec=None, bank: NoiseBank | None = None):
        self.jobs, self.clips_dir, self.spec, self.bank = jobs, Path(clips_dir), augment_spec, bank

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i: int) -> np.ndarray:
        job = self.jobs[i]
        audio = np.concatenate([decode_file(self.clips_dir / c, SAMPLE_RATE) for c in job.clips])
        if job.augment:
            audio = apply(audio, plan_for(job.clips[0], self.spec, self.bank), self.bank)
        return audio


def _write_done(path: Path, n: int) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(f"{n}\n")
    os.replace(tmp, path)


def run(jobs: list[Job], clips_dir: Path, out_dir: Path, extractor: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int], augment_spec=None, bank: NoiseBank | None = None, part: int = 0, parts: int = 1,
        workers: int = 0, log: Callable[[str], None] = print) -> dict[str, int]:
    """Extract every job of this part. Returns rows filled per set."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filled = {}
    for set_name in [s for s in SETS if any(j.set == s for j in jobs)]:
        mine = [j for k, j in enumerate(j for j in jobs if j.set == set_name) if k % parts == part]
        stem = f"{set_name}.part{part}of{parts}"        # not Path.with_suffix: ".part0of2" is no suffix
        ids_file, npy_file, done_file = (out_dir / f"{stem}{s}" for s in (".ids", ".npy", ".done"))
        ids = "".join(f"{j.id}\n" for j in mine)
        if ids_file.exists() and ids_file.read_text() != ids:
            raise RuntimeError(f"{ids_file} lists other items; use another --out")
        ids_file.write_text(ids)
        exists = npy_file.exists()
        rows = open_memmap(npy_file, mode="r+" if exists else "w+", dtype=np.float16, shape=(len(mine), *shape))
        if rows.shape != (len(mine), *shape):
            raise RuntimeError(f"{npy_file} has shape {rows.shape}, expected {(len(mine), *shape)}")
        done = int(done_file.read_text()) if exists and done_file.exists() else 0
        if done < len(mine):
            log(f"{set_name} part {part}/{parts}: {len(mine)} items, starting at {done}")
            source = AudioSource(mine[done:], clips_dir, augment_spec, bank)
            loader = torch.utils.data.DataLoader(source, batch_size=None, num_workers=workers)
            t0, seconds = time.perf_counter(), 0.0
            for k, audio in enumerate(loader, start=done):
                audio = np.asarray(audio, dtype=np.float32)
                seconds += len(audio) / SAMPLE_RATE
                rows[k] = extractor(audio)
                if (k + 1) % SAVE_EVERY == 0 or k + 1 == len(mine):
                    rows.flush()
                    _write_done(done_file, k + 1)
                if (k + 1) % 2000 == 0 or k + 1 == len(mine):
                    took = time.perf_counter() - t0
                    log(f"  {set_name}: {k + 1}/{len(mine)} done, {seconds / took:.1f} s of audio per second")
        del rows
        filled[set_name] = len(mine)
    return filled


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """How close two feature arrays are: lowest cosine similarity of any (item, layer, mean/std)
    vector, and the largest difference relative to the vector's size."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    half = a.shape[-1] // 2
    a = a.reshape(*a.shape[:-1], 2, half)
    b = b.reshape(*b.shape[:-1], 2, half)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    rel = np.abs(a - b).max(-1) / np.maximum(np.abs(a).max(-1), 1e-12)
    return {"min_cosine": float(cos.min()), "max_relative_difference": float(rel.max())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="folder with the base model's files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--ids", type=Path, help="only the item ids listed in this file")
    ap.add_argument("--workers", type=int, default=0, help="processes decoding audio while the model runs")
    args = ap.parse_args(argv)

    from training.features import FeatureExtractor, load_features_spec

    jobs = [j for j in read_upload_table(args.bundle / "items.tsv") if j.set in args.sets.split(",")]
    if args.ids:
        wanted = set(args.ids.read_text().split())
        jobs = [j for j in jobs if j.id in wanted]
    spec = load_augment_spec(args.bundle / "augment.yaml")
    bank = NoiseBank(args.bundle / "noise", spec.reverb.simulated_rooms) if any(j.augment for j in jobs) else None
    if args.device.startswith("cuda"):          # full float32, like the laptop's CPU: no TF32 shortcuts
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    features_spec = load_features_spec(args.bundle / "features.yaml")
    extractor = FeatureExtractor(args.model, args.device, features_spec.storage_scale)
    print(f"base model loaded on {args.device} ({extractor.unused_keys} pre-training weights not needed); "
          f"{len(jobs)} items", flush=True)
    filled = run(jobs, args.bundle / "clips", args.out, extractor, (extractor.layers, 2 * extractor.width),
                 spec, bank, args.part, args.parts, args.workers, log=lambda m: print(m, flush=True))
    print(f"finished: {filled}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
