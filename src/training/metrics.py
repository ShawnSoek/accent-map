"""How well a classifier does, counted so that big speakers and big classes don't dominate.

Main number, "balanced accuracy by speaker": for each class, the share of correct answers of each
of its speakers, averaged over the speakers; then averaged over the classes. A speaker with 50
clips counts as much as one with 5, and the Canaries count as much as Mexico.

Clips of one speaker are not independent, so the 95 % interval comes from resampling speakers
(within each class), not clips.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def speaker_accuracies(y: np.ndarray, pred: np.ndarray, speakers: list[str], n_classes: int) -> list[np.ndarray]:
    """Per class: the accuracy of each of its speakers (sorted by speaker id)."""
    hits = defaultdict(list)
    for t, p, s in zip(y, pred, speakers):
        hits[(int(t), s)].append(t == p)
    out = [[] for _ in range(n_classes)]
    for (t, s) in sorted(hits):
        out[t].append(float(np.mean(hits[(t, s)])))
    return [np.array(a) for a in out]


def balanced_accuracy(y, pred, speakers, n_classes: int) -> float:
    per = [a.mean() for a in speaker_accuracies(y, pred, speakers, n_classes) if len(a)]
    return float(np.mean(per))


def bootstrap_interval(y, pred, speakers, n_classes: int, rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    per = [a for a in speaker_accuracies(y, pred, speakers, n_classes) if len(a)]
    draws = np.stack([a[rng.integers(0, len(a), size=(rounds, len(a)))].mean(axis=1) for a in per])
    values = draws.mean(axis=0)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def confusion(y, pred, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (np.asarray(y), np.asarray(pred)), 1)
    return m


def calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far the top probability is from the real hit rate."""
    top = probs.max(axis=1)
    hit = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(top, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            err += sel.mean() * abs(top[sel].mean() - hit[sel].mean())
    return float(err)


def report(probs: np.ndarray, y: np.ndarray, speakers: list[str], genders: list[str], class_ids: list[str],
           with_interval: bool = True) -> dict:
    """Everything for one evaluation set."""
    n = len(class_ids)
    y = np.asarray(y)
    pred = probs.argmax(axis=1)
    per = speaker_accuracies(y, pred, speakers, n)
    out = {"items": int(len(y)), "speakers": int(sum(len(a) for a in per)),
           "balanced_accuracy": balanced_accuracy(y, pred, speakers, n),
           "clip_accuracy": float((pred == y).mean()),
           "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)))),
           "calibration_error": calibration_error(probs, y),
           "per_class": {c: {"speakers": int(len(per[k])), "items": int((y == k).sum()),
                             "accuracy": float(per[k].mean()) if len(per[k]) else None}
                         for k, c in enumerate(class_ids)},
           "confusion": confusion(y, pred, n).tolist()}
    if with_interval:
        out["interval_95"] = bootstrap_interval(y, pred, speakers, n)
    genders = np.asarray(genders)
    out["by_gender"] = {}
    for g in sorted(set(genders.tolist())):
        sel = genders == g
        out["by_gender"][g] = {                        # f, m, other (includes not given)
            "speakers": len({s for s, k in zip(speakers, sel) if k}),
            "balanced_accuracy": balanced_accuracy(y[sel], pred[sel], [s for s, k in zip(speakers, sel) if k], n)}
    return out
