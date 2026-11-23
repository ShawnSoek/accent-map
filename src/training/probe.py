"""Which of the base model's 25 layers carry the accent (stage 2, step 5a).

One linear classifier per layer, all trained together from the same blocks of rows, so the 9.9 GB
of features are read in order a few times instead of at random (this disk reads ~250 MB/s in
order, ~50 MB/s at random). Inputs are standardized per layer and dimension with statistics of
the clean training clips. The loss weights every class equally (Mexico has 12x the Canaries'
training audio).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from training.feature_store import FeatureSet, ItemInfo

BLOCK = 2000                        # rows read at once: ~200 MB as float16, ~140 speakers


def layer_stats(fset: FeatureSet, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std per (layer, dimension), over all rows of the set."""
    n, total, squares = 0, 0.0, 0.0
    for _, x in fset.blocks(block):
        total = total + x.sum(0, dtype=np.float64)          # float64 sums without a float64 copy
        squares = squares + np.square(x).sum(0, dtype=np.float64)
        n += len(x)
    mean = total / n
    std = np.sqrt(np.maximum(squares / n - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Weights that make every class count the same in the loss (mean weight per item is 1)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"classes without training items: {np.flatnonzero(counts == 0).tolist()}")
    return (len(labels) / (n_classes * counts)).astype(np.float32)


class LayerProbes(torch.nn.Module):
    """A linear classifier for every layer: (batch, layers, dim) -> (batch, layers, classes)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, n_classes: int):
        super().__init__()
        layers, dim = mean.shape
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))
        self.weight = torch.nn.Parameter(torch.zeros(layers, dim, n_classes))
        self.bias = torch.nn.Parameter(torch.zeros(layers, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        return torch.einsum("bld,ldc->blc", z, self.weight) + self.bias


def train_probes(fset: FeatureSet, info: dict[str, ItemInfo], mean: np.ndarray, std: np.ndarray, n_classes: int,
                 epochs: int = 4, batch: int = 256, lr: float = 1e-3, weight_decay: float = 1e-3, seed: int = 0,
                 block: int = BLOCK, log: Callable[[str], None] = print) -> LayerProbes:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.array([info[i].label for i in fset.ids])
    weights = torch.from_numpy(class_weights(labels, n_classes))
    model = LayerProbes(mean, std, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        total, seen = 0.0, 0
        for ids, x in fset.blocks(block, rng):
            y = torch.tensor([info[i].label for i in ids])
            x = torch.from_numpy(x)
            for idx in torch.from_numpy(rng.permutation(len(ids))).split(batch):
                logits = model(x[idx])                                       # (b, layers, classes)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), y[idx].repeat_interleave(logits.shape[1]),
                    weight=weights) * logits.shape[1]                         # sum over layers
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
                seen += len(idx)
        log(f"  epoch {epoch + 1}/{epochs}: loss per layer {total / seen / mean.shape[0]:.3f}")
    return model


@torch.no_grad()
def probe_probabilities(model: LayerProbes, fset: FeatureSet, block: int = BLOCK) -> tuple[list[str], np.ndarray]:
    """Item ids and probabilities (items, layers, classes), in the set's order."""
    ids, out = [], []
    for block_ids, x in fset.blocks(block):
        ids += block_ids
        out.append(torch.softmax(model(torch.from_numpy(x)), dim=-1).numpy())
    return ids, np.concatenate(out)
