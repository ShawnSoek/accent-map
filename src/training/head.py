"""Stage 2, step 5b: the accent classifier on top of the frozen base model.

Input: the pooled features of a few neighbouring layers, each standardized with the clean
training clips' statistics and then averaged, giving one vector (2 x 1024 numbers: mean and std
over time) per piece of audio. Which layers is chosen on dev, guided by the per-layer probe.

These vectors are written once (build_inputs: one pass over the 9.9 GB of features, ~400 MB
out), so classifier settings can be compared in seconds instead of re-reading the features.

The classifier: standardize the vector, dropout, an optional hidden layer, the output layer.
The loss counts every class equally. Training uses the clean and the noisy copies of the
training clips and keeps the epoch with the best speaker-balanced accuracy on the dev pairs
(~10 s, like the app's window).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from training.feature_store import FeatureSet
from training.metrics import balanced_accuracy
from training.probe import BLOCK, class_weights


def layer_choices(probe_scores: list[float], near: float = 0.05) -> dict[str, list[int]]:
    """Candidate layer sets from the per-layer probe's dev-pair scores: the best layer, windows
    of 3, 5 and 9 layers around it, every layer within `near` of the best, and all layers."""
    layers = len(probe_scores)
    best = int(np.argmax(probe_scores))
    out = {f"layer {best}": [best]}
    for width in (w for w in (3, 5, 9) if w < layers):
        lo = min(max(best - width // 2, 0), layers - width)
        out[f"layers {lo}-{lo + width - 1}"] = list(range(lo, lo + width))
    close = [k for k, s in enumerate(probe_scores) if s >= probe_scores[best] - near]
    out[f"within {near:.0%} of best ({len(close)} layers)"] = close
    out["all layers"] = list(range(layers))
    return out


def combine(x: np.ndarray, mean: np.ndarray, std: np.ndarray, layers: list[int]) -> np.ndarray:
    """(items, layers, dim) -> (items, dim): the chosen layers, standardized, averaged."""
    return ((x[:, layers] - mean[layers]) / std[layers]).mean(axis=1)


def build_inputs(fset: FeatureSet, mean: np.ndarray, std: np.ndarray, choices: dict[str, list[int]],
                 out_dir: Path, block: int = BLOCK) -> None:
    """One pass over `fset`: for each choice of layers, <out_dir>/<choice>/<set>.npy (float16, in
    the set's order) and <set>.ids."""
    out_dir = Path(out_dir)
    outs = {}
    for choice in choices:
        (out_dir / choice).mkdir(parents=True, exist_ok=True)
        (out_dir / choice / f"{fset.name}.ids").write_text("".join(f"{i}\n" for i in fset.ids))
        outs[choice] = open_memmap(out_dir / choice / f"{fset.name}.npy", mode="w+", dtype=np.float16,
                                   shape=(len(fset), fset.shape[1]))
    pos = 0
    for ids, x in fset.blocks(block):
        for choice, layers in choices.items():
            outs[choice][pos:pos + len(ids)] = combine(x, mean, std, layers)
        pos += len(ids)
    for out in outs.values():
        out.flush()


def load_inputs(folder: Path, set_name: str) -> tuple[list[str], np.ndarray]:
    ids = (Path(folder) / f"{set_name}.ids").read_text(encoding="utf-8").split()
    return ids, np.load(Path(folder) / f"{set_name}.npy")


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 0                   # 0: linear
    dropout: float = 0.0
    weight_decay: float = 1e-3
    lr: float = 1e-3
    batch: int = 256
    max_epochs: int = 30
    patience: int = 5                 # stop after this many epochs without a better dev score
    seed: int = 0


class AccentHead(torch.nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(dim))
        self.register_buffer("in_std", torch.ones(dim))
        layers: list[torch.nn.Module] = [torch.nn.Dropout(dropout)]
        if hidden:
            layers += [torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)]
        layers.append(torch.nn.Linear(hidden or dim, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.in_mean) / self.in_std)


def column_stats(x: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each column, in chunks (no full float32 copy of x)."""
    total = squares = 0.0
    for s in range(0, len(x), chunk):
        c = np.asarray(x[s:s + chunk], np.float32)
        total = total + c.sum(0, dtype=np.float64)
        squares = squares + np.square(c).sum(0, dtype=np.float64)
    mean = total / len(x)
    std = np.sqrt(np.maximum(squares / len(x) - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


@torch.no_grad()
def predict(model: AccentHead, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.softmax(model(torch.from_numpy(np.asarray(x[s:s + batch], np.float32))), -1).numpy()
                           for s in range(0, len(x), batch)])


def train_head(x: np.ndarray, y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, dev_speakers: list[str],
               n_classes: int, config: HeadConfig, log: Callable[[str], None] = print) -> tuple[AccentHead, dict]:
    """Returns the model at its best dev epoch and the history (dev score per epoch)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = AccentHead(x.shape[1], n_classes, config.hidden, config.dropout)
    mean, std = column_stats(x)
    model.in_mean.copy_(torch.from_numpy(mean))
    model.in_std.copy_(torch.from_numpy(std))
    weights = torch.from_numpy(class_weights(y, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    yt = torch.from_numpy(np.asarray(y, np.int64))
    best, best_state, history, waited = -1.0, None, [], 0
    for epoch in range(config.max_epochs):
        model.train()
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // config.batch)):
            idx = np.sort(idx)
            loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(np.asarray(x[idx], np.float32))), yt[idx],
                                                     weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = balanced_accuracy(dev_y, predict(model, dev_x).argmax(1), dev_speakers, n_classes)
        history.append(score)
        if score > best:
            best, waited = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        log(f"  epoch {epoch + 1}: dev {score:.1%}{' *' if waited == 0 else ''}")
        if waited >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"config": asdict(config), "dev_history": history, "best_epoch": int(np.argmax(history)) + 1,
                   "best_dev": best}


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """The single temperature that makes the probabilities fit best (lowest log loss) on held-out
    data. It changes no answer, only how sure the model claims to be."""
    logp = np.log(np.clip(probs, 1e-12, None))
    y = np.asarray(y)

    def loss(t: float) -> float:
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))

    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    return float(min(grid, key=loss))


def save_head(path: Path, model: AccentHead, class_ids: list[str], layers: list[int], mean: np.ndarray,
              std: np.ndarray, config: HeadConfig, temperature: float = 1.0) -> None:
    torch.save({"class_ids": list(class_ids), "layers": list(layers),
                "layer_mean": torch.from_numpy(np.ascontiguousarray(mean[layers])),
                "layer_std": torch.from_numpy(np.ascontiguousarray(std[layers])),
                "hidden": config.hidden, "dropout": config.dropout, "temperature": float(temperature),
                "state_dict": model.state_dict()}, path)


class TrainedHead:
    """A saved classifier: the base model's pooled features (items, layers, dim), with the storage
    scaling undone, -> class probabilities."""

    def __init__(self, path: Path):
        path = Path(path)
        saved = read_safetensors_head(path) if path.suffix == ".safetensors" else \
            torch.load(path, map_location="cpu", weights_only=True)
        self.class_ids: list[str] = saved["class_ids"]
        self.layers: list[int] = saved["layers"]
        self.layer_mean = saved["layer_mean"].numpy()
        self.layer_std = saved["layer_std"].numpy()
        self.temperature = float(saved.get("temperature", 1.0))
        dim = self.layer_mean.shape[1]
        self.model = AccentHead(dim, len(self.class_ids), saved["hidden"], saved["dropout"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.saved = saved

    def inputs(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, np.float32)[:, self.layers] - self.layer_mean) / self.layer_std).mean(axis=1)

    @torch.no_grad()
    def probabilities(self, x: np.ndarray, temperature: float | None = None) -> np.ndarray:
        t = self.temperature if temperature is None else temperature
        logits = self.model(torch.from_numpy(self.inputs(x).astype(np.float32)))
        return torch.softmax(logits / t, dim=-1).numpy()


HEAD_FORMAT = "accent-map-head/1"


def export_head(head: TrainedHead, path: Path, storage_scale: dict[int, float], info: dict) -> None:
    """Write a classifier for the app as safetensors: tensors plus a text header (class ids in
    output order, layers, temperature, storage scaling, where it came from). Unlike a .pt file,
    loading it cannot run code."""
    from safetensors.torch import save_file

    tensors = {"layer_mean": torch.from_numpy(np.ascontiguousarray(head.layer_mean)),
               "layer_std": torch.from_numpy(np.ascontiguousarray(head.layer_std))}
    tensors.update({f"head.{k}": v.contiguous() for k, v in head.model.state_dict().items()})
    meta = {"format": HEAD_FORMAT, "class_ids": head.class_ids, "layers": head.layers,
            "hidden": head.saved["hidden"], "dropout": head.saved["dropout"], "temperature": head.temperature,
            "storage_scale": {str(k): v for k, v in storage_scale.items()}, "info": info}
    save_file(tensors, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


def read_safetensors_head(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        raw = f.metadata() or {}
        if raw.get("format") != json.dumps(HEAD_FORMAT):
            raise ValueError(f"{path} is not an accent-map classifier (format {raw.get('format')!r})")
        meta = {k: json.loads(v) for k, v in raw.items()}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return {"class_ids": meta["class_ids"], "layers": meta["layers"], "hidden": meta["hidden"],
            "dropout": meta["dropout"], "temperature": meta["temperature"],
            "storage_scale": {int(k): v for k, v in meta["storage_scale"].items()}, "info": meta["info"],
            "layer_mean": tensors["layer_mean"], "layer_std": tensors["layer_std"],
            "state_dict": {k.removeprefix("head."): v for k, v in tensors.items() if k.startswith("head.")}}
