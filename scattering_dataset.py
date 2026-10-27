"""scattering_dataset.py -- loader for the binary CBS dataset written by CBS_GPU_gen.cu

Usage
-----
    python scattering_dataset.py --root cbs_data          # builds splits.json + sanity check
    from scattering_dataset import get_loaders
    train_dl, val_dl, test_dl = get_loaders("cbs_data", batch_size=16)
    
Each item is a dict:
    x      (2,128,128)  Re/Im of the field SHI, divided by max |SHI| (phase preserved)
    y      (2,128,128)  Re/Im of the source S,  divided by max |S|
    mask   (1,128,128)  1 inside the source disk, 0 outside
    cond   (1,)         kf*dx = 2*pi*freq_idx/256  (radians per pixel), for FiLM
    scale_x, scale_s    the divisors (to undo normalisation)
    id                  sample id
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

N = 256            # stored grid
LO, HI = 64, 192   # crop -> 128x128 (PML is 62 px, so the crop is PML-free)
EPS = 1e-30        # field values are large/small physical numbers; keep eps tiny


def read_metadata(root):
    with open(os.path.join(root, "metadata.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def make_splits(root, seed=0, frac=(0.8, 0.1, 0.1), only_converged=True, force=False):
    """Seeded random split by sample id; written to <root>/splits.json."""
    path = os.path.join(root, "splits.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    rows = read_metadata(root)
    ids = [int(r["id"]) for r in rows if (not only_converged or r["converged"] == "1")]
    ids = [i for i in ids if os.path.exists(os.path.join(root, f"sample_{i:05d}.bin"))]
    rng = np.random.RandomState(seed)
    ids = list(rng.permutation(ids))
    n = len(ids)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    splits = {
        "train": sorted(int(i) for i in ids[:n_tr]),
        "val": sorted(int(i) for i in ids[n_tr:n_tr + n_va]),
        "test": sorted(int(i) for i in ids[n_tr + n_va:]),
    }
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits


class ScatteringDataset(Dataset):
    def __init__(self, root, split="train", crop=True):
        self.root, self.crop = root, crop
        splits = make_splits(root)
        self.ids = splits[split]
        meta = {int(r["id"]): r for r in read_metadata(root)}
        self.meta = {i: meta[i] for i in self.ids}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        sid = self.ids[k]
        raw = np.fromfile(os.path.join(self.root, f"sample_{sid:05d}.bin"), dtype=np.float32)
        raw = raw.reshape(4, N, N)
        if self.crop:
            raw = raw[:, LO:HI, LO:HI]
        x, y = raw[0:2].astype(np.float64), raw[2:4].astype(np.float64)

        # joint max-|.| normalisation: keeps relative amplitude and phase of Re/Im
        scale_x = np.sqrt((x ** 2).sum(0)).max() + EPS
        scale_s = np.sqrt((y ** 2).sum(0)).max() + EPS
        x, y = x / scale_x, y / scale_s
        mask = (np.sqrt((y ** 2).sum(0)) > 0.5).astype(np.float32)[None]

        fidx = int(self.meta[sid]["freq_idx"])
        cond = np.array([2 * np.pi * fidx / N], dtype=np.float32)
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "cond": torch.from_numpy(cond),
            "scale_x": torch.tensor(scale_x, dtype=torch.float64),
            "scale_s": torch.tensor(scale_s, dtype=torch.float64),
            "id": sid,
        }


def get_loaders(root, batch_size=16, num_workers=2):
    mk = lambda s, sh: DataLoader(ScatteringDataset(root, s), batch_size=batch_size,
                                  shuffle=sh, num_workers=num_workers, drop_last=sh)
    return mk("train", True), mk("val", False), mk("test", False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cbs_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sp = make_splits(a.root, seed=a.seed, force=True)
    print({k: len(v) for k, v in sp.items()})
    tr, va, te = get_loaders(a.root, batch_size=4, num_workers=0)
    b = next(iter(tr))
    for k, v in b.items():
        print(k, tuple(v.shape) if hasattr(v, "shape") else v)
    print("x range", float(b["x"].min()), float(b["x"].max()),
          "| mask fraction", float(b["mask"].mean()))