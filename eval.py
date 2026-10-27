"""evaluate.py -- load a trained checkpoint, score it on any CBS dataset, and visualise.

    # held-out test split of the original data
    python evaluate.py --ckpt runs/exp1/best.pt --root cbs_data --split test --out eval_test

    # a different dataset (e.g. generated with another seed / settings): use every sample
    ./cbs_gen cbs_data2 200 123 0
    python evaluate.py --ckpt runs/exp1/best.pt --root cbs_data2 --split all --out eval_new

    # ablation: coarse stage only
    python evaluate.py ... --coarse

Writes into --out:
    per_sample.csv   metrics for every sample (id, radius, frequency, vs, dice, centroid err, ...)
    summary.json     mean / median of each metric
    examples.png     input, physics prior, ground truth, prediction, mask probability, overlay
    errors.png       error vs frequency and vs radius, Dice histogram
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PhysResAttnUNet, helmholtz_prior
from scattering_dataset import ScatteringDataset, read_metadata
from train import batch_metrics, PX_UM


def build_dataset(root, split):
    if split != "all":
        return ScatteringDataset(root, split)
    ds = ScatteringDataset(root, "test")          # builds splits.json if absent (harmless)
    meta = {int(r["id"]): r for r in read_metadata(root)}
    ids = sorted(i for i, r in meta.items()
                 if r["converged"] == "1" and os.path.exists(os.path.join(root, f"sample_{i:05d}.bin")))
    ds.ids, ds.meta = ids, {i: meta[i] for i in ids}
    return ds


def contour(ax, arr, color):
    try:
        ax.contour(arr, levels=[0.5], colors=color, linewidths=1.2)
    except Exception:
        pass


def plot_examples(vis, path, bg="raw"):
    """Five columns per sample: input | physics prior | ground truth | actual prediction | normalised prediction.
    Column 4 shows the network's raw regression output (before the mask gates it to 0/1) in physical
    units, or the input field if bg='input'; the predicted boundary (mask prob = 0.5) is the red
    contour and the true boundary the white one. Ground truth shares column 4's colour scale."""
    n = len(vis)
    heads = ["|SHI| input", "physics prior |S|", "ground truth |S|",
             "actual prediction (raw, pre-mask)\nred = predicted, white = true boundary" if bg == "raw"
             else "input field + contours\nred = predicted, white = true boundary",
             "normalised prediction |S|"]
    fig, axs = plt.subplots(n, 5, figsize=(17, 3.3 * n), squeeze=False)
    for i, v in enumerate(vis):
        vmax = float(v["gt"].max()) or 1.0
        ims = [
            axs[i, 0].imshow(v["x"], cmap="viridis"),
            axs[i, 1].imshow(v["prior"], cmap="viridis"),
            axs[i, 2].imshow(v["gt"], cmap="magma", vmin=0, vmax=vmax),
            (axs[i, 3].imshow(v["pred"], cmap="magma", vmin=0, vmax=vmax) if bg == "raw"
             else axs[i, 3].imshow(v["pred"], cmap="gray")),
            axs[i, 4].imshow(v["pred_n"], cmap="magma", vmin=0, vmax=1),
        ]
        contour(axs[i, 3], v["mask"], "w")
        contour(axs[i, 3], v["prob"], "r")
        axs[i, 0].set_ylabel(v["label"], fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
        for j in range(5):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j >= 2:
                fig.colorbar(ims[j], ax=axs[i, j], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
            if i == 0:
                axs[i, j].set_title(heads[j], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="cbs_data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_vis", type=int, default=8)
    ap.add_argument("--coarse", action="store_true", help="use the coarse stage only")
    ap.add_argument("--bg", default="raw", choices=["raw", "input"],
                    help="column 4 background: raw regression output before the mask (default) "
                         "or the input field |SHI| (rings)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysResAttnUNet().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    key = "coarse" if a.coarse else "fine"

    ds = build_dataset(a.root, a.split)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)
    print(f"{len(ds)} samples from {a.root} [{a.split}], stage={key}")

    rows, vis = [], []
    with torch.no_grad():
        for b in dl:
            x, y, mk, c = (b[k].to(dev) for k in ("x", "y", "mask", "cond"))
            out = model(x, c, use_refiner=not a.coarse)[key]
            dice, iou, cerr, rerr, valid = batch_metrics(out["m"], mk)
            prior = helmholtz_prior(x, c)
            S = out["S"].float()
            for j in range(x.shape[0]):
                sid = int(b["id"][j])
                m = ds.meta[sid]
                rows.append({
                    "id": sid, "radius_px": int(m["radius_px"]), "freq_idx": int(m["freq_idx"]),
                    "vs": float(m["vs"]), "dice": float(dice[j]), "iou": float(iou[j]),
                    "centroid_err_px": float(cerr[j]) if valid[j] else float("nan"),
                    "radius_err_px": float(rerr[j]) if valid[j] else float("nan"),
                    "empty_pred": int(not valid[j]),
                })
                if len(vis) < a.n_vis:
                    sc = float(b["scale_s"][j])              # undo the loader's max-|S| normalisation
                    gt_n = y[j].pow(2).sum(0).sqrt().cpu().numpy()
                    pr_n = S[j].pow(2).sum(0).sqrt().cpu().numpy()
                    ce = float(cerr[j]) if valid[j] else float("nan")
                    x_mag = x[j].pow(2).sum(0).sqrt().cpu().numpy()
                    raw_n = out["reg"][j].float().pow(2).sum(0).sqrt().cpu().numpy()
                    vis.append({
                        "x": x_mag,
                        "prior": prior[j].pow(2).sum(0).sqrt().cpu().numpy(),
                        "gt": gt_n * sc,                      # physical units
                        "pred": raw_n * sc if a.bg == "raw" else x_mag,   # column 4 background
                        "prob": torch.sigmoid(out["m"][j, 0].float()).cpu().numpy(),
                        "mask": mk[j, 0].cpu().numpy(),
                        "pred_n": np.clip(pr_n, 0, 1),        # normalised model output
                        "label": (f"id {sid} | r={m['radius_px']}px f={m['freq_idx']} vs={float(m['vs']):.0f}\n"
                                  f"Dice {float(dice[j]):.3f}\n"
                                  f"centroid err {ce:.3f} px ({ce * PX_UM:.4f} um)\n"
                                  f"radius err {float(rerr[j]):+.3f} px"),
                    })

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    summ = {"n": len(rows), "stage": key}
    for k in ("dice", "iou", "centroid_err_px", "radius_err_px"):
        v = np.array([r[k] for r in rows], dtype=float)
        summ[k + "_mean"], summ[k + "_median"] = float(np.nanmean(v)), float(np.nanmedian(v))
    summ["centroid_err_um_mean"] = summ["centroid_err_px_mean"] * PX_UM
    summ["abs_radius_err_px_mean"] = float(np.nanmean(np.abs([r["radius_err_px"] for r in rows])))
    summ["empty_pred_frac"] = float(np.mean([r["empty_pred"] for r in rows]))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))

    # ---- examples figure (absolute [0,1] colour scale on GT / prediction / probability)
    plot_examples(vis, os.path.join(a.out, "examples.png"), a.bg)

    # ---- error summary figure
    fr = np.array([r["freq_idx"] for r in rows]); rad = np.array([r["radius_px"] for r in rows])
    ce = np.array([r["centroid_err_px"] for r in rows], dtype=float)
    re = np.array([r["radius_err_px"] for r in rows], dtype=float)
    dc = np.array([r["dice"] for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fs = sorted(set(fr))
    groups = [ce[(fr == f) & ~np.isnan(ce)] for f in fs]
    try:
        axs[0, 0].boxplot(groups, tick_labels=[str(f) for f in fs])   # matplotlib >= 3.9
    except TypeError:
        axs[0, 0].boxplot(groups, labels=[str(f) for f in fs])
    axs[0, 0].set_xlabel("frequency index"); axs[0, 0].set_ylabel("centroid error (px)")
    axs[0, 1].scatter(rad, ce, s=10); axs[0, 1].set_xlabel("true radius (px)"); axs[0, 1].set_ylabel("centroid error (px)")
    axs[1, 0].scatter(rad, re, s=10); axs[1, 0].axhline(0, color="k", lw=0.6)
    axs[1, 0].set_xlabel("true radius (px)"); axs[1, 0].set_ylabel("radius error, pred - true (px)")
    axs[1, 1].hist(dc, bins=20); axs[1, 1].set_xlabel("Dice"); axs[1, 1].set_ylabel("samples")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "errors.png"), dpi=110)
    plt.close()
    print("saved to", a.out)


if __name__ == "__main__":
    main()