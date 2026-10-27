"""plot_log.py -- training curves from log.json written by train.py

    python plot_log.py --log runs/exp1/log.json --out runs/exp1/curves.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="runs/exp1/log.json")
ap.add_argument("--out", default="runs/exp1/curves.png")
a = ap.parse_args()

log = json.load(open(a.log))
ep = [r["epoch"] for r in log]
tr = [r["train_loss"] for r in log]
va = [r["val_loss"] for r in log]
switch = next((r["epoch"] for r in log if r["phase"] == "B"), None)   # first joint epoch

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))

ax = axs[0]
ax.plot(ep, tr, label="train loss")
ax.plot(ep, va, label="val loss")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)"); ax.set_title("Epoch vs loss")
if switch:
    ax.axvline(switch - 0.5, color="k", ls="--", lw=0.8)
    ax.text(switch - 0.3, ax.get_ylim()[1], " refiner on", va="top", fontsize=8)
ax.legend(); ax.grid(alpha=0.3)

ax = axs[1]
ax.plot(ep, [r["val_dice"] for r in log], label="val Dice")
ax.plot(ep, [r["val_iou"] for r in log], label="val IoU")
ax.set_xlabel("epoch"); ax.set_title("Validation overlap"); ax.grid(alpha=0.3); ax.legend()
if switch:
    ax.axvline(switch - 0.5, color="k", ls="--", lw=0.8)

ax = axs[2]
ax.plot(ep, [r["val_centroid_err_px"] for r in log], label="centroid err (px)")
ax.plot(ep, [abs(r["val_radius_err_px"]) for r in log], label="|radius err| (px)")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_title("Validation localisation error"); ax.grid(alpha=0.3); ax.legend()
if switch:
    ax.axvline(switch - 0.5, color="k", ls="--", lw=0.8)

plt.tight_layout()
plt.savefig(a.out, dpi=130)
print("saved", a.out)