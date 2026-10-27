"""train.py -- train PhysRes-AttnUNet on the CBS dataset.

    python train.py --root cbs_data --out runs/exp1               # full run
    python train.py --root cbs_data --out runs/smoke --smoke      # 2 quick epochs

Phase A (epochs < --stage1_epochs): only the coarse Attention Res-UNet is trained.
Phase B: refiner switched on, joint training with deep supervision.
Best checkpoint (lowest val loss) is evaluated on the test split at the end.
"""
import argparse
import json
import math
import os
import time

import torch

from model import CombinedLoss, PhysResAttnUNet
from scattering_dataset import get_loaders

PX_UM = 0.1   # 100 nm per pixel


@torch.no_grad()
def batch_metrics(m_logit, mask):
    pred = (torch.sigmoid(m_logit.float()) > 0.5).float()
    inter = (pred * mask).sum((1, 2, 3))
    union = ((pred + mask) > 0).float().sum((1, 2, 3))
    dice = (2 * inter + 1e-6) / (pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    B, _, H, W = mask.shape
    ys = torch.arange(H, device=mask.device).view(1, 1, H, 1).float()
    xs = torch.arange(W, device=mask.device).view(1, 1, 1, W).float()
    ap, ag = pred.sum((1, 2, 3)), mask.sum((1, 2, 3))
    cp = torch.stack([(pred * ys).sum((1, 2, 3)) / ap.clamp(min=1), (pred * xs).sum((1, 2, 3)) / ap.clamp(min=1)], 1)
    cg = torch.stack([(mask * ys).sum((1, 2, 3)) / ag.clamp(min=1), (mask * xs).sum((1, 2, 3)) / ag.clamp(min=1)], 1)
    cerr = (cp - cg).norm(dim=1)
    rerr = (ap / math.pi).sqrt() - (ag / math.pi).sqrt()          # signed radius error (px)
    valid = ap > 0
    return dice, iou, cerr, rerr, valid


def evaluate(model, dl, crit, dev, amp_dtype, use_refiner=True, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    D, I, C, R, V = [], [], [], [], []
    with torch.no_grad():
        for bi, b in enumerate(dl):
            if max_batches and bi >= max_batches:
                break
            x, y, mk, c = (b[k].to(dev) for k in ("x", "y", "mask", "cond"))
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x, c, use_refiner=use_refiner)
            tot += float(crit(out, y, mk)) * x.shape[0]
            n += x.shape[0]
            key = "fine" if use_refiner else "coarse"
            d, i, ce, re, v = batch_metrics(out[key]["m"], mk)
            D.append(d); I.append(i); C.append(ce); R.append(re); V.append(v)
    D, I, C, R, V = (torch.cat(t) for t in (D, I, C, R, V))
    return {
        "loss": tot / max(n, 1),
        "dice": float(D.mean()), "iou": float(I.mean()),
        "centroid_err_px": float(C[V].mean()) if V.any() else float("nan"),
        "centroid_err_um": float(C[V].mean()) * PX_UM if V.any() else float("nan"),
        "radius_err_px": float(R[V].mean()) if V.any() else float("nan"),
        "abs_radius_err_px": float(R[V].abs().mean()) if V.any() else float("nan"),
        "empty_pred_frac": float((~V).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cbs_data")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stage1_epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--obj_w", type=float, default=3.0)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.stage1_epochs = 2, 1

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if dev.type == "cuda" and not a.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    mb = 3 if a.smoke else None

    train_dl, val_dl, test_dl = get_loaders(a.root, batch_size=a.bs, num_workers=a.workers)
    print(f"device={dev} amp={amp_dtype} train/val/test = "
          f"{len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")

    model = PhysResAttnUNet().to(dev)
    crit = CombinedLoss(a.alpha, a.obj_w, a.pos_weight, a.dice_w).to(dev)
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (mb or len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    best, log = float("inf"), []
    for ep in range(a.epochs):
        joint = ep >= a.stage1_epochs
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for bi, b in enumerate(train_dl):
            if mb and bi >= mb:
                break
            x, y, mk, c = (b[k].to(dev) for k in ("x", "y", "mask", "cond"))
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x, c, use_refiner=joint)
            loss = crit(out, y, mk)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss); nb += 1
        val = evaluate(model, val_dl, crit, dev, amp_dtype, use_refiner=joint, max_batches=mb)
        row = {"epoch": ep + 1, "phase": "B" if joint else "A", "train_loss": run / max(nb, 1), **{f"val_{k}": v for k, v in val.items()}}
        log.append(row)
        print(f"ep {ep+1:3d} [{row['phase']}] train {row['train_loss']:.4f} | val loss {val['loss']:.4f} "
              f"dice {val['dice']:.3f} cen {val['centroid_err_px']:.2f}px rad {val['radius_err_px']:+.2f}px "
              f"({time.time()-t0:.0f}s)")
        # only compare val loss within the same phase mix (phase B loss includes the refiner term)
        if joint and val["loss"] < best:
            best = val["loss"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    ck = os.path.join(a.out, "best.pt")
    if not os.path.exists(ck):
        ck = os.path.join(a.out, "last.pt")
    model.load_state_dict(torch.load(ck, map_location=dev))
    res = evaluate(model, test_dl, crit, dev, amp_dtype, use_refiner=True, max_batches=mb)
    print("TEST:", json.dumps(res, indent=1))
    with open(os.path.join(a.out, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()