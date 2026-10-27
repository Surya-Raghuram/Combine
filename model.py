"""model.py -- PhysRes-AttnUNet

Two-stage network for inverse scattering / photoacoustic source reconstruction.

  physics prior (fixed)  ->  Attention Res-UNet (coarse)  ->  residual refiner (DeepNIS-style)

Origin of each piece
  Approach 2 : Helmholtz physics prior fed as input channels; deep encoder with small
               bottleneck (8x8) for a global receptive field; mask head (BCE + Dice).
  Approach 1 : residual blocks everywhere, cascaded residual refinement
               (output = coarse + learned correction), spatially weighted L1.
  Approach 3 : attention-gated skips, bilinear-upsample + conv decoder (no checkerboard),
               SSIM + L1 hybrid loss, multi-frequency conditioning.
  Added      : FiLM conditioning on kf*dx, coordinate channels, deep supervision.

Input : x    (B,2,H,W)  Re/Im of the field SHI (loader output, H=W=128)
        cond (B,1)      kf*dx in radians/pixel (loader output)
Output: dict(coarse={S,m}, fine={S,m});  S (B,2,H,W) = Re/Im of source, m (B,1,H,W) = mask logit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- physics
def helmholtz_prior(x, cond):
    """S_prior = -(lap + kf^2 + i*eps) E  in pixel units (kf = cond rad/px, eps = 0.8 kf^2).

    This is the inverse of the Green's-function operator used by the generator
    (G = 1/(k^2 - kf^2 - i*eps)), i.e. exact in the linear (no-contrast) limit and an
    approximation otherwise. Fixed, no parameters. Output is normalised per sample.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device)
        k = k.view(1, 1, 3, 3).repeat(2, 1, 1, 1)
        lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=2)
        kf2 = (cond.float() ** 2).view(-1, 1, 1, 1)
        eps = 0.8 * kf2
        re, im = x[:, 0:1], x[:, 1:2]
        p_re = -lap[:, 0:1] - kf2 * re + eps * im
        p_im = -lap[:, 1:2] - kf2 * im - eps * re
        p = torch.cat([p_re, p_im], 1)
        mx = p.pow(2).sum(1, keepdim=True).sqrt().amax(dim=(1, 2, 3), keepdim=True) + 1e-12
        return p / mx


# --------------------------------------------------------------------------- blocks
class CondEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer("w", 2.0 ** torch.arange(8) * 4.0)
        self.mlp = nn.Sequential(nn.Linear(16, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c):                       # c (B,1)
        a = c * self.w                          # (B,8)
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class ResBlock(nn.Module):
    """Pre-activation residual block with FiLM conditioning (Approach 1 residual idea)."""
    def __init__(self, cin, cout, emb=64, dil=1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=dil, dilation=dil)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(e)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class AttnGate(nn.Module):
    """Additive soft-attention gate on a skip connection (Approach 3 / Attention U-Net)."""
    def __init__(self, c, ci):
        super().__init__()
        self.wg = nn.Conv2d(c, ci, 1)
        self.wx = nn.Conv2d(c, ci, 1)
        self.psi = nn.Conv2d(ci, 1, 1)

    def forward(self, x, g):
        return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))


def up_block(cin, cout):                        # bilinear + conv: no checkerboard artefacts
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(cin, cout, 3, padding=1))


# --------------------------------------------------------------------------- model
class PhysResAttnUNet(nn.Module):
    def __init__(self, chs=(32, 64, 128, 192), bott=256, emb=64, ref_ch=64,
                 ref_dils=(1, 2, 4, 1)):
        super().__init__()
        self.emb = CondEmbed(emb)
        self.stem = nn.Conv2d(7, chs[0], 3, padding=1)
        self.enc0 = ResBlock(chs[0], chs[0], emb)
        self.downs = nn.ModuleList([nn.Conv2d(chs[i - 1], chs[i - 1], 3, stride=2, padding=1)
                                    for i in range(1, 4)])
        self.encs = nn.ModuleList([ResBlock(chs[i - 1], chs[i], emb) for i in range(1, 4)])
        self.down_b = nn.Conv2d(chs[3], chs[3], 3, stride=2, padding=1)
        self.b1 = ResBlock(chs[3], bott, emb)
        self.b2 = ResBlock(bott, bott, emb, dil=2)

        self.ups, self.gates, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in (3, 2, 1, 0):
            cin = bott if i == 3 else chs[i + 1]
            self.ups.append(up_block(cin, chs[i]))
            self.gates.append(AttnGate(chs[i], max(chs[i] // 2, 8)))
            self.decs.append(ResBlock(2 * chs[i], chs[i], emb))
        self.head = nn.Conv2d(chs[0], 3, 1)

        # DeepNIS-style cascaded residual refiner
        self.r_in = nn.Conv2d(7 + 3, ref_ch, 3, padding=1)
        self.r_blocks = nn.ModuleList([ResBlock(ref_ch, ref_ch, emb, dil=d) for d in ref_dils])
        self.r_out = nn.Conv2d(ref_ch, 3, 3, padding=1)
        nn.init.zeros_(self.r_out.weight)
        nn.init.zeros_(self.r_out.bias)

    @staticmethod
    def _compose(t):
        reg, m = t[:, :2], t[:, 2:3]
        return {"S": torch.sigmoid(m) * reg, "m": m, "reg": reg}   # reg = raw regression, before mask gating

    def forward(self, x, cond, use_refiner=True):
        B, _, H, W = x.shape
        prior = helmholtz_prior(x, cond)
        mag = x.pow(2).sum(1, keepdim=True).sqrt()
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        inp = torch.cat([x, prior.to(x.dtype), mag, ys.to(x.dtype), xs.to(x.dtype)], 1)  # 7 ch
        e = self.emb(cond)

        h = self.enc0(self.stem(inp), e)
        feats = [h]
        for d, enc in zip(self.downs, self.encs):
            h = enc(d(h), e)
            feats.append(h)
        h = self.b2(self.b1(self.down_b(h), e), e)
        for k, (up, gate, dec) in enumerate(zip(self.ups, self.gates, self.decs)):
            u = up(h)
            sk = gate(feats[3 - k], u)
            h = dec(torch.cat([u, sk], 1), e)
        coarse = self.head(h)
        out = {"coarse": self._compose(coarse)}

        if use_refiner:
            r = self.r_in(torch.cat([inp, coarse], 1))
            for blk in self.r_blocks:
                r = blk(r, e)
            out["fine"] = self._compose(coarse + self.r_out(F.silu(r)))
        return out


# --------------------------------------------------------------------------- losses
def _gauss(win=11, sigma=1.5, device="cpu"):
    g = torch.arange(win, device=device, dtype=torch.float32) - win // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, L=2.0, win=11, sigma=1.5):
    C = a.shape[1]
    k = _gauss(win, sigma, a.device).repeat(C, 1, 1, 1)
    f = lambda t: F.conv2d(t, k, padding=win // 2, groups=C)
    mu_a, mu_b = f(a), f(b)
    va = f(a * a) - mu_a ** 2
    vb = f(b * b) - mu_b ** 2
    cab = f(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean()


class CombinedLoss(nn.Module):
    """L_rec  = alpha*(1-SSIM) + (1-alpha)*WeightedL1      (Approach 3 hybrid + Approach 1 weighting)
       L_mask = BCE(pos_weight) + dice_w*Dice               (Approach 2)
       total  = L_rec + w_mask*L_mask
    NOTE: with radii 20-50 px on a 128x128 crop the disk covers ~8-48% of the image, so the
    extreme imbalance of the thesis (pos_weight=50) does not apply; defaults are mild.
    """
    def __init__(self, alpha=0.5, obj_w=3.0, pos_weight=1.0, dice_w=1.0, w_mask=1.0):
        super().__init__()
        self.alpha, self.obj_w, self.dice_w, self.w_mask = alpha, obj_w, dice_w, w_mask
        self.register_buffer("pw", torch.tensor(float(pos_weight)))

    def one(self, o, y, mask):
        S, m = o["S"].float(), o["m"].float()
        w = 1.0 + (self.obj_w - 1.0) * mask
        wl1 = (w * (S - y).abs()).mean()
        rec = self.alpha * (1 - ssim(S, y)) + (1 - self.alpha) * wl1
        bce = F.binary_cross_entropy_with_logits(m, mask, pos_weight=self.pw)
        p = torch.sigmoid(m)
        inter = (p * mask).sum((1, 2, 3))
        dice = 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
        return rec + self.w_mask * (bce + self.dice_w * dice)

    def forward(self, out, y, mask, aux=0.5):
        if "fine" in out:
            return self.one(out["fine"], y, mask) + aux * self.one(out["coarse"], y, mask)
        return self.one(out["coarse"], y, mask)


if __name__ == "__main__":
    m = PhysResAttnUNet()
    print("params: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    x, c = torch.randn(2, 2, 128, 128), torch.full((2, 1), 0.3)
    o = m(x, c)
    print({k: (tuple(v["S"].shape), tuple(v["m"].shape)) for k, v in o.items()})
    y, mk = torch.randn(2, 2, 128, 128).clamp(-1, 1), (torch.rand(2, 1, 128, 128) > 0.7).float()
    print("loss", float(CombinedLoss()(o, y, mk)))