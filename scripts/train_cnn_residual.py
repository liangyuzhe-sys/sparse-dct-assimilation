"""Stage 2: small CNN that learns the residual r = truth - bg for one variable.

Parametrized over --target. For the slow-decay variables the sparse-DCT prior
adds little, so the CNN just learns r = truth - bg directly. It outputs a mean
and a log-variance, trained with a latitude-weighted Gaussian NLL so the
variance is calibrated.

Inputs (z-scored): bg_<target>, the innovation d=(obs-bg) on the grid, the obs
mask, and a few context fields (see CONTEXT_FOR). Target: r = truth - bg.
Trained on 128x128 patches; inference is fully convolutional on the full field.
Data: /root/train_pairs/*.npz (2021-2022, one .npz per date with bg+truth for
all targets plus z500/t700 context).

Usage:
    uv run python -u scripts/train_cnn_residual.py --target q850
"""

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import argparse
import glob
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = "/root/train_pairs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, BATCH = 128, 16
STEPS_PER_EPOCH, EPOCHS = 200, 80
OBS_FRACTION, SIGMA_FRAC = 0.005, 0.25
VAL_FRACTION, SEED = 0.15, 0

# Physically-relevant context (bg only) for each target. Every name here must
# exist as bg_<name> in the npz (targets q500/q700/q850/q925/t500/t850/u700/v700
# + context z500/t700).
CONTEXT_FOR = {
    "q500": ["t500", "u700", "v700", "z500"],
    "q700": ["t700", "u700", "v700", "z500"],
    "q850": ["t850", "u700", "v700", "z500"],
    "q925": ["t850", "u700", "v700", "z500"],   # 925 has no temperature target; use 850 neighbour
    "t500": ["q500", "u700", "v700", "z500"],
    "t850": ["q850", "u700", "v700", "z500"],
    "u700": ["v700", "t700", "z500"],
    "v700": ["u700", "t700", "z500"],
}


def load_all(target, ctx):
    paths = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    assert paths, f"no npz in {DATA_DIR}"
    bg_keys = [f"bg_{target}"] + [f"bg_{c}" for c in ctx]
    data = []
    for p in paths:
        z = np.load(p)
        d = {k: z[k].astype(np.float32) for k in bg_keys + [f"truth_{target}"]}
        d["lat"] = z["lat"].astype(np.float32)
        data.append(d)
    return data, bg_keys


def compute_stats(data, idx, target, bg_keys):
    s = {}
    for k in bg_keys:
        v = np.concatenate([data[i][k].ravel() for i in idx])
        s[k] = (float(v.mean()), float(v.std() + 1e-12))
    r = np.concatenate([(data[i][f"truth_{target}"] - data[i][f"bg_{target}"]).ravel() for i in idx])
    s["r_std"] = float(r.std() + 1e-12)
    s["bg_rms"] = float(np.sqrt((r ** 2).mean()))
    return s


class PatchDS(torch.utils.data.Dataset):
    def __init__(self, data, idx, stats, steps, target, ctx):
        self.data, self.idx, self.s, self.steps = data, idx, stats, steps
        self.target, self.ctx = target, ctx
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return self.steps * BATCH

    def __getitem__(self, _):
        s, tg = self.s, self.target
        d = self.data[self.rng.choice(self.idx)]
        H, W = d[f"bg_{tg}"].shape
        i, j = self.rng.integers(0, H - PATCH), self.rng.integers(0, W - PATCH)
        sl = (slice(i, i + PATCH), slice(j, j + PATCH))
        bg, truth = d[f"bg_{tg}"][sl], d[f"truth_{tg}"][sl]
        r = truth - bg
        n = PATCH * PATCH
        k = max(1, int(OBS_FRACTION * n))
        flat = self.rng.choice(n, k, replace=False)
        mask = np.zeros(n, np.float32); mask[flat] = 1.0; mask = mask.reshape(PATCH, PATCH)
        obs = truth + self.rng.normal(0, SIGMA_FRAC * s["bg_rms"], (PATCH, PATCH)).astype(np.float32)
        d_innov = (obs - bg) * mask
        chans = [
            (bg - s[f"bg_{tg}"][0]) / s[f"bg_{tg}"][1],
            d_innov / s["r_std"], mask,
        ]
        for c in self.ctx:
            chans.append((d[f"bg_{c}"][sl] - s[f"bg_{c}"][0]) / s[f"bg_{c}"][1])
        x = np.stack(chans, 0).astype(np.float32)
        y = (r / s["r_std"]).astype(np.float32)[None]
        latw = np.broadcast_to(np.cos(np.deg2rad(d["lat"][i:i+PATCH]))[:, None],
                               (PATCH, PATCH)).astype(np.float32)[None]
        return x, y, latw


class UQNet(nn.Module):
    def __init__(self, in_ch, h=64):
        super().__init__()
        blk = lambda ci, co, dl: nn.Sequential(nn.Conv2d(ci, co, 3, padding=dl, dilation=dl), nn.ReLU())
        self.body = nn.Sequential(blk(in_ch, h, 1), blk(h, h, 2), blk(h, h, 4), blk(h, h, 2), blk(h, h, 1))
        self.mean = nn.Conv2d(h, 1, 1)
        self.logvar = nn.Conv2d(h, 1, 1)

    def forward(self, x):
        z = self.body(x)
        return self.mean(z), torch.clamp(self.logvar(z), -10, 10)


def nll(mean, logvar, y, w):
    loss = 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))
    return (loss * w).sum() / w.sum()


def evaluate(model, data, idx, s, target, ctx):
    model.eval(); rng = np.random.default_rng(123); imps, covs = [], []
    tg = target
    with torch.no_grad():
        for ii in idx:
            d = data[ii]; H, W = d[f"bg_{tg}"].shape
            bg, truth = d[f"bg_{tg}"], d[f"truth_{tg}"]; r = truth - bg
            n = H * W; k = int(OBS_FRACTION * n)
            flat = rng.choice(n, k, replace=False)
            mask = np.zeros(n, np.float32); mask[flat] = 1.0; mask = mask.reshape(H, W)
            obs = truth + rng.normal(0, SIGMA_FRAC * s["bg_rms"], (H, W)).astype(np.float32)
            d_innov = (obs - bg) * mask
            chans = [
                (bg - s[f"bg_{tg}"][0]) / s[f"bg_{tg}"][1], d_innov / s["r_std"], mask,
            ]
            for c in ctx:
                chans.append((d[f"bg_{c}"] - s[f"bg_{c}"][0]) / s[f"bg_{c}"][1])
            x = torch.from_numpy(np.stack(chans, 0)[None]).float().to(DEVICE)
            mean, logvar = model(x)
            v = mean[0, 0].cpu().numpy() * s["r_std"]
            sd = np.sqrt(np.exp(logvar[0, 0].cpu().numpy())) * s["r_std"]
            ana = bg + v
            if tg.startswith("q"):
                ana = np.maximum(ana, 0.0)        # humidity >= 0
            latw = np.broadcast_to(np.cos(np.deg2rad(d["lat"]))[:, None], (H, W)).astype(np.float32)
            wrms = lambda a: np.sqrt((a ** 2 * latw).sum() / latw.sum())
            imps.append(100 * (wrms(r) - wrms(truth - ana)) / wrms(r))
            covs.append(float(((np.abs(r - v) < 1.6449 * sd) * latw).sum() / latw.sum()))
    model.train()
    return float(np.mean(imps)), float(np.mean(covs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="q700", choices=list(CONTEXT_FOR))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = args.target
    ctx = CONTEXT_FOR[target]
    out_model = args.out or f"/root/cnn_{target}_best.pt"
    in_ch = 3 + len(ctx)

    torch.manual_seed(SEED)
    data, bg_keys = load_all(target, ctx)
    n = len(data)
    perm = np.random.default_rng(SEED).permutation(n)
    n_val = max(1, int(VAL_FRACTION * n))
    val_idx, train_idx = list(perm[:n_val]), list(perm[n_val:])
    print(f"target={target}  context={ctx}  in_ch={in_ch}")
    print(f"{n} dates -> {len(train_idx)} train / {len(val_idx)} val")
    s = compute_stats(data, train_idx, target, bg_keys)
    print(f"bg_rms({target})={s['bg_rms']:.3e}  r_std={s['r_std']:.3e}")

    dl = torch.utils.data.DataLoader(PatchDS(data, train_idx, s, STEPS_PER_EPOCH, target, ctx),
                                     batch_size=BATCH, num_workers=0, drop_last=True)
    model = UQNet(in_ch).to(DEVICE)
    print(f"UQNet params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    best, best_state, bad, patience = -1e9, None, 0, 12
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for x, y, w in dl:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            mean, logvar = model(x); loss = nll(mean, logvar, y, w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        imp, cov = evaluate(model, data, val_idx, s, target, ctx); sched.step(-imp)
        print(f"ep{ep:3d}  train_nll={tot/len(dl):+.4f}  val_imp={imp:+.1f}%  cov90={cov:.2f}")
        if imp > best:
            best, best_state, bad = imp, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val_imp={best:+.1f}%)"); break
    torch.save({"state": best_state, "stats": s, "target": target,
                "context": ctx, "in_ch": in_ch}, out_model)
    print(f"saved best (val_imp={best:+.1f}%) -> {out_model}")


if __name__ == "__main__":
    main()