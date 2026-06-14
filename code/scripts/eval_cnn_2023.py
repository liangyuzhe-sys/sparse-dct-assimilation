"""Evaluate a trained residual CNN on the 2023 dates (--target).

Training/val used 2021-2022, so testing on 2023 checks generalization to an
unseen year. The checkpoint stores target/context/in_ch/stats, which lets this
script rebuild the exact input stack used in training.

Obs setup matches the main table (sparse fraction + sigma_obs, 3 seeds). sigma_obs
uses baseline_stats[target] when available, otherwise falls back to
0.25 * (training residual RMS) from the checkpoint (same 0.25*bg_rmse convention).

Usage:
    uv run python -u scripts/eval_cnn_2023.py --target q850
"""

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
import yaml

from aurora_da.data.schema import snapshot_basename
from aurora_da.observations import extract_variable, generate_observation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRACTION, N_SEEDS, SIGMA_FRAC = 0.005, 3, 0.25

# var spec for every field a target or its context might reference
ALL_SPECS = {
    "q500": {"name": "q500", "group": "atmospheric", "var": "q", "level": 500},
    "q700": {"name": "q700", "group": "atmospheric", "var": "q", "level": 700},
    "q850": {"name": "q850", "group": "atmospheric", "var": "q", "level": 850},
    "q925": {"name": "q925", "group": "atmospheric", "var": "q", "level": 925},
    "t500": {"name": "t500", "group": "atmospheric", "var": "t", "level": 500},
    "t700": {"name": "t700", "group": "atmospheric", "var": "t", "level": 700},
    "t850": {"name": "t850", "group": "atmospheric", "var": "t", "level": 850},
    "u700": {"name": "u700", "group": "atmospheric", "var": "u", "level": 700},
    "v700": {"name": "v700", "group": "atmospheric", "var": "v", "level": 700},
    "z500": {"name": "z500", "group": "atmospheric", "var": "z", "level": 500},
}


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


def eval_dates(cfg):
    e = cfg["evaluation"]
    dr = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    return [datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="q700")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    target = args.target
    model_path = args.model or f"/root/cnn_{target}_best.pt"

    cfg = yaml.safe_load(open("configs/data.yaml"))
    dr = Path(cfg["paths"]["data_root"])
    bd = Path(cfg["paths"]["background_dir"])
    td = Path(cfg["paths"]["era5_ic_dir"])

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    s = ckpt["stats"]; ctx = ckpt["context"]; in_ch = ckpt["in_ch"]
    assert ckpt["target"] == target, f"ckpt target {ckpt['target']} != {target}"
    model = UQNet(in_ch).to(DEVICE)
    model.load_state_dict(ckpt["state"]); model.eval()
    print(f"loaded {model_path}  target={target}  context={ctx}  in_ch={in_ch}")

    # obs noise: prefer baseline_stats[target], else 0.25 * training residual RMS
    try:
        sigma_obs = json.load(open(dr / "baseline_stats.json"))[target]["sigma_obs_native"]
        print(f"sigma_obs from baseline_stats: {sigma_obs:.3e}")
    except (KeyError, FileNotFoundError):
        sigma_obs = SIGMA_FRAC * s["bg_rms"]
        print(f"sigma_obs fallback (0.25*bg_rms): {sigma_obs:.3e}")

    needed = [target] + ctx
    dates = eval_dates(cfg)
    imps, covs = [], []
    with torch.no_grad():
        for di, d in enumerate(dates, 1):
            bn = snapshot_basename(d)
            try:
                bgds = xr.open_zarr(bd / f"{bn}.zarr")
                trds = xr.open_zarr(td / f"{bn}.zarr")
            except Exception as exc:
                print(f"[{di:2d}] {bn}  skip ({type(exc).__name__})")
                continue
            lat = bgds.latitude.values; H = len(lat)
            bg = {k: extract_variable(bgds, ALL_SPECS[k]).astype(np.float32)[:H] for k in needed}
            truth = extract_variable(trds, ALL_SPECS[target]).astype(np.float32)[:H]
            r = truth - bg[target]
            H, W = r.shape
            latw = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], r.shape).astype(np.float32)
            wrms = lambda a: np.sqrt((a ** 2 * latw).sum() / latw.sum())

            di_imps, di_covs = [], []
            for seed in range(N_SEEDS):
                obs = generate_observation(truth, FRACTION, sigma_obs, seed=seed)
                hi, wi, vals = obs["h_indices"], obs["w_indices"], obs["values"]
                mask = np.zeros((H, W), np.float32); mask[hi, wi] = 1.0
                d_innov = np.zeros((H, W), np.float32); d_innov[hi, wi] = vals - bg[target][hi, wi]
                chans = [
                    (bg[target] - s[f"bg_{target}"][0]) / s[f"bg_{target}"][1],
                    d_innov / s["r_std"], mask,
                ]
                for c in ctx:
                    chans.append((bg[c] - s[f"bg_{c}"][0]) / s[f"bg_{c}"][1])
                x = torch.from_numpy(np.stack(chans, 0)[None]).float().to(DEVICE)
                mean, logvar = model(x)
                v = mean[0, 0].cpu().numpy() * s["r_std"]
                sd = np.sqrt(np.exp(logvar[0, 0].cpu().numpy())) * s["r_std"]
                ana = bg[target] + v
                if target.startswith("q"):
                    ana = np.maximum(ana, 0.0)
                di_imps.append(100 * (wrms(r) - wrms(truth - ana)) / wrms(r))
                di_covs.append(float(((np.abs(r - v) < 1.6449 * sd) * latw).sum() / latw.sum()))
            imps.append(float(np.mean(di_imps))); covs.append(float(np.mean(di_covs)))
            print(f"[{di:2d}/{len(dates)}] {bn}  imp={imps[-1]:+.1f}%  cov90={covs[-1]:.2f}")

    imps, covs = np.array(imps), np.array(covs)
    print(f"\n{'='*52}")
    print(f"{target} CNN on 2023 eval ({len(imps)} dates x {N_SEEDS} seeds):")
    print(f"  imp   = {imps.mean():+.1f}% +/- {imps.std():.1f}%")
    print(f"  cov90 = {covs.mean():.2f} +/- {covs.std():.2f}")


if __name__ == "__main__":
    main()