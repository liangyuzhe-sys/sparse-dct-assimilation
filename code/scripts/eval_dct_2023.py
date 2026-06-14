"""Evaluate the zero-training sparse-DCT stage on the 2023 dates.

Same obs setup as eval_cnn_2023.py so the numbers line up. Also reports cov90
(90% credible-interval coverage) from the Gaussian posterior (chol_L), using
the same lat-weighted formula as the CNN eval.

Usage:
    uv run python -u scripts/eval_dct_2023.py --target z500
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
import xarray as xr
import yaml
from scipy.linalg import solve_triangular
try:
    from scipy.fft import idctn as _idctn
except Exception:  # noqa: BLE001
    from scipy.fftpack import idctn as _idctn
from aurora_da.data.schema import snapshot_basename
from aurora_da.observations import extract_variable, generate_observation
from aurora_da.methods import sparse_dct

FRACTION, N_SEEDS = 0.005, 3
M_COV = 128  # posterior samples for the cov90 std field
# sparse_dct.fit hyperparameters
FIT_KW = dict(kh_init=64, kw_init=128, tau_E=0.95, n_outer=5, n_fista=200, lam_init=1e-3)


def eval_dates(cfg):
    e = cfg["evaluation"]
    dr = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    return [datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="z500")
    args = ap.parse_args()
    tg = args.target
    cfg = yaml.safe_load(open("configs/data.yaml"))
    dr = Path(cfg["paths"]["data_root"])
    bd = Path(cfg["paths"]["background_dir"])
    td = Path(cfg["paths"]["era5_ic_dir"])
    var_spec = next(v for v in cfg["variables"]["evaluation"] if v["name"] == tg)
    s = json.load(open(dr / "baseline_stats.json"))[tg]
    sigma_b = s["bg_rmse_native"]
    sigma_obs = s["sigma_obs_native"]
    print(f"{tg}: sigma_b={sigma_b:.3e}  sigma_obs={sigma_obs:.3e}  (DCT fit: {FIT_KW})")
    dates = eval_dates(cfg)
    lat = lon = latw = None
    imps, covs = [], []
    for di, d in enumerate(dates, 1):
        bn = snapshot_basename(d)
        try:
            bgds = xr.open_zarr(bd / f"{bn}.zarr")
            trds = xr.open_zarr(td / f"{bn}.zarr")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {bn} ({type(exc).__name__})")
            continue
        if lat is None:
            lat = bgds.latitude.values
            lon = bgds.longitude.values
            H, W = len(lat), len(lon)
            latw = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (H, W)).astype(np.float32)
        H = len(lat)
        bg = extract_variable(bgds, var_spec).astype(np.float32)[:H]
        tr = extract_variable(trds, var_spec).astype(np.float32)[:H]
        wrms = lambda a: float(np.sqrt((a ** 2 * latw).sum() / latw.sum()))
        r = tr - bg
        seed_imps, seed_covs = [], []
        for seed in range(N_SEEDS):
            obs = generate_observation(tr, fraction=FRACTION, sigma_obs=sigma_obs, seed=seed)
            state = sparse_dct.fit(bg, obs, var_spec, lat, sigma_b=sigma_b,
                                   lon_degrees=lon, **FIT_KW)
            residual = sparse_dct._reconstruct_field(
                state["mu_S"], state["kh_star"], state["kw_star"], bg.shape[0], bg.shape[1])
            xa = bg + residual.astype(bg.dtype)
            seed_imps.append(100.0 * (wrms(r) - wrms(tr - xa)) / wrms(r))

            # --- cov90: posterior std field from the explicit Gaussian posterior ---
            # Sigma_S = (C C^T)^{-1}, C = chol_L (lower).  sample x = C^{-T} z ~ N(0, Sigma_S),
            # reconstruct each sample to a field, take std -> per-gridpoint sd.
            chol_L = state["chol_L"]
            kh, kw = state["kh_star"], state["kw_star"]
            Hf, Wf = residual.shape
            N = chol_L.shape[0]
            rng = np.random.default_rng(1000 + seed)
            zz = rng.standard_normal((N, M_COV))
            dcoef = solve_triangular(chol_L.T, zz, lower=False)          # (N, M)
            full = np.zeros((Hf, Wf, M_COV), dtype=np.float64)
            full[:kh, :kw, :] = dcoef.reshape(kh, kw, M_COV)
            dfields = _idctn(full, type=2, norm="ortho", axes=(0, 1))    # (H, W, M)
            sd = dfields.std(axis=2).astype(np.float32) + 1e-12
            cov = float(((np.abs(r - residual) < 1.6449 * sd) * latw).sum() / latw.sum())
            seed_covs.append(cov)
        imps.append(float(np.mean(seed_imps)))
        covs.append(float(np.mean(seed_covs)))
        print(f"[{di:2d}/{len(dates)}] {bn}  imp={imps[-1]:+.1f}%  cov90={covs[-1]:.2f}")
    imps, covs = np.array(imps), np.array(covs)
    print(f"\n{'='*52}")
    print(f"{tg} sparse-DCT on 2023 eval ({len(imps)} dates x {N_SEEDS} seeds):")
    print(f"  imp   = {imps.mean():+.1f}% +/- {imps.std():.1f}%")
    print(f"  cov90 = {covs.mean():.2f} +/- {covs.std():.2f}")


if __name__ == "__main__":
    main()
