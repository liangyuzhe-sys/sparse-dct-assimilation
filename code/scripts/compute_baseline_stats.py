"""Compute baseline statistics: cos-lat weighted bg-truth RMSE per evaluation
variable, aggregated over all evaluation dates.

Output: a JSON file at `<data_root>/baseline_stats.json` with bg_rmse and
sigma_obs per variable, in both native (ERA5) units and display units (gpm
for z, hPa for msl, etc.). This file is the input to all subsequent DA
methods (it sets sigma_obs).

Sanity check: bg_rmse for z500 should be near 14.69 gpm.

Usage:
    uv run python scripts/compute_baseline_stats.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from aurora_da.data.schema import REQUIRED_LEVELS_HPA, snapshot_basename
from aurora_da.metrics import weighted_rmse_stacked


# Standard gravity used to convert ERA5 geopotential (m^2/s^2) to gpm.
G = 9.80665


def _display_conversion(var_spec: dict) -> tuple[float, str, str]:
    """Return (multiplicative scale, native unit, display unit) for one variable.

    Convention: display_value = native_value * scale.
    """
    var = var_spec["var"]
    if var == "z":
        return 1.0 / G, "m^2/s^2", "gpm"
    if var == "t":
        return 1.0, "K", "K"
    if var == "msl":
        return 0.01, "Pa", "hPa"
    if var == "q":
        return 1000.0, "kg/kg", "g/kg"
    if var in ("u", "v"):
        return 1.0, "m/s", "m/s"
    return 1.0, "?", "?"


def compute_eval_dates(cfg: dict) -> list[datetime]:
    """52 weekly evaluation datetimes from config."""
    e = cfg["evaluation"]
    dates = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    return [datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dates]


def main() -> None:
    with open("configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    eval_dates = compute_eval_dates(cfg)
    eval_vars = cfg["variables"]["evaluation"]
    bg_dir = Path(cfg["paths"]["background_dir"])
    truth_dir = Path(cfg["paths"]["era5_ic_dir"])
    noise_param = cfg["observations"]["noise_param"]

    # Probe one snapshot for grid metadata
    sample_bg = xr.open_zarr(bg_dir / f"{snapshot_basename(eval_dates[0])}.zarr")
    lat = sample_bg.latitude.values
    H, W = len(lat), len(sample_bg.longitude.values)

    print(f"Evaluating {len(eval_vars)} variables over {len(eval_dates)} dates")
    print(f"  bg grid   : ({H}, {W})   ({lat[0]:.2f} to {lat[-1]:.2f})")
    print(f"  bg_dir    : {bg_dir}")
    print(f"  truth_dir : {truth_dir}")
    print(f"  sigma_obs = {noise_param} * bg_rmse (per variable)")
    print()

    # Pre-allocate stacks
    T = len(eval_dates)
    bg_stacks = {v["name"]: np.empty((T, H, W), dtype=np.float32) for v in eval_vars}
    truth_stacks = {v["name"]: np.empty((T, H, W), dtype=np.float32) for v in eval_vars}

    # Load all snapshots, caching full per-variable arrays within each date
    # so that e.g. z500 and z200 share one disk read for ds["z"].
    print("Loading snapshots...")
    t0 = time.monotonic()
    for i, date in enumerate(eval_dates):
        bg_ds = xr.open_zarr(bg_dir / f"{snapshot_basename(date)}.zarr")
        truth_ds = xr.open_zarr(truth_dir / f"{snapshot_basename(date)}.zarr")
        bg_full: dict[str, np.ndarray] = {}
        truth_full: dict[str, np.ndarray] = {}

        for var_spec in eval_vars:
            name = var_spec["name"]
            var = var_spec["var"]
            if var not in bg_full:
                bg_full[var] = bg_ds[var].values
            if var not in truth_full:
                truth_full[var] = truth_ds[var].values

            if var_spec["group"] == "atmospheric":
                lvl = REQUIRED_LEVELS_HPA.index(var_spec["level"])
                bg_arr = bg_full[var][lvl]
                truth_arr = truth_full[var][lvl, :H]
            else:
                bg_arr = bg_full[var]
                truth_arr = truth_full[var][:H]

            bg_stacks[name][i] = bg_arr.astype(np.float32, copy=False)
            truth_stacks[name][i] = truth_arr.astype(np.float32, copy=False)

        if (i + 1) % 10 == 0 or (i + 1) == T:
            elapsed = time.monotonic() - t0
            eta = elapsed * (T - i - 1) / (i + 1)
            print(f"  {i + 1:>2d}/{T}  elapsed {elapsed:>5.1f}s  ETA {eta:>5.1f}s")
    print(f"Total load time: {time.monotonic() - t0:.1f}s")
    print()

    # Compute RMSEs
    results: dict = {}
    print("Per-variable cos-lat weighted RMSE:")
    print(f"{'var':<6}  {'bg_rmse (native)':>22}   "
          f"{'bg_rmse (display)':>22}   {'sigma_obs (display)':>22}")
    print("-" * 84)
    for var_spec in eval_vars:
        name = var_spec["name"]
        bg_rmse = weighted_rmse_stacked(bg_stacks[name], truth_stacks[name], lat)
        sigma_obs = noise_param * bg_rmse
        scale, u_native, u_display = _display_conversion(var_spec)

        results[name] = {
            "var_spec": var_spec,
            "n_dates": T,
            "bg_rmse_native": bg_rmse,
            "sigma_obs_native": sigma_obs,
            "units_native": u_native,
            "bg_rmse_display": bg_rmse * scale,
            "sigma_obs_display": sigma_obs * scale,
            "units_display": u_display,
        }

        print(
            f"{name:<6}  "
            f"{bg_rmse:>14.4f} {u_native:<7}   "
            f"{bg_rmse * scale:>14.4f} {u_display:<7}   "
            f"{sigma_obs * scale:>14.4f} {u_display:<7}"
        )

    out_path = Path(cfg["paths"]["data_root"]) / "baseline_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()