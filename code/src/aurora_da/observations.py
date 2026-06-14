"""Synthetic observation generation and variable extraction.

Two layers:
1. Variable extraction: turn a zarr-backed Dataset + variable spec into a 2D
   (H, W) numpy array, hiding the surface-vs-atmospheric distinction.
2. Observation synthesis: given a truth field, subsample uniformly at random
   and add iid Gaussian noise. Reproducible via integer seed.

Grid convention:
- Truth (ERA5) snapshots are at (721, 1440).
- Background (Aurora) snapshots are at (720, 1440).
- `load_bg_truth_pair` crops truth to 720 by dropping the south-pole row.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from aurora_da.data.schema import REQUIRED_LEVELS_HPA, snapshot_basename


# ---------------------------------------------------------------------------
# Variable extraction
# ---------------------------------------------------------------------------

def extract_variable(ds: xr.Dataset, var_spec: dict) -> np.ndarray:
    """Extract one evaluation variable from a snapshot Dataset as (H, W).

    `var_spec` follows the format in configs/data.yaml, e.g.:
        {"name": "z500", "group": "atmospheric", "var": "z", "level": 500}
        {"name": "msl",  "group": "surface",     "var": "msl"}
    """
    var_name = var_spec["var"]
    group = var_spec["group"]

    if group == "atmospheric":
        level = var_spec["level"]
        level_idx = REQUIRED_LEVELS_HPA.index(level)
        return ds[var_name].values[level_idx]
    elif group == "surface":
        return ds[var_name].values
    else:
        raise ValueError(
            f"unknown group {group!r}; expected 'atmospheric' or 'surface'"
        )


def load_variable_from_zarr(
    snapshot_dir: Path,
    timestamp: datetime,
    var_spec: dict,
) -> np.ndarray:
    """Open a snapshot zarr by timestamp and extract one variable as (H, W)."""
    path = Path(snapshot_dir) / f"{snapshot_basename(timestamp)}.zarr"
    ds = xr.open_zarr(path)
    return extract_variable(ds, var_spec)


def load_bg_truth_pair(
    bg_dir: Path,
    truth_dir: Path,
    timestamp: datetime,
    var_spec: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (background, truth_cropped) for one date+variable, aligned on the
    Aurora 720-lat grid.

    The ERA5 truth at 721 lat is cropped to match the background's 720 lat
    (we drop the south-pole row, lat = -90 degrees).
    """
    bg = load_variable_from_zarr(bg_dir, timestamp, var_spec)
    truth = load_variable_from_zarr(truth_dir, timestamp, var_spec)
    H_bg = bg.shape[0]
    truth = truth[:H_bg]
    return bg, truth


# ---------------------------------------------------------------------------
# Observation synthesis
# ---------------------------------------------------------------------------

def generate_observation(
    truth_field: np.ndarray,
    fraction: float,
    sigma_obs: float,
    seed: int,
) -> dict:
    """Generate sparse, iid-Gaussian-noisy observations of one variable.

    Algorithm:
        1. Sample `int(round(H * W * fraction))` grid points uniformly
           without replacement using a single RNG seeded by `seed`.
        2. Read the truth value at each sampled point.
        3. Add iid Gaussian noise with std dev `sigma_obs` to each.

    Args:
        truth_field: (H, W) ground-truth field for one variable.
        fraction: observation density in (0, 1]. Typical: 0.003 (~3 obs per
            1000 grid cells, matching radiosonde density on a 0.25 deg grid).
        sigma_obs: per-observation noise std dev, same units as truth_field.
        seed: integer seed for reproducibility.

    Returns:
        dict with keys:
            mask        : (H, W) bool, True at observed points.
            h_indices   : (N,) int, latitude row of each obs.
            w_indices   : (N,) int, longitude column of each obs.
            values      : (N,) float, noisy observation values.
            sigma_obs   : float, the noise std passed in.
            seed        : int, the seed used.
            n_obs       : int, number of observations generated.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if sigma_obs < 0:
        raise ValueError(f"sigma_obs must be non-negative, got {sigma_obs}")
    if truth_field.ndim != 2:
        raise ValueError(
            f"truth_field must be 2D (H, W), got shape {truth_field.shape}"
        )

    H, W = truth_field.shape
    rng = np.random.default_rng(seed)

    n_total = H * W
    n_obs = max(int(round(n_total * fraction)), 1)

    flat_idx = rng.choice(n_total, size=n_obs, replace=False)
    h_idx = flat_idx // W
    w_idx = flat_idx % W

    truth_at_obs = truth_field[h_idx, w_idx].astype(np.float64)
    noise = (
        rng.normal(0.0, sigma_obs, size=n_obs)
        if sigma_obs > 0
        else np.zeros(n_obs)
    )
    values = truth_at_obs + noise

    mask = np.zeros((H, W), dtype=bool)
    mask[h_idx, w_idx] = True

    return {
        "mask": mask,
        "h_indices": h_idx,
        "w_indices": w_idx,
        "values": values,
        "sigma_obs": float(sigma_obs),
        "seed": int(seed),
        "n_obs": n_obs,
    }