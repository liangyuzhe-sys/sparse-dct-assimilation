"""Fetch ERA5 snapshots from Google ARCO into Aurora-format Zarr files.

Conceptual model:
- ARCO ERA5 stores ECMWF reanalysis as one giant cloud Zarr with
  ECMWF long variable names (e.g. "2m_temperature") and 37 pressure levels.
- We want individual per-timestamp Zarr files using Aurora's short names
  (e.g. "2t") restricted to 13 pressure levels.
- Each fetched snapshot is idempotent: if a valid output already exists,
  we skip.

Two kinds of snapshots:
- IC/background/truth (all same layout): 4 surface vars + 5 atmos vars x 13 levels.
- Static: 3 time-invariant fields (lsm, slt, surface z).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from aurora_da.data.schema import (
    REQUIRED_LEVELS_HPA,
    snapshot_basename,
    validate_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ARCO ERA5's time axis spans 1900..2050 but real data starts at the dataset's
# `valid_time_start` attribute (~1940). The static (time-invariant) fields are
# stored with a time dimension whose pre-valid entries are NaN. To fetch them
# we MUST select a time inside the valid range. We use the same reference time
# as the smoke test for consistency.
STATIC_REFERENCE_TIME = datetime(2023, 1, 2, 6)

# ARCO long names -> Aurora batch keys.
# Two atmospheric/surface maps because they are selected together for IC/truth.
RENAME_SURFACE: dict[str, str] = {
    "2m_temperature": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
}
RENAME_ATMOS: dict[str, str] = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}
RENAME_STATIC: dict[str, str] = {
    "land_sea_mask": "lsm",
    "soil_type": "slt",
    "geopotential_at_surface": "z",
}


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------

def open_arco(source_uri: str = DEFAULT_ARCO_URL) -> xr.Dataset:
    """Open the ARCO ERA5 store. Metadata only; no data transferred yet."""
    return xr.open_zarr(
        source_uri,
        chunks=None,
        storage_options=dict(token="anon"),
        consolidated=True,
    )


# ---------------------------------------------------------------------------
# Fetch one IC / truth snapshot
# ---------------------------------------------------------------------------

def fetch_snapshot(
    timestamp: datetime,
    out_path: Path,
    ds: xr.Dataset | None = None,
    source_uri: str = DEFAULT_ARCO_URL,
) -> dict:
    """Fetch one ERA5 snapshot at `timestamp` into the project's Zarr layout.

    Idempotent: if `out_path` already exists and passes schema validation,
    nothing is downloaded and `result["skipped"]` is True.

    Args:
        timestamp: requested time, hour-precision (UTC).
        out_path: destination .zarr directory.
        ds: optional pre-opened ARCO dataset. Pass this when fetching many
            snapshots in a loop to avoid the ~7s metadata open per call.
        source_uri: ARCO Zarr URL.

    Returns:
        {"size_mb": float, "fetch_time_s": float, "skipped": bool}
    """
    out_path = Path(out_path)

    if out_path.exists():
        report = validate_snapshot(out_path, kind="ic")
        if report.ok:
            return {
                "size_mb": _dir_size_mb(out_path),
                "fetch_time_s": 0.0,
                "skipped": True,
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ds is None:
        ds = open_arco(source_uri)

    arco_vars = list(RENAME_SURFACE) + list(RENAME_ATMOS)
    sub = (
        ds[arco_vars]
        .sel(time=timestamp)
        .sel(level=list(REQUIRED_LEVELS_HPA))
        .rename({**RENAME_SURFACE, **RENAME_ATMOS})
    )

    t0 = time.monotonic()
    sub.to_zarr(out_path, mode="w", consolidated=True)
    elapsed = time.monotonic() - t0

    return {
        "size_mb": _dir_size_mb(out_path),
        "fetch_time_s": elapsed,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Fetch static
# ---------------------------------------------------------------------------

def fetch_static(
    out_path: Path,
    ds: xr.Dataset | None = None,
    source_uri: str = DEFAULT_ARCO_URL,
) -> dict:
    """Fetch the time-invariant static fields (lsm, slt, surface z).

    Idempotent (same semantics as `fetch_snapshot`).
    """
    out_path = Path(out_path)

    if out_path.exists():
        report = validate_snapshot(out_path, kind="static")
        if report.ok:
            return {
                "size_mb": _dir_size_mb(out_path),
                "fetch_time_s": 0.0,
                "skipped": True,
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ds is None:
        ds = open_arco(source_uri)

    # Pre-1940 entries on ARCO's time axis are NaN, so we cannot use isel(time=0).
    # Select a known-valid time instead; the values are identical across time.
    sub = (
        ds[list(RENAME_STATIC)]
        .sel(time=STATIC_REFERENCE_TIME)
        .reset_coords("time", drop=True)
        .rename(RENAME_STATIC)
    )

    t0 = time.monotonic()
    sub.to_zarr(out_path, mode="w", consolidated=True)
    elapsed = time.monotonic() - t0

    return {
        "size_mb": _dir_size_mb(out_path),
        "fetch_time_s": elapsed,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Plan the full fetch
# ---------------------------------------------------------------------------

def compute_timestamps(cfg: dict) -> list[datetime]:
    """Return the sorted, unique list of ERA5 timestamps needed by the project.

    Reads `evaluation` (date range, cadence, hour) and `aurora` (IC offsets +
    prediction offset) sections of the project config. Each evaluation date d
    contributes timestamps `d + offset_h` for offset_h in
    `aurora.ic_offsets_h + [aurora.pred_offset_h]`.

    Returns datetime objects (not pandas Timestamps).
    """
    import pandas as pd  # pulled in transitively by xarray

    e = cfg["evaluation"]
    a = cfg["aurora"]

    dates = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    eval_datetimes = [
        datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dates
    ]

    offsets_h = list(a["ic_offsets_h"]) + [a["pred_offset_h"]]

    unique: set[datetime] = set()
    for d in eval_datetimes:
        for h in offsets_h:
            unique.add(d + timedelta(hours=h))

    return sorted(unique)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)