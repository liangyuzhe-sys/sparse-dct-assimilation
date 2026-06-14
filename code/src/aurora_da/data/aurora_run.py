"""Run Aurora 0.25 Pretrained inference to produce background fields.

Public functions:
  - load_aurora_model(...) : load + move to GPU + eval mode
  - run_inference(...) : one date end-to-end with idempotence

Inputs: two consecutive ERA5 ICs (t_0-12h, t_0-6h) plus static fields.
Output: one zarr group at out_path with Aurora-format vars on 720 x 1440 grid.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata

from aurora_da.data.schema import (
    REQUIRED_ATMOS_VARS,
    REQUIRED_LEVELS_HPA,
    REQUIRED_STATIC_VARS,
    REQUIRED_SURFACE_VARS,
    snapshot_basename,
    validate_snapshot,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_aurora_model(
    checkpoint_repo: str = "microsoft/aurora",
    checkpoint_name: str = "aurora-0.25-pretrained.ckpt",
    device: str = "cuda",
) -> AuroraPretrained:
    """Build AuroraPretrained, download/cache checkpoint, move to device, eval mode."""
    model = AuroraPretrained(autocast=True, use_lora=False)
    model.load_checkpoint(checkpoint_repo, checkpoint_name)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vars(zarr_path: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Load specified variables from a zarr group as float32 tensors."""
    ds = xr.open_zarr(zarr_path)
    return {v: torch.from_numpy(ds[v].values.copy()).float() for v in names}


def _build_batch(
    ic1_path: Path,
    ic2_path: Path,
    static_path: Path,
    ic2_time: datetime,
) -> Batch:
    """Construct an Aurora Batch from three Zarr inputs."""
    ic1_surf = _load_vars(ic1_path, list(REQUIRED_SURFACE_VARS))
    ic1_atmos = _load_vars(ic1_path, list(REQUIRED_ATMOS_VARS))
    ic2_surf = _load_vars(ic2_path, list(REQUIRED_SURFACE_VARS))
    ic2_atmos = _load_vars(ic2_path, list(REQUIRED_ATMOS_VARS))
    static_vars = _load_vars(static_path, list(REQUIRED_STATIC_VARS))

    ds1 = xr.open_zarr(ic1_path)
    lat = torch.from_numpy(ds1.latitude.values.copy()).float()
    lon = torch.from_numpy(ds1.longitude.values.copy()).float()

    # surf (B=1, T=2, H, W); atmos (B=1, T=2, C, H, W)
    surf_vars = {
        v: torch.stack([ic1_surf[v], ic2_surf[v]], dim=0).unsqueeze(0)
        for v in REQUIRED_SURFACE_VARS
    }
    atmos_vars = {
        v: torch.stack([ic1_atmos[v], ic2_atmos[v]], dim=0).unsqueeze(0)
        for v in REQUIRED_ATMOS_VARS
    }

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(ic2_time,),  # time of latest input
            atmos_levels=REQUIRED_LEVELS_HPA,
        ),
    )


def _save_prediction(
    pred: Batch,
    out_path: Path,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
) -> None:
    """Save Aurora prediction as a Zarr group in Aurora-format layout (720 x 1440)."""
    H_pred = pred.surf_vars["2t"].shape[-2]
    lat_out = lat_full[:H_pred]

    data_vars: dict = {}
    for v in REQUIRED_SURFACE_VARS:
        arr = pred.surf_vars[v][0, 0].cpu().float().numpy()
        data_vars[v] = (("latitude", "longitude"), arr)
    for v in REQUIRED_ATMOS_VARS:
        arr = pred.atmos_vars[v][0, 0].cpu().float().numpy()
        data_vars[v] = (("level", "latitude", "longitude"), arr)

    ds_pred = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": lat_out,
            "longitude": lon_full,
            "level": list(REQUIRED_LEVELS_HPA),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_pred.to_zarr(out_path, mode="w", consolidated=True)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


# ---------------------------------------------------------------------------
# Public: one-date inference
# ---------------------------------------------------------------------------

def run_inference(
    timestamp: datetime,
    ic_dir: Path,
    static_path: Path,
    out_path: Path,
    model: AuroraPretrained,
    device: str = "cuda",
) -> dict:
    """Generate the background field for `timestamp`.

    IC1 = `timestamp - 12h`, IC2 = `timestamp - 6h`. Output saved to `out_path`.
    Idempotent: if `out_path` exists and is valid, skip without running inference.

    Returns: {"size_mb", "inference_s", "peak_gib", "skipped"}.

    Raises:
        FileNotFoundError if any input (IC1, IC2, static) is missing.
    """
    out_path = Path(out_path)

    if out_path.exists():
        report = validate_snapshot(out_path, kind="background")
        if report.ok:
            return {
                "size_mb": _dir_size_mb(out_path),
                "inference_s": 0.0,
                "peak_gib": 0.0,
                "skipped": True,
            }

    ic1_time = timestamp - timedelta(hours=12)
    ic2_time = timestamp - timedelta(hours=6)
    ic1_path = ic_dir / f"{snapshot_basename(ic1_time)}.zarr"
    ic2_path = ic_dir / f"{snapshot_basename(ic2_time)}.zarr"

    for p, label in ((ic1_path, "IC1"), (ic2_path, "IC2"), (static_path, "static")):
        if not p.exists():
            raise FileNotFoundError(f"{label} missing: {p}")

    batch = _build_batch(ic1_path, ic2_path, static_path, ic2_time)
    batch = batch.to(device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with torch.inference_mode():
        pred = model.forward(batch)
    elapsed = time.monotonic() - t0
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

    ds1 = xr.open_zarr(ic1_path)
    lat_full = ds1.latitude.values
    lon_full = ds1.longitude.values

    _save_prediction(pred, out_path, lat_full, lon_full)

    del batch, pred
    torch.cuda.empty_cache()

    return {
        "size_mb": _dir_size_mb(out_path),
        "inference_s": elapsed,
        "peak_gib": peak_gib,
        "skipped": False,
    }