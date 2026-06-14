"""Stream training-data prep for the Class-B CNN residual models.

Per date: fetch ERA5 (d-12h, d-6h, d) to /dev/shm, run frozen Aurora, extract
targets + context, save ONE .npz, delete scratch. Resumable (existing .npz are
skipped).

Robustness: each date is wrapped in a wall-clock timeout (SIGALRM) so that a
hung ARCO download cannot stall the whole run. On timeout the date is retried;
after MAX_RETRY failures it is skipped and recorded, and the run continues.

Targets (bg AND truth): q500 q700 q850 q925 t500 t850 u700 v700
Context (bg only):       z500 t700
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import argparse
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from aurora_da.data.era5_fetch import fetch_snapshot, fetch_static, open_arco
from aurora_da.data.aurora_run import load_aurora_model, run_inference
from aurora_da.data.schema import snapshot_basename
from aurora_da.observations import extract_variable

SCRATCH = Path("/dev/shm/aurora_scratch")
PER_DATE_TIMEOUT = 240   # seconds; a date taking longer is treated as hung
MAX_RETRY = 2            # attempts per date before skipping it
RETRY_SLEEP = 5          # seconds between attempts
SKIP_DATES = set()   # known-bad ARCO dates: skip without downloading

# Targets: need bg AND truth (train a CNN residual for each)
TARGET_SPECS = {
    "q500": {"name": "q500", "group": "atmospheric", "var": "q", "level": 500},
    "q700": {"name": "q700", "group": "atmospheric", "var": "q", "level": 700},
    "q850": {"name": "q850", "group": "atmospheric", "var": "q", "level": 850},
    "q925": {"name": "q925", "group": "atmospheric", "var": "q", "level": 925},
    "t500": {"name": "t500", "group": "atmospheric", "var": "t", "level": 500},
    "t850": {"name": "t850", "group": "atmospheric", "var": "t", "level": 850},
    "u700": {"name": "u700", "group": "atmospheric", "var": "u", "level": 700},
    "v700": {"name": "v700", "group": "atmospheric", "var": "v", "level": 700},
}
# Context: bg only (large-scale circulation + same-level temperature)
CONTEXT_SPECS = {
    "z500": {"name": "z500", "group": "atmospheric", "var": "z", "level": 500},
    "t700": {"name": "t700", "group": "atmospheric", "var": "t", "level": 700},
}


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def train_dates(cfg):
    e = cfg["evaluation"]
    dr = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    return [datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dr]


def _process_date(d, bn, out_path, ic_offsets, static_path, ds, model):
    """Fetch + infer + extract + save for one date. Raises on failure/timeout."""
    shutil.rmtree(SCRATCH, ignore_errors=True)
    scratch_ic = SCRATCH / "ic"
    scratch_ic.mkdir(parents=True, exist_ok=True)
    scratch_bg = SCRATCH / f"bg_{bn}.zarr"
    try:
        need = [d + timedelta(hours=h) for h in ic_offsets] + [d]
        for ts in need:
            fetch_snapshot(ts, scratch_ic / f"{snapshot_basename(ts)}.zarr", ds=ds)
        run_inference(d, scratch_ic, static_path, scratch_bg, model)
        bgds = xr.open_zarr(scratch_bg)
        trds = xr.open_zarr(scratch_ic / f"{snapshot_basename(d)}.zarr")
        H = len(bgds.latitude)
        arrays = {}
        for k, sp in {**TARGET_SPECS, **CONTEXT_SPECS}.items():
            arrays[f"bg_{k}"] = extract_variable(bgds, sp).astype(np.float32)[:H]
        for k, sp in TARGET_SPECS.items():
            arrays[f"truth_{k}"] = extract_variable(trds, sp).astype(np.float32)[:H]
        arrays["lat"] = bgds.latitude.values.astype(np.float32)[:H]
        arrays["lon"] = bgds.longitude.values.astype(np.float32)
        np.savez_compressed(out_path, **arrays)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_train.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ic_offsets = list(cfg["aurora"]["ic_offsets_h"])          # [-12, -6]
    static_path = Path(cfg["paths"]["era5_static"])
    out_dir = Path("/root/train_pairs")                        # system disk
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = train_dates(cfg)
    if args.limit:
        dates = dates[: args.limit]
    print(f"{len(dates)} training dates -> {out_dir}")
    print(f"targets: {list(TARGET_SPECS)}")
    print(f"context: {list(CONTEXT_SPECS)}")
    print(f"per-date timeout={PER_DATE_TIMEOUT}s  retries={MAX_RETRY}")

    print("Opening ARCO...")
    ds = open_arco()
    fetch_static(static_path, ds=ds)
    print("Loading Aurora...")
    model = load_aurora_model()

    signal.signal(signal.SIGALRM, _on_alarm)

    n_done = n_skip = 0
    failed = []
    t0 = time.monotonic()
    for i, d in enumerate(dates, 1):
        bn = snapshot_basename(d)
        out_path = out_dir / f"{bn}.npz"
        print(f"[{i:3d}/{len(dates)}] {bn}", end="  ", flush=True)
        if out_path.exists():
            print("skip (exists)")
            n_skip += 1
            continue
        if bn in SKIP_DATES:
            print("skip (known-bad)")
            n_skip += 1
            continue

        ok = False
        for attempt in range(1, MAX_RETRY + 1):
            try:
                signal.alarm(PER_DATE_TIMEOUT)
                _process_date(d, bn, out_path, ic_offsets, static_path, ds, model)
                signal.alarm(0)
                print(f"OK ({out_path.stat().st_size / 1e6:.1f} MB)")
                n_done += 1
                ok = True
                break
            except _Timeout:
                signal.alarm(0)
                print(f"timeout({attempt}/{MAX_RETRY})", end="  ", flush=True)
            except KeyboardInterrupt:
                signal.alarm(0)
                shutil.rmtree(SCRATCH, ignore_errors=True)
                print("\ninterrupted; rerun to resume")
                return 130
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                print(f"err({attempt}/{MAX_RETRY}):{type(exc).__name__}",
                      end="  ", flush=True)
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP)
        if not ok:
            print("-> SKIPPED after retries")
            failed.append(bn)

    dt = (time.monotonic() - t0) / 60
    print(f"\nDone in {dt:.1f} min.  done={n_done} skip={n_skip} failed={len(failed)}")
    if failed:
        print("failed dates (rerun later to fill in):")
        for bn in failed:
            print(f"  {bn}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())