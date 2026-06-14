"""Prepare Aurora background fields for all evaluation dates.

Reads configs/data.yaml, loads the Aurora model once, then for each evaluation
date runs inference on the corresponding ICs and saves the background as zarr.
Maintains a JSON manifest. Idempotent and resumable.

Usage:
    uv run python scripts/prepare_background.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from aurora_da.data.aurora_run import load_aurora_model, run_inference
from aurora_da.data.schema import snapshot_basename


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"backgrounds": {}}
    with path.open() as f:
        m = json.load(f)
    if "backgrounds" not in m:
        m["backgrounds"] = {}
    return m


def _save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    tmp.replace(path)


def compute_eval_dates(cfg: dict) -> list[datetime]:
    """Return the 52 weekly evaluation datetimes."""
    import pandas as pd
    e = cfg["evaluation"]
    dates = pd.date_range(start=e["start_date"], end=e["end_date"], freq=e["cadence"])
    return [datetime(d.year, d.month, d.day, e["hour_utc"]) for d in dates]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_dates = compute_eval_dates(cfg)
    if args.limit:
        eval_dates = eval_dates[: args.limit]

    bg_dir = Path(cfg["paths"]["background_dir"])
    static_path = Path(cfg["paths"]["era5_static"])
    ic_dir = Path(cfg["paths"]["era5_ic_dir"])
    manifest_path = Path(cfg["paths"]["manifest"])

    print("Plan:")
    print(f"  {len(eval_dates)} background fields -> {bg_dir}/")
    print(f"  range: {eval_dates[0]}  to  {eval_dates[-1]}")
    print()

    if args.dry_run:
        for d in eval_dates[:5]:
            print(f"  {snapshot_basename(d)}.zarr")
        if len(eval_dates) > 5:
            print(f"  ... ({len(eval_dates) - 5} more)")
        return 0

    print("Loading Aurora model (~30s)...")
    t0 = time.monotonic()
    model = load_aurora_model()
    print(f"  OK in {time.monotonic() - t0:.1f}s")
    print()

    manifest = _load_manifest(manifest_path)
    n_done = 0
    n_skip = 0
    n_fail = 0
    total_start = time.monotonic()

    for i, d in enumerate(eval_dates, 1):
        bn = snapshot_basename(d)
        out_path = bg_dir / f"{bn}.zarr"
        print(f"[{i:2d}/{len(eval_dates)}] {bn}", end="  ", flush=True)

        try:
            result = run_inference(d, ic_dir, static_path, out_path, model)
        except KeyboardInterrupt:
            print("interrupted by user")
            return 130
        except FileNotFoundError as exc:
            # Most likely the ERA5 fetch hasn't gotten to this date yet.
            # Don't write to manifest; we'll retry on the next run.
            print(f"SKIP (data not yet available): {exc}")
            n_fail += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {type(exc).__name__}: {exc}")
            manifest["backgrounds"][bn] = {
                "path": str(out_path),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {exc}",
                "valid": False,
            }
            _save_manifest(manifest, manifest_path)
            n_fail += 1
            continue

        if result["skipped"]:
            print(f"skipped         ({result['size_mb']:5.1f} MB)")
            n_skip += 1
        else:
            print(
                f"OK {result['inference_s']:5.1f}s  "
                f"({result['size_mb']:5.1f} MB, peak {result['peak_gib']:.1f} GiB)"
            )
            n_done += 1

        manifest["backgrounds"][bn] = {
            "path": str(out_path),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "valid": True,
            **result,
        }
        _save_manifest(manifest, manifest_path)

    elapsed = time.monotonic() - total_start
    print()
    print(f"Done in {elapsed / 60:.1f} minutes.")
    print(f"  inferred : {n_done}")
    print(f"  skipped  : {n_skip}")
    print(f"  failed   : {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())