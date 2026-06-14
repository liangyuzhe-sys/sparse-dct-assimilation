"""Prepare all ERA5 snapshots required by the project.

Reads configs/data.yaml, fetches the static snapshot plus N IC/truth snapshots,
maintains a JSON manifest, and is fully idempotent and resumable.

Usage:
    uv run python scripts/prepare_era5.py [--dry-run] [--static-only] [--limit N]

  --dry-run     : print the plan, don't download anything
  --static-only : only fetch the static snapshot (~15 MB, fast)
  --limit N     : fetch only the first N snapshots (smoke testing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from aurora_da.data.era5_fetch import (
    compute_timestamps,
    fetch_snapshot,
    fetch_static,
    open_arco,
)
from aurora_da.data.schema import snapshot_basename


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"static": None, "snapshots": {}}
    with path.open() as f:
        return json.load(f)


def _save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    tmp.replace(path)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    manifest_path = Path(cfg["paths"]["manifest"])
    manifest = _load_manifest(manifest_path)

    # ---- Plan
    timestamps = compute_timestamps(cfg)
    if args.limit:
        timestamps = timestamps[: args.limit]
    static_path = Path(cfg["paths"]["era5_static"])
    ic_dir = Path(cfg["paths"]["era5_ic_dir"])

    print(f"Plan:")
    print(f"  static -> {static_path}")
    print(f"  {len(timestamps)} snapshots -> {ic_dir}/")
    print(f"  range: {timestamps[0]}  to  {timestamps[-1]}")
    print()

    if args.dry_run:
        for ts in timestamps[:5]:
            print(f"  {snapshot_basename(ts)}.zarr")
        if len(timestamps) > 5:
            print(f"  ... ({len(timestamps) - 5} more)")
        return 0

    # ---- Open ARCO once and reuse
    print("Opening ARCO...")
    t0 = time.monotonic()
    ds = open_arco()
    print(f"  OK in {time.monotonic() - t0:.1f}s")
    print()

    # ---- Static
    print("Fetching static...")
    result = fetch_static(static_path, ds=ds)
    action = "skipped (already valid)" if result["skipped"] else f"fetched in {result['fetch_time_s']:.1f}s"
    print(f"  {action}  ({result['size_mb']:.1f} MB)")
    manifest["static"] = {
        "path": str(static_path),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        **result,
    }
    _save_manifest(manifest, manifest_path)

    if args.static_only:
        return 0

    # ---- Snapshots
    print()
    total_start = time.monotonic()
    n_fetched = 0
    n_skipped = 0
    n_failed = 0

    for i, ts in enumerate(timestamps, 1):
        bn = snapshot_basename(ts)
        out_path = ic_dir / f"{bn}.zarr"
        print(f"[{i:3d}/{len(timestamps)}] {bn}", end="  ", flush=True)

        try:
            result = fetch_snapshot(ts, out_path, ds=ds)
        except KeyboardInterrupt:
            print("interrupted by user")
            print()
            print("Progress is saved in the manifest. Re-run to resume.")
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {type(exc).__name__}: {exc}")
            manifest["snapshots"][bn] = {
                "path": str(out_path),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {exc}",
                "valid": False,
            }
            _save_manifest(manifest, manifest_path)
            n_failed += 1
            continue

        if result["skipped"]:
            print(f"skipped         ({result['size_mb']:5.1f} MB)")
            n_skipped += 1
        else:
            print(f"OK {result['fetch_time_s']:5.1f}s  ({result['size_mb']:5.1f} MB)")
            n_fetched += 1

        manifest["snapshots"][bn] = {
            "path": str(out_path),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "valid": True,
            **result,
        }
        _save_manifest(manifest, manifest_path)

    elapsed = time.monotonic() - total_start
    print()
    print(f"Done in {elapsed/60:.1f} minutes.")
    print(f"  fetched : {n_fetched}")
    print(f"  skipped : {n_skipped}")
    print(f"  failed  : {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())