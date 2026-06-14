"""Data schema for the project.

Collects in one place:
- variable name conventions (matching Aurora's expected input format)
- the pressure-level set
- grid shapes (ICs at 721 lat, backgrounds at 720 lat; Aurora drops one row)
- the snapshot filename convention (ISO-8601, hour precision)
- zarr snapshot validation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import zarr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SURFACE_VARS: tuple[str, ...] = ("2t", "10u", "10v", "msl")

REQUIRED_ATMOS_VARS: tuple[str, ...] = ("t", "u", "v", "q", "z")

# Note: static `z` is SURFACE geopotential, distinct from atmospheric `z`.
REQUIRED_STATIC_VARS: tuple[str, ...] = ("lsm", "slt", "z")

REQUIRED_LEVELS_HPA: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400,
    500, 600, 700, 850, 925, 1000,
)

# ERA5 native grid at 0.25 degree (poles inclusive).
GRID_SHAPE: tuple[int, int] = (721, 1440)

# Aurora 0.25 Pretrained internally drops one latitude row because 721 is not
# divisible by its patch size. So Aurora outputs (and hence background fields)
# live on a 720 x 1440 grid. ICs/truth from ERA5 remain at 721.
BACKGROUND_GRID_SHAPE: tuple[int, int] = (720, 1440)

ISO_FORMAT: str = "%Y-%m-%dT%H"

SnapshotKind = Literal["ic", "background", "static"]


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def snapshot_basename(timestamp: datetime) -> str:
    """Return the canonical filename stem for a snapshot at the given timestamp."""
    return timestamp.strftime(ISO_FORMAT)


def parse_basename(basename: str) -> datetime:
    """Inverse of `snapshot_basename`. Raises ValueError on malformed input."""
    return datetime.strptime(basename, ISO_FORMAT)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating a snapshot directory against the project schema."""

    ok: bool
    path: Path
    kind: str
    missing_surface: tuple[str, ...] = ()
    missing_atmos: tuple[str, ...] = ()
    missing_static: tuple[str, ...] = ()
    missing_levels: tuple[int, ...] = ()
    bad_shape: tuple[str, ...] = ()
    error: str | None = None


def validate_snapshot(path: Path, kind: SnapshotKind) -> ValidationReport:
    """Open a Zarr group at `path` and check schema compliance.

    Args:
        path: directory containing a Zarr group.
        kind:
            - "ic": ERA5 IC/truth snapshot, grid 721 x 1440, surface + atmos.
            - "background": Aurora output, grid 720 x 1440, surface + atmos.
            - "static": time-invariant fields, grid 721 x 1440.

    TODO(spec-02): partial-write and NaN detection.
    """
    path = Path(path)

    if not path.exists():
        return ValidationReport(
            ok=False, path=path, kind=kind, error="path does not exist",
        )

    try:
        group = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # noqa: BLE001
        return ValidationReport(
            ok=False, path=path, kind=kind, error=f"failed to open zarr group: {exc}",
        )

    existing_vars = set(group.array_keys())

    missing_surface: tuple[str, ...] = ()
    missing_atmos: tuple[str, ...] = ()
    missing_static: tuple[str, ...] = ()
    bad_shape: list[str] = []

    if kind in ("ic", "background"):
        # Backgrounds (Aurora output) are at 720 lat; ICs at 721.
        expected_grid = BACKGROUND_GRID_SHAPE if kind == "background" else GRID_SHAPE

        for var in REQUIRED_SURFACE_VARS:
            if var not in existing_vars:
                missing_surface = (*missing_surface, var)
            elif tuple(group[var].shape) != expected_grid:
                bad_shape.append(var)

        expected_atmos_shape = (len(REQUIRED_LEVELS_HPA),) + expected_grid
        for var in REQUIRED_ATMOS_VARS:
            if var not in existing_vars:
                missing_atmos = (*missing_atmos, var)
            elif tuple(group[var].shape) != expected_atmos_shape:
                bad_shape.append(var)

    elif kind == "static":
        for var in REQUIRED_STATIC_VARS:
            if var not in existing_vars:
                missing_static = (*missing_static, var)
            elif tuple(group[var].shape) != GRID_SHAPE:
                bad_shape.append(var)

    else:
        return ValidationReport(
            ok=False, path=path, kind=kind,
            error=f"unknown kind {kind!r}; expected 'ic', 'background', or 'static'",
        )

    ok = (
        not missing_surface and not missing_atmos
        and not missing_static and not bad_shape
    )

    return ValidationReport(
        ok=ok, path=path, kind=kind,
        missing_surface=missing_surface,
        missing_atmos=missing_atmos,
        missing_static=missing_static,
        bad_shape=tuple(bad_shape),
    )