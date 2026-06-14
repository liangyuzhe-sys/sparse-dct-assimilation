"""Tests for aurora_da.data.schema.

Pure logic tests: no network, no GPU, no zarr files on disk. The validate_snapshot
function is tested only on the path-does-not-exist code path here; the
valid-zarr code path is exercised by integration tests in later specs.
"""

from datetime import datetime
from pathlib import Path

import pytest

from aurora_da.data.schema import (
    GRID_SHAPE,
    REQUIRED_ATMOS_VARS,
    REQUIRED_LEVELS_HPA,
    REQUIRED_STATIC_VARS,
    REQUIRED_SURFACE_VARS,
    parse_basename,
    snapshot_basename,
    validate_snapshot,
)


def test_required_constants_match_aurora_docs() -> None:
    """Guard against typos in the canonical Aurora variable lists."""
    assert REQUIRED_SURFACE_VARS == ("2t", "10u", "10v", "msl")
    assert REQUIRED_ATMOS_VARS == ("t", "u", "v", "q", "z")
    assert REQUIRED_STATIC_VARS == ("lsm", "slt", "z")
    assert REQUIRED_LEVELS_HPA == (
        50, 100, 150, 200, 250, 300, 400,
        500, 600, 700, 850, 925, 1000,
    )


def test_grid_shape_is_721_by_1440() -> None:
    """0.25 degree global grid: 721 lat (poles inclusive) x 1440 lon."""
    assert GRID_SHAPE == (721, 1440)


def test_snapshot_basename_roundtrip() -> None:
    """snapshot_basename and parse_basename must be exact inverses at hour precision."""
    for t in (
        datetime(2023, 1, 2, 6),
        datetime(2023, 12, 25, 6),
        datetime(2023, 7, 15, 18),
        datetime(2024, 2, 29, 0),  # leap day, midnight
    ):
        assert parse_basename(snapshot_basename(t)) == t


def test_snapshot_basename_strips_minutes() -> None:
    """Sub-hour precision in the input must be silently dropped."""
    t = datetime(2023, 1, 2, 6, 30, 15)
    assert snapshot_basename(t) == "2023-01-02T06"


def test_parse_basename_rejects_garbage() -> None:
    """Malformed input must raise ValueError, not return a default."""
    with pytest.raises(ValueError):
        parse_basename("not-a-date")


def test_validate_snapshot_missing_path() -> None:
    """A non-existent path must yield ok=False with a helpful error message."""
    report = validate_snapshot(Path("/nonexistent/path/abc"), kind="ic")
    assert not report.ok
    assert report.error is not None
    assert "exist" in report.error.lower()