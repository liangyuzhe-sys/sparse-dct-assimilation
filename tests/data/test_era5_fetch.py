"""Tests for aurora_da.data.era5_fetch.

Pure-logic tests only: no network, no file I/O. Variable rename maps and the
timestamp planner are covered. The actual fetch functions require network and
are exercised by the smoke test in scripts/fetch_one_snapshot.py.
"""

from datetime import datetime

import pytest

from aurora_da.data.era5_fetch import (
    RENAME_ATMOS,
    RENAME_STATIC,
    RENAME_SURFACE,
    compute_timestamps,
)
from aurora_da.data.schema import (
    REQUIRED_ATMOS_VARS,
    REQUIRED_STATIC_VARS,
    REQUIRED_SURFACE_VARS,
)


def test_rename_surface_covers_all_required() -> None:
    assert set(RENAME_SURFACE.values()) == set(REQUIRED_SURFACE_VARS)


def test_rename_atmos_covers_all_required() -> None:
    assert set(RENAME_ATMOS.values()) == set(REQUIRED_ATMOS_VARS)


def test_rename_static_covers_all_required() -> None:
    assert set(RENAME_STATIC.values()) == set(REQUIRED_STATIC_VARS)


def test_compute_timestamps_count_for_weekly_2023() -> None:
    """52 weekly dates x 3 offsets per date = 156 unique timestamps (no overlap)."""
    cfg = {
        "evaluation": {
            "start_date": "2023-01-02",
            "end_date":   "2023-12-25",
            "cadence":    "7D",
            "hour_utc":   6,
        },
        "aurora": {
            "ic_offsets_h":  [-12, -6],
            "pred_offset_h": 0,
        },
    }
    ts = compute_timestamps(cfg)
    assert len(ts) == 156
    assert ts[0] == datetime(2023, 1, 1, 18)
    assert ts[-1] == datetime(2023, 12, 25, 6)


def test_compute_timestamps_sorted_and_unique() -> None:
    cfg = {
        "evaluation": {
            "start_date": "2023-01-02",
            "end_date":   "2023-01-09",
            "cadence":    "7D",
            "hour_utc":   6,
        },
        "aurora": {
            "ic_offsets_h":  [-12, -6],
            "pred_offset_h": 0,
        },
    }
    ts = compute_timestamps(cfg)
    assert len(ts) == 6
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)
    