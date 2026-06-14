"""Tests for aurora_da.observations.

No network, no file I/O. extract_variable tested with in-memory Datasets.
"""

import numpy as np
import pytest
import xarray as xr

from aurora_da.data.schema import REQUIRED_LEVELS_HPA
from aurora_da.observations import (
    extract_variable,
    generate_observation,
)


# ---------------------------------------------------------------------------
# extract_variable
# ---------------------------------------------------------------------------

def _make_synthetic_ds(H: int = 8, W: int = 12) -> xr.Dataset:
    """Synthetic dataset matching the Aurora-format layout used in this project."""
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "msl": (("latitude", "longitude"), rng.standard_normal((H, W))),
            "2t":  (("latitude", "longitude"), 270 + rng.standard_normal((H, W))),
            "t":   (("level", "latitude", "longitude"),
                    rng.standard_normal((len(REQUIRED_LEVELS_HPA), H, W))),
            "z":   (("level", "latitude", "longitude"),
                    rng.standard_normal((len(REQUIRED_LEVELS_HPA), H, W))),
        }
    )


def test_extract_variable_surface() -> None:
    ds = _make_synthetic_ds()
    arr = extract_variable(ds, {"name": "msl", "group": "surface", "var": "msl"})
    assert arr.shape == (8, 12)
    np.testing.assert_array_equal(arr, ds["msl"].values)


def test_extract_variable_atmospheric_at_500_hpa() -> None:
    ds = _make_synthetic_ds()
    arr = extract_variable(
        ds, {"name": "t500", "group": "atmospheric", "var": "t", "level": 500}
    )
    assert REQUIRED_LEVELS_HPA[7] == 500
    np.testing.assert_array_equal(arr, ds["t"].values[7])


def test_extract_variable_atmospheric_at_850_hpa() -> None:
    ds = _make_synthetic_ds()
    arr = extract_variable(
        ds, {"name": "t850", "group": "atmospheric", "var": "t", "level": 850}
    )
    assert REQUIRED_LEVELS_HPA[10] == 850
    np.testing.assert_array_equal(arr, ds["t"].values[10])


def test_extract_variable_unknown_group_raises() -> None:
    ds = _make_synthetic_ds()
    with pytest.raises(ValueError, match="unknown group"):
        extract_variable(ds, {"name": "x", "group": "garbage", "var": "msl"})


# ---------------------------------------------------------------------------
# generate_observation
# ---------------------------------------------------------------------------

def test_generate_observation_reproducible() -> None:
    """Same seed -> bit-identical result."""
    truth = np.random.default_rng(0).standard_normal((50, 100))
    o1 = generate_observation(truth, fraction=0.01, sigma_obs=0.5, seed=42)
    o2 = generate_observation(truth, fraction=0.01, sigma_obs=0.5, seed=42)
    np.testing.assert_array_equal(o1["h_indices"], o2["h_indices"])
    np.testing.assert_array_equal(o1["w_indices"], o2["w_indices"])
    np.testing.assert_array_equal(o1["values"], o2["values"])


def test_generate_observation_different_seeds_differ() -> None:
    truth = np.random.default_rng(0).standard_normal((50, 100))
    o1 = generate_observation(truth, fraction=0.01, sigma_obs=0.5, seed=0)
    o2 = generate_observation(truth, fraction=0.01, sigma_obs=0.5, seed=1)
    assert not np.array_equal(o1["h_indices"], o2["h_indices"])


def test_generate_observation_count() -> None:
    H, W = 100, 200
    truth = np.zeros((H, W))
    obs = generate_observation(truth, fraction=0.05, sigma_obs=1.0, seed=0)
    expected = int(round(H * W * 0.05))
    assert obs["n_obs"] == expected
    assert len(obs["values"]) == expected
    assert len(obs["h_indices"]) == expected
    assert obs["mask"].sum() == expected


def test_generate_observation_zero_noise_recovers_truth() -> None:
    H, W = 50, 100
    truth = np.random.default_rng(0).standard_normal((H, W))
    obs = generate_observation(truth, fraction=0.02, sigma_obs=0.0, seed=0)
    truth_at_obs = truth[obs["h_indices"], obs["w_indices"]]
    np.testing.assert_array_equal(obs["values"], truth_at_obs)


def test_generate_observation_noise_std_close_to_sigma() -> None:
    """With many obs, empirical std of (values - truth) should ~= sigma_obs."""
    H, W = 200, 400  # n_obs = 16000 -> tight std estimate
    truth = np.zeros((H, W))
    sigma = 1.5
    obs = generate_observation(truth, fraction=0.2, sigma_obs=sigma, seed=0)
    truth_at_obs = truth[obs["h_indices"], obs["w_indices"]]
    empirical_std = float((obs["values"] - truth_at_obs).std())
    assert abs(empirical_std - sigma) / sigma < 0.05


def test_generate_observation_no_duplicate_locations() -> None:
    H, W = 50, 100
    truth = np.zeros((H, W))
    obs = generate_observation(truth, fraction=0.5, sigma_obs=1.0, seed=0)
    flat = obs["h_indices"].astype(np.int64) * W + obs["w_indices"]
    assert len(np.unique(flat)) == len(flat)


def test_generate_observation_mask_matches_indices() -> None:
    H, W = 50, 100
    truth = np.zeros((H, W))
    obs = generate_observation(truth, fraction=0.05, sigma_obs=1.0, seed=0)
    assert obs["mask"].sum() == obs["n_obs"]
    # mask must be True exactly at the (h, w) indices
    mask_from_indices = np.zeros((H, W), dtype=bool)
    mask_from_indices[obs["h_indices"], obs["w_indices"]] = True
    np.testing.assert_array_equal(obs["mask"], mask_from_indices)


def test_generate_observation_invalid_fraction_raises() -> None:
    truth = np.zeros((10, 10))
    with pytest.raises(ValueError, match="fraction"):
        generate_observation(truth, fraction=0.0, sigma_obs=1.0, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        generate_observation(truth, fraction=1.5, sigma_obs=1.0, seed=0)


def test_generate_observation_negative_sigma_raises() -> None:
    truth = np.zeros((10, 10))
    with pytest.raises(ValueError, match="sigma_obs"):
        generate_observation(truth, fraction=0.1, sigma_obs=-0.5, seed=0)