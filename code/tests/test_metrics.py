"""Tests for aurora_da.metrics."""

import numpy as np
import pytest

from aurora_da.metrics import (
    cos_lat_weights,
    weighted_rmse,
    weighted_rmse_stacked,
)


def test_cos_lat_weights_sum_to_one() -> None:
    lat = np.linspace(90, -90, 721)
    w = cos_lat_weights(lat)
    assert w.shape == (721,)
    assert abs(w.sum() - 1.0) < 1e-12


def test_cos_lat_weights_peak_at_equator() -> None:
    """At 0.25 deg grid with 721 points, index 360 is exactly the equator."""
    lat = np.linspace(90, -90, 721)
    w = cos_lat_weights(lat)
    assert lat[360] == pytest.approx(0.0, abs=1e-12)
    assert w[360] == w.max()


def test_cos_lat_weights_near_zero_at_poles() -> None:
    """Poles should contribute ~zero weight."""
    lat = np.linspace(90, -90, 721)
    w = cos_lat_weights(lat)
    assert w[0] < 1e-3
    assert w[-1] < 1e-3


def test_cos_lat_weights_order_independent() -> None:
    """Reversing lat order must not change the set of weights."""
    lat_a = np.linspace(90, -90, 11)
    lat_b = lat_a[::-1].copy()
    w_a = cos_lat_weights(lat_a)
    w_b = cos_lat_weights(lat_b)
    np.testing.assert_allclose(np.sort(w_a), np.sort(w_b))


def test_weighted_rmse_identical_arrays_zero() -> None:
    H, W = 10, 20
    lat = np.linspace(90, -90, H)
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((H, W))
    truth = pred.copy()
    assert weighted_rmse(pred, truth, lat) == 0.0


def test_weighted_rmse_constant_error_equals_abs() -> None:
    """If pred - truth is the constant c everywhere, weighted RMSE = |c|."""
    H, W = 10, 20
    lat = np.linspace(90, -90, H)
    truth = np.zeros((H, W))
    pred = np.full((H, W), 2.5)
    assert weighted_rmse(pred, truth, lat) == pytest.approx(2.5)


def test_weighted_rmse_shape_mismatch_raises() -> None:
    lat = np.linspace(90, -90, 10)
    with pytest.raises(ValueError, match="shape mismatch"):
        weighted_rmse(np.zeros((10, 20)), np.zeros((10, 21)), lat)


def test_weighted_rmse_wrong_lat_length_raises() -> None:
    with pytest.raises(ValueError, match="lat length"):
        weighted_rmse(
            np.zeros((10, 20)),
            np.zeros((10, 20)),
            np.linspace(90, -90, 15),
        )


def test_weighted_rmse_stacked_consistency_with_single() -> None:
    """Stacked of shape (1, H, W) must equal single-date weighted_rmse."""
    H, W = 10, 20
    lat = np.linspace(90, -90, H)
    rng = np.random.default_rng(42)
    pred = rng.standard_normal((H, W))
    truth = rng.standard_normal((H, W))
    single = weighted_rmse(pred, truth, lat)
    stacked = weighted_rmse_stacked(pred[None], truth[None], lat)
    assert stacked == pytest.approx(single)


def test_weighted_rmse_stacked_is_sqrt_of_mean_mse() -> None:
    """Two dates with MSE 9 and 16: stacked RMSE = sqrt((9+16)/2) ~ 3.54.
    NOT mean of RMSEs = (3+4)/2 = 3.5.
    """
    H, W = 10, 20
    lat = np.linspace(90, -90, H)
    truth_stack = np.zeros((2, H, W))
    pred_stack = np.stack([
        np.full((H, W), 3.0),   # MSE = 9
        np.full((H, W), 4.0),   # MSE = 16
    ])
    rmse = weighted_rmse_stacked(pred_stack, truth_stack, lat)
    assert rmse == pytest.approx(np.sqrt(12.5))
    # Sanity: it should NOT equal the mean of RMSEs
    assert rmse != pytest.approx(3.5)