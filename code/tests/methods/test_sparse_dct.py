"""Sanity tests for sparse-DCT method, including Phase 4 UQ."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import idctn

from aurora_da.methods import sparse_dct
from aurora_da.methods.sparse_dct import (
    _dct_basis,
    _Operator,
    _freq_weights,
    _select_shape,
)


# ----------------------------------------------------------------------------
# Phase 1-3 sanity
# ----------------------------------------------------------------------------

def test_dct_basis_orthonormal():
    phi = _dct_basis(32, 16)
    np.testing.assert_allclose(phi.T @ phi, np.eye(16), atol=1e-10)


def test_dct_basis_matches_scipy_idctn():
    H, W = 16, 32
    i, j = 3, 7
    alpha = np.zeros((H, W))
    alpha[i, j] = 1.0
    field = idctn(alpha, type=2, norm="ortho", axes=(0, 1))
    phi_H = _dct_basis(H, i + 1)[:, i]
    phi_W = _dct_basis(W, j + 1)[:, j]
    np.testing.assert_allclose(field, np.outer(phi_H, phi_W), atol=1e-10)


def test_operator_adjoint():
    rng = np.random.default_rng(0)
    H, W, kh, kw, m = 16, 32, 4, 8, 25
    h_idx = rng.integers(0, H, size=m)
    w_idx = rng.integers(0, W, size=m)
    op = _Operator(H, W, kh, kw, h_idx, w_idx)
    x = rng.standard_normal((kh, kw))
    y = rng.standard_normal(m)
    np.testing.assert_allclose(
        op.apply(x) @ y, (x * op.adjoint(y)).sum(), atol=1e-10
    )


def test_operator_materialize_consistent_with_apply():
    rng = np.random.default_rng(1)
    H, W, kh, kw, m = 16, 32, 4, 8, 25
    h_idx = rng.integers(0, H, size=m)
    w_idx = rng.integers(0, W, size=m)
    op = _Operator(H, W, kh, kw, h_idx, w_idx)
    A_S = op.materialize()
    x = rng.standard_normal((kh, kw))
    np.testing.assert_allclose(op.apply(x), A_S @ x.ravel(), atol=1e-10)


def test_freq_weights_p2():
    W = _freq_weights(3, 4, p=2.0)
    i = np.arange(3)[:, None]
    j = np.arange(4)[None, :]
    np.testing.assert_allclose(W, 1.0 + i ** 2 + j ** 2, atol=1e-10)


def test_shape_selection_picks_compact_corner():
    alpha = np.zeros((10, 10))
    alpha[:3, :3] = [[10.0, 5.0, 3.0],
                     [5.0, 3.0, 1.0],
                     [3.0, 1.0, 0.5]]
    assert _select_shape(alpha, tau_E=0.95) == (3, 3)


def test_shape_selection_handles_zero():
    assert _select_shape(np.zeros((5, 5)), tau_E=0.99) == (1, 1)


def test_analyze_recovers_smooth_signal():
    rng = np.random.default_rng(0)
    H, W = 32, 64
    alpha_true = np.zeros((H, W))
    alpha_true[:4, :4] = rng.standard_normal((4, 4)) * 3.0
    x_true = idctn(alpha_true, type=2, norm="ortho", axes=(0, 1))
    bg = np.zeros_like(x_true)
    n_obs = 400
    h_idx = rng.integers(0, H, size=n_obs)
    w_idx = rng.integers(0, W, size=n_obs)
    sigma_obs = 0.01
    y = x_true[h_idx, w_idx] + sigma_obs * rng.standard_normal(n_obs)
    obs = dict(h_indices=h_idx, w_indices=w_idx,
               values=y, sigma_obs=sigma_obs)
    analysis = sparse_dct.analyze(
        bg, obs, var_spec={"name": "test"},
        lat_degrees=np.linspace(90, -90, H),
        sigma_b=1.0,
        lon_degrees=np.linspace(0, 360, W, endpoint=False),
        kh_init=16, kw_init=32,
        n_outer=4, n_fista=200, tau_E=0.99,
    )
    bg_rmse = np.sqrt(((bg - x_true) ** 2).mean())
    a_rmse = np.sqrt(((analysis - x_true) ** 2).mean())
    assert a_rmse < 0.30 * bg_rmse


def test_analyze_returns_background_when_no_obs():
    bg = np.random.default_rng(0).standard_normal((16, 16))
    obs = dict(h_indices=np.array([], dtype=int),
               w_indices=np.array([], dtype=int),
               values=np.array([]), sigma_obs=1.0)
    out = sparse_dct.analyze(
        bg, obs, var_spec={}, lat_degrees=np.linspace(90, -90, 16),
        sigma_b=1.0, lon_degrees=np.linspace(0, 360, 16, endpoint=False),
        kh_init=4, kw_init=4,
    )
    np.testing.assert_array_equal(out, bg)


# ----------------------------------------------------------------------------
# Phase 4 UQ
# ----------------------------------------------------------------------------

def _build_toy_problem(seed=0, H=32, W=64, sigma_obs=0.3, n_obs=200):
    """Toy problem with REPRESENTATION ERROR: signal lives in an 8x8 DCT block,
    but we will fit with a 4x4 subspace, so the truth has substantial energy
    outside the fit subspace.  This is required to exercise Phase 4 inflation.
    """
    rng = np.random.default_rng(seed)
    alpha_true = np.zeros((H, W))
    alpha_true[:8, :8] = rng.standard_normal((8, 8)) * 3.0
    x_true = idctn(alpha_true, type=2, norm="ortho", axes=(0, 1))
    bg = np.zeros_like(x_true)
    h_idx = rng.integers(0, H, size=n_obs)
    w_idx = rng.integers(0, W, size=n_obs)
    y = x_true[h_idx, w_idx] + sigma_obs * rng.standard_normal(n_obs)
    obs = dict(h_indices=h_idx, w_indices=w_idx,
               values=y, sigma_obs=sigma_obs)
    lat = np.linspace(90, -90, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    return x_true, bg, obs, lat, lon


def test_fit_returns_expected_keys():
    _x, bg, obs, lat, lon = _build_toy_problem()
    state = sparse_dct.fit(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=16, kw_init=32, n_outer=3, n_fista=150, tau_E=0.99,
    )
    for key in ("bg", "mu_S", "chol_L", "A_S",
                "sigma_rep", "lam_hat", "kh_star", "kw_star"):
        assert key in state, f"missing key {key}"
    assert state["chol_L"].shape[0] == state["kh_star"] * state["kw_star"]
    assert state["sigma_rep"] >= 0.0


def test_posterior_sample_shape_and_mean():
    _x, bg, obs, lat, lon = _build_toy_problem()
    state = sparse_dct.fit(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=16, kw_init=32, n_outer=3, n_fista=150, tau_E=0.99,
    )
    samples = sparse_dct.posterior_sample(
        state, n_samples=200, seed=0, inflation="none",
    )
    assert samples.shape == (200, *bg.shape)
    analysis = sparse_dct.analyze(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=16, kw_init=32, n_outer=3, n_fista=150, tau_E=0.99,
    )
    err = np.abs(samples.mean(axis=0) - analysis).max()
    assert err < 0.2 * analysis.std()


def test_uniform_residual_increases_coverage():
    """With deliberately-too-small fit subspace, 'none' under-covers and
    uniform_residual should pull cov_90 up substantially."""
    x_true, bg, obs, lat, lon = _build_toy_problem()
    state = sparse_dct.fit(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=4, kw_init=4,                 # << smaller than 8x8 signal
        n_outer=3, n_fista=200, tau_E=0.99,
    )
    assert state["sigma_rep"] > 0.0, (
        f"toy problem failed to produce representation error "
        f"(sigma_rep={state['sigma_rep']})"
    )
    s_none = sparse_dct.posterior_sample(
        state, n_samples=400, seed=0, inflation="none",
    )
    s_uni = sparse_dct.posterior_sample(
        state, n_samples=400, seed=0, inflation="uniform_residual",
    )

    def cov(samples):
        lo = np.quantile(samples, 0.05, axis=0)
        hi = np.quantile(samples, 0.95, axis=0)
        return float(((x_true >= lo) & (x_true <= hi)).mean())

    cov_none = cov(s_none)
    cov_uni = cov(s_uni)
    assert cov_uni > cov_none + 0.05, (
        f"uniform inflation did not help enough: "
        f"{cov_none:.3f} -> {cov_uni:.3f}"
    )


def test_spatial_dist_requires_truth():
    _x, bg, obs, lat, lon = _build_toy_problem()
    state = sparse_dct.fit(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=16, kw_init=32, n_outer=3, n_fista=150, tau_E=0.99,
    )
    with pytest.raises(ValueError, match="truth"):
        sparse_dct.posterior_sample(
            state, n_samples=50, seed=0, inflation="spatial_dist",
        )


def test_spatial_dist_calibrates_to_target_coverage():
    """With oracle k from bisection, spatial_dist should hit the target
    LAT-WEIGHTED cov_90 (which is what _calibrate_k actually optimizes).
    """
    x_true, bg, obs, lat, lon = _build_toy_problem(seed=1)
    state = sparse_dct.fit(
        bg, obs, var_spec={}, lat_degrees=lat,
        sigma_b=1.0, lon_degrees=lon,
        kh_init=4, kw_init=4,
        n_outer=3, n_fista=200, tau_E=0.99,
    )
    samples = sparse_dct.posterior_sample(
        state, n_samples=400, seed=0, inflation="spatial_dist",
        truth=x_true, cov_target=0.90, k_tol=1e-3,
    )
    lo = np.quantile(samples, 0.05, axis=0)
    hi = np.quantile(samples, 0.95, axis=0)
    inside = ((x_true >= lo) & (x_true <= hi)).astype(np.float64)
    # Match the lat-weighting used inside _calibrate_k
    cl = np.cos(np.deg2rad(lat))
    cl = np.maximum(cl, 0.0)
    cl /= cl.mean()
    area_w = np.broadcast_to(cl[:, None], inside.shape)
    cov_90_weighted = float((inside * area_w).mean())
    assert 0.86 <= cov_90_weighted <= 0.94, (
        f"weighted cov_90 not in band: {cov_90_weighted:.3f}"
    )