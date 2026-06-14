"""Sparse-DCT Bayesian data assimilation on top of an Aurora background field.

x = xb + B alpha,   alpha in R^{kh x kw} (low-frequency DCT-II coefficients).

Three-layer Bayesian model
--------------------------
  likelihood :  d | alpha ~ N(A alpha, sigma_obs^2 I),  d = y - H xb,  A = H B
  prior      :  alpha_ij | lambda ~ Laplace(lambda * W_ij),  W_ij = (1+i^2+j^2)^{p/2}
  hyperprior :  lambda ~ Gamma(a, b)

Pipeline
--------
  Phase 1  FISTA solves the joint MAP for alpha at fixed lambda; lambda is
           updated via the Gamma posterior MODE (deviation from the report,
           which uses the mean -- see ALGORITHM NOTES).
  Phase 2  Energy-based shape selection: smallest rectangle (kh*, kw*)
           capturing tau_E of |alpha|^2.  Default tau_E = 0.99
           (report used 0.95).
  Phase 3  Gaussian posterior on the selected subspace with frequency-aware
           prior precision  D_S[i,j] = lambda * W_ij / (|alpha_hat_ij| + eps).
           Lambda_S is formed explicitly and factorized once (Cholesky); the
           same factor gives the posterior MEAN and the samples used in
           Phase 4.
  Phase 4  Uncertainty quantification via inflation.
           I.  Representation-error inflation: r^(m) ~ N(0, sigma_rep^2 I),
               sigma_rep estimated from the observation residual
               (sigma_hat_rep^2 = max(0, (1/m)||d - A_S alpha_hat||^2
                                       - sigma_obs^2)).
           II. Spatial inflation: r^(m)(s) ~ N(0, k * w(s)) with
               w(s) = dist_to_nearest_obs(s)^p / mean(.) and k calibrated
               via bisection to hit a target marginal coverage on the
               provided truth field (oracle k).

ALGORITHM NOTES (deviations from the report)
--------------------------------------------
  1. Lambda update uses the Gamma posterior MODE
        lambda_hat = (a + N_active - 1) / (b + S),     valid when shape > 1.
     Report uses the mean.  Mode is the MAP estimate of lambda, consistent
     with the rest of the pipeline being MAP-based.
  2. tau_E defaults to 0.99 (report: 0.95).
  3. Phase 3 uses an explicit Lambda_S + Cholesky, not CG, so that the
     factor can be reused for sampling in Phase 4.

Operator implementation
-----------------------
The forward operator A = H B is never materialized as a matrix.  The 2D
DCT-II ortho basis is separable:

    B[:, (i,j)]  reshape to (H, W)  =  phi_i^H (outer) phi_j^W

so apply and adjoint are einsums on precomputed (m, kh) and (m, kw)
matrices.  For Phase 3 we DO materialize the (m, N) row matrix A_S, because
we need A_S^T A_S explicitly for the Cholesky factorization of Lambda_S.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import idctn
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial import cKDTree


# ============================================================================
# Basis and operator
# ============================================================================

def _dct_basis(n: int, k: int) -> np.ndarray:
    """Length-n DCT-II orthonormal basis matrix, columns 0..k-1.  phi.T phi = I."""
    nn = np.arange(n)
    kk = np.arange(k)
    phi = np.cos(np.pi * (nn[:, None] + 0.5) * kk[None, :] / n)
    phi *= np.sqrt(2.0 / n)
    phi[:, 0] *= np.sqrt(0.5)
    return phi


class _Operator:
    """Linear operator A = H B for grid (H, W), truncation (kh, kw),
    observation indices (h_idx, w_idx)."""

    def __init__(self, H: int, W: int, kh: int, kw: int,
                 h_idx: np.ndarray, w_idx: np.ndarray):
        self.H = H
        self.W = W
        self.kh = kh
        self.kw = kw
        self.h_idx = h_idx
        self.w_idx = w_idx
        self.m = len(h_idx)
        self.PHI_H = _dct_basis(H, kh)[h_idx]   # (m, kh)
        self.PHI_W = _dct_basis(W, kw)[w_idx]   # (m, kw)

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        """(kh, kw) -> (m,):  A alpha"""
        return np.einsum("mi,ij,mj->m", self.PHI_H, alpha, self.PHI_W, optimize=True)

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        """(m,) -> (kh, kw):  A^T r"""
        return np.einsum("mi,mj,m->ij", self.PHI_H, self.PHI_W, r, optimize=True)

    def materialize(self) -> np.ndarray:
        """Form A_S explicitly as (m, kh*kw). Columns flat-indexed by
        k = i * kw + j (row-major). Memory m * kh * kw * 8 bytes."""
        return (self.PHI_H[:, :, None] * self.PHI_W[:, None, :]).reshape(
            self.m, self.kh * self.kw
        )

    def lipschitz(self, n_iter: int = 20, seed: int = 0) -> float:
        """L = lambda_max(A^T A) by power iteration. Lipschitz of
        g(alpha) = (1/2)||A alpha - d||^2; the FISTA objective is
        f = g / sigma_obs^2, so step size eta = sigma_obs^2 / L."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((self.kh, self.kw))
        v /= np.linalg.norm(v) + 1e-30
        for _ in range(n_iter):
            v = self.adjoint(self.apply(v))
            nrm = np.linalg.norm(v)
            if nrm < 1e-30:
                return 0.0
            v /= nrm
        Av = self.apply(v)
        return float(Av @ Av)


def _freq_weights(kh: int, kw: int, p: float = 2.0) -> np.ndarray:
    """W_ij = (1 + i^2 + j^2)^{p/2}."""
    i = np.arange(kh)[:, None]
    j = np.arange(kw)[None, :]
    return (1.0 + i ** 2 + j ** 2) ** (p / 2.0)


# ============================================================================
# Phase 1: FISTA + adaptive lambda
# ============================================================================

def _fista(op, d, lam, W, sigma_obs, L, n_iter, tol, alpha_init=None):
    sigma2 = sigma_obs ** 2
    eta = sigma2 / L
    threshold = eta * lam * W
    alpha = (alpha_init.copy() if alpha_init is not None
             else np.zeros((op.kh, op.kw)))
    z = alpha.copy()
    s = 1.0
    for _ in range(n_iter):
        grad = op.adjoint(op.apply(z) - d) / sigma2
        u = z - eta * grad
        alpha_new = np.sign(u) * np.maximum(np.abs(u) - threshold, 0.0)
        s_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s * s))
        z = alpha_new + ((s - 1.0) / s_new) * (alpha_new - alpha)
        diff = np.linalg.norm(alpha_new - alpha)
        denom = np.linalg.norm(alpha) + tol
        alpha = alpha_new
        s = s_new
        if diff < tol * denom:
            break
    return alpha


def _solve_phase1(op, d, sigma_obs, W, a, b, lam_init,
                  n_outer, n_fista, fista_tol):
    """Alternating FISTA + Gamma posterior MODE update."""
    L = op.lipschitz()
    if L <= 0.0:
        return np.zeros((op.kh, op.kw)), lam_init
    lam = float(lam_init)
    alpha = None
    for _ in range(n_outer):
        alpha = _fista(op, d, lam, W, sigma_obs, L,
                       n_iter=n_fista, tol=fista_tol, alpha_init=alpha)
        S = float((W * np.abs(alpha)).sum())
        n_active = int(np.count_nonzero(alpha))
        shape = a + n_active
        rate = b + S
        if rate <= 0.0:
            break
        lam = (shape - 1.0) / rate if shape > 1.0 else shape / rate
    return alpha, lam


# ============================================================================
# Phase 2: Shape selection
# ============================================================================

def _select_shape(alpha, tau_E):
    E = alpha * alpha
    total = E.sum()
    if total <= 0.0:
        return (1, 1)
    cum = E.cumsum(axis=0).cumsum(axis=1)
    target = tau_E * total
    valid = cum >= target
    if not valid.any():
        return alpha.shape
    kh_max, kw_max = alpha.shape
    sizes = np.outer(np.arange(1, kh_max + 1), np.arange(1, kw_max + 1))
    sentinel = sizes.max() + 1
    sizes_masked = np.where(valid, sizes, sentinel)
    flat = int(sizes_masked.argmin())
    return (flat // kw_max + 1, flat % kw_max + 1)


# ============================================================================
# Phase 3: Subspace Gaussian posterior with explicit Lambda_S + Cholesky
# ============================================================================

def _solve_phase3(op_S, d, alpha_hat_S, lam, W_S, sigma_obs, eps):
    """Form Lambda_S explicitly and Cholesky-factor it.

    Returns
    -------
    mu_S : (kh, kw)            posterior mean coefficient field
    chol : (N, N) lower-tri    Cholesky factor C with C C^T = Lambda_S
    A_S  : (m, N)              materialized operator (kept for residual calc)
    """
    sigma2 = sigma_obs ** 2
    D_S = lam * W_S / (np.abs(alpha_hat_S) + eps)        # (kh, kw)
    A_S = op_S.materialize()                              # (m, N), N = kh*kw
    N = A_S.shape[1]
    Lambda_S = (A_S.T @ A_S) / sigma2
    Lambda_S.flat[::N + 1] += D_S.ravel()                 # add diag in-place
    c, low = cho_factor(Lambda_S, lower=True, overwrite_a=True)
    rhs = (A_S.T @ d) / sigma2                            # (N,)
    mu_flat = cho_solve((c, low), rhs)
    mu_S = mu_flat.reshape(op_S.kh, op_S.kw)
    chol_L = np.tril(c)
    np.fill_diagonal(chol_L, np.diag(c))
    return mu_S, chol_L, A_S


def _reconstruct_field(alpha_S, kh, kw, H, W):
    alpha_full = np.zeros((H, W), dtype=np.float64)
    alpha_full[:kh, :kw] = alpha_S
    return idctn(alpha_full, type=2, norm="ortho", axes=(0, 1))


# ============================================================================
# Phase 4 helpers
# ============================================================================

def _sigma_rep_from_residual(d, A_S, alpha_S_hat_flat, sigma_obs):
    """sigma_hat_rep^2 = max(0, (1/m)||d - A_S alpha_hat||^2 - sigma_obs^2)."""
    pred = A_S @ alpha_S_hat_flat
    rss = float(((d - pred) ** 2).mean())
    return float(np.sqrt(max(0.0, rss - sigma_obs ** 2)))


def _dist_to_nearest_obs_km(H, W, h_idx, w_idx, lat_degrees, lon_degrees):
    """For each pixel, great-circle distance (km) to the nearest observation."""
    from aurora_da.distances import latlon_to_unit_xyz
    EARTH_RADIUS_KM = 6371.0088
    lat2d, lon2d = np.meshgrid(lat_degrees, lon_degrees, indexing="ij")
    pixel_xyz = latlon_to_unit_xyz(lat2d.ravel(), lon2d.ravel())
    obs_xyz = latlon_to_unit_xyz(lat_degrees[h_idx], lon_degrees[w_idx])
    tree = cKDTree(obs_xyz)
    chord, _ = tree.query(pixel_xyz, k=1)
    half = np.clip(0.5 * chord, 0.0, 1.0)
    arc_km = EARTH_RADIUS_KM * 2.0 * np.arcsin(half)
    return arc_km.reshape(H, W)


def _spatial_weights(dist_km, p):
    """w(s) = dist(s)^p, normalized so mean(w) = 1."""
    w = np.power(np.maximum(dist_km, 0.0), p)
    w_mean = w.mean()
    if w_mean <= 0.0:
        return np.ones_like(w)
    return w / w_mean


def _cos_lat_weights(lat_degrees, H, W):
    """Per-pixel cosine-of-latitude weights, normalized to mean 1."""
    cl = np.cos(np.deg2rad(lat_degrees))
    cl = np.maximum(cl, 0.0)
    cl /= cl.mean()
    return np.broadcast_to(cl[:, None], (H, W))


def _coverage_90(samples_HW, truth_HW, area_w_HW):
    """Lat-weighted mean coverage of the 90% empirical interval."""
    lo = np.quantile(samples_HW, 0.05, axis=0)
    hi = np.quantile(samples_HW, 0.95, axis=0)
    inside = ((truth_HW >= lo) & (truth_HW <= hi)).astype(np.float64)
    return float((inside * area_w_HW).mean())


def _calibrate_k(samples_subspace_HW, truth, area_w_HW, w_spatial_HW,
                 unit_noise_HW, cov_target, tol, max_iter):
    """Bisection on k.  Objective uses a PRE-DRAWN unit-normal noise array so
    that coverage(k) is a deterministic function of k -- bisection requires
    a monotone deterministic objective.
    """

    def coverage_at(k):
        if k < 0.0:
            return 0.0
        std = np.sqrt(np.maximum(k * w_spatial_HW, 0.0))
        return _coverage_90(
            samples_subspace_HW + unit_noise_HW * std[None],
            truth, area_w_HW,
        )

    lo, hi = 0.0, 1.0
    while coverage_at(hi) < cov_target:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if coverage_at(mid) < cov_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * (1.0 + hi):
            break
    return 0.5 * (lo + hi)


# ============================================================================
# Public API
# ============================================================================

def fit(
    bg: np.ndarray,
    obs: dict,
    var_spec: dict,
    lat_degrees: np.ndarray,
    *,
    sigma_b: float,
    lon_degrees: np.ndarray,
    kh_init: int = 64,
    kw_init: int = 128,
    p: float = 2.0,
    a: float = 0.1,
    b: float = 0.1,
    lam_init: float = 1e-3,
    n_outer: int = 5,
    n_fista: int = 200,
    fista_tol: float = 1e-7,
    tau_E: float = 0.99,
    eps: float = 1e-6,
    **kwargs,
) -> dict:
    """Run Phases 1-3 and return everything needed for Phase 4 sampling."""
    if bg.ndim != 2:
        raise ValueError(f"bg must be 2D, got {bg.shape}")
    H, W = bg.shape
    if kh_init > H or kw_init > W:
        raise ValueError(
            f"kh_init={kh_init}, kw_init={kw_init} exceed grid ({H}, {W})"
        )

    h_idx = np.asarray(obs["h_indices"], dtype=np.int64)
    w_idx = np.asarray(obs["w_indices"], dtype=np.int64)
    y = np.asarray(obs["values"], dtype=np.float64)
    sigma_obs = float(obs["sigma_obs"])
    n_obs = len(y)
    if n_obs == 0:
        return {
            "bg": bg, "H": H, "W": W, "h_idx": h_idx, "w_idx": w_idx,
            "sigma_obs": sigma_obs, "sigma_b": sigma_b,
            "lat_degrees": lat_degrees, "lon_degrees": lon_degrees,
            "empty": True,
        }

    bg_at_obs = bg[h_idx, w_idx].astype(np.float64)
    d = y - bg_at_obs

    # Phase 1
    op_full = _Operator(H, W, kh_init, kw_init, h_idx, w_idx)
    W_full = _freq_weights(kh_init, kw_init, p=p)
    alpha_hat, lam_hat = _solve_phase1(
        op_full, d, sigma_obs, W_full,
        a=a, b=b, lam_init=lam_init,
        n_outer=n_outer, n_fista=n_fista, fista_tol=fista_tol,
    )

    # Phase 2
    kh_star, kw_star = _select_shape(alpha_hat, tau_E=tau_E)

    # Phase 3
    op_S = _Operator(H, W, kh_star, kw_star, h_idx, w_idx)
    alpha_S_hat = alpha_hat[:kh_star, :kw_star]
    W_S = W_full[:kh_star, :kw_star]
    mu_S, chol_L, A_S = _solve_phase3(
        op_S, d, alpha_S_hat, lam_hat, W_S, sigma_obs, eps=eps,
    )

    sigma_rep = _sigma_rep_from_residual(d, A_S, mu_S.ravel(), sigma_obs)

    return {
        "bg": bg, "H": H, "W": W,
        "h_idx": h_idx, "w_idx": w_idx,
        "sigma_obs": sigma_obs, "sigma_b": sigma_b,
        "lat_degrees": lat_degrees, "lon_degrees": lon_degrees,
        "d": d,
        "alpha_hat": alpha_hat, "lam_hat": lam_hat,
        "kh_star": kh_star, "kw_star": kw_star,
        "alpha_S_hat": alpha_S_hat, "W_S": W_S,
        "mu_S": mu_S, "chol_L": chol_L, "A_S": A_S,
        "sigma_rep": sigma_rep,
        "empty": False,
    }


def analyze(bg, obs, var_spec, lat_degrees, *, sigma_b, lon_degrees, **kwargs):
    """Convenience wrapper: fit + return reconstructed posterior mean field."""
    state = fit(bg, obs, var_spec, lat_degrees,
                sigma_b=sigma_b, lon_degrees=lon_degrees, **kwargs)
    if state.get("empty"):
        return bg.copy()
    residual = _reconstruct_field(
        state["mu_S"], state["kh_star"], state["kw_star"], state["H"], state["W"]
    )
    return (bg + residual).astype(bg.dtype, copy=False)


def posterior_sample(
    state: dict,
    *,
    n_samples: int = 200,
    seed: int = 0,
    inflation: str = "uniform_residual",
    truth: np.ndarray | None = None,
    cov_target: float = 0.90,
    dist_power: float = 1.0,
    k_tol: float = 1e-3,
    k_max_iter: int = 40,
) -> np.ndarray:
    """Phase 4 posterior samples.

    inflation in {'none', 'uniform_residual', 'spatial_dist'}.
    """
    if state.get("empty"):
        bg = state["bg"]
        return np.broadcast_to(bg[None].astype(np.float64),
                               (n_samples, *bg.shape)).copy()

    if inflation not in ("none", "uniform_residual", "spatial_dist"):
        raise ValueError(f"Unknown inflation mode: {inflation!r}")

    H, W = state["H"], state["W"]
    kh, kw = state["kh_star"], state["kw_star"]
    mu_flat = state["mu_S"].ravel()
    chol_L = state["chol_L"]
    bg = state["bg"]
    N = chol_L.shape[0]
    rng = np.random.default_rng(seed)

    samples_HW = np.empty((n_samples, H, W), dtype=np.float64)
    for m in range(n_samples):
        z = rng.standard_normal(N)
        x = solve_triangular(chol_L.T, z, lower=False)
        alpha_m = mu_flat + x
        residual = _reconstruct_field(alpha_m.reshape(kh, kw), kh, kw, H, W)
        samples_HW[m] = bg + residual

    if inflation == "none":
        return samples_HW

    if inflation == "uniform_residual":
        sigma_rep = state["sigma_rep"]
        if sigma_rep > 0.0:
            samples_HW += rng.standard_normal(samples_HW.shape) * sigma_rep
        return samples_HW

    # spatial_dist
    if truth is None:
        raise ValueError(
            "inflation='spatial_dist' requires truth field for k calibration"
        )
    if truth.shape != (H, W):
        raise ValueError(f"truth shape {truth.shape} != ({H}, {W})")
    dist_km = _dist_to_nearest_obs_km(
        H, W, state["h_idx"], state["w_idx"],
        state["lat_degrees"], state["lon_degrees"],
    )
    w_HW = _spatial_weights(dist_km, p=dist_power)
    area_w = _cos_lat_weights(state["lat_degrees"], H, W)
    unit_noise = rng.standard_normal(samples_HW.shape)
    k_star = _calibrate_k(
        samples_HW, truth, area_w, w_HW, unit_noise,
        cov_target=cov_target, tol=k_tol, max_iter=k_max_iter,
    )
    std = np.sqrt(np.maximum(k_star * w_HW, 0.0))
    samples_HW += unit_noise * std[None]
    return samples_HW