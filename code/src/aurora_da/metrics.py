"""Cosine-latitude weighted RMSE and aggregation helpers.

Used by all DA methods for fair per-variable, per-date scoring. All functions
take numpy arrays; no torch here.
"""

from __future__ import annotations

import numpy as np


def cos_lat_weights(lat_degrees: np.ndarray) -> np.ndarray:
    """Normalized cos-latitude weights for area-fair spatial means.

    Args:
        lat_degrees: latitudes in degrees, shape (H,). Order does not matter.

    Returns:
        Weights of shape (H,) summing exactly to 1. Negative cos values
        (which can happen at numerical-precision-level near poles) are clipped
        to 0.
    """
    w = np.cos(np.deg2rad(np.asarray(lat_degrees, dtype=np.float64)))
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total == 0:
        raise ValueError("All cos-lat weights are zero; check lat input")
    return w / total


def weighted_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
    lat_degrees: np.ndarray,
) -> float:
    """Cos-lat weighted RMSE for a single (H, W) field.

    RMSE^2 = sum_i w_i * mean_j (pred[i,j] - truth[i,j])^2

    where w_i is the normalized cos-lat weight for row i.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected 2D (H, W), got shape {pred.shape}")
    if pred.shape[0] != lat_degrees.shape[0]:
        raise ValueError(
            f"lat length {lat_degrees.shape[0]} != H {pred.shape[0]}"
        )

    w = cos_lat_weights(lat_degrees)
    sq_err = (pred - truth) ** 2          # (H, W)
    mse_per_lat = sq_err.mean(axis=-1)    # (H,)
    weighted_mse = (mse_per_lat * w).sum()
    return float(np.sqrt(weighted_mse))


def weighted_rmse_stacked(
    preds: np.ndarray,
    truths: np.ndarray,
    lat_degrees: np.ndarray,
) -> float:
    """Aggregated cos-lat weighted RMSE across multiple dates.

    Args:
        preds: shape (T, H, W).
        truths: shape (T, H, W).
        lat_degrees: shape (H,).

    Returns:
        RMSE = sqrt( mean_t [ sum_i w_i * mean_j (preds[t,i,j] - truths[t,i,j])^2 ] )

    This is the "sqrt of mean MSE" convention used in DA / NWP papers, not
    "mean of RMSEs" (which would over-weight bad dates).
    """
    preds = np.asarray(preds, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if preds.shape != truths.shape:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs truths {truths.shape}")
    if preds.ndim != 3:
        raise ValueError(f"expected 3D (T, H, W), got shape {preds.shape}")
    if preds.shape[1] != lat_degrees.shape[0]:
        raise ValueError(
            f"lat length {lat_degrees.shape[0]} != H {preds.shape[1]}"
        )

    w = cos_lat_weights(lat_degrees)
    sq_err = (preds - truths) ** 2                  # (T, H, W)
    mse_per_lat = sq_err.mean(axis=-1)              # (T, H)
    mse_per_date = (mse_per_lat * w).sum(axis=-1)   # (T,)
    return float(np.sqrt(mse_per_date.mean()))