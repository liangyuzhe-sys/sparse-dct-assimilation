"""Great-circle distances on the Earth's surface.

All distances returned in kilometers. Uses the haversine formula because it
is numerically stable near zero distance (the spherical law of cosines loses
precision there).
"""

from __future__ import annotations

import numpy as np


# Mean Earth radius in km (IUGG 2015 mean radius R_1).
EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle distance between two sets of points, in km.

    All inputs are broadcastable arrays of latitudes/longitudes in degrees.
    Returns an array shaped by broadcasting.

    Formula (haversine):
        a = sin^2(d_phi / 2) + cos(phi1) cos(phi2) sin^2(d_lambda / 2)
        d = 2 R asin(sqrt(a))
    """
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)  # guard against tiny negative or >1 from FP
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def pairwise_distances_km(
    lats_a_deg: np.ndarray,
    lons_a_deg: np.ndarray,
    lats_b_deg: np.ndarray,
    lons_b_deg: np.ndarray,
) -> np.ndarray:
    """Pairwise distances between two sets of points.

    Parameters
    ----------
    lats_a_deg, lons_a_deg : (M,) arrays
    lats_b_deg, lons_b_deg : (N,) arrays

    Returns
    -------
    distances : (M, N) array of great-circle distances in km.
    """
    lats_a = np.asarray(lats_a_deg).reshape(-1, 1)
    lons_a = np.asarray(lons_a_deg).reshape(-1, 1)
    lats_b = np.asarray(lats_b_deg).reshape(1, -1)
    lons_b = np.asarray(lons_b_deg).reshape(1, -1)
    return haversine_distance_km(lats_a, lons_a, lats_b, lons_b)


def latlon_to_unit_xyz(lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in degrees to (x, y, z) on the unit sphere.

    Useful for feeding into scipy.spatial.cKDTree (which uses Euclidean
    distance). Chord distance on the unit sphere is monotonic in great-circle
    distance, so radius queries are valid after appropriate conversion.

    Returns array of shape lats.shape + (3,).
    """
    lats = np.deg2rad(lats_deg)
    lons = np.deg2rad(lons_deg)
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)
    return np.stack([x, y, z], axis=-1)


def arc_km_to_chord(arc_km: float) -> float:
    """Convert great-circle arc length (km) to chord length on the unit sphere.

    The relationship is chord = 2 sin(arc / 2R). For small arcs this is
    nearly arc/R; for arc = pi R (antipode), chord = 2.
    """
    return 2.0 * np.sin(arc_km / (2.0 * EARTH_RADIUS_KM))