"""Tests for great-circle distance computations."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_da.distances import (
    EARTH_RADIUS_KM,
    arc_km_to_chord,
    haversine_distance_km,
    latlon_to_unit_xyz,
    pairwise_distances_km,
)


def test_zero_distance_to_self():
    d = haversine_distance_km(40.0, -75.0, 40.0, -75.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_antipodal_points():
    """Distance from a point to its antipode is half the great circle = pi R."""
    d = haversine_distance_km(40.0, -74.0, -40.0, 106.0)
    assert d == pytest.approx(np.pi * EARTH_RADIUS_KM, rel=1e-6)


def test_equator_quarter_circle():
    d = haversine_distance_km(0.0, 0.0, 0.0, 90.0)
    assert d == pytest.approx(0.5 * np.pi * EARTH_RADIUS_KM, rel=1e-6)


def test_pole_to_equator():
    d = haversine_distance_km(90.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(0.5 * np.pi * EARTH_RADIUS_KM, rel=1e-6)


def test_known_distance_paris_newyork():
    """Paris to NYC ~ 5837 km (reference: online haversine calculators)."""
    d = haversine_distance_km(48.8566, 2.3522, 40.7128, -74.0060)
    assert d == pytest.approx(5837.0, rel=0.005)


def test_symmetric():
    d_ab = haversine_distance_km(10.0, 20.0, -30.0, 100.0)
    d_ba = haversine_distance_km(-30.0, 100.0, 10.0, 20.0)
    assert d_ab == pytest.approx(d_ba, rel=1e-12)


def test_broadcasting():
    lats1 = np.array([0.0, 30.0, 60.0])
    lons1 = np.array([0.0, 0.0, 0.0])
    d = haversine_distance_km(lats1, lons1, lats1, lons1)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, 0.0, atol=1e-9)


def test_pairwise_shape():
    lats_a = np.array([0.0, 30.0])
    lons_a = np.array([0.0, 10.0])
    lats_b = np.array([0.0, 60.0, -30.0])
    lons_b = np.array([0.0, 90.0, -20.0])
    D = pairwise_distances_km(lats_a, lons_a, lats_b, lons_b)
    assert D.shape == (2, 3)


def test_pairwise_diagonal_zero():
    lats = np.linspace(-80, 80, 10)
    lons = np.linspace(-170, 170, 10)
    D = pairwise_distances_km(lats, lons, lats, lons)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)


def test_pairwise_symmetric_when_same_set():
    lats = np.array([0.0, 30.0, -45.0, 60.0])
    lons = np.array([0.0, 100.0, -50.0, 170.0])
    D = pairwise_distances_km(lats, lons, lats, lons)
    np.testing.assert_allclose(D, D.T, rtol=1e-12, atol=1e-9)


def test_no_negative_distances():
    rng = np.random.default_rng(0)
    lats = rng.uniform(-90, 90, size=50)
    lons = rng.uniform(-180, 180, size=50)
    D = pairwise_distances_km(lats, lons, lats, lons)
    assert (D >= 0).all()


def test_distance_bounded_by_half_circumference():
    rng = np.random.default_rng(1)
    lats_a = rng.uniform(-90, 90, size=30)
    lons_a = rng.uniform(-180, 180, size=30)
    lats_b = rng.uniform(-90, 90, size=30)
    lons_b = rng.uniform(-180, 180, size=30)
    D = pairwise_distances_km(lats_a, lons_a, lats_b, lons_b)
    assert D.max() <= np.pi * EARTH_RADIUS_KM + 1e-6


def test_latlon_to_unit_xyz_shape():
    lats = np.array([0.0, 30.0, 60.0])
    lons = np.array([0.0, 90.0, 180.0])
    xyz = latlon_to_unit_xyz(lats, lons)
    assert xyz.shape == (3, 3)


def test_latlon_to_unit_xyz_on_sphere():
    """All output points lie on the unit sphere."""
    rng = np.random.default_rng(2)
    lats = rng.uniform(-90, 90, size=20)
    lons = rng.uniform(-180, 180, size=20)
    xyz = latlon_to_unit_xyz(lats, lons)
    norms = np.linalg.norm(xyz, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_latlon_to_unit_xyz_known_points():
    """North pole -> (0,0,1); equator at (0,0) -> (1,0,0)."""
    xyz = latlon_to_unit_xyz(np.array([90.0, 0.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(xyz[0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(xyz[1], [1, 0, 0], atol=1e-12)


def test_arc_to_chord_small_arc():
    """For small arcs, chord ~ arc / R (unit sphere)."""
    chord = arc_km_to_chord(100.0)
    expected = 100.0 / EARTH_RADIUS_KM
    assert chord == pytest.approx(expected, rel=1e-3)


def test_arc_to_chord_antipode():
    """At arc = pi R, chord = 2 (diameter of unit sphere)."""
    chord = arc_km_to_chord(np.pi * EARTH_RADIUS_KM)
    assert chord == pytest.approx(2.0, abs=1e-12)