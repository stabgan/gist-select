import numpy as np
import pytest

from gist.distances import (
    CallableDistance,
    CosineDistance,
    EuclideanDistance,
    approximate_diameter,
)


# ---- EuclideanDistance -----------------------------------------------------

class TestEuclideanDistance:
    def test_correctness(self):
        rng = np.random.default_rng(0)
        points = rng.standard_normal((100, 16)).astype(np.float64)
        metric = EuclideanDistance()
        points = metric.prepare(points)

        source = 0
        targets = np.array([1, 2, 3, 50, 99], dtype=np.intp)
        dists = metric.from_point(points, source, targets)

        expected = np.array([
            np.linalg.norm(points[source] - points[t]) for t in targets
        ])
        np.testing.assert_allclose(dists, expected, rtol=1e-6)

    def test_self_distance_is_zero(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        metric = EuclideanDistance()
        points = metric.prepare(points)
        d = metric.from_point(points, 0, np.array([0], dtype=np.intp))
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_prepare_stores_norms(self):
        points = np.array([[3.0, 4.0], [1.0, 0.0]])
        metric = EuclideanDistance()
        metric.prepare(points)
        np.testing.assert_allclose(metric._norms_sq, [25.0, 1.0])

    def test_float32(self):
        points = np.random.default_rng(1).standard_normal((50, 8)).astype(np.float32)
        metric = EuclideanDistance()
        points = metric.prepare(points)
        dists = metric.from_point(points, 0, np.arange(50, dtype=np.intp))
        assert dists.shape == (50,)
        assert dists[0] == pytest.approx(0.0, abs=1e-3)  # float32 precision


# ---- CosineDistance --------------------------------------------------------

class TestCosineDistance:
    def test_correctness(self):
        rng = np.random.default_rng(2)
        points = rng.standard_normal((100, 16))
        metric = CosineDistance()
        normed = metric.prepare(points)

        source = 5
        targets = np.array([0, 10, 20, 99], dtype=np.intp)
        dists = metric.from_point(normed, source, targets)

        # Reference: scipy cosine distance.
        from scipy.spatial.distance import cosine
        expected = np.array([cosine(points[source], points[t]) for t in targets])
        np.testing.assert_allclose(dists, expected, rtol=1e-6)

    def test_prepare_normalises(self):
        points = np.array([[3.0, 4.0], [0.0, 5.0]])
        metric = CosineDistance()
        normed = metric.prepare(points)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)

    def test_identical_points_zero_distance(self):
        points = np.array([[1.0, 2.0], [1.0, 2.0]])
        metric = CosineDistance()
        normed = metric.prepare(points)
        d = metric.from_point(normed, 0, np.array([1], dtype=np.intp))
        assert d[0] == pytest.approx(0.0, abs=1e-10)


# ---- CallableDistance ------------------------------------------------------

class TestCallableDistance:
    def test_wraps_function(self):
        def manhattan(source_vec, target_matrix):
            return np.abs(target_matrix - source_vec).sum(axis=1)

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
        metric = CallableDistance(manhattan)
        targets = np.array([1, 2], dtype=np.intp)
        dists = metric.from_point(points, 0, targets)
        np.testing.assert_allclose(dists, [1.0, 3.0])


# ---- approximate_diameter --------------------------------------------------

class TestApproximateDiameter:
    def test_line_of_points(self):
        """For evenly spaced points on a line, diameter should be exact."""
        points = np.arange(10, dtype=np.float64).reshape(-1, 1)
        metric = EuclideanDistance()
        points = metric.prepare(points)
        rng = np.random.default_rng(42)
        d_max, u, v = approximate_diameter(points, metric, rng)
        assert d_max == pytest.approx(9.0)
        assert {u, v} == {0, 9}

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        points = np.random.default_rng(0).standard_normal((200, 4))
        metric = EuclideanDistance()
        points = metric.prepare(points)
        r1 = approximate_diameter(points, metric, rng1)
        r2 = approximate_diameter(points, metric, rng2)
        assert r1 == r2

    def test_multi_start_improves(self):
        """More starts should find diameter >= single-start."""
        rng = np.random.default_rng(7)
        points = rng.standard_normal((500, 32))
        metric = EuclideanDistance()
        points = metric.prepare(points)
        d1, _, _ = approximate_diameter(points, metric, np.random.default_rng(7), n_starts=1)
        d5, _, _ = approximate_diameter(points, metric, np.random.default_rng(7), n_starts=5)
        assert d5 >= d1 - 1e-10
