import numpy as np
import pytest

from gist import EuclideanDistance, GISTResult, LinearUtility, gist
from gist.algorithm import _greedy_independent_set


# ---- GreedyIndependentSet --------------------------------------------------

class TestGreedyIndependentSet:
    def test_d_zero_standard_greedy(self):
        """With d=0 (no diversity constraint), should match standard greedy."""
        rng = np.random.default_rng(0)
        points = rng.standard_normal((20, 4))
        weights = np.arange(20, dtype=np.float64)
        metric = EuclideanDistance()
        points = metric.prepare(points)
        util = LinearUtility(weights)

        sel, min_pw = _greedy_independent_set(points, util, metric, d=0.0, k=5)

        # Standard greedy on linear utility picks the 5 heaviest.
        assert sorted(sel) == [15, 16, 17, 18, 19]

    def test_all_pairs_satisfy_distance(self):
        """All selected pairs must have pairwise distance >= d."""
        rng = np.random.default_rng(1)
        points = rng.standard_normal((100, 8))
        weights = rng.random(100)
        metric = EuclideanDistance()
        points = metric.prepare(points)
        util = LinearUtility(weights)

        d = 2.0
        sel, min_pw = _greedy_independent_set(points, util, metric, d=d, k=20)

        # Check all pairs.
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                dist = np.linalg.norm(points[sel[i]] - points[sel[j]])
                assert dist >= d - 1e-10

        # min_pw should match the minimum we just computed.
        if len(sel) >= 2:
            actual_min = min(
                np.linalg.norm(points[sel[i]] - points[sel[j]])
                for i in range(len(sel))
                for j in range(i + 1, len(sel))
            )
            assert min_pw == pytest.approx(actual_min, rel=1e-6)

    def test_large_d_returns_few_points(self):
        """Very large d should return only 1 point."""
        points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        weights = np.array([1.0, 2.0, 3.0])
        metric = EuclideanDistance()
        points = metric.prepare(points)
        util = LinearUtility(weights)

        sel, _ = _greedy_independent_set(points, util, metric, d=100.0, k=3)
        assert len(sel) == 1
        assert sel[0] == 2  # Highest weight.

    def test_celf_matches_brute_force(self):
        """CELF should produce the same result as brute-force argmax."""
        rng = np.random.default_rng(42)
        points = rng.standard_normal((30, 4))
        weights = rng.random(30)
        metric = EuclideanDistance()
        points = metric.prepare(points)
        util = LinearUtility(weights)

        d = 1.5
        sel, _ = _greedy_independent_set(points, util, metric, d=d, k=10)

        # Brute-force: manually run greedy independent set.
        n = len(points)
        active = set(range(n))
        bf_sel = []
        for _ in range(10):
            candidates = sorted(active)
            if not candidates:
                break
            best = max(candidates, key=lambda v: weights[v])
            bf_sel.append(best)
            active.discard(best)
            to_remove = set()
            for v in active:
                if np.linalg.norm(points[v] - points[best]) < d:
                    to_remove.add(v)
            active -= to_remove

        assert sel == bf_sel


# ---- GIST main function ---------------------------------------------------

class TestGist:
    def test_basic_run(self):
        rng = np.random.default_rng(10)
        points = rng.standard_normal((200, 8))
        weights = rng.random(200)

        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=20, lam=1.0, eps=0.1, seed=10,
        )

        assert isinstance(result, GISTResult)
        assert len(result.indices) <= 20
        assert len(result.indices) > 0

        # f(S) = g(S) + lam * div(S).
        assert result.objective_value == pytest.approx(
            result.utility_value + 1.0 * result.diversity, rel=1e-10,
        )

    def test_objective_consistency(self):
        """Manually verify f(S) = g(S) + lam * div(S)."""
        rng = np.random.default_rng(20)
        points = rng.standard_normal((50, 4))
        weights = rng.random(50)
        lam = 2.5

        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=10, lam=lam, eps=0.1, seed=20,
        )

        # Recompute g(S).
        g = float(weights[result.indices].sum())
        assert result.utility_value == pytest.approx(g, rel=1e-10)

        # Recompute div(S).
        metric = EuclideanDistance()
        pts = metric.prepare(points.copy())
        sel = result.indices
        if len(sel) >= 2:
            min_dist = min(
                np.linalg.norm(pts[sel[i]] - pts[sel[j]])
                for i in range(len(sel))
                for j in range(i + 1, len(sel))
            )
            assert result.diversity == pytest.approx(min_dist, rel=1e-6)

        assert result.objective_value == pytest.approx(g + lam * result.diversity, rel=1e-10)

    def test_all_pairs_diverse(self):
        """All selected points should have pairwise distance >= diversity."""
        rng = np.random.default_rng(30)
        points = rng.standard_normal((100, 8))
        weights = rng.random(100)

        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=15, lam=1.0, eps=0.1, seed=30,
        )

        sel = result.indices
        if len(sel) >= 2:
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    d = np.linalg.norm(points[sel[i]] - points[sel[j]])
                    assert d >= result.diversity - 1e-6

    def test_k_equals_1(self):
        points = np.array([[0.0, 0.0], [10.0, 0.0]])
        weights = np.array([5.0, 3.0])
        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=1, lam=1.0, eps=0.1, seed=0,
        )
        assert len(result.indices) == 1

    def test_k_equals_2(self):
        """k=2 should consider the diametrical pair."""
        points = np.array([[0.0], [10.0], [5.0]])
        weights = np.array([0.0, 0.0, 100.0])
        # With lam=10, the diametrical pair {0,1} gives
        # f = 0 + 10*10 = 100.
        # The greedy picks point 2 (weight 100), then gets a pair with less diversity.
        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=2, lam=10.0, eps=0.1, seed=0,
        )
        assert len(result.indices) == 2

    def test_determinism(self):
        rng_points = np.random.default_rng(40)
        points = rng_points.standard_normal((100, 8))
        weights = rng_points.random(100)

        r1 = gist(points, LinearUtility(weights), EuclideanDistance(),
                   k=10, lam=1.0, seed=42)
        r2 = gist(points, LinearUtility(weights), EuclideanDistance(),
                   k=10, lam=1.0, seed=42)

        np.testing.assert_array_equal(r1.indices, r2.indices)
        assert r1.objective_value == r2.objective_value

    def test_precomputed_diameter(self):
        points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        weights = np.array([1.0, 1.0, 1.0])
        d_max = 5.0  # dist(0, 1)
        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=2, lam=1.0, eps=0.1, diameter=(d_max, 0, 1),
        )
        assert len(result.indices) == 2

    def test_empty_input(self):
        points = np.empty((0, 4))
        weights = np.empty(0)
        result = gist(points, LinearUtility(weights), EuclideanDistance(), k=5)
        assert len(result.indices) == 0
        assert result.objective_value == 0.0

    def test_k_zero(self):
        points = np.array([[1.0, 2.0]])
        result = gist(points, LinearUtility(np.array([1.0])), EuclideanDistance(), k=0)
        assert len(result.indices) == 0

    def test_identical_points(self):
        """All points at the same location: d_max = 0, skip threshold sweep."""
        points = np.ones((10, 3))
        weights = np.arange(10, dtype=np.float64)
        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=5, lam=1.0, seed=0,
        )
        assert len(result.indices) <= 5
        assert result.diversity == pytest.approx(0.0, abs=1e-10)

    def test_cosine_distance(self):
        from gist import CosineDistance

        rng = np.random.default_rng(50)
        points = rng.standard_normal((80, 16))
        weights = rng.random(80)

        result = gist(
            points, LinearUtility(weights), CosineDistance(),
            k=10, lam=0.5, eps=0.1, seed=50,
        )
        assert len(result.indices) <= 10
        assert result.objective_value > 0

    def test_lam_zero_pure_utility(self):
        """With lam=0, should pick top-k by utility."""
        weights = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        points = np.random.default_rng(0).standard_normal((5, 2))
        result = gist(
            points, LinearUtility(weights), EuclideanDistance(),
            k=3, lam=0.0, eps=0.1, seed=0,
        )
        # Top 3 weights: indices 0 (5.0), 2 (4.0), 4 (3.0).
        assert set(result.indices) == {0, 2, 4}

    def test_invalid_inputs(self):
        points = np.array([1.0, 2.0, 3.0])  # 1-D
        with pytest.raises(ValueError, match="2-D"):
            gist(points, LinearUtility(np.array([1.0])), EuclideanDistance(), k=1)

        points = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="lam"):
            gist(points, LinearUtility(np.array([1.0])), EuclideanDistance(), k=1, lam=-1)

        with pytest.raises(ValueError, match="eps"):
            gist(points, LinearUtility(np.array([1.0])), EuclideanDistance(), k=1, eps=0)
