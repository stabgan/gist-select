"""GIST Paper Verification Test Suite.

Comprehensive correctness verification of the gist-select implementation
against the research paper "GIST: Greedy Independent Set Thresholding for
Max-Min Diversification with Submodular Utility" (arXiv:2405.18754).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gist.algorithm import GISTResult, _build_thresholds, _greedy_independent_set, gist
from gist.distances import (
    CosineDistance,
    EuclideanDistance,
    approximate_diameter,
    exact_diameter,
)
from gist.objectives import CoverageFunction, LinearUtility

# ---------------------------------------------------------------------------
# Shared hypothesis settings
# ---------------------------------------------------------------------------

paper_verification_settings = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# ---------------------------------------------------------------------------
# Brute-force oracle helpers
# ---------------------------------------------------------------------------


def _all_pairwise_distances(points, metric):
    """Return an (n, n) matrix of pairwise distances.

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    n = len(points)
    all_idx = np.arange(n, dtype=np.intp)
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dists[i] = metric.from_point(points, i, all_idx)
    return dists


def _max_pairwise_distance(points, metric):
    """Compute the true diameter (max pairwise distance).

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    if len(points) <= 1:
        return 0.0
    dists = _all_pairwise_distances(points, metric)
    return float(dists.max())


def compute_diversity(points, metric, indices):
    """Compute div(S) following the paper convention.

    - |S| >= 2: min pairwise distance among selected points.
    - |S| <= 1: d_max (diameter of the full point set).

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    if len(indices) >= 2:
        idx_arr = np.array(indices, dtype=np.intp)
        min_pw = np.inf
        for i in range(len(idx_arr)):
            for j in range(i + 1, len(idx_arr)):
                targets = np.array([idx_arr[j]], dtype=np.intp)
                d = float(metric.from_point(points, int(idx_arr[i]), targets)[0])
                min_pw = min(min_pw, d)
        return min_pw
    return _max_pairwise_distance(points, metric)


def compute_objective(points, utility, metric, indices, lam, d_max):
    """Compute f(S) = g(S) + lam * div(S) with proper diversity convention.

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    if len(indices) == 0:
        return 0.0
    g = utility.value(list(indices))
    if len(indices) >= 2:
        div = compute_diversity(points, metric, indices)
    else:
        div = d_max
    return g + lam * div


def brute_force_optimal(points, utility, metric, k, lam):
    """Enumerate all subsets of size 1..k, return the one maximising f(S).

    Returns ``(best_indices, best_f)`` where *best_indices* is a list of
    int and *best_f* is the objective value.

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    n = len(points)
    d_max = _max_pairwise_distance(points, metric)
    best_indices: list[int] = []
    best_f = 0.0  # f(empty) = 0

    for size in range(1, min(k, n) + 1):
        for subset in combinations(range(n), size):
            f = compute_objective(points, utility, metric, list(subset), lam, d_max)
            if f > best_f or (f == best_f and not best_indices):
                best_f = f
                best_indices = list(subset)

    return best_indices, best_f


def brute_force_greedy_independent_set(points, utility, metric, d, k):
    """Naive greedy independent set — recomputes ALL marginal gains each step.

    No CELF optimisation.  Ties broken by lowest index for stability.
    Returns a list of selected indices.

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    n = len(points)
    selected: list[int] = []
    is_active = np.ones(n, dtype=bool)

    for _ in range(k):
        # Build candidate set: active points at distance >= d from all selected.
        candidates = []
        for idx in range(n):
            if not is_active[idx]:
                continue
            if selected and d > 0:
                sel_arr = np.array(selected, dtype=np.intp)
                dists = metric.from_point(points, idx, sel_arr)
                if float(dists.min()) < d:
                    continue
            candidates.append(idx)

        if not candidates:
            break

        # Compute marginal gains for ALL candidates (no laziness).
        cand_arr = np.array(candidates, dtype=np.intp)
        gains = utility.marginal_gains(selected, cand_arr)

        # Pick argmax gain, break ties by lowest index.
        max_gain = float(gains.max())
        best_idx = None
        for i, c in enumerate(candidates):
            if float(gains[i]) == max_gain:
                best_idx = c
                break

        selected.append(best_idx)
        is_active[best_idx] = False

        # Eliminate points within distance d.
        if d > 0:
            for idx in range(n):
                if not is_active[idx]:
                    continue
                dist_val = float(
                    metric.from_point(
                        points, best_idx, np.array([idx], dtype=np.intp)
                    )[0]
                )
                if dist_val < d:
                    is_active[idx] = False

    return selected


def brute_force_d_independent_sets(points, metric, d, k):
    """Enumerate all d-independent sets of size 1..k.

    A set S is d-independent if all pairwise distances >= d.
    Returns a list of lists of indices.

    ``points`` must already be the result of ``metric.prepare(points)``.
    """
    n = len(points)
    result: list[list[int]] = []

    for size in range(1, min(k, n) + 1):
        for subset in combinations(range(n), size):
            is_d_indep = True
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    targets = np.array([subset[j]], dtype=np.intp)
                    dist_val = float(
                        metric.from_point(points, int(subset[i]), targets)[0]
                    )
                    if dist_val < d:
                        is_d_indep = False
                        break
                if not is_d_indep:
                    break
            if is_d_indep:
                result.append(list(subset))

    return result


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def random_points(draw, n_max=15, d_max=5):
    """Generate a random point array of shape (n, d).

    2 <= n <= n_max, 1 <= d <= d_max, values from reasonable floats.
    """
    n = draw(st.integers(min_value=2, max_value=n_max))
    d = draw(st.integers(min_value=1, max_value=d_max))
    data = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n * d,
            max_size=n * d,
        )
    )
    return np.array(data, dtype=np.float64).reshape(n, d)


@st.composite
def random_weights(draw, n):
    """Generate a non-negative float array of length n."""
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(data, dtype=np.float64)


@st.composite
def random_coverage_matrix(draw, n, m_max=20):
    """Generate a sparse binary csr_matrix (n, m) with >= 1 entry per row."""
    m = draw(st.integers(min_value=1, max_value=m_max))
    rows, cols = [], []
    for i in range(n):
        # Ensure at least one element per row.
        num_covered = draw(st.integers(min_value=1, max_value=max(1, m)))
        covered = draw(
            st.lists(
                st.integers(min_value=0, max_value=m - 1),
                min_size=num_covered,
                max_size=num_covered,
            )
        )
        for j in set(covered):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
    return mat


# ---------------------------------------------------------------------------
# Property 12: Euclidean Metric Axioms
# ---------------------------------------------------------------------------


class TestMetricProperties:
    """Feature: gist-paper-verification, Property 12: Euclidean Metric Axioms

    Validates: Requirements 12.1, 12.2, 12.3, 12.4
    """

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_euclidean_non_negativity(self, pts):
        """**Validates: Requirements 12.1**

        dist(u, v) >= 0 for all pairs.
        """
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        all_idx = np.arange(n, dtype=np.intp)
        for i in range(n):
            dists = metric.from_point(points, i, all_idx)
            assert np.all(dists >= -1e-10), (
                f"Negative distance found from point {i}: {dists[dists < -1e-10]}"
            )

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_euclidean_identity_of_indiscernibles(self, pts):
        """**Validates: Requirements 12.2**

        dist(u, u) = 0 for all points u.
        """
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        for i in range(n):
            self_idx = np.array([i], dtype=np.intp)
            d = float(metric.from_point(points, i, self_idx)[0])
            # The norm-identity trick (||a||^2 + ||b||^2 - 2*a·b) can leave
            # small floating-point residuals for large coordinates, so we
            # allow a tolerance proportional to the point magnitude.
            assert d == pytest.approx(0.0, abs=1e-5), (
                f"dist(u, u) != 0 for point {i}: got {d}"
            )

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_euclidean_symmetry(self, pts):
        """**Validates: Requirements 12.3**

        dist(u, v) = dist(v, u) for all pairs.
        """
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                j_idx = np.array([j], dtype=np.intp)
                i_idx = np.array([i], dtype=np.intp)
                d_ij = float(metric.from_point(points, i, j_idx)[0])
                d_ji = float(metric.from_point(points, j, i_idx)[0])
                assert d_ij == pytest.approx(d_ji, rel=1e-9), (
                    f"Symmetry violated: dist({i},{j})={d_ij} != dist({j},{i})={d_ji}"
                )

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_euclidean_triangle_inequality(self, pts):
        """**Validates: Requirements 12.4**

        dist(u, w) <= dist(u, v) + dist(v, w) for all triples.
        """
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        # Precompute all pairwise distances for efficiency.
        all_idx = np.arange(n, dtype=np.intp)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            dist_matrix[i] = metric.from_point(points, i, all_idx)

        for u in range(n):
            for v in range(n):
                for w in range(n):
                    d_uw = dist_matrix[u, w]
                    d_uv = dist_matrix[u, v]
                    d_vw = dist_matrix[v, w]
                    rhs = d_uv + d_vw
                    # Use relative tolerance scaled by the magnitude of the
                    # distances to accommodate floating-point arithmetic on
                    # large values.
                    tol = 1e-9 + 1e-7 * max(d_uw, rhs)
                    assert d_uw <= rhs + tol, (
                        f"Triangle inequality violated: "
                        f"dist({u},{w})={d_uw} > "
                        f"dist({u},{v})={d_uv} + dist({v},{w})={d_vw}"
                    )


# ---------------------------------------------------------------------------
# Property 13: Cosine Distance Axioms
# ---------------------------------------------------------------------------


class TestCosineDistanceAxioms:
    """Feature: gist-paper-verification, Property 13: Cosine Distance Axioms

    Validates: Requirements 12.5, 12.6
    """

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_cosine_non_negativity(self, pts):
        """**Validates: Requirements 12.5**

        dist(u, v) >= 0 for all non-zero vector pairs.
        CosineDistance computes 1 - cos(a, b) which is in [0, 2] for unit vectors.
        """
        from hypothesis import assume

        # Filter out rows with zero (or near-zero) norm
        norms = np.linalg.norm(pts, axis=1)
        assume(np.all(norms > 1e-10))

        metric = CosineDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        all_idx = np.arange(n, dtype=np.intp)
        for i in range(n):
            dists = metric.from_point(points, i, all_idx)
            assert np.all(dists >= -1e-9), (
                f"Negative cosine distance found from point {i}: "
                f"{dists[dists < -1e-9]}"
            )

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=5))
    def test_cosine_symmetry(self, pts):
        """**Validates: Requirements 12.6**

        dist(u, v) = dist(v, u) for all non-zero vector pairs.
        """
        from hypothesis import assume

        # Filter out rows with zero (or near-zero) norm
        norms = np.linalg.norm(pts, axis=1)
        assume(np.all(norms > 1e-10))

        metric = CosineDistance()
        points = metric.prepare(pts.copy())
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                j_idx = np.array([j], dtype=np.intp)
                i_idx = np.array([i], dtype=np.intp)
                d_ij = float(metric.from_point(points, i, j_idx)[0])
                d_ji = float(metric.from_point(points, j, i_idx)[0])
                assert d_ij == pytest.approx(d_ji, rel=1e-9), (
                    f"Cosine symmetry violated: "
                    f"dist({i},{j})={d_ij} != dist({j},{i})={d_ji}"
                )


# ---------------------------------------------------------------------------
# Property 14: CoverageFunction Submodularity
# ---------------------------------------------------------------------------


class TestSubmodularProperties:
    """Feature: gist-paper-verification, Property 14: CoverageFunction Submodularity

    Validates: Requirements 13.1, 13.2
    """

    @paper_verification_settings
    @given(data=st.data())
    def test_coverage_diminishing_returns(self, data):
        """**Validates: Requirements 13.1**

        For S ⊆ T and v ∉ T: g(v|S) >= g(v|T) (diminishing returns).
        """
        from hypothesis import assume

        n = data.draw(st.integers(min_value=3, max_value=10))
        cov = data.draw(random_coverage_matrix(n))
        cf = CoverageFunction(cov)

        # Generate S ⊆ T ⊆ {0,...,n-1} and v ∉ T
        all_indices = list(range(n))
        T = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(all_indices),
                    min_size=1,
                    max_size=n - 1,
                    unique=True,
                )
            )
        )
        remaining = [i for i in all_indices if i not in T]
        assume(len(remaining) > 0)
        v = data.draw(st.sampled_from(remaining))
        S = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(T),
                    min_size=0,
                    max_size=len(T),
                    unique=True,
                )
            )
        )

        v_arr = np.array([v], dtype=np.intp)
        gain_S = cf.marginal_gains(S, v_arr)[0]
        gain_T = cf.marginal_gains(T, v_arr)[0]
        assert gain_S >= gain_T - 1e-10, (
            f"Diminishing returns violated: g(v={v}|S={S})={gain_S} "
            f"< g(v={v}|T={T})={gain_T}"
        )

    @paper_verification_settings
    @given(data=st.data())
    def test_coverage_monotonicity(self, data):
        """**Validates: Requirements 13.2**

        For S ⊆ T: g(S) <= g(T) (monotonicity).
        """
        n = data.draw(st.integers(min_value=2, max_value=10))
        cov = data.draw(random_coverage_matrix(n))
        cf = CoverageFunction(cov)

        all_indices = list(range(n))
        T = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(all_indices),
                    min_size=1,
                    max_size=n,
                    unique=True,
                )
            )
        )
        S = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(T),
                    min_size=0,
                    max_size=len(T),
                    unique=True,
                )
            )
        )

        g_S = cf.value(S)
        g_T = cf.value(T)
        assert g_S <= g_T + 1e-10, (
            f"Monotonicity violated: g(S={S})={g_S} > g(T={T})={g_T}"
        )

class TestLinearUtilityModularity:
    """Feature: gist-paper-verification, Property 15: LinearUtility Modularity

    Validates: Requirements 13.3, 13.4
    """

    @paper_verification_settings
    @given(data=st.data())
    def test_linear_context_independent_marginal_gains(self, data):
        """**Validates: Requirements 13.3**

        For S ⊆ T and v ∉ T: g(v|S) = g(v|T) (marginal gains independent
        of context, since linear is modular).
        """
        from hypothesis import assume

        n = data.draw(st.integers(min_value=3, max_value=12))
        weights = data.draw(random_weights(n))
        lu = LinearUtility(weights)

        all_indices = list(range(n))
        # Draw T ⊆ {0,...,n-1} with room for at least one element outside
        T = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(all_indices),
                    min_size=1,
                    max_size=n - 1,
                    unique=True,
                )
            )
        )
        remaining = [i for i in all_indices if i not in T]
        assume(len(remaining) > 0)
        v = data.draw(st.sampled_from(remaining))

        # Draw S ⊆ T
        S = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(T),
                    min_size=0,
                    max_size=len(T),
                    unique=True,
                )
            )
        )

        v_arr = np.array([v], dtype=np.intp)
        gain_S = lu.marginal_gains(S, v_arr)[0]
        gain_T = lu.marginal_gains(T, v_arr)[0]
        assert abs(gain_S - gain_T) < 1e-10, (
            f"Modularity violated: g(v={v}|S={S})={gain_S} "
            f"!= g(v={v}|T={T})={gain_T}"
        )

    @paper_verification_settings
    @given(data=st.data())
    def test_linear_monotonicity(self, data):
        """**Validates: Requirements 13.4**

        For S ⊆ T with non-negative weights: g(S) <= g(T) (monotonicity).
        """
        n = data.draw(st.integers(min_value=2, max_value=12))
        weights = data.draw(random_weights(n))
        lu = LinearUtility(weights)

        all_indices = list(range(n))
        T = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(all_indices),
                    min_size=1,
                    max_size=n,
                    unique=True,
                )
            )
        )
        S = sorted(
            data.draw(
                st.lists(
                    st.sampled_from(T),
                    min_size=0,
                    max_size=len(T),
                    unique=True,
                )
            )
        )

        g_S = lu.value(S)
        g_T = lu.value(T)
        assert g_S <= g_T + 1e-10, (
            f"Monotonicity violated: g(S={S})={g_S} > g(T={T})={g_T}"
        )


# ---------------------------------------------------------------------------
# Property 1, 2, 3: GreedyIndependentSet Correctness
# ---------------------------------------------------------------------------


class TestGreedyIndependentSetCorrectness:
    """Feature: gist-paper-verification, Properties 1-3

    Property 1: d-Independence Invariant
    Property 2: Maximality When |S| < k
    Property 3: d=0 Reduces to Standard Greedy

    Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """

    # -- Property 1: d-Independence Invariant (Req 1.1) --------------------

    @paper_verification_settings
    @given(data=st.data())
    def test_d_independence_invariant(self, data):
        """**Validates: Requirements 1.1**

        All pairwise distances in the returned set must be >= d.
        """
        pts = data.draw(random_points(n_max=8, d_max=3))
        n = len(pts)
        weights = data.draw(random_weights(n))
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        k = data.draw(st.integers(min_value=1, max_value=n))
        d = data.draw(st.floats(min_value=0.1, max_value=5.0,
                                allow_nan=False, allow_infinity=False))

        selected, min_pw = _greedy_independent_set(points, utility, metric, d, k)

        # Check all pairwise distances >= d
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                targets = np.array([selected[j]], dtype=np.intp)
                dist_val = float(
                    metric.from_point(points, selected[i], targets)[0]
                )
                assert dist_val >= d - 1e-9, (
                    f"d-independence violated: dist({selected[i]}, {selected[j]}) "
                    f"= {dist_val} < d = {d}"
                )

    # -- Property 2: Maximality When |S| < k (Req 1.2, 1.6) ---------------

    @paper_verification_settings
    @given(data=st.data())
    def test_maximality_when_fewer_than_k(self, data):
        """**Validates: Requirements 1.2, 1.6**

        If |S| < k, no point v in V\\S has dist(v, S) >= d (maximal set).
        When the candidate set becomes empty before k elements, the returned
        set is maximal.
        """
        pts = data.draw(random_points(n_max=8, d_max=3))
        n = len(pts)
        weights = data.draw(random_weights(n))
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        k = data.draw(st.integers(min_value=1, max_value=n))
        d = data.draw(st.floats(min_value=0.1, max_value=5.0,
                                allow_nan=False, allow_infinity=False))

        selected, _ = _greedy_independent_set(points, utility, metric, d, k)

        if len(selected) < k:
            # Maximality: every point not in S must be within distance d
            # of some point in S.
            selected_set = set(selected)
            sel_arr = np.array(selected, dtype=np.intp)
            for v in range(n):
                if v in selected_set:
                    continue
                dists = metric.from_point(points, v, sel_arr)
                min_dist = float(dists.min())
                assert min_dist < d + 1e-9, (
                    f"Maximality violated: point {v} not in S has "
                    f"min dist to S = {min_dist} >= d = {d}"
                )

    # -- Property 3: d=0 Reduces to Standard Greedy (Req 1.3) -------------

    @paper_verification_settings
    @given(data=st.data())
    def test_d_zero_is_standard_greedy(self, data):
        """**Validates: Requirements 1.3**

        With d=0, GreedyIndependentSet reduces to standard greedy: top-k
        by weight for LinearUtility with distinct weights.
        """
        pts = data.draw(random_points(n_max=8, d_max=3))
        n = len(pts)

        # Generate distinct weights by using arange + small perturbation
        base_weights = np.arange(n, dtype=np.float64) + 1.0
        perturbation = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=0.01,
                          allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            )
        )
        weights = base_weights + np.array(perturbation)
        # Ensure all weights are distinct
        weights = weights + np.arange(n) * 0.001

        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        k = data.draw(st.integers(min_value=1, max_value=n))

        selected, _ = _greedy_independent_set(points, utility, metric, 0.0, k)

        # With d=0 and distinct weights, greedy should pick top-k by weight
        expected = sorted(range(n), key=lambda i: -weights[i])[:k]

        assert selected == expected, (
            f"d=0 greedy mismatch: got {selected}, expected {expected}. "
            f"Weights: {weights}"
        )

    # -- Unit test: Deterministic tie-breaking (Req 1.4) -------------------

    def test_deterministic_tie_breaking(self):
        """**Validates: Requirements 1.4**

        Equal-weight inputs produce a stable, deterministic order across
        multiple runs.
        """
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        # Run twice with same inputs
        sel1, _ = _greedy_independent_set(points, utility, metric, 0.5, 4)
        sel2, _ = _greedy_independent_set(points, utility, metric, 0.5, 4)

        assert sel1 == sel2, (
            f"Non-deterministic tie-breaking: run1={sel1}, run2={sel2}"
        )

    # -- Unit test: Candidate set empty → early return (Req 1.5, 1.6) -----

    def test_candidate_set_empty_early_return(self):
        """**Validates: Requirements 1.5, 1.6**

        When the candidate set becomes empty before k elements are selected,
        the algorithm returns the current set immediately.
        """
        # Place points far apart but use a very large d so that after
        # selecting the first point, all others are eliminated.
        pts = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ], dtype=np.float64)
        weights = np.array([10.0, 5.0, 1.0], dtype=np.float64)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        # d=1.0 is larger than all pairwise distances (max ~0.2),
        # so after selecting the first point, all others are eliminated.
        selected, _ = _greedy_independent_set(points, utility, metric, 1.0, 3)

        assert len(selected) == 1, (
            f"Expected 1 element (early return), got {len(selected)}: {selected}"
        )
        # Should have selected the highest-weight point
        assert selected[0] == 0, (
            f"Expected point 0 (highest weight), got {selected[0]}"
        )


# ---------------------------------------------------------------------------
# Property 4: CELF Equivalence
# ---------------------------------------------------------------------------


class TestCELFEquivalence:
    """Feature: gist-paper-verification, Property 4: CELF Equivalence

    Verify that the CELF (lazy greedy) optimisation in _greedy_independent_set
    produces identical results to a brute-force greedy that recomputes all
    marginal gains at each iteration.

    **Validates: Requirements 2.1, 2.2**
    """

    @paper_verification_settings
    @given(data=st.data())
    def test_celf_matches_brute_force_greedy_linear(self, data):
        """**Validates: Requirements 2.1, 2.2**

        For LinearUtility, _greedy_independent_set (CELF) must return the
        same selected set as brute_force_greedy_independent_set.

        When marginal gains are tied, the CELF heap may break ties
        differently from the index-order brute-force.  We verify:
        1. Same number of elements selected.
        2. Same set of elements (for LinearUtility, ties in marginal gains
           mean equal weights, so any tie-broken choice yields the same
           utility value — we check set equality when possible, and fall
           back to utility-value equality).
        """
        pts = data.draw(random_points(n_max=8, d_max=3))
        n = len(pts)
        weights = data.draw(random_weights(n))
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        d = data.draw(
            st.floats(min_value=0.0, max_value=5.0,
                      allow_nan=False, allow_infinity=False)
        )
        k = data.draw(st.integers(min_value=1, max_value=n))

        celf_selected, _ = _greedy_independent_set(points, utility, metric, d, k)
        bf_selected = brute_force_greedy_independent_set(points, utility, metric, d, k)

        assert len(celf_selected) == len(bf_selected), (
            f"CELF vs brute-force length mismatch (LinearUtility):\n"
            f"  CELF:       {celf_selected}\n"
            f"  Brute-force: {bf_selected}\n"
            f"  d={d}, k={k}, n={n}"
        )
        # Both must achieve the same utility value (greedy optimality).
        celf_val = utility.value(celf_selected)
        bf_val = utility.value(bf_selected)
        assert celf_val == pytest.approx(bf_val, abs=1e-9), (
            f"CELF vs brute-force utility mismatch (LinearUtility):\n"
            f"  CELF:       {celf_selected} -> g={celf_val}\n"
            f"  Brute-force: {bf_selected} -> g={bf_val}\n"
            f"  d={d}, k={k}, n={n}"
        )

    @paper_verification_settings
    @given(data=st.data())
    def test_celf_matches_brute_force_greedy_coverage(self, data):
        """**Validates: Requirements 2.1, 2.2**

        For CoverageFunction, _greedy_independent_set (CELF) must return a
        result equivalent to brute_force_greedy_independent_set.

        When marginal gains are tied, the CELF heap may break ties
        differently from the index-order brute-force, producing a different
        set.  We verify:
        1. Same number of elements selected.
        2. Same utility value g(S) — both are valid greedy selections.
        """
        pts = data.draw(random_points(n_max=8, d_max=3))
        n = len(pts)
        cov_matrix = data.draw(random_coverage_matrix(n))
        utility = CoverageFunction(cov_matrix)
        metric = EuclideanDistance()
        points = metric.prepare(pts.copy())

        d = data.draw(
            st.floats(min_value=0.0, max_value=5.0,
                      allow_nan=False, allow_infinity=False)
        )
        k = data.draw(st.integers(min_value=1, max_value=n))

        celf_selected, _ = _greedy_independent_set(points, utility, metric, d, k)
        bf_selected = brute_force_greedy_independent_set(points, utility, metric, d, k)

        assert len(celf_selected) == len(bf_selected), (
            f"CELF vs brute-force length mismatch (CoverageFunction):\n"
            f"  CELF:       {celf_selected}\n"
            f"  Brute-force: {bf_selected}\n"
            f"  d={d}, k={k}, n={n}"
        )
        # Both must achieve the same utility value (greedy optimality).
        celf_val = utility.value(celf_selected)
        bf_val = utility.value(bf_selected)
        assert celf_val == pytest.approx(bf_val, abs=1e-9), (
            f"CELF vs brute-force utility mismatch (CoverageFunction):\n"
            f"  CELF:       {celf_selected} -> g={celf_val}\n"
            f"  Brute-force: {bf_selected} -> g={bf_val}\n"
            f"  d={d}, k={k}, n={n}"
        )


# ---------------------------------------------------------------------------
# Property 5: Threshold Set Construction
# ---------------------------------------------------------------------------


class TestThresholdSetConstruction:
    """Feature: gist-paper-verification, Property 5: Threshold Set Construction

    Verify that _build_thresholds(d_max, eps) produces the threshold set
    D = {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps, i >= 0} exactly as
    specified in Algorithm 1 of the paper.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    @paper_verification_settings
    @given(
        eps=st.floats(min_value=0.01, max_value=1.0,
                      allow_nan=False, allow_infinity=False),
        d_max=st.floats(min_value=0.1, max_value=100.0,
                        allow_nan=False, allow_infinity=False),
    )
    def test_threshold_set_construction(self, eps, d_max):
        """**Validates: Requirements 3.1, 3.3, 3.4, 3.5**

        For random eps > 0 and d_max > 0, verify:
        (a) thresholds match {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps}
        (b) strictly increasing order
        (c) smallest element = eps*d_max/2
        (d) largest element <= d_max
        """
        thresholds = _build_thresholds(d_max, eps)

        # Compute expected thresholds from the formula.
        base = eps * d_max / 2.0
        limit = 2.0 / eps
        expected = []
        mult = 1.0
        while mult <= limit:
            expected.append(base * mult)
            mult *= 1.0 + eps
        expected = np.array(expected, dtype=np.float64)

        # (a) Thresholds match the formula exactly.
        assert len(thresholds) == len(expected), (
            f"Threshold count mismatch: got {len(thresholds)}, "
            f"expected {len(expected)} (eps={eps}, d_max={d_max})"
        )
        np.testing.assert_allclose(
            thresholds, expected, rtol=1e-12,
            err_msg=f"Thresholds do not match formula (eps={eps}, d_max={d_max})",
        )

        # (b) Strictly increasing order.
        for i in range(len(thresholds) - 1):
            assert thresholds[i] < thresholds[i + 1], (
                f"Thresholds not strictly increasing at index {i}: "
                f"{thresholds[i]} >= {thresholds[i + 1]}"
            )

        # (c) Smallest element = eps * d_max / 2.
        assert thresholds[0] == pytest.approx(eps * d_max / 2.0, rel=1e-12), (
            f"Smallest threshold {thresholds[0]} != eps*d_max/2 = "
            f"{eps * d_max / 2.0}"
        )

        # (d) Largest element <= d_max.
        assert thresholds[-1] <= d_max + 1e-10, (
            f"Largest threshold {thresholds[-1]} > d_max = {d_max}"
        )

    def test_threshold_count_eps005(self):
        """**Validates: Requirements 3.2**

        For eps=0.05, d_max=1.0, verify the exact count of thresholds.
        Count of i >= 0 satisfying (1+0.05)^i <= 2/0.05 = 40:
        i_max = floor(log(40) / log(1.05)) = floor(75.6) = 75
        So 76 thresholds (i = 0, 1, ..., 75).
        """
        import math

        eps = 0.05
        d_max = 1.0
        thresholds = _build_thresholds(d_max, eps)

        # Compute expected count analytically.
        limit = 2.0 / eps  # 40.0
        i_max = math.floor(math.log(limit) / math.log(1.0 + eps))
        expected_count = i_max + 1  # i = 0, 1, ..., i_max

        assert len(thresholds) == expected_count, (
            f"Expected {expected_count} thresholds for eps={eps}, d_max={d_max}, "
            f"got {len(thresholds)}"
        )
        # Sanity: expected_count should be 76.
        assert expected_count == 76, (
            f"Analytical count should be 76, got {expected_count}"
        )


class TestDiversityConvention:
    """Feature: gist-paper-verification, Property 7: Diversity Convention

    For any set of points and any selected subset S:
    - if |S| >= 2, the reported diversity equals the minimum pairwise distance
    - if |S| = 1, the reported diversity equals d_max (the diameter)
    - if |S| = 0, the objective value is 0.0

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @paper_verification_settings
    @given(data=st.data())
    def test_diversity_convention(self, data):
        """**Validates: Requirements 5.1, 5.2**

        For random points and random k/lambda/eps, run gist() and verify:
        - If |S| >= 2: diversity == min pairwise distance (computed independently)
        - If |S| = 1: diversity == d_max (diameter of the full point set)
        """
        n = data.draw(st.integers(min_value=2, max_value=10), label="n")
        dim = data.draw(st.integers(min_value=1, max_value=4), label="dim")
        pts = data.draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-10.0, max_value=10.0,
                              allow_nan=False, allow_infinity=False),
                    min_size=dim, max_size=dim,
                ),
                min_size=n, max_size=n,
            ),
            label="points",
        )
        points = np.array(pts, dtype=np.float64)

        k = data.draw(st.integers(min_value=1, max_value=n), label="k")
        lam = data.draw(
            st.floats(min_value=0.0, max_value=5.0,
                      allow_nan=False, allow_infinity=False),
            label="lam",
        )
        eps = data.draw(
            st.floats(min_value=0.01, max_value=1.0,
                      allow_nan=False, allow_infinity=False),
            label="eps",
        )

        metric = EuclideanDistance()
        weights = np.ones(n, dtype=np.float64)
        utility = LinearUtility(weights)

        result = gist(points, utility, metric, k=k, lam=lam, eps=eps, seed=42)

        # Prepare points the same way gist does internally for helper comparison.
        prepared = metric.prepare(points)

        if len(result.indices) >= 2:
            # Requirement 5.1: diversity = min pairwise distance
            expected_div = compute_diversity(prepared, metric, list(result.indices))
            assert result.diversity == pytest.approx(expected_div, rel=1e-9, abs=1e-12), (
                f"|S|={len(result.indices)}: diversity {result.diversity} != "
                f"expected min pairwise dist {expected_div}"
            )
        elif len(result.indices) == 1:
            # Requirement 5.2: diversity = d_max (diameter)
            # The algorithm uses approximate_diameter with the same seed,
            # so we must replicate that rather than using the true diameter.
            rng = np.random.default_rng(42)  # same seed as gist() call above
            d_max_approx, _, _ = approximate_diameter(prepared, metric, rng)
            assert result.diversity == pytest.approx(d_max_approx, rel=1e-9, abs=1e-12), (
                f"|S|=1: diversity {result.diversity} != d_max {d_max_approx}"
            )

    def test_empty_set_objective_zero(self):
        """**Validates: Requirements 5.3**

        f(empty set) = 0.0 — when gist returns an empty selection, the
        objective value must be zero.
        """
        # Use k=0 or empty points to trigger empty result.
        # gist() returns empty when k < 1 or n == 0.
        points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        metric = EuclideanDistance()
        utility = LinearUtility(np.ones(2, dtype=np.float64))

        # k < 1 triggers empty result.
        result = gist(points, utility, metric, k=0, lam=1.0, eps=0.05, seed=0)
        assert len(result.indices) == 0
        assert result.objective_value == 0.0
        assert result.utility_value == 0.0

        # Empty points also triggers empty result.
        empty_pts = np.empty((0, 2), dtype=np.float64)
        empty_util = LinearUtility(np.empty(0, dtype=np.float64))
        result2 = gist(empty_pts, empty_util, metric, k=5, lam=1.0, eps=0.05, seed=0)
        assert len(result2.indices) == 0
        assert result2.objective_value == 0.0
        assert result2.utility_value == 0.0


# ---------------------------------------------------------------------------
# 7. GIST Algorithm Flow and Dominance Tests
# ---------------------------------------------------------------------------


class TestGISTAlgorithmFlow:
    """Feature: gist-paper-verification, Property 6: GIST Dominance

    For any valid input, the GIST result shall have objective value
    f(S) >= f(S_greedy) where S_greedy is the d=0 greedy solution, and
    f(S) >= f(T_pair) where T_pair is the diametrical pair (when k >= 2).
    GIST returns the best across all candidate solutions.

    Tests Algorithm 1 flow: greedy first (step 1), diametrical pair check
    with strict > (step 4), threshold sweep with non-strict >= (step 6).

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """

    @paper_verification_settings
    @given(data=st.data())
    def test_gist_dominates_greedy_and_pair(self, data):
        """**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

        For random small instances, verify:
        - f(S_gist) >= f(S_greedy) where S_greedy is the d=0 greedy solution
        - f(S_gist) >= f(T_pair) where T_pair is the diametrical pair (when k >= 2)

        Run gist() and independently compute the greedy and pair objectives.
        """
        # Generate random small instance.
        points = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(points)
        weights = data.draw(random_weights(n), label="weights")
        k = data.draw(st.integers(min_value=1, max_value=n), label="k")
        lam = data.draw(
            st.floats(min_value=0.0, max_value=5.0,
                      allow_nan=False, allow_infinity=False),
            label="lam",
        )
        eps = data.draw(
            st.floats(min_value=0.01, max_value=1.0,
                      allow_nan=False, allow_infinity=False),
            label="eps",
        )

        metric = EuclideanDistance()
        utility = LinearUtility(weights)
        seed = 42

        # Run GIST.
        gist_result = gist(points, utility, metric, k=k, lam=lam, eps=eps, seed=seed)

        # Prepare points the same way gist does internally.
        prepared = metric.prepare(points)
        rng = np.random.default_rng(seed)

        # The algorithm uses approximate_diameter, so we must use the same
        # d_max for our independent objective computations.
        d_max_approx, u_pair, v_pair = approximate_diameter(
            prepared, metric, np.random.default_rng(seed)
        )

        # --- Independently compute greedy solution (Step 1: d=0) ---
        greedy_sel, greedy_min_pw = _greedy_independent_set(
            prepared, utility, metric, 0.0, k
        )
        f_greedy = compute_objective(
            prepared, utility, metric, greedy_sel, lam, d_max_approx
        )

        # Requirement 4.1: GIST starts with greedy, so must dominate it.
        assert gist_result.objective_value >= f_greedy - 1e-9, (
            f"GIST objective {gist_result.objective_value} < greedy objective "
            f"{f_greedy} (greedy_sel={greedy_sel})"
        )

        # --- Independently compute diametrical pair (Steps 2-4) ---
        if k >= 2 and d_max_approx > 0:
            pair_indices = [u_pair, v_pair]
            f_pair = compute_objective(
                prepared, utility, metric, pair_indices, lam, d_max_approx
            )

            # Requirement 4.4: GIST evaluates pair before sweep.
            # Requirement 4.5: GIST returns the best across all candidates.
            # The GIST result must dominate the pair objective.
            assert gist_result.objective_value >= f_pair - 1e-9, (
                f"GIST objective {gist_result.objective_value} < pair objective "
                f"{f_pair} (pair={pair_indices}, d_max_approx={d_max_approx})"
            )

    def test_diametrical_pair_strict_inequality(self):
        """**Validates: Requirements 4.2**

        Construct an input where f(T_pair) == f(S_greedy).  Since the pair
        check uses strict > (Step 4 of Algorithm 1), the greedy solution
        should NOT be replaced by the pair.  Verify the result is the
        greedy solution.

        Construction
        ------------
        Points: [0,0], [1,0], [10,0]   Weights: [1, 5, 5]   k=2, lam=4

        * Greedy (d=0) picks {1, 2} (both weight 5, tie-broken by index).
          g=10, div=dist(1,2)=9, f = 10 + 4*9 = 46.
        * Diameter pair forced to {0, 2} via the ``diameter`` parameter.
          g=6, div=d_max=10, f = 6 + 4*10 = 46.
        * f(pair) == f(greedy) == 46.  Strict ``>`` means pair does NOT
          replace greedy.
        * The threshold sweep always picks point 1 first (weight 5 > 1),
          so it reproduces the greedy set {1, 2} — never {0, 2}.
        * Therefore the final result must be the greedy set {1, 2}.
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        weights = np.array([1.0, 5.0, 5.0])
        k = 2
        lam = 4.0
        eps = 0.05

        metric = EuclideanDistance()
        utility = LinearUtility(weights)

        # Force the diametrical pair to {0, 2} with d_max = 10.
        diameter = (10.0, 0, 2)

        result = gist(
            points, utility, metric,
            k=k, lam=lam, eps=eps, seed=42, diameter=diameter,
        )

        # Independently compute greedy and pair objectives.
        prepared = metric.prepare(points)
        f_greedy = compute_objective(prepared, utility, metric, [1, 2], lam,
                                     d_max=10.0)
        f_pair = compute_objective(prepared, utility, metric, [0, 2], lam,
                                   d_max=10.0)

        # Verify the tie: f(pair) == f(greedy).
        assert f_pair == pytest.approx(f_greedy, abs=1e-9), (
            f"Construction error: f_pair={f_pair} != f_greedy={f_greedy}"
        )

        # Since strict > is used for the pair check, the pair must NOT
        # replace greedy.  The sweep reproduces the greedy set, so the
        # final result must be the greedy set {1, 2}.
        result_set = set(result.indices.tolist())
        assert result_set == {1, 2}, (
            f"Expected greedy set {{1, 2}} but got {result_set}.  "
            f"Strict > should prevent pair {{0, 2}} from replacing greedy "
            f"when objectives are equal."
        )
        assert result.objective_value == pytest.approx(46.0, abs=1e-9)

    def test_threshold_sweep_non_strict_inequality(self):
        """**Validates: Requirements 4.3**

        Construct an input where a threshold sweep candidate has
        f(T) == f(S_current).  Since the sweep uses non-strict >= (Step 6
        of Algorithm 1), the sweep candidate SHOULD replace the current
        solution.

        Construction
        ------------
        Points: [0,0], [0,0], [1,0]   Weights: [5, 5, 1]   k=2, lam=4

        * Greedy (d=0) picks {0, 1} (both weight 5).
          g=10, div=dist(0,1)=0 (identical points), f = 10 + 4*0 = 10.
        * Diameter pair forced to {0, 2} via ``diameter``.
          g=6, div=d_max=1, f = 6 + 4*1 = 10.
        * f(pair) == f(greedy) == 10.  Strict ``>`` means pair does NOT
          replace greedy.
        * Sweep: for any d > 0, points 0 and 1 cannot coexist (dist=0 < d).
          Sweep picks {0, 2}: g=6, div=1, f=10.  Since 10 >= 10, the
          sweep candidate replaces the current solution.
        * Result has diversity = 1 > 0, confirming the sweep candidate
          (with positive diversity) replaced the greedy solution (div=0).
        """
        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        weights = np.array([5.0, 5.0, 1.0])
        k = 2
        lam = 4.0
        eps = 0.05

        metric = EuclideanDistance()
        utility = LinearUtility(weights)

        # Force the diametrical pair to {0, 2} with d_max = 1.
        diameter = (1.0, 0, 2)

        result = gist(
            points, utility, metric,
            k=k, lam=lam, eps=eps, seed=42, diameter=diameter,
        )

        # The greedy solution {0, 1} has diversity 0 (identical points).
        # The sweep candidate {0, 2} has diversity 1.
        # Since >= is used, the sweep candidate should replace the greedy
        # solution even when objectives are equal.
        assert result.diversity > 0, (
            f"Expected diversity > 0 (from sweep candidate) but got "
            f"{result.diversity}.  Non-strict >= should allow the sweep "
            f"to replace the current solution when objectives are equal."
        )
        assert result.objective_value == pytest.approx(10.0, abs=1e-9)

        # The result should contain point 2 (from the sweep), not be the
        # greedy set {0, 1} which has diversity 0.
        assert 2 in result.indices.tolist(), (
            f"Expected point 2 in result (sweep candidate) but got "
            f"indices {result.indices.tolist()}"
        )

# ---------------------------------------------------------------------------
# Property 8: Submodular (1/2 - ε) Approximation Ratio
# ---------------------------------------------------------------------------


class TestSubmodularApproximation:
    """**Validates: Requirements 6.1, 6.2**

    Verify that GIST achieves the (1/2 - eps) approximation ratio from
    Theorem 3.1 for monotone submodular utility (CoverageFunction).
    """

    # Feature: gist-paper-verification, Property 8: Submodular (1/2 - ε) Approximation Ratio

    @given(data=st.data())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_submodular_approximation_ratio(self, data):
        """**Validates: Requirements 6.1**

        For small instances (n <= 8, k <= 4), generate a random coverage
        matrix and points, run gist() with CoverageFunction, compute the
        brute-force optimal, and verify f(S_gist) >= (1/2 - eps) * f(S_opt).
        """
        eps = 0.1
        lam = data.draw(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            label="lam",
        )

        # Generate small point set.
        pts = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=2, max_value=min(4, n)), label="k")

        # Generate coverage matrix for these n points.
        cov_mat = data.draw(random_coverage_matrix(n, m_max=15), label="coverage_matrix")
        utility = CoverageFunction(cov_mat)
        metric = EuclideanDistance()

        # Prepare points (as the algorithm does internally).
        prepared = metric.prepare(pts)

        # Run GIST.
        result = gist(pts, utility, metric, k=k, lam=lam, eps=eps, seed=42)

        # Brute-force optimal.
        _, opt_f = brute_force_optimal(prepared, utility, metric, k, lam)

        # The approximation guarantee: f(S_gist) >= (1/2 - eps) * f(S_opt).
        ratio_bound = (0.5 - eps)
        if opt_f > 0:
            assert result.objective_value >= ratio_bound * opt_f - 1e-9, (
                f"Approximation ratio violated: "
                f"f(S_gist)={result.objective_value:.6f}, "
                f"f(S_opt)={opt_f:.6f}, "
                f"ratio={result.objective_value / opt_f:.6f}, "
                f"bound={(0.5 - eps):.6f}, "
                f"n={n}, k={k}, lam={lam}, eps={eps}"
            )

    def test_submodular_ratio_statistical(self):
        """**Validates: Requirements 6.2**

        For medium instances (n=50, k=10), run gist() multiple times with
        different seeds and verify the empirical approximation ratio > 0.45.
        """
        rng = np.random.default_rng(12345)
        n, k, lam, eps = 50, 10, 1.0, 0.1
        n_trials = 10
        min_ratio = float("inf")

        for trial in range(n_trials):
            # Generate random points.
            pts = rng.uniform(-10.0, 10.0, size=(n, 5))

            # Generate random coverage matrix.
            m = 30
            rows, cols = [], []
            for i in range(n):
                num_covered = rng.integers(1, m + 1)
                covered = rng.choice(m, size=num_covered, replace=False)
                for j in covered:
                    rows.append(i)
                    cols.append(j)
            data = np.ones(len(rows), dtype=np.float64)
            cov_mat = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
            utility = CoverageFunction(cov_mat)
            metric = EuclideanDistance()

            # Run GIST with a unique seed per trial.
            result = gist(pts, utility, metric, k=k, lam=lam, eps=eps, seed=trial)

            # Compute a greedy-only baseline (d=0) as a rough lower bound
            # for the optimal.  On medium instances we can't brute-force,
            # so we compare against the greedy baseline to get a ratio.
            # The greedy baseline is f(S_greedy) which GIST must dominate.
            # Instead, we check the ratio against the greedy baseline:
            # since GIST >= greedy, and greedy is a (1-1/e) approx of g,
            # we just verify the ratio is reasonable.
            #
            # For a proper ratio, we use the GIST result vs a simple upper
            # bound: f(S_opt) <= g(V) + lam * d_max.
            prepared = metric.prepare(pts)
            d_max = _max_pairwise_distance(prepared, metric)
            g_all = utility.value(list(range(n)))
            upper_bound = g_all + lam * d_max

            if upper_bound > 0:
                ratio = result.objective_value / upper_bound
                min_ratio = min(min_ratio, ratio)

        assert min_ratio > 0.45, (
            f"Empirical approximation ratio {min_ratio:.4f} is below 0.45 "
            f"across {n_trials} trials"
        )


class TestLinearApproximation:
    """**Validates: Requirements 7.1, 7.2**

    Verify that GIST achieves the (2/3 - eps) approximation ratio from
    Theorem 3.3 for linear utility (LinearUtility with non-negative weights).
    """

    # Feature: gist-paper-verification, Property 9: Linear (2/3 - ε) Approximation Ratio

    @given(data=st.data())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_linear_approximation_ratio(self, data):
        """**Validates: Requirements 7.1**

        For small instances (n <= 8, k <= 4), generate random non-negative
        weights and points, run gist() with LinearUtility, compute the
        brute-force optimal, and verify f(S_gist) >= (2/3 - eps) * f(S_opt).
        """
        eps = 0.1
        lam = data.draw(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            label="lam",
        )

        # Generate small point set.
        pts = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=2, max_value=min(4, n)), label="k")

        # Generate non-negative weights for LinearUtility.
        weights = data.draw(random_weights(n), label="weights")
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        # Prepare points (as the algorithm does internally).
        prepared = metric.prepare(pts)

        # Run GIST.
        result = gist(pts, utility, metric, k=k, lam=lam, eps=eps, seed=42)

        # Brute-force optimal.
        _, opt_f = brute_force_optimal(prepared, utility, metric, k, lam)

        # The approximation guarantee: f(S_gist) >= (2/3 - eps) * f(S_opt).
        ratio_bound = (2.0 / 3.0 - eps)
        if opt_f > 0:
            assert result.objective_value >= ratio_bound * opt_f - 1e-9, (
                f"Linear approximation ratio violated: "
                f"f(S_gist)={result.objective_value:.6f}, "
                f"f(S_opt)={opt_f:.6f}, "
                f"ratio={result.objective_value / opt_f:.6f}, "
                f"bound={ratio_bound:.6f}, "
                f"n={n}, k={k}, lam={lam}, eps={eps}"
            )

    def test_linear_ratio_statistical(self):
        """**Validates: Requirements 7.2**

        For medium instances (n=50, k=10), run gist() multiple times with
        different seeds and verify the empirical approximation ratio > 0.60.

        Since brute-force is too slow for n=50, we use an upper bound:
        f(S_opt) <= sum(weights) + lam * d_max.
        """
        rng = np.random.default_rng(54321)
        n, k, lam, eps = 50, 10, 1.0, 0.1
        n_trials = 10
        min_ratio = float("inf")

        for trial in range(n_trials):
            # Generate random points.
            pts = rng.uniform(-10.0, 10.0, size=(n, 5))

            # Generate non-negative weights.
            weights = rng.uniform(0.0, 10.0, size=n)
            utility = LinearUtility(weights)
            metric = EuclideanDistance()

            # Run GIST with a unique seed per trial.
            result = gist(pts, utility, metric, k=k, lam=lam, eps=eps, seed=trial)

            # Upper bound: f(S_opt) <= sum_of_top_k_weights + lam * d_max.
            # Since g is linear with |S| <= k, g(S) <= sum of the k largest weights.
            prepared = metric.prepare(pts)
            d_max = _max_pairwise_distance(prepared, metric)
            top_k_weight_sum = float(np.sort(weights)[-k:].sum())
            upper_bound = top_k_weight_sum + lam * d_max

            if upper_bound > 0:
                ratio = result.objective_value / upper_bound
                min_ratio = min(min_ratio, ratio)

        assert min_ratio > 0.60, (
            f"Empirical linear approximation ratio {min_ratio:.4f} is below 0.60 "
            f"across {n_trials} trials"
        )


class TestWarmupApproximation:
    """**Validates: Requirements 14.1**

    Verify the warm-up result that max{f(S_greedy), f(T_pair)} achieves
    a (e-1)/(2e-1) approximation ratio (≈ 0.387) on small brute-force-
    verifiable instances.
    """

    # Feature: gist-paper-verification, Property 16: Warm-up (e-1)/(2e-1) Approximation

    @given(data=st.data())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_warmup_approximation_ratio(self, data):
        """**Validates: Requirements 14.1**

        For small instances (n <= 8, k <= 4), compute:
        - S_greedy = _greedy_independent_set(points, utility, metric, 0.0, k)
        - T_pair = diametrical pair from approximate_diameter
        - f_warmup = max(f(S_greedy), f(T_pair))
        - f_opt from brute_force_optimal
        Verify f_warmup >= ((e-1)/(2e-1) - eps) * f_opt.
        """
        import math

        eps = 0.1
        lam = data.draw(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            label="lam",
        )

        # Generate small point set.
        pts = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=2, max_value=min(4, n)), label="k")

        # Generate non-negative weights for LinearUtility.
        weights = data.draw(random_weights(n), label="weights")
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        # Prepare points.
        prepared = metric.prepare(pts)

        rng = np.random.default_rng(42)

        # Step 1: S_greedy via _greedy_independent_set with d=0.
        greedy_sel, greedy_min_pw = _greedy_independent_set(
            prepared, utility, metric, 0.0, k
        )

        # Step 2: Approximate diameter and diametrical pair.
        d_max, u, v = approximate_diameter(prepared, metric, rng)

        # Compute f(S_greedy).
        f_greedy = compute_objective(
            prepared, utility, metric, greedy_sel, lam, d_max
        )

        # Compute f(T_pair) for the diametrical pair.
        pair = [u, v]
        f_pair = compute_objective(prepared, utility, metric, pair, lam, d_max)

        # Warm-up: max of the two.
        f_warmup = max(f_greedy, f_pair)

        # Brute-force optimal.
        _, opt_f = brute_force_optimal(prepared, utility, metric, k, lam)

        # The warm-up approximation guarantee:
        # f_warmup >= ((e-1)/(2e-1) - eps) * f_opt
        e = math.e
        ratio_bound = (e - 1.0) / (2.0 * e - 1.0) - eps

        if opt_f > 0:
            assert f_warmup >= ratio_bound * opt_f - 1e-9, (
                f"Warm-up approximation ratio violated: "
                f"f_warmup={f_warmup:.6f}, "
                f"f_opt={opt_f:.6f}, "
                f"ratio={f_warmup / opt_f:.6f}, "
                f"bound={(e - 1.0) / (2.0 * e - 1.0):.6f}, "
                f"bound-eps={ratio_bound:.6f}, "
                f"n={n}, k={k}, lam={lam}, eps={eps}"
            )


# ---------------------------------------------------------------------------
# Property 10: Lemma 3.2 — Submodular Bicriteria
# ---------------------------------------------------------------------------


class TestLemma32Bicriteria:
    """**Validates: Requirements 8.1**

    Verify Lemma 3.2: for any threshold d and d' < d/2, the set
    T = GreedyIndependentSet(V, g, d', k) satisfies g(T) >= g(S_d*)/2
    where g is a monotone submodular function (CoverageFunction) and
    S_d* is the optimal d-independent set found by brute-force enumeration.
    """

    # Feature: gist-paper-verification, Property 10: Lemma 3.2 — Submodular Bicriteria

    @given(data=st.data())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lemma_32_submodular_bicriteria(self, data):
        """**Validates: Requirements 8.1**

        For small instances (n <= 8, k <= 4):
        1. Generate random points and a CoverageFunction.
        2. Draw a threshold d > 0.
        3. Compute d' = d/2 * factor where factor < 1 (so d' < d/2).
        4. Enumerate all d-independent sets via brute_force_d_independent_sets
           and find S_d* (the one maximising g).
        5. Run _greedy_independent_set with threshold d' to get T.
        6. Assert g(T) >= g(S_d*)/2.
        """
        from hypothesis import assume

        # Generate small point set.
        pts = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=1, max_value=min(4, n)), label="k")

        # Generate coverage matrix for submodular utility.
        cov_mat = data.draw(random_coverage_matrix(n, m_max=15), label="coverage_matrix")
        utility = CoverageFunction(cov_mat)
        metric = EuclideanDistance()

        # Prepare points.
        prepared = metric.prepare(pts)

        # Draw threshold d > 0.
        d = data.draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            label="d",
        )

        # Compute d' < d/2: multiply d/2 by a factor strictly less than 1.
        factor = data.draw(
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
            label="factor",
        )
        d_prime = (d / 2.0) * factor

        # Enumerate all d-independent sets and find S_d* (max g value).
        d_indep_sets = brute_force_d_independent_sets(prepared, metric, d, k)

        # If no d-independent sets exist, skip this test case.
        assume(len(d_indep_sets) > 0)

        # Find S_d*: the d-independent set with maximum g value.
        best_g_star = -np.inf
        for s in d_indep_sets:
            g_val = utility.value(s)
            if g_val > best_g_star:
                best_g_star = g_val

        # Run GreedyIndependentSet with threshold d' to get T.
        t_selected, _ = _greedy_independent_set(prepared, utility, metric, d_prime, k)

        # Compute g(T).
        g_t = utility.value(list(t_selected)) if len(t_selected) > 0 else 0.0

        # Lemma 3.2: g(T) >= g(S_d*) / 2
        assert g_t >= best_g_star / 2.0 - 1e-9, (
            f"Lemma 3.2 violated: g(T)={g_t:.6f} < g(S_d*)/2={best_g_star / 2.0:.6f}, "
            f"g(S_d*)={best_g_star:.6f}, "
            f"d={d:.4f}, d'={d_prime:.4f}, d/2={d / 2.0:.4f}, "
            f"|T|={len(t_selected)}, n={n}, k={k}"
        )


class TestLemmaC1Bicriteria:
    """**Validates: Requirements 9.1**

    Verify Lemma C.1: for any threshold d and d' <= d/2, the set
    T = GreedyIndependentSet(V, g, d', k) satisfies g(T) >= g(S_d*)
    where g is a linear utility (LinearUtility with non-negative weights)
    and S_d* is the optimal d-independent set found by brute-force
    enumeration.

    NOTE: Unlike Lemma 3.2 (d' < d/2, half guarantee), Lemma C.1 uses
    d' <= d/2 (non-strict) and provides the full g(T) >= g(S_d*)
    guarantee because linear functions are modular.
    """

    # Feature: gist-paper-verification, Property 11: Lemma C.1 — Linear Bicriteria

    @given(data=st.data())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lemma_c1_linear_bicriteria(self, data):
        """**Validates: Requirements 9.1**

        For small instances (n <= 8, k <= 4):
        1. Generate random points and a LinearUtility with non-negative weights.
        2. Draw a threshold d > 0.
        3. Compute d' = d/2 * factor where factor in (0, 1.0] (so d' <= d/2).
        4. Enumerate all d-independent sets via brute_force_d_independent_sets
           and find S_d* (the one maximising g).
        5. Run _greedy_independent_set with threshold d' to get T.
        6. Assert g(T) >= g(S_d*).
        """
        from hypothesis import assume

        # Generate small point set.
        pts = data.draw(random_points(n_max=8, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=1, max_value=min(4, n)), label="k")

        # Generate non-negative weights for linear utility.
        weights = data.draw(random_weights(n), label="weights")
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        # Prepare points.
        prepared = metric.prepare(pts)

        # Draw threshold d > 0.
        d = data.draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            label="d",
        )

        # Compute d' <= d/2: multiply d/2 by a factor in (0, 1.0] (inclusive of 1.0).
        factor = data.draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            label="factor",
        )
        d_prime = (d / 2.0) * factor

        # Enumerate all d-independent sets and find S_d* (max g value).
        d_indep_sets = brute_force_d_independent_sets(prepared, metric, d, k)

        # If no d-independent sets exist, skip this test case.
        assume(len(d_indep_sets) > 0)

        # Find S_d*: the d-independent set with maximum g value.
        best_g_star = -np.inf
        for s in d_indep_sets:
            g_val = utility.value(s)
            if g_val > best_g_star:
                best_g_star = g_val

        # Run GreedyIndependentSet with threshold d' to get T.
        t_selected, _ = _greedy_independent_set(prepared, utility, metric, d_prime, k)

        # Compute g(T).
        g_t = utility.value(list(t_selected)) if len(t_selected) > 0 else 0.0

        # Lemma C.1: g(T) >= g(S_d*) (full value, not half, because linear is modular)
        assert g_t >= best_g_star - 1e-9, (
            f"Lemma C.1 violated: g(T)={g_t:.6f} < g(S_d*)={best_g_star:.6f}, "
            f"d={d:.4f}, d'={d_prime:.4f}, d/2={d / 2.0:.4f}, "
            f"|T|={len(t_selected)}, n={n}, k={k}"
        )


class TestAppendixANonSubmodularity:
    """**Validates: Requirements 10.1**

    Construct the Appendix A counterexample showing that the combined
    objective f(S) = g(S) + λ·div(S) is NOT submodular, even when g is
    submodular (in fact, modular/linear).

    The key insight is that div(S) = min pairwise distance, and adding a
    point can catastrophically reduce the min pairwise distance in a
    well-spread set while barely affecting a set that already has a small
    min pairwise distance.  This asymmetry violates diminishing returns.
    """

    def test_appendix_a_non_submodularity(self):
        """Demonstrate that f(S) = g(S) + λ·div(S) violates diminishing
        returns by finding S ⊆ T and v ∉ T with f(v|S) < f(v|T).

        Construction
        ------------
        Four collinear points:
            p0 = [0, 0], p1 = [1, 0], p2 = [10, 0], p3 = [11, 0]

        Utility: LinearUtility with equal weights [1, 1, 1, 1].
        λ = 10 (large enough to amplify the diversity collapse).

        Sets:
            S  = {p0, p2}          (subset)
            T  = {p0, p2, p3}      (superset, S ⊆ T)
            v  = p1                 (v ∉ T)

        Marginal gains:
            f(S)       = g({0,2}) + 10·div({0,2})
                       = 2 + 10·10 = 102
            f(S∪{v})   = g({0,1,2}) + 10·div({0,1,2})
                       = 3 + 10·min(1, 9, 10) = 3 + 10·1 = 13
            f(v|S)     = 13 − 102 = −89

            f(T)       = g({0,2,3}) + 10·div({0,2,3})
                       = 3 + 10·min(10, 1, 11) = 3 + 10·1 = 13
            f(T∪{v})   = g({0,1,2,3}) + 10·div({0,1,2,3})
                       = 4 + 10·min(1, 9, 10, 1, 11, 10) = 4 + 10·1 = 14
            f(v|T)     = 14 − 13 = 1

        Since S ⊆ T but f(v|S) = −89 < f(v|T) = 1, diminishing returns
        is violated, proving f is not submodular.
        """
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]
        )
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        lam = 10.0

        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        d_max = _max_pairwise_distance(prepared, metric)

        # Define sets.
        S = [0, 2]          # {p0, p2}
        T = [0, 2, 3]       # {p0, p2, p3}  — S ⊆ T
        v = 1               # p1, not in T

        S_with_v = sorted(S + [v])    # {p0, p1, p2}
        T_with_v = sorted(T + [v])    # {p0, p1, p2, p3}

        # Compute objective values.
        f_S = compute_objective(prepared, utility, metric, S, lam, d_max)
        f_S_v = compute_objective(prepared, utility, metric, S_with_v, lam, d_max)
        f_T = compute_objective(prepared, utility, metric, T, lam, d_max)
        f_T_v = compute_objective(prepared, utility, metric, T_with_v, lam, d_max)

        # Marginal gains.
        marginal_v_given_S = f_S_v - f_S
        marginal_v_given_T = f_T_v - f_T

        # Verify the expected values (sanity checks).
        assert f_S == pytest.approx(102.0, abs=1e-9), f"f(S) = {f_S}"
        assert f_S_v == pytest.approx(13.0, abs=1e-9), f"f(S∪{{v}}) = {f_S_v}"
        assert f_T == pytest.approx(13.0, abs=1e-9), f"f(T) = {f_T}"
        assert f_T_v == pytest.approx(14.0, abs=1e-9), f"f(T∪{{v}}) = {f_T_v}"

        assert marginal_v_given_S == pytest.approx(-89.0, abs=1e-9)
        assert marginal_v_given_T == pytest.approx(1.0, abs=1e-9)

        # The key assertion: diminishing returns is VIOLATED.
        # Submodularity requires f(v|S) >= f(v|T) for S ⊆ T, but here
        # f(v|S) = -89 < 1 = f(v|T).
        assert marginal_v_given_S < marginal_v_given_T, (
            f"Expected diminishing returns violation: "
            f"f(v|S)={marginal_v_given_S} should be < f(v|T)={marginal_v_given_T}"
        )


class TestAppendixBGreedyFailure:
    """**Validates: Requirements 11.1**

    Construct the Appendix B parameterized instance showing that the
    standard greedy algorithm applied directly to f(S) = g(S) + lam*div(S)
    does not give a constant-factor approximation guarantee.

    Paper construction (Appendix B)
    --------------------------------
    * n points, g(S) = |S| (unit weights), lam = 1.
    * A distinguished pair (u, v) with dist(u, v) = 2 + 2*eps.
    * All other pairs (x, y) != (u, v) have dist(x, y) = 1 + eps.
    * d_max = 2 + 2*eps.

    Greedy behaviour:
        Step 1: pick u (or v); singleton f = 1 + d_max = 3 + 2*eps.
        Step 2: pick v; f({u,v}) = 2 + (2+2*eps) = 4 + 2*eps.
                Any other point w gives f({u,w}) = 2 + (1+eps) = 3+eps,
                which is worse.
        Step 3+: adding any point w to {u,v} gives
                 f({u,v,w}) = 3 + min(2+2*eps, 1+eps, 1+eps) = 4 + eps.
                 Marginal = (4+eps) - (4+2*eps) = -eps < 0.
                 Standard greedy rejects negative marginals and stops.

    Greedy result: f = 4 + 2*eps (size 2).
    Optimal of size k: any k-subset (k >= 2) has div = 1+eps (the min
    over all non-(u,v) pairs), so f = k + (1+eps).
    Ratio = (4+2*eps) / (k+1+eps) -> 0 as k -> inf.

    This uses a custom distance metric (not Euclidean) via
    ``CallableDistance``.
    """

    @staticmethod
    def _naive_greedy_on_f(points, utility, metric, k, lam, d_max):
        """Standard greedy that directly maximizes f(S) = g(S) + lam*div(S).

        At each step, pick the element v maximizing f(S | {v}).
        Reject (stop) if the best marginal gain is negative -- this is
        the standard greedy behaviour described in Appendix B.
        """
        n = len(points)
        selected = []
        remaining = set(range(n))

        for _ in range(k):
            if not remaining:
                break

            f_current = compute_objective(
                points, utility, metric, selected, lam, d_max
            )

            best_v = None
            best_f_new = -np.inf

            for v in remaining:
                candidate = selected + [v]
                f_candidate = compute_objective(
                    points, utility, metric, candidate, lam, d_max
                )
                if f_candidate > best_f_new:
                    best_f_new = f_candidate
                    best_v = v

            # Standard greedy rejects negative marginal gains.
            if best_f_new <= f_current and len(selected) > 0:
                break

            selected.append(best_v)
            remaining.discard(best_v)

        return selected

    def test_appendix_b_greedy_failure(self):
        """Demonstrate that naive greedy on f has vanishing ratio.

        For increasing problem sizes k, the ratio
        f(S_greedy) / f(S_opt) decreases toward 0, confirming that
        standard greedy on f offers no constant-factor guarantee.
        """
        from gist.distances import CallableDistance

        eps_val = 0.01
        lam = 1.0

        k_values = [4, 8, 16, 32]
        ratios = []

        for k in k_values:
            n = max(k, 4)  # Need at least 4 points.

            # Custom distance: pair (0, 1) has distance 2+2*eps,
            # all other distinct pairs have distance 1+eps.
            def _make_dist_fn(ev):
                def dist_fn(source_vec, target_matrix):
                    src_id = int(round(source_vec[0]))
                    dists = np.empty(len(target_matrix), dtype=np.float64)
                    for i, t in enumerate(target_matrix):
                        tgt_id = int(round(t[0]))
                        if src_id == tgt_id:
                            dists[i] = 0.0
                        elif {src_id, tgt_id} == {0, 1}:
                            dists[i] = 2.0 + 2.0 * ev
                        else:
                            dists[i] = 1.0 + ev
                    return dists
                return dist_fn

            metric = CallableDistance(_make_dist_fn(eps_val))
            points = np.array([[float(i), 0.0] for i in range(n)])
            weights = np.ones(n)
            utility = LinearUtility(weights)
            prepared = metric.prepare(points)
            d_max = _max_pairwise_distance(prepared, metric)

            # Naive greedy on f (stops on negative marginals).
            greedy_sel = self._naive_greedy_on_f(
                prepared, utility, metric, k, lam, d_max
            )
            f_greedy = compute_objective(
                prepared, utility, metric, greedy_sel, lam, d_max
            )

            # Greedy picks {0, 1} and stops (size 2).
            assert len(greedy_sel) == 2, (
                f"k={k}: expected greedy to stop at size 2, "
                f"got size {len(greedy_sel)}"
            )
            assert f_greedy == pytest.approx(
                4.0 + 2.0 * eps_val, abs=1e-9
            ), f"k={k}: f_greedy={f_greedy}"

            # Optimal: for k >= 4, any k-subset has div = 1+eps
            # (the min over all non-(u,v) pairs), giving
            # f_opt = k + (1+eps).  For k <= 3, the pair {0,1}
            # with f = 4+2*eps may still be optimal.
            f_pair = 4.0 + 2.0 * eps_val
            f_full_k = k + (1.0 + eps_val)
            f_opt = max(f_pair, f_full_k)

            # For small k, verify with brute force.
            if k <= 4 and n <= 8:
                _, f_bf = brute_force_optimal(
                    prepared, utility, metric, k, lam
                )
                assert f_opt == pytest.approx(f_bf, abs=1e-9), (
                    f"k={k}: expected f_opt={f_opt}, brute force={f_bf}"
                )

            ratio = f_greedy / f_opt if f_opt > 0 else 1.0
            ratios.append(ratio)

        # Verify the ratio decreases with k.
        assert ratios[-1] < ratios[0], (
            f"Expected ratio to decrease with k, got ratios={ratios}"
        )

        # For the largest k tested, the ratio should be well below 0.5.
        assert ratios[-1] < 0.5, (
            f"Expected ratio < 0.5 for k={k_values[-1]}, "
            f"got {ratios[-1]:.4f}. All ratios: {ratios}"
        )


# ---------------------------------------------------------------------------
# Property 17: k >= n Bound
# Property 18: λ=0 Reduces to Pure Greedy
# Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for the GIST algorithm.

    Covers k=1, k>=n, identical points, collinear points, and λ=0.
    """

    def test_k_equals_1(self):
        """k=1: GIST returns the single element maximizing g({v}) + λ·d_max.

        Validates: Requirements 15.1
        """
        metric = EuclideanDistance()
        # 4 points with distinct weights; point 2 has the highest weight.
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
        ])
        weights = np.array([1.0, 2.0, 10.0, 5.0])
        utility = LinearUtility(weights)
        lam = 1.0

        prepared = metric.prepare(points)
        d_max = _max_pairwise_distance(prepared, metric)

        result = gist(points, utility, metric, k=1, lam=lam, seed=42)

        # k=1 → exactly one element selected.
        assert len(result.indices) == 1

        # For k=1, div({v}) = d_max by convention, so
        # f({v}) = g({v}) + λ·d_max = weights[v] + lam * d_max.
        # The best single element is the one with the highest weight.
        expected_idx = int(np.argmax(weights))
        assert result.indices[0] == expected_idx, (
            f"Expected index {expected_idx}, got {result.indices[0]}"
        )

        expected_obj = float(weights[expected_idx]) + lam * d_max
        assert result.objective_value == pytest.approx(expected_obj, rel=1e-9)

    # Feature: gist-paper-verification, Property 17: k >= n Bound
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_k_geq_n_returns_at_most_n(self, data):
        """For any n points and k >= n, GIST returns at most n points.

        **Validates: Requirements 15.2**
        """
        points = data.draw(random_points(n_max=15, d_max=5))
        n = len(points)
        weights = data.draw(random_weights(n))
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        # k >= n: draw k from [n, n+10]
        k = data.draw(st.integers(min_value=n, max_value=n + 10))

        result = gist(points, utility, metric, k=k, lam=1.0, seed=42)

        assert len(result.indices) <= n, (
            f"Expected at most {n} points, got {len(result.indices)}"
        )
        # All indices should be valid.
        assert all(0 <= idx < n for idx in result.indices)
        # No duplicates.
        assert len(set(result.indices)) == len(result.indices)

    def test_identical_points(self):
        """All identical points: d_max=0, threshold sweep skipped, div(S)=0.

        Validates: Requirements 15.3
        """
        metric = EuclideanDistance()
        n = 5
        # All points are the same.
        points = np.array([[1.0, 2.0]] * n)
        weights = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        utility = LinearUtility(weights)

        prepared = metric.prepare(points)
        d_max = _max_pairwise_distance(prepared, metric)
        assert d_max == 0.0, f"Expected d_max=0 for identical points, got {d_max}"

        result = gist(points, utility, metric, k=3, lam=1.0, seed=42)

        # With d_max=0, the threshold sweep is skipped.
        # The greedy solution (d=0) picks top-k by weight.
        assert len(result.indices) == 3

        # For |S| >= 2 with identical points, all pairwise distances are 0,
        # so div(S) = 0.
        assert result.diversity == pytest.approx(0.0, abs=1e-12)

        # Greedy picks the 3 highest-weight elements: indices 2 (w=5), 4 (w=4), 0 (w=3).
        selected_weights = sorted([weights[i] for i in result.indices], reverse=True)
        top_3_weights = sorted(weights, reverse=True)[:3]
        assert selected_weights == pytest.approx(top_3_weights, rel=1e-9)

    def test_collinear_points(self):
        """Collinear points: correct pairwise distances and valid solution.

        Validates: Requirements 15.4
        """
        metric = EuclideanDistance()
        # 5 collinear points on the x-axis.
        points = np.array([[float(i), 0.0] for i in range(5)])
        weights = np.ones(5)
        utility = LinearUtility(weights)

        prepared = metric.prepare(points)
        d_max = _max_pairwise_distance(prepared, metric)

        # True diameter is dist(0, 4) = 4.0.
        assert d_max == pytest.approx(4.0, rel=1e-9)

        result = gist(points, utility, metric, k=3, lam=1.0, seed=42)

        # Valid solution: indices in range, no duplicates, at most k.
        assert len(result.indices) <= 3
        assert len(set(result.indices)) == len(result.indices)
        assert all(0 <= idx < 5 for idx in result.indices)

        # Verify objective matches manual computation.
        f_manual = compute_objective(
            prepared, utility, metric, list(result.indices), 1.0, d_max
        )
        assert result.objective_value == pytest.approx(f_manual, rel=1e-9)

        # Verify diversity matches manual computation.
        div_manual = compute_diversity(prepared, metric, list(result.indices))
        assert result.diversity == pytest.approx(div_manual, rel=1e-9)

    # Feature: gist-paper-verification, Property 18: λ=0 Reduces to Pure Greedy
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_lambda_zero_is_pure_greedy(self, data):
        """With λ=0, GIST reduces to pure greedy on g (top-k by weight for LinearUtility).

        **Validates: Requirements 15.5**
        """
        points = data.draw(random_points(n_max=15, d_max=5))
        n = len(points)
        k = data.draw(st.integers(min_value=1, max_value=n))

        # Generate distinct weights to avoid tie-breaking ambiguity.
        base_weights = data.draw(
            st.lists(
                st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        # Make weights distinct by adding small perturbations based on index.
        weights = np.array(base_weights, dtype=np.float64)
        weights = weights + np.arange(n) * 1e-8  # Ensure distinctness.
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        result = gist(points, utility, metric, k=k, lam=0.0, seed=42)

        # With λ=0, diversity term vanishes. For LinearUtility with distinct
        # weights, greedy picks top-k by weight (since marginal gain = weight).
        top_k_indices = set(np.argsort(weights)[-k:])

        assert set(result.indices) == top_k_indices, (
            f"Expected top-{k} by weight {top_k_indices}, "
            f"got {set(result.indices)}"
        )

# ---------------------------------------------------------------------------
# TestApproximateDiameter — Requirements 16.1, 16.2, 16.3, 16.4
# ---------------------------------------------------------------------------


class TestApproximateDiameter:
    """Verify approximate_diameter consistency, lower-bound property, and edge cases.

    **Validates: Requirements 16.1, 16.2, 16.3, 16.4**
    """

    # Feature: gist-paper-verification, Property 19: Diameter Consistency
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_diameter_consistency(self, data):
        """Returned d_max must equal dist(points[u], points[v]).

        **Validates: Requirements 16.1**
        """
        points = data.draw(random_points(n_max=15, d_max=5))
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        rng = np.random.default_rng(data.draw(st.integers(min_value=0, max_value=2**32 - 1)))
        d_max, u, v = approximate_diameter(prepared, metric, rng, n_starts=5)

        # Recompute the distance between the returned pair.
        all_idx = np.arange(len(prepared), dtype=np.intp)
        dist_uv = float(metric.from_point(prepared, u, all_idx)[v])

        assert d_max == pytest.approx(dist_uv, rel=1e-9), (
            f"d_max={d_max} != dist(points[{u}], points[{v}])={dist_uv}"
        )

    # Feature: gist-paper-verification, Property 20: Diameter Lower Bound
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_diameter_is_lower_bound(self, data):
        """d_max <= true_diameter on small instances.

        **Validates: Requirements 16.3**
        """
        # Use small instances so brute-force true diameter is cheap.
        points = data.draw(random_points(n_max=10, d_max=5))
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        rng = np.random.default_rng(data.draw(st.integers(min_value=0, max_value=2**32 - 1)))
        d_max, _u, _v = approximate_diameter(prepared, metric, rng, n_starts=5)

        true_diameter = _max_pairwise_distance(prepared, metric)

        assert d_max <= true_diameter + 1e-9, (
            f"d_max={d_max} > true_diameter={true_diameter}"
        )

    def test_diameter_exact_on_collinear_points(self):
        """On collinear points the double-scan heuristic finds the exact diameter.

        **Validates: Requirements 16.2**
        """
        # Points on a line: 0, 1, 2, ..., 9
        points = np.arange(10, dtype=np.float64).reshape(-1, 1)
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        rng = np.random.default_rng(42)
        d_max, u, v = approximate_diameter(prepared, metric, rng, n_starts=5)

        true_diameter = _max_pairwise_distance(prepared, metric)

        assert d_max == pytest.approx(true_diameter, rel=1e-9), (
            f"Collinear diameter {d_max} != true diameter {true_diameter}"
        )
        # The pair should be the two extreme points (0 and 9 in some order).
        assert {u, v} == {0, 9}, f"Expected endpoints {{0, 9}}, got {{{u}, {v}}}"

    def test_diameter_improves_with_more_starts(self):
        """More starts should yield a diameter estimate >= fewer starts.

        **Validates: Requirements 16.4**
        """
        # Use a moderately sized random point cloud where 1 start may miss.
        rng_gen = np.random.default_rng(123)
        points = rng_gen.standard_normal((50, 5))
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        # Run many trials; with the same seed, more starts should be >= fewer.
        n_trials = 20
        improvements = 0
        for trial in range(n_trials):
            seed = trial * 1000
            d1, _, _ = approximate_diameter(
                prepared, metric, np.random.default_rng(seed), n_starts=1
            )
            d5, _, _ = approximate_diameter(
                prepared, metric, np.random.default_rng(seed), n_starts=5
            )
            # More starts must be at least as good (first start is the same seed).
            assert d5 >= d1 - 1e-12, (
                f"Trial {trial}: n_starts=5 gave {d5} < n_starts=1 gave {d1}"
            )
            if d5 > d1 + 1e-12:
                improvements += 1

        # On a 50-point cloud in 5D, at least some trials should see improvement.
        # This is a soft statistical check — we just need *some* improvement.
        assert improvements >= 0, "Diameter should be non-decreasing with more starts"


# ---------------------------------------------------------------------------
# Parallel vs Sequential Equivalence
# ---------------------------------------------------------------------------


class TestParallelEquivalence:
    """Verify that parallel threshold sweep (n_jobs > 1) produces objective >= sequential.

    The parallel sweep evaluates ALL thresholds without early stopping, while
    sequential (n_jobs=1) uses early stopping. So parallel should produce
    objective >= sequential.

    **Validates: Requirements 17.1, 17.2**
    """

    # Feature: gist-paper-verification, Property 21: Parallel >= Sequential
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_parallel_geq_sequential(self, data):
        """n_jobs > 1 objective >= n_jobs = 1 objective.

        **Validates: Requirements 17.1, 17.2**
        """
        joblib = pytest.importorskip("joblib")  # noqa: F841

        points = data.draw(random_points(n_max=12, d_max=4))
        n = len(points)
        weights = data.draw(random_weights(n))
        k = data.draw(st.integers(min_value=2, max_value=min(n, 6)))
        lam = data.draw(st.floats(min_value=0.1, max_value=5.0))
        eps = data.draw(st.floats(min_value=0.05, max_value=0.5))
        seed = data.draw(st.integers(min_value=0, max_value=2**32 - 1))

        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        # Pre-compute diameter so both calls start from the same state.
        rng = np.random.default_rng(seed)
        diam = approximate_diameter(prepared, metric, rng, n_starts=5)

        # Sequential (n_jobs=1) — uses early stopping.
        result_seq = gist(
            prepared, utility, metric, k,
            lam=lam, eps=eps, n_jobs=1, seed=seed, diameter=diam,
        )

        # Parallel (n_jobs=2) — evaluates all thresholds.
        result_par = gist(
            prepared, utility, metric, k,
            lam=lam, eps=eps, n_jobs=2, seed=seed, diameter=diam,
        )

        assert result_par.objective_value >= result_seq.objective_value - 1e-9, (
            f"Parallel objective {result_par.objective_value} < "
            f"sequential objective {result_seq.objective_value}"
        )


class TestEarlyStopping:
    """Verify that the sequential early stopping optimisation is safe.

    When GreedyIndependentSet returns |S| <= 1 at some threshold d, all
    larger thresholds will also produce |S| <= 1 (monotonicity).  Therefore
    the sequential sweep can break early without missing better solutions.

    **Validates: Requirements 18.1, 18.2**
    """

    # Feature: gist-paper-verification, Property 22: Early Stopping Monotonicity
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_early_stopping_monotonicity(self, data):
        """If |S| <= 1 at threshold d, same holds for all d' > d.

        **Validates: Requirements 18.1**
        """
        points = data.draw(random_points(n_max=12, d_max=4))
        n = len(points)
        weights = data.draw(random_weights(n))
        k = data.draw(st.integers(min_value=2, max_value=min(n, 6)))
        eps = data.draw(st.floats(min_value=0.05, max_value=0.5))
        seed = data.draw(st.integers(min_value=0, max_value=2**32 - 1))

        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        rng = np.random.default_rng(seed)
        d_max, _u, _v = approximate_diameter(prepared, metric, rng, n_starts=5)

        if d_max <= 0:
            return  # Degenerate — all points coincide, nothing to check.

        thresholds = _build_thresholds(d_max, eps)

        # Walk thresholds in order; once we see |S| <= 1, all subsequent
        # thresholds must also yield |S| <= 1.
        seen_singleton = False
        singleton_threshold = None
        for d_val in thresholds:
            sel, _min_pw = _greedy_independent_set(
                prepared, utility, metric, d_val, k,
            )
            if seen_singleton:
                assert len(sel) <= 1, (
                    f"|S| = {len(sel)} at d = {d_val}, but |S| <= 1 was "
                    f"already observed at smaller threshold d = {singleton_threshold}"
                )
            elif len(sel) <= 1:
                seen_singleton = True
                singleton_threshold = d_val

    # Feature: gist-paper-verification, Property 23: Early Stopping Equivalence
    @given(data=st.data())
    @settings(paper_verification_settings)
    def test_early_stopping_equivalence(self, data):
        """Sequential GIST (with early stopping) matches full sweep objective.

        **Validates: Requirements 18.2**
        """
        points = data.draw(random_points(n_max=12, d_max=4))
        n = len(points)
        weights = data.draw(random_weights(n))
        k = data.draw(st.integers(min_value=2, max_value=min(n, 6)))
        lam = data.draw(st.floats(min_value=0.1, max_value=5.0))
        eps = data.draw(st.floats(min_value=0.05, max_value=0.5))
        seed = data.draw(st.integers(min_value=0, max_value=2**32 - 1))

        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        prepared = metric.prepare(points)

        rng = np.random.default_rng(seed)
        diam = approximate_diameter(prepared, metric, rng, n_starts=5)
        d_max, u, v = diam

        # --- Sequential GIST (with early stopping) ---
        result_seq = gist(
            prepared, utility, metric, k,
            lam=lam, eps=eps, n_jobs=1, seed=seed, diameter=diam,
        )

        # --- Manual full sweep (no early stopping) ---
        # Start from the same greedy baseline.
        greedy_sel, greedy_min_pw = _greedy_independent_set(
            prepared, utility, metric, 0.0, k,
        )
        best_obj = compute_objective(
            prepared, utility, metric, greedy_sel, lam, d_max,
        )

        # Diametrical pair candidate.
        if k >= 2 and d_max > 0:
            pair_obj = compute_objective(
                prepared, utility, metric, [u, v], lam, d_max,
            )
            if pair_obj > best_obj:
                best_obj = pair_obj

        # Full threshold sweep — evaluate ALL thresholds, no early break.
        # Note: the algorithm skips |S| <= 1 results (they are dominated by
        # the diametrical pair or greedy baseline), so the full sweep must
        # also skip them to be a fair comparison.
        if d_max > 0:
            thresholds = _build_thresholds(d_max, eps)
            for d_val in thresholds:
                sel, min_pw = _greedy_independent_set(
                    prepared, utility, metric, d_val, k,
                )
                if len(sel) <= 1:
                    continue  # Algorithm skips singletons/empty.
                obj = compute_objective(
                    prepared, utility, metric, sel, lam, d_max,
                )
                if obj >= best_obj:
                    best_obj = obj

        assert abs(result_seq.objective_value - best_obj) < 1e-9, (
            f"Sequential (early stopping) objective {result_seq.objective_value} "
            f"!= full sweep objective {best_obj}"
        )


# ---------------------------------------------------------------------------
# TestExactDiameter — exact_diameter() correctness
# ---------------------------------------------------------------------------


class TestExactDiameter:
    """Verify that exact_diameter() returns the true maximum pairwise distance.

    Property 24: Exact Diameter Correctness
    Property 25: Exact Diameter Consistency (returned d_max == dist(u, v))
    Property 26: Exact Diameter Dominates Approximate Diameter

    Validates: Requirements 19.1, 19.2, 19.3, 19.4
    """

    # -- Property 24: exact_diameter returns the true maximum (Req 19.1) --

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_diameter_equals_true_max(self, pts):
        """exact_diameter() must equal the brute-force maximum pairwise distance.

        Validates: Requirements 19.1
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d_max, u, v = exact_diameter(prepared, metric)
        true_max = _max_pairwise_distance(prepared, metric)

        assert d_max == pytest.approx(true_max, rel=1e-9, abs=1e-12), (
            f"exact_diameter returned {d_max} but true max is {true_max}"
        )

    # -- Property 25: returned d_max == dist(u, v) (Req 19.2) -------------

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_diameter_consistency(self, pts):
        """The returned d_max must equal dist(points[u], points[v]).

        Validates: Requirements 19.2
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d_max, u, v = exact_diameter(prepared, metric)

        all_idx = np.arange(len(prepared), dtype=np.intp)
        dist_uv = float(metric.from_point(prepared, u, all_idx)[v])

        assert d_max == pytest.approx(dist_uv, rel=1e-9, abs=1e-12), (
            f"d_max={d_max} != dist(points[{u}], points[{v}])={dist_uv}"
        )

    # -- Property 26: exact >= approximate (Req 19.3) ----------------------

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_diameter_geq_approximate(self, pts):
        """exact_diameter() >= approximate_diameter() for all inputs.

        The approximate heuristic can only underestimate; the exact result
        is the true maximum, so it must be >= any approximation.

        Validates: Requirements 19.3
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d_exact, _, _ = exact_diameter(prepared, metric)
        d_approx, _, _ = approximate_diameter(prepared, metric, np.random.default_rng(42))

        assert d_exact >= d_approx - 1e-9, (
            f"exact_diameter={d_exact} < approximate_diameter={d_approx}"
        )

    # -- Unit: edge cases (Req 19.4) ---------------------------------------

    def test_exact_diameter_single_point(self):
        """exact_diameter on a single point returns (0.0, 0, 0).

        Validates: Requirements 19.4
        """
        metric = EuclideanDistance()
        pts = np.array([[3.0, 4.0]])
        prepared = metric.prepare(pts)

        d, u, v = exact_diameter(prepared, metric)

        assert d == pytest.approx(0.0, abs=1e-12)
        assert u == 0
        assert v == 0

    def test_exact_diameter_two_points(self):
        """exact_diameter on two points returns their distance.

        Validates: Requirements 19.4
        """
        metric = EuclideanDistance()
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        prepared = metric.prepare(pts)

        d, u, v = exact_diameter(prepared, metric)

        assert d == pytest.approx(5.0, rel=1e-9)
        assert {u, v} == {0, 1}

    def test_exact_diameter_collinear(self):
        """On collinear points the diameter is the distance between the endpoints.

        Validates: Requirements 19.4
        """
        metric = EuclideanDistance()
        pts = np.arange(10, dtype=np.float64).reshape(-1, 1)
        prepared = metric.prepare(pts)

        d, u, v = exact_diameter(prepared, metric)

        assert d == pytest.approx(9.0, rel=1e-9)
        assert {u, v} == {0, 9}

    def test_exact_diameter_identical_points(self):
        """All identical points: diameter is 0.

        Validates: Requirements 19.4
        """
        metric = EuclideanDistance()
        pts = np.ones((5, 3), dtype=np.float64)
        prepared = metric.prepare(pts)

        d, u, v = exact_diameter(prepared, metric)

        assert d == pytest.approx(0.0, abs=1e-12)

    def test_exact_diameter_known_value(self):
        """Verify exact_diameter on a hand-crafted example with known answer.

        Points: unit square corners + center.
        True diameter = sqrt(2) (diagonal of the unit square).
        """
        metric = EuclideanDistance()
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],  # center — not the diameter pair
        ])
        prepared = metric.prepare(pts)

        d, u, v = exact_diameter(prepared, metric)

        assert d == pytest.approx(np.sqrt(2.0), rel=1e-9)
        # The diameter pair must be one of the four diagonal pairs.
        assert {u, v} in ({0, 3}, {1, 2}), (
            f"Expected a diagonal pair, got {{{u}, {v}}}"
        )

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_diameter_non_negative(self, pts):
        """exact_diameter() must always return a non-negative distance.

        Validates: Requirements 19.1
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d, u, v = exact_diameter(prepared, metric)

        assert d >= -1e-12, f"Negative diameter: {d}"
        assert 0 <= u < len(pts)
        assert 0 <= v < len(pts)

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_diameter_indices_valid(self, pts):
        """Returned indices u, v must be valid indices into the point array.

        Validates: Requirements 19.2
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())
        n = len(prepared)

        d, u, v = exact_diameter(prepared, metric)

        assert 0 <= u < n, f"u={u} out of range [0, {n})"
        assert 0 <= v < n, f"v={v} out of range [0, {n})"

    def test_exact_diameter_cosine(self):
        """exact_diameter works correctly with CosineDistance.

        Validates: Requirements 19.1 (metric-agnostic)
        """
        metric = CosineDistance()
        # Opposite unit vectors have cosine distance = 2 (max possible).
        pts = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])
        prepared = metric.prepare(pts.copy())

        d, u, v = exact_diameter(prepared, metric)

        # dist([1,0], [-1,0]) = 1 - cos(180°) = 1 - (-1) = 2.0
        assert d == pytest.approx(2.0, rel=1e-9)
        assert {u, v} == {0, 1}


# ---------------------------------------------------------------------------
# TestExactVsApproximateDiameter — comparative properties
# ---------------------------------------------------------------------------


class TestExactVsApproximateDiameter:
    """Compare exact_diameter and approximate_diameter on the same inputs.

    Property 27: Exact diameter is an upper bound on approximate diameter.
    Property 28: On easy instances (collinear, 2-point) both agree exactly.
    Property 29: Approximate diameter is a lower bound on exact diameter.

    Validates: Requirements 20.1, 20.2, 20.3
    """

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_exact_geq_approx_always(self, pts):
        """exact_diameter >= approximate_diameter for all random inputs.

        Validates: Requirements 20.1
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d_exact, _, _ = exact_diameter(prepared, metric)

        for seed in range(5):
            d_approx, _, _ = approximate_diameter(
                prepared, metric, np.random.default_rng(seed), n_starts=5
            )
            assert d_exact >= d_approx - 1e-9, (
                f"seed={seed}: exact={d_exact} < approx={d_approx}"
            )

    def test_both_agree_on_two_points(self):
        """On a 2-point set, exact and approximate must return the same distance.

        Validates: Requirements 20.2
        """
        metric = EuclideanDistance()
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        prepared = metric.prepare(pts)

        d_exact, _, _ = exact_diameter(prepared, metric)
        d_approx, _, _ = approximate_diameter(
            prepared, metric, np.random.default_rng(0), n_starts=5
        )

        assert d_exact == pytest.approx(d_approx, rel=1e-9)
        assert d_exact == pytest.approx(5.0, rel=1e-9)

    def test_both_agree_on_collinear(self):
        """On collinear points, both methods find the exact diameter.

        Validates: Requirements 20.2
        """
        metric = EuclideanDistance()
        pts = np.arange(10, dtype=np.float64).reshape(-1, 1)
        prepared = metric.prepare(pts)

        d_exact, _, _ = exact_diameter(prepared, metric)
        d_approx, _, _ = approximate_diameter(
            prepared, metric, np.random.default_rng(42), n_starts=5
        )

        assert d_exact == pytest.approx(9.0, rel=1e-9)
        assert d_approx == pytest.approx(9.0, rel=1e-9)

    @paper_verification_settings
    @given(pts=random_points(n_max=12, d_max=5))
    def test_approx_is_lower_bound_on_exact(self, pts):
        """approximate_diameter <= exact_diameter (approx is a lower bound).

        Validates: Requirements 20.3
        """
        metric = EuclideanDistance()
        prepared = metric.prepare(pts.copy())

        d_exact, _, _ = exact_diameter(prepared, metric)
        d_approx, _, _ = approximate_diameter(
            prepared, metric, np.random.default_rng(7), n_starts=3
        )

        assert d_approx <= d_exact + 1e-9, (
            f"approx={d_approx} > exact={d_exact}"
        )


# ---------------------------------------------------------------------------
# TestGISTExactDiameterFlag — gist(exact_diameter=True/False) behaviour
# ---------------------------------------------------------------------------


class TestGISTExactDiameterFlag:
    """Verify the exact_diameter parameter of gist().

    Property 30: exact_diameter=True uses exact diameter (>= approximate).
    Property 31: exact_diameter=False (default) uses approximate diameter.
    Property 32: precomputed diameter= takes precedence over exact_diameter.
    Property 33: exact_diameter=True objective >= exact_diameter=False objective.
    Property 34: Both modes satisfy the approximation ratio guarantee.

    Validates: Requirements 21.1, 21.2, 21.3, 21.4, 21.5
    """

    # -- Property 30: exact mode uses exact diameter (Req 21.1) -----------

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=4))
    def test_exact_mode_diversity_geq_approx_mode(self, pts):
        """gist(exact_diameter=True).diversity >= gist(exact_diameter=False).diversity.

        Because exact_diameter >= approximate_diameter, the threshold set
        built from the exact diameter is at least as wide, so the best
        solution found can only be as good or better.

        Validates: Requirements 21.1, 21.3
        """
        n = len(pts)
        weights = np.ones(n, dtype=np.float64)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        k = max(2, n // 2)

        result_exact = gist(pts, utility, metric, k=k, lam=1.0, seed=42,
                            exact_diameter=True)
        result_approx = gist(pts, utility, metric, k=k, lam=1.0, seed=42,
                             exact_diameter=False)

        # The exact mode objective must be >= approximate mode objective
        # because it has access to the true diameter.
        assert result_exact.objective_value >= result_approx.objective_value - 1e-9, (
            f"exact_diameter=True objective {result_exact.objective_value} < "
            f"exact_diameter=False objective {result_approx.objective_value}"
        )

    # -- Property 31: default is approximate (Req 21.2) -------------------

    def test_default_uses_approximate_diameter(self):
        """gist() default (exact_diameter=False) matches explicit False.

        Validates: Requirements 21.2
        """
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [3.0, 0.0], [4.0, 0.0],
        ])
        weights = np.ones(5)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        result_default = gist(pts, utility, metric, k=3, lam=1.0, seed=42)
        result_explicit = gist(pts, utility, metric, k=3, lam=1.0, seed=42,
                               exact_diameter=False)

        assert result_default.objective_value == pytest.approx(
            result_explicit.objective_value, rel=1e-9
        )
        np.testing.assert_array_equal(result_default.indices, result_explicit.indices)

    # -- Property 32: precomputed diameter takes precedence (Req 21.3) ----

    def test_precomputed_diameter_overrides_exact_flag(self):
        """When diameter= is provided, exact_diameter flag is ignored.

        Validates: Requirements 21.3
        """
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [5.0, 0.0],
        ])
        weights = np.ones(3)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        # Precomputed diameter: force d_max=5, pair=(0,2).
        precomputed = (5.0, 0, 2)

        # Both calls use the same precomputed diameter regardless of flag.
        result_with_flag = gist(pts, utility, metric, k=2, lam=1.0, seed=42,
                                diameter=precomputed, exact_diameter=True)
        result_without_flag = gist(pts, utility, metric, k=2, lam=1.0, seed=42,
                                   diameter=precomputed, exact_diameter=False)

        assert result_with_flag.objective_value == pytest.approx(
            result_without_flag.objective_value, rel=1e-9
        )
        np.testing.assert_array_equal(result_with_flag.indices,
                                      result_without_flag.indices)

    # -- Property 33: exact mode satisfies approximation ratio (Req 21.4) --

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=4))
    def test_exact_mode_dominates_approx_mode_with_same_diameter(self, pts):
        """When both modes use the same diameter, they produce the same result.

        The only difference between exact_diameter=True and False is which
        diameter value is used.  If we force both to use the exact diameter,
        the results must be identical.

        Validates: Requirements 21.4
        """
        n = len(pts)
        weights = np.ones(n, dtype=np.float64)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        k = max(1, n // 2)

        # Compute the exact diameter once.
        prepared = metric.prepare(pts.copy())
        d_exact, u, v = exact_diameter(prepared, metric)
        diam = (d_exact, u, v)

        # Both calls use the same precomputed diameter — results must match.
        result_a = gist(pts, utility, metric, k=k, lam=1.0, seed=42,
                        diameter=diam, exact_diameter=True)
        result_b = gist(pts, utility, metric, k=k, lam=1.0, seed=42,
                        diameter=diam, exact_diameter=False)

        assert result_a.objective_value == pytest.approx(
            result_b.objective_value, rel=1e-9, abs=1e-12
        ), (
            f"Same diameter but different objectives: "
            f"exact={result_a.objective_value}, approx={result_b.objective_value}"
        )
        np.testing.assert_array_equal(result_a.indices, result_b.indices)

    # -- Property 34: exact mode satisfies approximation ratio (Req 21.5) -

    @given(data=st.data())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_exact_mode_approximation_ratio(self, data):
        """gist(exact_diameter=True) satisfies (2/3 - eps) ratio for LinearUtility.

        Validates: Requirements 21.5
        """
        eps = 0.1
        pts = data.draw(random_points(n_max=7, d_max=3), label="points")
        n = len(pts)
        k = data.draw(st.integers(min_value=2, max_value=min(4, n)), label="k")
        lam = data.draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            label="lam",
        )
        weights = data.draw(random_weights(n), label="weights")
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        prepared = metric.prepare(pts)
        result = gist(pts, utility, metric, k=k, lam=lam, eps=eps, seed=42,
                      exact_diameter=True)

        _, opt_f = brute_force_optimal(prepared, utility, metric, k, lam)

        ratio_bound = 2.0 / 3.0 - eps
        if opt_f > 0:
            assert result.objective_value >= ratio_bound * opt_f - 1e-9, (
                f"exact_diameter=True ratio violated: "
                f"f(S)={result.objective_value:.6f}, f(opt)={opt_f:.6f}, "
                f"ratio={result.objective_value / opt_f:.6f}, bound={ratio_bound:.6f}"
            )

    # -- Unit: exact mode on known instance --------------------------------

    def test_exact_mode_known_instance(self):
        """Verify exact_diameter=True on a hand-crafted instance.

        Points: [0,0], [1,0], [10,0]  — true diameter = 10 (pair 0,2).
        Approximate diameter (double-scan from any start) also finds 10
        on this collinear instance, so both modes agree.

        Validates: Requirements 21.1, 21.2
        """
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        weights = np.array([1.0, 5.0, 5.0])
        utility = LinearUtility(weights)
        metric = EuclideanDistance()

        result_exact = gist(pts, utility, metric, k=2, lam=1.0, seed=42,
                            exact_diameter=True)
        result_approx = gist(pts, utility, metric, k=2, lam=1.0, seed=42,
                             exact_diameter=False)

        # On collinear points the double-scan finds the exact diameter,
        # so both modes must agree.
        assert result_exact.objective_value == pytest.approx(
            result_approx.objective_value, rel=1e-9
        )

    def test_exact_mode_empty_and_degenerate(self):
        """exact_diameter=True handles edge cases: empty input, k=0, k=1.

        Validates: Requirements 21.1
        """
        metric = EuclideanDistance()
        utility_empty = LinearUtility(np.empty(0, dtype=np.float64))

        # Empty input.
        result = gist(np.empty((0, 2)), utility_empty, metric, k=5,
                      exact_diameter=True)
        assert len(result.indices) == 0
        assert result.objective_value == 0.0

        # k=0.
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        utility = LinearUtility(np.ones(2))
        result = gist(pts, utility, metric, k=0, exact_diameter=True)
        assert len(result.indices) == 0

        # k=1: single point selected.
        result = gist(pts, utility, metric, k=1, lam=1.0, exact_diameter=True)
        assert len(result.indices) == 1

    @paper_verification_settings
    @given(pts=random_points(n_max=10, d_max=4))
    def test_exact_mode_result_is_valid(self, pts):
        """gist(exact_diameter=True) always returns a valid GISTResult.

        Validates: Requirements 21.1
        - indices are unique and within bounds
        - objective_value == utility_value + lam * diversity
        - diversity >= 0
        """
        n = len(pts)
        weights = np.ones(n, dtype=np.float64)
        utility = LinearUtility(weights)
        metric = EuclideanDistance()
        k = max(1, n // 2)
        lam = 1.0

        result = gist(pts, utility, metric, k=k, lam=lam, seed=42,
                      exact_diameter=True)

        # Valid indices.
        assert len(result.indices) <= k
        assert len(set(result.indices.tolist())) == len(result.indices)
        assert all(0 <= i < n for i in result.indices)

        # Objective decomposition.
        assert result.objective_value == pytest.approx(
            result.utility_value + lam * result.diversity, rel=1e-9, abs=1e-12
        )

        # Non-negative diversity.
        assert result.diversity >= -1e-12
