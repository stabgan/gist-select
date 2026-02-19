import numpy as np
import pytest
from scipy import sparse

from gist.objectives import CoverageFunction, LinearUtility


# ---- LinearUtility ---------------------------------------------------------

class TestLinearUtility:
    def test_marginal_gains_are_weights(self):
        weights = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        util = LinearUtility(weights)
        cands = np.array([0, 2, 4], dtype=np.intp)
        gains = util.marginal_gains([], cands)
        np.testing.assert_allclose(gains, [3.0, 4.0, 5.0])

    def test_marginal_gains_independent_of_selected(self):
        weights = np.array([10.0, 20.0, 30.0])
        util = LinearUtility(weights)
        cands = np.array([0, 1, 2], dtype=np.intp)
        g1 = util.marginal_gains([], cands)
        g2 = util.marginal_gains([0, 1], cands)
        np.testing.assert_allclose(g1, g2)

    def test_value_is_sum(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        util = LinearUtility(weights)
        assert util.value([0, 2]) == pytest.approx(4.0)
        assert util.value([]) == pytest.approx(0.0)
        assert util.value([0, 1, 2, 3]) == pytest.approx(10.0)


# ---- CoverageFunction -----------------------------------------------------

class TestCoverageFunction:
    def _make_simple(self):
        """3 points covering 4 elements:
        point 0 covers {0, 1}
        point 1 covers {1, 2}
        point 2 covers {2, 3}
        """
        row = [0, 0, 1, 1, 2, 2]
        col = [0, 1, 1, 2, 2, 3]
        data = [1, 1, 1, 1, 1, 1]
        cov = sparse.csr_matrix((data, (row, col)), shape=(3, 4))
        return CoverageFunction(cov)

    def test_value_empty(self):
        cf = self._make_simple()
        assert cf.value([]) == pytest.approx(0.0)

    def test_value_single(self):
        cf = self._make_simple()
        assert cf.value([0]) == pytest.approx(2.0)  # covers {0,1}

    def test_value_union(self):
        cf = self._make_simple()
        assert cf.value([0, 1]) == pytest.approx(3.0)  # covers {0,1,2}
        assert cf.value([0, 1, 2]) == pytest.approx(4.0)  # covers all

    def test_marginal_gains_empty_selected(self):
        cf = self._make_simple()
        cands = np.array([0, 1, 2], dtype=np.intp)
        gains = cf.marginal_gains([], cands)
        np.testing.assert_allclose(gains, [2.0, 2.0, 2.0])

    def test_marginal_gains_diminishing_returns(self):
        cf = self._make_simple()
        cands = np.array([1, 2], dtype=np.intp)
        g_empty = cf.marginal_gains([], cands)
        g_with_0 = cf.marginal_gains([0], cands)
        # Submodularity: g(v|S) >= g(v|T) for S ⊆ T.
        assert np.all(g_empty >= g_with_0 - 1e-10)

    def test_marginal_gains_overlap(self):
        cf = self._make_simple()
        # After selecting point 0 (covers {0,1}), point 1 covers only {2}.
        cands = np.array([1], dtype=np.intp)
        gain = cf.marginal_gains([0], cands)
        assert gain[0] == pytest.approx(1.0)

    def test_weighted_elements(self):
        row = [0, 0, 1]
        col = [0, 1, 1]
        data = [1, 1, 1]
        cov = sparse.csr_matrix((data, (row, col)), shape=(2, 2))
        ew = np.array([10.0, 1.0])
        cf = CoverageFunction(cov, element_weights=ew)
        assert cf.value([0]) == pytest.approx(11.0)
        assert cf.value([1]) == pytest.approx(1.0)
        gain = cf.marginal_gains([0], np.array([1], dtype=np.intp))
        assert gain[0] == pytest.approx(0.0)  # element 1 already covered
