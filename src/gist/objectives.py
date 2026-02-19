from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class SubmodularFunction(ABC):
    """Abstract base class for monotone submodular utility functions."""

    @abstractmethod
    def marginal_gains(
        self,
        selected: list[int],
        candidates: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        """Marginal gain ``g(v | S)`` for each *v* in *candidates*.

        Parameters
        ----------
        selected : list[int]
            Indices currently in the selected set *S*, in selection order.
        candidates : ndarray of int
            Indices of candidate points to evaluate.

        Returns
        -------
        ndarray
            1-D float array of marginal gains, same length as *candidates*.
        """
        ...

    @abstractmethod
    def value(self, selected: list[int]) -> float:
        """Evaluate ``g(S)`` for the given set of indices."""
        ...


class LinearUtility(SubmodularFunction):
    """Additive (linear) utility: ``g(S) = sum_{v in S} weights[v]``.

    Marginal gains are simply the individual weights and do not depend
    on *S*.  This is the special case for which GIST achieves a tight
    ``2/3``-approximation guarantee.
    """

    def __init__(self, weights: npt.NDArray[np.floating]) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)

    def marginal_gains(
        self,
        selected: list[int],
        candidates: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        return self._weights[candidates]

    def value(self, selected: list[int]) -> float:
        if not selected:
            return 0.0
        return float(np.sum(self._weights[selected]))


class CoverageFunction(SubmodularFunction):
    """Set-coverage utility: ``g(S) = |union_{i in S} cover(i)|``.

    Optionally weighted: ``g(S) = sum of element_weights[j]`` for all
    elements *j* covered by at least one point in *S*.

    Parameters
    ----------
    coverage_matrix : scipy.sparse.csr_matrix
        Binary ``(n_points, n_elements)`` sparse matrix where entry
        ``(i, j) = 1`` means point *i* covers element *j*.
    element_weights : ndarray, optional
        Per-element weights.  If ``None``, all elements have weight 1.
    """

    def __init__(
        self,
        coverage_matrix: "scipy.sparse.csr_matrix",
        element_weights: npt.NDArray[np.floating] | None = None,
    ) -> None:
        self._cov = coverage_matrix
        self._n_elements = coverage_matrix.shape[1]
        self._ew = element_weights

    def _covered_mask(self, selected: list[int]) -> npt.NDArray[np.bool_]:
        covered = np.zeros(self._n_elements, dtype=bool)
        for idx in selected:
            covered[self._cov[idx].indices] = True
        return covered

    def marginal_gains(
        self,
        selected: list[int],
        candidates: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        covered = self._covered_mask(selected)
        gains = np.empty(len(candidates), dtype=np.float64)
        for i, c in enumerate(candidates):
            new = self._cov[c].indices[~covered[self._cov[c].indices]]
            if self._ew is None:
                gains[i] = len(new)
            else:
                gains[i] = float(self._ew[new].sum())
        return gains

    def value(self, selected: list[int]) -> float:
        covered = self._covered_mask(selected)
        if self._ew is None:
            return float(covered.sum())
        return float(self._ew[covered].sum())
