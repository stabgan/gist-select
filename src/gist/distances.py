from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""

    def prepare(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Optional preprocessing called once before the algorithm runs.

        Can store precomputed data on ``self`` and return (possibly
        transformed) points.  The default implementation is the identity.
        """
        return points

    @abstractmethod
    def from_point(
        self,
        points: npt.NDArray[np.floating],
        source: int,
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        """Distances from ``points[source]`` to ``points[targets]``.

        Returns a 1-D array with the same length as *targets*.
        """
        ...


class EuclideanDistance(DistanceMetric):
    """Euclidean distance using the norm identity trick.

    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a . b)``

    ``prepare`` precomputes squared norms so the hot path is a single
    BLAS GEMV (dot product) plus cheap vector arithmetic.
    """

    _norms_sq: npt.NDArray[np.floating]

    def prepare(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        self._norms_sq = np.einsum("ij,ij->i", points, points)
        return points

    def from_point(
        self,
        points: npt.NDArray[np.floating],
        source: int,
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        dots = points[targets] @ points[source]
        sq = self._norms_sq[targets] + self._norms_sq[source] - 2.0 * dots
        np.maximum(sq, 0.0, out=sq)
        return np.sqrt(sq)


class CosineDistance(DistanceMetric):
    """Cosine distance: ``1 - cos(a, b)``.

    ``prepare`` normalises the points to unit length so the hot path
    reduces to ``1 - dot(a, b)``.
    """

    def prepare(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        np.maximum(norms, 1e-10, out=norms)
        return (points / norms).astype(points.dtype)

    def from_point(
        self,
        points: npt.NDArray[np.floating],
        source: int,
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        dots = points[targets] @ points[source]
        return 1.0 - dots


class CallableDistance(DistanceMetric):
    """Wraps a user-provided vectorised distance function.

    Parameters
    ----------
    fn : callable
        ``fn(source_vector, target_matrix) -> distances`` where
        *source_vector* has shape ``(d,)`` and *target_matrix* has shape
        ``(m, d)``.  Must return a 1-D array of length *m*.
    """

    def __init__(
        self,
        fn: "callable[[npt.NDArray, npt.NDArray], npt.NDArray]",
    ) -> None:
        self._fn = fn

    def from_point(
        self,
        points: npt.NDArray[np.floating],
        source: int,
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating]:
        return self._fn(points[source], points[targets])


def approximate_diameter(
    points: npt.NDArray[np.floating],
    metric: DistanceMetric,
    rng: np.random.Generator,
    n_starts: int = 5,
) -> tuple[float, int, int]:
    """Multi-start double-scan heuristic for the approximate diameter.

    Runs *n_starts* independent double-scans from random starting points
    and returns the pair with the largest distance found.  Cost is
    ``2 * n_starts`` one-to-all distance computations.
    """
    n = len(points)
    all_indices = np.arange(n, dtype=np.intp)

    best_dist = -1.0
    best_u = 0
    best_v = 0

    for _ in range(n_starts):
        start = int(rng.integers(n))

        dists = metric.from_point(points, start, all_indices)
        u = int(np.argmax(dists))

        dists = metric.from_point(points, u, all_indices)
        v = int(np.argmax(dists))

        d = float(dists[v])
        if d > best_dist:
            best_dist = d
            best_u = u
            best_v = v

    return best_dist, best_u, best_v
