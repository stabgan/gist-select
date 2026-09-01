from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .distances import DistanceMetric, approximate_diameter, exact_diameter as _exact_diameter
from .objectives import SubmodularFunction


@dataclass(frozen=True)
class GISTResult:
    """Result returned by :func:`gist`."""

    indices: npt.NDArray[np.intp]
    """Indices of the selected points."""
    objective_value: float
    """``f(S) = g(S) + lambda * div(S)``."""
    utility_value: float
    """``g(S)``."""
    diversity: float
    """``div(S)`` — the minimum pairwise distance in the selected set."""


# ---------------------------------------------------------------------------
# GreedyIndependentSet with CELF
# ---------------------------------------------------------------------------

def _greedy_independent_set(
    points: npt.NDArray[np.floating],
    utility: SubmodularFunction,
    metric: DistanceMetric,
    d: float,
    k: int,
) -> tuple[list[int], float]:
    """GreedyIndependentSet from Algorithm 1, accelerated with CELF.

    Returns ``(selected_indices, min_pairwise_distance)``.
    ``min_pairwise_distance`` is ``inf`` when ``|selected| <= 1``.
    """
    n = len(points)
    selected: list[int] = []
    min_pairwise = np.inf

    # Active set — boolean for O(1) lookup, index array for distance ops.
    is_active = np.ones(n, dtype=bool)
    active_indices = np.arange(n, dtype=np.intp)

    # CELF heap: entries are (-gain, eval_iteration, index).
    # Python heapq is a min-heap, so we negate gains for max-heap behaviour.
    all_indices = np.arange(n, dtype=np.intp)
    initial_gains = utility.marginal_gains([], all_indices)
    # Use a tiebreaker counter to ensure stable ordering when gains are equal.
    counter = 0
    heap: list[tuple[float, int, int, int]] = []
    for idx in range(n):
        heap.append((-float(initial_gains[idx]), counter, -1, idx))
        counter += 1
    heapq.heapify(heap)

    for iteration in range(k):
        found = False
        while heap:
            neg_gain, _cnt, eval_iter, idx = heapq.heappop(heap)

            # Lazy deletion: skip eliminated points.
            if not is_active[idx]:
                continue

            # If gain was evaluated in this iteration, it is current — select.
            if eval_iter == iteration:
                # --- SELECT idx ---
                selected.append(idx)
                is_active[idx] = False

                # Track min pairwise distance incrementally.
                if len(selected) >= 2:
                    prev = np.array(selected[:-1], dtype=np.intp)
                    dists_to_prev = metric.from_point(points, idx, prev)
                    min_pairwise = min(min_pairwise, float(dists_to_prev.min()))

                # Eliminate points within distance d.
                if d > 0:
                    active_indices = np.where(is_active)[0]
                    if len(active_indices) > 0:
                        dists = metric.from_point(points, idx, active_indices)
                        to_elim = active_indices[dists < d]
                        is_active[to_elim] = False
                        active_indices = np.where(is_active)[0]

                found = True
                break

            # Gain is stale — re-evaluate and push back.
            new_gain = utility.marginal_gains(selected, np.array([idx], dtype=np.intp))
            heapq.heappush(heap, (-float(new_gain[0]), counter, iteration, idx))
            counter += 1

        if not found:
            break  # No more candidates.

    return selected, float(min_pairwise)


# ---------------------------------------------------------------------------
# Threshold construction
# ---------------------------------------------------------------------------

def _build_thresholds(d_max: float, eps: float) -> npt.NDArray[np.floating]:
    """D = {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps, i >= 0}."""
    base = eps * d_max / 2.0
    limit = 2.0 / eps
    thresholds: list[float] = []
    mult = 1.0
    while mult <= limit:
        thresholds.append(base * mult)
        mult *= 1.0 + eps
    return np.array(thresholds, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main GIST function
# ---------------------------------------------------------------------------

def gist(
    points: npt.NDArray[np.floating],
    utility: SubmodularFunction,
    distance: DistanceMetric,
    k: int,
    lam: float = 1.0,
    eps: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
    diameter: tuple[float, int, int] | None = None,
    exact_diameter: bool = False,
) -> GISTResult:
    """GIST algorithm for max-min diversification with submodular utility.

    Maximises ``f(S) = g(S) + lam * div(S)`` subject to ``|S| <= k``,
    where ``g`` is a monotone submodular function and ``div(S)`` is the
    minimum pairwise distance in *S*.

    Parameters
    ----------
    points : ndarray, shape ``(n, d)``
        Data points in a *d*-dimensional space.
    utility : SubmodularFunction
        Monotone submodular utility function ``g``.
    distance : DistanceMetric
        Distance metric for the point set.
    k : int
        Cardinality constraint (maximum selected points).
    lam : float
        Diversity strength ``lambda >= 0``.
    eps : float
        Approximation parameter ``epsilon > 0``.  Controls the number of
        distance thresholds (~76 for ``eps=0.05``).
    n_jobs : int
        Number of threads for the threshold sweep.  Requires *joblib*
        when ``n_jobs != 1``.
    seed : int, optional
        Random seed for reproducibility.
    diameter : tuple, optional
        Pre-computed ``(d_max, idx_u, idx_v)``.  If ``None``, the
        diameter is computed according to ``exact_diameter``.
    exact_diameter : bool
        If ``True``, compute the exact diameter via O(n²) exhaustive
        search. When using a true metric (satisfying the triangle
        inequality) and a monotone submodular utility, this guarantees
        the paper's theoretical approximation bounds hold. Recommended
        only for small datasets (n ≲ 5 000). If ``False`` (default), use
        the fast multi-start double-scan heuristic. Ignored when
        ``diameter`` is provided.

    Returns
    -------
    GISTResult
    """
    # --- Input validation (system boundary) --------------------------------
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D, got {points.ndim}-D")

    n = len(points)
    if n == 0 or k < 1:
        return GISTResult(np.array([], dtype=np.intp), 0.0, 0.0, 0.0)
    k = min(k, n)
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    rng = np.random.default_rng(seed)

    # --- Preprocess --------------------------------------------------------
    points = distance.prepare(points)

    # --- Step 1: Classic greedy (d = 0) ------------------------------------
    best_sel, best_min_pw = _greedy_independent_set(points, utility, distance, 0.0, k)

    # --- Step 2: Diameter --------------------------------------------------
    if diameter is None:
        if exact_diameter:
            d_max, u, v = _exact_diameter(points, distance)
        else:
            d_max, u, v = approximate_diameter(points, distance, rng)
    else:
        d_max, u, v = diameter

    # Helper to compute div(S) from tracked min pairwise distance.
    def _diversity(sel: list[int], min_pw: float) -> float:
        if len(sel) >= 2:
            return min_pw
        return d_max  # Paper convention: div(S) = diameter when |S| <= 1.

    def _objective(sel: list[int], min_pw: float) -> tuple[float, float, float]:
        g = utility.value(sel)
        div = _diversity(sel, min_pw)
        return g + lam * div, g, div

    best_obj, best_g, best_div = _objective(best_sel, best_min_pw)

    # --- Step 3: Diametrical pair ------------------------------------------
    if k >= 2 and d_max > 0:
        pair = [u, v]
        pair_obj, pair_g, pair_div = _objective(pair, d_max)
        if pair_obj > best_obj:
            best_sel, best_obj, best_g, best_div = pair, pair_obj, pair_g, pair_div

    # --- Step 4: Threshold sweep -------------------------------------------
    if d_max <= 0:
        # Degenerate: all points coincide.
        return GISTResult(np.array(best_sel, dtype=np.intp), best_obj, best_g, best_div)

    thresholds = _build_thresholds(d_max, eps)

    def _run_threshold(d_val: float) -> tuple[list[int], float, float, float] | None:
        sel, min_pw = _greedy_independent_set(points, utility, distance, d_val, k)
        if len(sel) <= 1:
            return None  # Dominated by diametrical pair.
        obj, g, div = _objective(sel, min_pw)
        return sel, obj, g, div

    if n_jobs == 1:
        # Sequential with early stopping.
        for d_val in thresholds:
            result = _run_threshold(d_val)
            if result is None:
                break  # Larger thresholds only make it harder.
            sel, obj, g, div = result
            if obj >= best_obj:
                best_sel, best_obj, best_g, best_div = sel, obj, g, div
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError(
                "joblib is required for n_jobs != 1. "
                "Install with:  pip install 'gist-select[parallel]'"
            ) from None

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_threshold)(d_val) for d_val in thresholds
        )
        for result in results:
            if result is None:
                continue
            sel, obj, g, div = result
            if obj >= best_obj:
                best_sel, best_obj, best_g, best_div = sel, obj, g, div

    return GISTResult(
        indices=np.array(best_sel, dtype=np.intp),
        objective_value=best_obj,
        utility_value=best_g,
        diversity=best_div,
    )
