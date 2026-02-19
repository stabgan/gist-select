# gist-select

**Greedy Independent Set Thresholding for Max-Min Diversification with Submodular Utility**

A production-grade Python implementation of the GIST algorithm from [Fahrbach et al. (NeurIPS 2025)](https://arxiv.org/abs/2405.18754). Select subsets that are both **high-quality** and **diverse** — with provable approximation guarantees.

---

## The Problem

You have a large pool of items (data points, images, documents, candidates) and need to select *k* of them. You want items that are **individually valuable** *and* **collectively diverse** — no redundancy, maximum coverage.

GIST solves this by maximizing:

```
f(S) = g(S) + λ · div(S)
```

| Term | Meaning |
|------|---------|
| `g(S)` | Monotone submodular utility — how valuable the selected set is |
| `div(S)` | Max-min diversity — the minimum pairwise distance in the set |
| `λ` | Trade-off knob between utility and diversity |

**Approximation guarantees:**
- General submodular utility: **1/2 - ε**
- Linear utility: **2/3 - ε** *(tight — matches the NP-hardness lower bound)*

---

## Features

- **Provably good** — constant-factor approximation guarantees from the paper
- **Fast at scale** — tested up to 2M points with high-dimensional embeddings
- **CELF acceleration** — lazy greedy evaluation reduces oracle calls by orders of magnitude
- **Optimised numerics** — BLAS-backed distance computation, precomputed norms, no large temporaries
- **Flexible metrics** — Euclidean, cosine, or bring your own distance function
- **Flexible utilities** — linear weights, set coverage, or bring your own submodular function
- **Parallel threshold sweep** — optional multi-threaded execution via joblib
- **Deterministic** — seed parameter for full reproducibility

---

## Installation

```bash
pip install gist-select
```

With optional parallel support:

```bash
pip install "gist-select[parallel]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10

---

## Quick Start

```python
import numpy as np
from gist import gist, LinearUtility, EuclideanDistance

# 10,000 points in 64 dimensions
rng = np.random.default_rng(42)
points = rng.standard_normal((10_000, 64)).astype(np.float32)
weights = rng.random(10_000)

# Select the 50 best-and-diverse points
result = gist(
    points=points,
    utility=LinearUtility(weights),
    distance=EuclideanDistance(),
    k=50,
    lam=1.0,
    seed=42,
)

print(f"Selected {len(result.indices)} points")
print(f"Objective: {result.objective_value:.4f}")
print(f"Utility:   {result.utility_value:.4f}")
print(f"Diversity: {result.diversity:.4f}")
```

---

## API Reference

### `gist()`

```python
gist(
    points,              # np.ndarray (n, d) — your data
    utility,             # SubmodularFunction — how to score subsets
    distance,            # DistanceMetric — how to measure spread
    k,                   # int — how many points to select
    lam=1.0,             # float — diversity weight (λ ≥ 0)
    eps=0.05,            # float — approximation granularity (ε > 0)
    n_jobs=1,            # int — threads for threshold sweep
    seed=None,           # int — random seed for reproducibility
    diameter=None,       # tuple — precomputed (d_max, idx_u, idx_v)
) -> GISTResult
```

**Returns** a `GISTResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `np.ndarray` | Indices of selected points |
| `objective_value` | `float` | `g(S) + λ · div(S)` |
| `utility_value` | `float` | `g(S)` |
| `diversity` | `float` | `div(S)` — minimum pairwise distance |

**Parameters in detail:**

- **`lam`** — Controls the utility/diversity trade-off. Higher values favour more spread-out selections. Set to `0` for pure utility maximisation (standard greedy).
- **`eps`** — Controls the number of distance thresholds swept (~76 for `eps=0.05`, ~38 for `eps=0.1`). Smaller is more thorough but slower.
- **`n_jobs`** — Number of threads for the threshold sweep. Requires `joblib`. Uses threading backend to share memory.
- **`diameter`** — Skip the automatic diameter estimation by providing `(d_max, idx_u, idx_v)`. Useful when calling `gist()` repeatedly on the same point set.

---

### Distance Metrics

```python
from gist import EuclideanDistance, CosineDistance, CallableDistance
```

| Class | Description | Hot Path |
|-------|-------------|----------|
| `EuclideanDistance()` | L2 distance with precomputed norms | Single BLAS GEMV |
| `CosineDistance()` | `1 - cos(a, b)`, auto-normalises | Single BLAS GEMV |
| `CallableDistance(fn)` | User-provided `fn(vec, matrix) → dists` | Your function |

**Custom distance example:**

```python
from gist import CallableDistance

def manhattan(source_vec, target_matrix):
    """L1 distance — must be vectorised."""
    return np.abs(target_matrix - source_vec).sum(axis=1)

distance = CallableDistance(manhattan)
```

> **Note:** The callable signature is `fn(source: ndarray shape (d,), targets: ndarray shape (m, d)) → ndarray shape (m,)`. It *must* be vectorised — a scalar `dist(a, b)` function will not work at scale.

---

### Submodular Utilities

```python
from gist import LinearUtility, CoverageFunction, SubmodularFunction
```

#### `LinearUtility(weights)`

Additive utility: `g(S) = Σ weights[i]` for `i ∈ S`.

This is the most common case and the fastest — marginal gains are just individual weights. GIST achieves the tight **2/3-approximation** for linear utilities.

```python
weights = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
utility = LinearUtility(weights)
```

#### `CoverageFunction(coverage_matrix, element_weights=None)`

Set-coverage utility: `g(S) = |⋃_{i ∈ S} cover(i)|`.

Each point covers a set of elements. The utility is the total number (or weighted sum) of distinct elements covered by the selected set. Classic diminishing returns.

```python
from scipy import sparse

# 1000 points, each covering some of 500 elements
coverage_matrix = sparse.random(1000, 500, density=0.05, format="csr")
coverage_matrix.data[:] = 1  # binary

utility = CoverageFunction(coverage_matrix)
```

#### Custom Submodular Functions

Subclass `SubmodularFunction` and implement two methods:

```python
from gist import SubmodularFunction

class MyUtility(SubmodularFunction):
    def marginal_gains(self, selected: list[int], candidates: np.ndarray) -> np.ndarray:
        """Return g(v | S) for each v in candidates."""
        gains = np.empty(len(candidates))
        for i, v in enumerate(candidates):
            gains[i] = self._compute_marginal(v, selected)
        return gains

    def value(self, selected: list[int]) -> float:
        """Return g(S)."""
        return self._compute_value(selected)
```

> **Performance tip:** GIST uses CELF (lazy greedy) internally, so `marginal_gains` is called infrequently on small batches after the initial pass. But the initial call evaluates *all* points, so make sure it handles large `candidates` arrays efficiently.

---

## Examples

### Data Sampling for Model Training

Select a representative training subset that balances uncertainty and diversity — inspired by the paper's ImageNet experiment:

```python
import numpy as np
from gist import gist, LinearUtility, CosineDistance

# embeddings: (n, 2048) from a pretrained model
# uncertainty: (n,) margin-based uncertainty scores
embeddings = np.load("embeddings.npy")
uncertainty = np.load("uncertainty_scores.npy")

# Select 50K diverse, uncertain examples
result = gist(
    points=embeddings,
    utility=LinearUtility(uncertainty),
    distance=CosineDistance(),
    k=50_000,
    lam=0.5,          # balance uncertainty with diversity
    eps=0.05,
    n_jobs=4,          # parallel threshold sweep
    seed=0,
)

train_indices = result.indices
print(f"Selected {len(train_indices)} training examples")
print(f"Min pairwise cosine distance: {result.diversity:.4f}")
```

### Feature Selection

Select a diverse subset of features that individually have high relevance:

```python
import numpy as np
from gist import gist, LinearUtility, EuclideanDistance

# features: (n_features, n_samples) — each row is a feature vector
features = np.load("feature_matrix.npy")
relevance_scores = np.load("relevance.npy")

result = gist(
    points=features,
    utility=LinearUtility(relevance_scores),
    distance=EuclideanDistance(),
    k=20,
    lam=2.0,          # strongly penalise redundant features
    seed=42,
)

selected_features = result.indices
```

### Tuning the Diversity Trade-off

Sweep over `lam` to find the right balance for your task:

```python
import numpy as np
from gist import gist, LinearUtility, EuclideanDistance

rng = np.random.default_rng(0)
points = rng.standard_normal((5000, 32))
weights = rng.random(5000)

for lam in [0.0, 0.5, 1.0, 2.0, 5.0]:
    result = gist(
        points, LinearUtility(weights), EuclideanDistance(),
        k=50, lam=lam, eps=0.1, seed=0,
    )
    print(f"λ={lam:<4}  utility={result.utility_value:.2f}  "
          f"diversity={result.diversity:.3f}  "
          f"objective={result.objective_value:.2f}")
```

```
λ=0.0   utility=49.47  diversity=3.041  objective=49.47
λ=0.5   utility=48.91  diversity=3.255  objective=50.54
λ=1.0   utility=48.31  diversity=3.442  objective=51.75
λ=2.0   utility=47.06  diversity=3.819  objective=54.70
λ=5.0   utility=43.98  diversity=4.607  objective=67.02
```

### Pure Utility Maximisation

Set `lam=0` to recover standard greedy submodular maximisation (no diversity term):

```python
result = gist(points, utility, distance, k=100, lam=0.0)
# Equivalent to the classic (1 - 1/e)-approximation greedy
```

---

## Performance

Benchmarked on Apple M-series, single-threaded, `eps=0.1`:

| Points | Dimensions | k | Time |
|--------|-----------|-----|------|
| 10K | 64 | 50 | 0.3s |
| 100K | 128 | 100 | 6s |
| 500K | 128 | 100 | ~30s |
| 2M | 128 | 100 | ~2 min |

**Scaling tips:**
- Use `float32` points — 2x memory savings and faster BLAS
- Increase `eps` (e.g., `0.1` → `0.2`) to halve the number of thresholds
- Use `n_jobs=-1` with joblib for parallel threshold sweep
- Precompute `diameter` when calling `gist()` repeatedly on the same data

---

## How It Works

GIST sweeps over a geometric sequence of distance thresholds. For each threshold *d*, it runs a greedy algorithm that builds a maximal independent set of the intersection graph (points within distance *d* are "neighbours") while maximising the submodular utility.

```
GIST(V, g, k, ε):
  1. S ← GreedyIndependentSet(V, g, d=0, k)           # pure utility baseline
  2. T ← diametrical pair with max distance             # pure diversity baseline
  3. For each threshold d in geometric sequence:
       T ← GreedyIndependentSet(V, g, d, k)             # utility + diversity
       Keep best f(T) = g(T) + λ · div(T)
  4. Return the best solution found
```

The `GreedyIndependentSet` subroutine uses CELF (lazy greedy) to minimise submodular oracle calls. Points within distance *d* of a selected point are eliminated, ensuring all selected points are pairwise at distance ≥ *d*.

For the full details, see the paper: [arXiv:2405.18754](https://arxiv.org/abs/2405.18754)

---

## Citation

If you use this package in your research, please cite the original paper:

```bibtex
@inproceedings{fahrbach2025gist,
  title={GIST: Greedy Independent Set Thresholding for Max-Min Diversification with Submodular Utility},
  author={Fahrbach, Matthew and Ramalingam, Srikumar and Zadimoghaddam, Morteza and Ahmadian, Sara and Citovsky, Gui and DeSalvo, Giulia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## License

MIT
