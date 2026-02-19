
import time
import numpy as np
import pytest
from gist import GISTResult, LinearUtility, EuclideanDistance, gist

@pytest.mark.slow
def test_gist_large_scale():
    """Test GIST with 1.5M points to verify scalability."""
    n = 1_500_000
    d = 64
    k = 100
    
    print(f"\nGenerating {n} points in {d} dimensions...")
    rng = np.random.default_rng(42)
    # Use float32 to save memory, though float64 is fine for 1.5M * 64 * 8 bytes ≈ 768MB
    points = rng.standard_normal((n, d), dtype=np.float32)
    weights = rng.random(n, dtype=np.float32)

    print("Running GIST...")
    start_time = time.time()
    
    result = gist(
        points, 
        LinearUtility(weights), 
        EuclideanDistance(),
        k=k, 
        lam=1.0, 
        eps=0.1, 
        seed=42,
        n_jobs=4 # Use parallel processing if available/supported
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nGIST completed in {duration:.2f} seconds")
    print(f"Objective Value: {result.objective_value}")
    print(f"Utility Value: {result.utility_value}")
    print(f"Diversity: {result.diversity}")
    print(f"Selected indices: {len(result.indices)}")

    assert isinstance(result, GISTResult)
    assert len(result.indices) <= k
    assert len(result.indices) > 0
    assert result.objective_value > 0

if __name__ == "__main__":
    test_gist_large_scale()
