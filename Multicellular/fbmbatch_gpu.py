import cupy as cp
from functools import lru_cache

@lru_cache(maxsize=None)
def _fbm_spectral_weights(n, h):
    """
    Calculates the eigenvalues of the circulant embedding using FFT.
    From O(n^3) to O(n log n) in comparison to the Cholesky decomposition.
    """
    # Embed the covariance matrix into a circulant matrix of size 2m
    m = n - 1
    r = cp.zeros(2*m, dtype=cp.float64)

    # First row of the autocovariance matrix for fractional Gaussian noise
    indices = cp.arange(m + 1, dtype=cp.float64)
    # gamma(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    phi = 0.5 * (
        cp.abs(indices - 1.0) ** (2.0 * h)
        - 2.0 * (indices ** (2.0 * h))
        + cp.abs(indices + 1.0) ** (2.0 * h)
    )

    # Build the first row of the circulant matrix
    r[:m + 1] = phi
    r[m + 1:] = phi[m-1:0:-1]

    # Compute the eigenvalues using FFT
    eigenvalues = cp.fft.fft(r).real

    # Clip eigenvalues to ensure non-negativity due to accuracy issues
    return cp.sqrt(cp.maximum(eigenvalues, 0.0))
    
def fgn_increments(n, h, dt, batch_size, n_cells, dtype=cp.float64):
    """
    Scaled fractioanl Gaussian increments of shape (n_cells, n)
    """
    out = cp.zeros((batch_size, n_cells, n), dtype=dtype)
    if n < 2:
        return out
    
    weights = _fbm_spectral_weights(n, float(h))
    m = n - 1
    total = batch_size * n_cells
    n_pairs = (total + 1) // 2  # Number of pairs of simulations to generate

    z = cp.random.randn(n_pairs, 2*m) + 1j * cp.random.randn(n_pairs, 2*m)
    c = cp.fft.ifft(weights * z, axis=1) * cp.sqrt(2*m)

    paths = cp.concatenate([c.real[:, :m], c.imag[:, :m]], axis=0)[:total]
    paths = (paths * (dt ** float(h))).astype(dtype)

    out[:, :, 1:] = paths.reshape(batch_size, n_cells, m)
    return out

def fbm_batch(n, h, dt, batch_size, n_cells, dtype=cp.float64):
    """Generates a 2D array of fBm noise: shape (num_sims, n)"""
    return cp.cumsum(fgn_increments(n, h, dt, batch_size, n_cells, dtype), axis=2)