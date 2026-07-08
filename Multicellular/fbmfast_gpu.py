import cupy as cp
from functools import lru_cache

@lru_cache(maxsize=None)
def _fbm_spectral_weights(n, h):
    """
    Eigenvalues (sqrt) of the circulant embedding of the fGn covariance,
    computed with one FFT -> O(n log n). Memoized on (n, h).
    """
    m = n - 1
    r = cp.zeros(2 * m, dtype=cp.float64)
 
    idx = cp.arange(m + 1, dtype=cp.float64)
    # gamma(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    phi = 0.5 * (
        cp.abs(idx - 1.0) ** (2.0 * h)
        - 2.0 * (idx ** (2.0 * h))
        + cp.abs(idx + 1.0) ** (2.0 * h)
    )
 
    r[:m + 1] = phi
    r[m + 1:] = phi[m - 1:0:-1]
 
    eigenvalues = cp.fft.fft(r).real
    return cp.sqrt(cp.maximum(eigenvalues, 0.0))
 
 
def fgn_increments(n, h, dt, n_cells, dtype=cp.float64):
    """
    Scaled fractional Gaussian increments of shape (n_cells, n), with a
    leading zero at t=0 so out[:, t] is the increment consumed at step t.
    Returns the increments directly (no fBm cumsum + diff round-trip) and
    uses real/imag independence to halve RNG and FFT work.
    """
    out = cp.zeros((n_cells, n), dtype=dtype)
    if n < 2:
        return out
 
    weights = _fbm_spectral_weights(n, float(h))
    m = n - 1
    n_pairs = (n_cells + 1) // 2
 
    z = cp.random.randn(n_pairs, 2 * m) + 1j * cp.random.randn(n_pairs, 2 * m)
    c = cp.fft.ifft(weights * z, axis=1) * cp.sqrt(2 * m)
 
    paths = cp.concatenate([c.real[:, :m], c.imag[:, :m]], axis=0)[:n_cells]
    out[:, 1:] = (paths * (dt ** float(h))).astype(dtype)
    return out
 
 
def fbm_batch(n, h, dt, n_cells, dtype=cp.float64):
    """Cumulative fBm path (kept for backward compatibility). Prefer
    fgn_increments() in the integrator."""
    return cp.cumsum(fgn_increments(n, h, dt, n_cells, dtype=dtype), axis=1)