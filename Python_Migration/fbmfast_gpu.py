import cupy as cp

def _fbm_spectral_weights(n, h):
    """
    Calculates the eigenvalues of the circulant embedding using FFT.
    From O(n^3) to O(n log n) in comparison to the Cholesky decomposition.
    """
    # Embed the covariance matrix into a circulant matrix of size 2m
    m = n - 1
    r = cp.zeros(2*m)

    # First row of the autocovariance matrix for fractional Gaussian noise
    indices = cp.arange(m + 1, dtype=float)
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
    return cp.sqrt(cp.maximum(eigenvalues, 0))

def fbm_batch(n, h, dt, num_sims):
    """Generates a 2D array of fBm noise: shape (num_sims, n)"""
    if n < 2:
        return cp.zeros((num_sims, n))

    weights = _fbm_spectral_weights(n, float(h))
    m = n - 1

    # Generate complex Gaussian samples for all simulations at once
    # Shape: (num_sims, 2*m)
    z = cp.random.randn(num_sims, 2*m) + 1j * cp.random.randn(num_sims, 2*m)

    # Weights will be broadcasted across all simulations
    fgn = cp.fft.ifft(weights * z, axis=1).real * cp.sqrt(2*m)

    # Integrate fGn to get fBm
    fbm_values = cp.zeros((num_sims, n))
    fbm_values[:, 1:] = cp.cumsum(fgn[:, :n-1], axis=1)

    # Scale
    fbm_values *= dt ** float(h)
    return fbm_values