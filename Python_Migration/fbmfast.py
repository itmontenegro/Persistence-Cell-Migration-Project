from functools import lru_cache
import numpy as np
import time

@lru_cache(maxsize=None)
def _fbm_spectral_weights(n, h):
    """
    Calculates the eigenvalues of the circulant embedding using FFT.
    From O(n^3) to O(n log n) in comparison to the Cholesky decomposition.
    """
    # Embed the covariance matrix into a circulant matrix of size 2m
    m = n - 1
    r = np.zeros(2*m)

    # First row of the autocovariance matrix for fractional Gaussian noise
    indices = np.arange(m + 1, dtype=float)
    # gamma(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    phi = 0.5 * (
        np.abs(indices - 1.0) ** (2.0 * h)
        - 2.0 * (indices ** (2.0 * h))
        + np.abs(indices + 1.0) ** (2.0 * h)
    )

    # Build the first row of the circulant matrix
    r[:m + 1] = phi
    r[m + 1:] = phi[m-1:0:-1]

    # Compute the eigenvalues using FFT
    eigenvalues = np.fft.fft(r).real

    # Clip eigenvalues to ensure non-negativity due to accuracy issues
    return np.sqrt(np.maximum(eigenvalues, 0))

def fbm(t, h):
    start_time = time.time()

    if np.isscalar(t):
        n = int(t)
        if n < 0:
            raise ValueError("Number of time points must be non-negative")
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("Time grid must be a 1D array")

    n = int(t.size)
    if n < 2:
        return np.zeros(n)

    dt = 1.0 if n < 2 else float((t[-1] - t[0]) / (n - 1))

    # Get the spectral weights
    weights = _fbm_spectral_weights(n, float(h))

    # Generate random noise in the frequency domain
    m = n - 1
    # Generate complex Gaussian samples
    z = np.random.randn(2*m) + 1j * np.random.randn(2*m)

    # Convolve with the spectral weights, generating fGn.
    fgn = np.fft.ifft(weights * z).real * np.sqrt(2*m)

    # Integrate fGn to get fBm
    fbm_values = np.zeros(n)
    fbm_values[1:] = np.cumsum(fgn[:n-1])

    # Scale for non-unit timestep so variance follows t^(2H).
    fbm_values *= dt ** float(h)

    print(f"FBM generated in {time.time() - start_time:.2f} seconds")
    return fbm_values