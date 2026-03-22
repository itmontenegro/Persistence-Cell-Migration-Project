from functools import lru_cache
import numpy as np
import time

@lru_cache(maxsize=None)
def _fbm_basis(n, h, t0, t1):
    t = np.linspace(t0, t1, n)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ti, tj = t[i], t[j]
            C[i, j] = 0.5 * (ti**(2*h) + tj**(2*h) - abs(ti - tj)**(2*h))
    eigvals, U = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 0, None)
    return U, np.sqrt(eigvals)

def fbm(t, h):
    start_time = time.time()
    U, sqrt_eig = _fbm_basis(len(t), float(h), float(t[0]), float(t[-1]))
    xi = np.random.randn(len(t))
    X = U @ (sqrt_eig * xi)
    print(f"FBM generated in {time.time() - start_time:.2f} seconds")
    return X