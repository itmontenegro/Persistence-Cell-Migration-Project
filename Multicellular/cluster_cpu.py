import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
 
 
def cluster_arrays(x, y, R, boundary, L_box):
    """
    x, y : (N, Nts) numpy position arrays for one replicate.
 
    Returns cluster_size_array (N, Nts) and n_clusters_array (Nts,), cast to
    x.dtype so they match the other saved arrays. Two cells are in contact when
    their (minimum-image, if periodic) distance is <= 2R -- identical to the
    criterion used in the simulation.
 
    All timesteps are packed into one block-diagonal contact graph, so
    connected_components is called once per replicate (blocks stay independent).
    """
    N, Nts = x.shape
    Xt = x.T.astype(np.float64)                  # (Nts, N)
    Yt = y.T.astype(np.float64)
 
    dX = Xt[:, :, None] - Xt[:, None, :]
    dY = Yt[:, :, None] - Yt[:, None, :]
    if boundary == 'periodic':
        dX -= 2.0 * L_box * np.round(dX / (2.0 * L_box))
        dY -= 2.0 * L_box * np.round(dY / (2.0 * L_box))
 
    dist = np.sqrt(dX * dX + dY * dY)
    dist[:, np.arange(N), np.arange(N)] = np.inf
    mask = dist <= 2.0 * R                        # (Nts, N, N)
 
    tt, ii, jj = np.nonzero(mask)
    big = sparse.coo_matrix(
        (np.ones(tt.size, np.int8), (tt * N + ii, tt * N + jj)),
        shape=(Nts * N, Nts * N),
    ).tocsr()
    _, labels = connected_components(big, directed=False, connection='weak')
 
    counts = np.bincount(labels)
    cluster_size = counts[labels].reshape(Nts, N).T
 
    sl = np.sort(labels.reshape(Nts, N), axis=1)
    is_new = np.ones_like(sl, dtype=bool)
    is_new[:, 1:] = sl[:, 1:] != sl[:, :-1]
    n_clusters = is_new.sum(axis=1)
 
    return cluster_size.astype(x.dtype), n_clusters.astype(x.dtype)
 