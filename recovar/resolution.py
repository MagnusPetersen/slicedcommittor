"""
Committor resolution via graph-Laplacian eigenbands (RECOVAR transfer, §2).

The FSC analogue for the committor: correlate two independently fitted
half-committors band-by-band in the eigenbasis of a graph Laplacian built on
the samples -- the Fourier basis on the data manifold. The finest band at
which the two halves still agree is the "committor resolution".

This is the ONLY resolution variant that works. The real-space variants --
correlating smoothed half-committors at increasing smoothing scales, and the
band-pass of successive smoothing differences -- saturate at ~0.93-1.00 on
every scale, because the committor's global 0 -> 1 sweep across the domain
dominates every band (REPORT.md §2). They are refuted and deliberately NOT
ported. Only the Laplacian-eigenband version produces an FSC-shaped falloff,
and its crossing band moves out with data volume
(corr(resolution band, -log RMSE) = 0.67 across five sample sizes).

Status: research-grade / unfinished (REPORT.md §2). Before it is quotable it
needs more eigenvectors, a larger subsample, averaging over splits, and a
calibrated threshold.

Memory: the dense pdist/squareform path is O(n_sub^2) -- at n_sub = 20000
each dense (n_sub, n_sub) float64 matrix costs ~3.2 GB, and several are alive
at once (distance matrix, kernel, normalized Laplacian, eigenvectors).
Recommend n_sub <= 10000 with this dense path; a sparse kNN kernel +
scipy.sparse.linalg.eigsh upgrade is listed future work.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


def laplacian_shell_correlation(q1, q2, Xq, n_sub=3000, n_eig=200, n_bands=10,
                                rng=None, eps=None):
    """Faithful FSC analogue: correlate the two committors' coefficients in
    bands of graph-Laplacian eigenvectors (the Fourier basis on the data
    manifold).

    Parameters
    ----------
    q1, q2 : (N,) half-committor values at the query points Xq.
    Xq     : (N, d) query points (feature space).
    n_sub  : subsample size for the dense kernel; O(n_sub^2) memory -- keep
             <= 10000 on this dense path (see module docstring).
    n_eig  : number of leading Laplacian eigenvectors kept.
    n_bands: number of eigenvector bands.
    rng    : seed / Generator for the subsample draw.
    eps    : Gaussian kernel scale; None -> median heuristic
             (median nonzero distance)^2 / 4.

    Returns (n_bands, 3) array of rows (band center index, mean eigenvalue,
    band correlation).
    """
    rng = np.random.default_rng(rng)
    idx = rng.choice(Xq.shape[0], size=min(n_sub, Xq.shape[0]), replace=False)
    Z = Xq[idx]
    Dm = squareform(pdist(Z))
    if eps is None:
        eps = np.median(Dm[Dm > 0]) ** 2 / 4
    K = np.exp(-Dm ** 2 / eps)
    dg = K.sum(1)
    Ln = np.eye(len(Z)) - (K / np.sqrt(np.outer(dg, dg)))
    lam, V = eigh(Ln)
    V = V[:, :n_eig]
    c1 = V.T @ (q1[idx] - q1[idx].mean())
    c2 = V.T @ (q2[idx] - q2[idx].mean())
    edges = np.linspace(0, n_eig, n_bands + 1).astype(int)
    out = []
    for k in range(n_bands):
        sl = slice(edges[k], edges[k + 1])
        a, b = c1[sl], c2[sl]
        c = float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2) + 1e-300))
        out.append((0.5 * (edges[k] + edges[k + 1]), float(lam[sl].mean()), c))
    return np.array(out)
