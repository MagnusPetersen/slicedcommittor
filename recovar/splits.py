"""
Data splits for half-set / cross-validation diagnostics.

Frames must be time-ordered within each group (trajectory): block_split
assumes contiguous indices are contiguous in time. On time-correlated data a
random frame split does not merely understate overfitting, it inverts its
sign (a random split reported -2.7% "overfitting" where contiguous blocks
reported +31% to +64%; _reference/REPORT.md section 10). Use block_split on
MD/Langevin data and check convergence in block length; iid_split is only
valid for genuinely i.i.d. samples.
"""
import numpy as np


def iid_split(n, rng):
    """Random half split of n frames.

    Only valid for i.i.d. samples: on time-correlated data random frame
    splits invert the sign of the overfitting diagnostic (REPORT.md
    section 10); use block_split instead.
    """
    rng = np.random.default_rng(rng)
    p = rng.permutation(n)
    return p[:n // 2], p[n // 2:]


def block_split(n, block_len, rng=None, group=None):
    """Alternate contiguous blocks between halves.

    `group` (e.g. walker id) keeps blocks from crossing trajectory boundaries.

    Frames must be time-ordered within each group, so contiguous index blocks
    are genuinely time-correlated. Random frame splits invert the overfitting
    sign on correlated data (REPORT.md section 10); the diagnostic keeps
    climbing with block length, so check convergence in block_len rather than
    picking one value.
    """
    if group is None:
        group = np.zeros(n, dtype=int)
    h1, h2 = [], []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        nb = int(np.ceil(len(idx) / block_len))
        for k in range(nb):
            blk = idx[k * block_len:(k + 1) * block_len]
            (h1 if k % 2 == 0 else h2).append(blk)
    return np.concatenate(h1), np.concatenate(h2)
