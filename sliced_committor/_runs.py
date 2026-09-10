"""Contiguous runs of a time-ordered array of frames.

A run is a maximal stretch of consecutive frames over which every label
(umbrella window, replicate, ...) is constant. Nothing in the library reads
across a run boundary: the block bootstrap resamples within runs, an
autocorrelation is integrated within one, and a displacement pair has both
ends in one. A label that recurs after an interruption starts a new run, so
frames are never spliced by label value alone.
"""

from itertools import pairwise

import numpy as np


def _changes(labels):
    """``(n - 1,)`` bool, True between frames ``t`` and ``t + 1`` where any label changes."""
    labels = [np.asarray(ids).reshape(-1) for ids in labels]
    if not labels:
        raise ValueError("at least one label array is needed")
    n = labels[0].shape[0]
    if any(ids.shape[0] != n for ids in labels):
        raise ValueError(f"label arrays must all have one entry per frame ({n})")
    change = np.zeros(max(n - 1, 0), dtype=bool)
    for ids in labels:
        change |= ids[1:] != ids[:-1]
    return change


def segment_ids(*labels):
    """``(n,)`` int: the run index of every frame."""
    return np.concatenate([[0], np.cumsum(_changes(labels))]).astype(np.int64)


def segments(*labels):
    """``(lo, hi)`` index ranges of the runs, in frame order."""
    change = _changes(labels)
    n = change.shape[0] + 1
    bounds = np.concatenate([[0], np.flatnonzero(change) + 1, [n]])
    return list(pairwise(bounds))
