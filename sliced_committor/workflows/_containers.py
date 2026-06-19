"""Container types for the umbrella-sampling workflow.

The pipeline is functional: loaders return an immutable :class:`USDataset`, every
downstream step takes it (plus options) and returns plain values or new datasets.
Nothing mutates in place.

A :class:`USDataset` keeps frames in per-window, time-ordered layout (windows
concatenated, frames within a window in simulation order). That single layout
serves all three consumers:

* the static equilibrium ensemble (``features`` + MBAR ``sample_weights``) that
  :func:`sliced_committor.compute_sliced_committor` and the density / population
  reductions consume;
* the time-ordered trajectory that :func:`sliced_committor.diffusion_coefficient`
  consumes (``window_ids`` keeps lag-pairs inside a single window);
* the per-window lists that
  :func:`sliced_committor.directions_tica_ema_decomposed` consumes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


def split_by_window(arr: np.ndarray, window_ids: np.ndarray) -> list[np.ndarray]:
    """Split a per-frame array into a list of per-window arrays.

    Windows are returned in ascending window-id order; the original
    within-window frame order (assumed temporal) is preserved.

    Args:
        arr: ``(N, ...)`` per-frame array.
        window_ids: ``(N,)`` integer window label per frame.

    Returns:
        List of ``(N_k, ...)`` arrays, one per distinct window id.
    """
    arr = np.asarray(arr)
    window_ids = np.asarray(window_ids).reshape(-1)
    out = []
    for w in np.unique(window_ids):
        out.append(arr[window_ids == w])
    return out


class USDataset(NamedTuple):
    """Immutable umbrella-sampling dataset in per-window, time-ordered layout.

    Attributes:
        features: ``(N, d)`` committor-input features (frames in per-window
            temporal order). For toy systems this is the raw configuration; for
            proteins it is the chosen featurization output.
        cvs: ``(N, n_cv)`` biased collective-variable value(s) per frame.
        window_ids: ``(N,)`` integer window index per frame (ascending, blocked).
        window_centers: ``(K, n_cv)`` harmonic-restraint centers per window.
        window_kappa: ``(K, n_cv)`` harmonic force constants in physical energy
            units per CV-unit squared (e.g. kJ/mol/nm^2). Combined with ``beta``
            to build the reduced bias in :mod:`reweight`.
        beta: inverse temperature ``1/kT`` in the CV energy units, so the reduced
            bias is ``0.5 * beta * kappa * (cv - center)**2``.
        dt: time between consecutive frames (system time units).
        in_A: ``(N,)`` bool basin-A membership (committor boundary q=0).
        in_B: ``(N,)`` bool basin-B membership (committor boundary q=1).
        system_name: registry key / label for the system.
        cv_names: per-CV-dimension names (length ``n_cv``).
        cv_periodic: per-CV ``(min, max)`` period bounds, or ``None`` for a
            non-periodic CV (length ``n_cv``).
        meta: free-form provenance (topology path, featurization spec, engine,
            units, source folder, reweighting notes, ...).
    """

    features: np.ndarray
    cvs: np.ndarray
    window_ids: np.ndarray
    window_centers: np.ndarray
    window_kappa: np.ndarray
    beta: float
    dt: float
    in_A: np.ndarray
    in_B: np.ndarray
    system_name: str
    cv_names: tuple[str, ...]
    cv_periodic: tuple[tuple[float, float] | None, ...]
    meta: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_cv(self) -> int:
        return int(self.cvs.shape[1])

    @property
    def n_windows(self) -> int:
        return int(self.window_centers.shape[0])

    def per_window_features(self) -> list[np.ndarray]:
        """List of ``(N_k, d)`` per-window feature blocks (for TICA-EMA)."""
        return split_by_window(self.features, self.window_ids)

    def per_window_weights(self, sample_weights: np.ndarray) -> list[np.ndarray]:
        """List of ``(N_k,)`` per-window weight blocks parallel to features."""
        return split_by_window(np.asarray(sample_weights).reshape(-1), self.window_ids)

    def subsample(self, max_frames: int, *, seed: int = 0) -> tuple[USDataset, np.ndarray]:
        """Return a frame-subsampled copy (stratified per window) and the kept index.

        When ``n_frames <= max_frames`` the dataset is returned unchanged with the
        identity index. Otherwise frames are kept proportionally per window so the
        per-window time order (needed for diffusion / TICA) is preserved.

        Args:
            max_frames: target maximum number of frames.
            seed: RNG seed for the per-window thinning offset.

        Returns:
            ``(subsampled_dataset, kept_index)`` where ``kept_index`` maps rows of
            the new dataset back into the original ``features``.
        """
        n = self.n_frames
        if n <= max_frames:
            return self, np.arange(n)
        keep_frac = max_frames / n
        rng = np.random.default_rng(seed)
        keep_parts = []
        for w in np.unique(self.window_ids):
            idx_w = np.nonzero(self.window_ids == w)[0]
            n_keep = max(1, int(round(keep_frac * idx_w.size)))
            # Even temporal thinning preserves autocorrelation structure better
            # than random sampling, with a random phase to avoid aliasing.
            stride = max(1, idx_w.size // n_keep)
            phase = int(rng.integers(0, stride))
            keep_parts.append(idx_w[phase::stride][:n_keep])
        kept = np.sort(np.concatenate(keep_parts))
        sub = self._replace(
            features=self.features[kept],
            cvs=self.cvs[kept],
            window_ids=self.window_ids[kept],
            in_A=self.in_A[kept],
            in_B=self.in_B[kept],
        )
        return sub, kept


class Reweighting(NamedTuple):
    """Output of a reweighting (MBAR or WHAM) solve.

    Attributes:
        sample_weights: ``(N,)`` per-frame equilibrium weights, summing to 1.
        f_k: ``(K,)`` per-window dimensionless free energies (reference subtracted).
        n_eff: Kish effective sample size ``1 / sum(w**2)``.
        method: ``"mbar"`` or ``"wham"`` (which engine actually ran).
        pmf_edges: histogram bin edges used for the PMF (per CV dim) or ``None``.
        pmf: free energy on the PMF grid in kT units (min subtracted) or ``None``.
        meta: free-form diagnostics (iterations, residual, fallback reason, ...).
    """

    sample_weights: np.ndarray
    f_k: np.ndarray
    n_eff: float
    method: str
    pmf_edges: Any
    pmf: Any
    meta: dict[str, Any]
