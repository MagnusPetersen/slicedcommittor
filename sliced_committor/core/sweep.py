"""Sweep many fit settings and keep the best by the variational objective.

:func:`sweep_committor` fits a :func:`~sliced_committor.fit_committor` for every
combination of the settings in a ``grid`` dict (a Cartesian product), scores
each by its Dirichlet energy :func:`~sliced_committor.committor_dirichlet_energy`
(label-free, no ground truth), and returns a :class:`SweepResult` holding all
fits plus the best one. The Dirichlet energy is a *relative* ranker: lower is
closer to the true committor by the variational principle. The mean boundary
error is reported alongside as a guard, since a fit can lower its Dirichlet
energy by undershooting the q=0 / q=1 boundaries.

Example::

    res = sweep_committor(
        samples, in_A=in_A, in_B=in_B,
        grid={"n_directions": [128, 256], "weights": ["ebmc", "full_gram"]},
        n_bins=200,                 # fixed across the sweep
    )
    q = res.best_committor          # just the winner
    print(res.summary())            # the full table
"""

import warnings
from collections.abc import Callable
from itertools import product
from typing import Any, NamedTuple

import jax.numpy as jnp

from .committor import CommittorFit, _energy_basis, fit_committor

# fit_committor arguments that are data, not sweepable settings.
_RESERVED_KEYS = frozenset({"samples", "in_A", "in_B", "return_details"})


class SweepResult(NamedTuple):
    """Outcome of :func:`sweep_committor`.

    Holds one entry per grid combination (in Cartesian-product order) plus the
    index of the best. ``fits[i]`` is ``None`` only for a combo that failed
    under ``on_error='skip'`` (its ``dirichlet_energies[i]`` is then ``inf``).
    """

    fits: list  # list[CommittorFit | None]
    configs: list  # list[dict] of the swept keys only
    dirichlet_energies: list  # list[float]
    boundary_errors: list  # list[float] mean eps over valid slices
    n_valid: list  # list[int] valid slices per fit
    best_idx: int

    @property
    def best_fit(self) -> CommittorFit:
        """The :class:`CommittorFit` with the lowest selection score."""
        return self.fits[self.best_idx]

    @property
    def best_committor(self) -> Callable:
        """The callable ``q(x)`` of the best fit."""
        return self.fits[self.best_idx].committor

    def summary(self) -> str:
        """Human-readable table, one row per grid combination.

        Columns: the swept settings, the Dirichlet energy 𝓓 (the selection
        objective, lower better), the mean boundary error (guard), and the
        valid-slice count. The best row is marked ``*``. Format may change
        between minor versions; do not parse it.
        """
        keys = sorted({k for cfg in self.configs for k in cfg})
        header = ["", "idx", *keys, "D[q]", "eps", "n_valid"]
        rows = [header]
        for i, cfg in enumerate(self.configs):
            mark = "*" if i == self.best_idx else " "
            vals = [_fmt(cfg.get(k)) for k in keys]
            rows.append(
                [
                    mark,
                    str(i),
                    *vals,
                    f"{self.dirichlet_energies[i]:.4g}",
                    f"{self.boundary_errors[i]:.4f}",
                    str(self.n_valid[i]),
                ]
            )
        widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
        lines = ["SweepResult", "-----------"]
        for r in rows:
            lines.append("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))
        lines.append(
            f"  best: idx {self.best_idx}  (D[q] = {self.dirichlet_energies[self.best_idx]:.4g})"
        )
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    """Compact one-cell rendering of a grid value for the summary table."""
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 18 else s[:15] + "..."


def _mean_boundary_error(result) -> float:
    """Mean cached boundary error over valid slices (NaN if none/unavailable)."""
    if result.boundary_errors is None:
        return float("nan")
    eps = jnp.where(result.valid_mask, result.boundary_errors, jnp.nan)
    return float(jnp.nanmean(eps))  # all-invalid -> NaN, no separate guard needed


def sweep_committor(
    samples: jnp.ndarray,
    *,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    grid: dict,
    select_by: str | Callable = "dirichlet",
    on_error: str = "skip",
    **fixed: Any,
) -> SweepResult:
    """Fit a sliced committor for every grid combination; keep the best.

    Each setting in ``grid`` is expanded as an independent axis (full Cartesian
    product), and every combination is fitted sequentially via
    :func:`~sliced_committor.fit_committor` with ``return_details=True``. Fits
    are ranked by ``select_by`` (default: lowest Dirichlet energy).

    Args:
        samples: ``(N, dim)`` configurations.
        in_A, in_B: ``(N,)`` bool basin-membership masks.
        grid: maps a ``fit_committor`` parameter name to a list of values to
            sweep, e.g. ``{"n_directions": [128, 256], "weights": ["ebmc",
            "full_gram"], "rd_kappa": [1e10, 1e12]}``. Any ``fit_committor``
            keyword is allowed except the data args (``samples``/``in_A``/
            ``in_B``/``return_details``). A value that is itself an array or a
            config object (e.g. a single ``DirectionSamplingConfig`` or a
            ``directions`` array) is one list element, never auto-expanded.
        select_by: ``"dirichlet"`` (default, minimise the Dirichlet energy) or a
            callable ``(CommittorFit) -> float`` to minimise (e.g. to penalise
            the boundary error). NaN / failed scores sort last. The Dirichlet
            energy is only comparable *within* the Gram-family solvers
            (ebmc / pesb / bmc / full_gram); a sweep that also varies ``weights``
            onto the diagonal solver is ranked on mismatched scales, so
            ``"dirichlet"`` emits a warning and you should pass a custom
            ``select_by`` (or sweep one family at a time) instead.
        on_error: ``"skip"`` (default) records a failed combination with
            ``fit=None`` and energy ``inf`` and continues; ``"raise"`` re-raises
            the first failure.
        **fixed: any other ``fit_committor`` keywords held constant across the
            sweep (e.g. ``n_bins=200``, ``seed=0``). Must not overlap ``grid``.

    Returns:
        a :class:`SweepResult`. Read ``.best_committor`` for just the winner, or
        iterate ``.fits`` / ``.dirichlet_energies`` for the full sweep.

    Note:
        Every fit is retained, and each carries its ``SlicedCommittorResult``
        and weights (the ``(M, N)`` projected samples and ``(M, M)`` Gram). For
        a large grid with large ``M`` / ``N``, pass ``store_projected_samples=
        False`` to drop the dominant per-fit buffer.
    """
    if not grid:
        raise ValueError("grid is empty; pass at least one setting to sweep.")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise'; got {on_error!r}.")
    bad = (set(grid) | set(fixed)) & _RESERVED_KEYS
    if bad:
        raise ValueError(
            f"these keys are data/control args, not sweepable: {sorted(bad)}. "
            "Pass samples/in_A/in_B positionally."
        )
    overlap = set(grid) & set(fixed)
    if overlap:
        raise ValueError(f"keys appear in both grid and fixed kwargs: {sorted(overlap)}.")
    for k, v in grid.items():
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError(
                f"grid[{k!r}] must be a non-empty list/tuple of values; got {type(v).__name__}."
            )

    keys = sorted(grid)
    combos = list(product(*(list(grid[k]) for k in keys)))

    if select_by == "dirichlet":
        score_fn = None  # use the recorded dirichlet energy directly
    elif callable(select_by):
        score_fn = select_by
    else:
        raise ValueError(f"select_by must be 'dirichlet' or a callable; got {select_by!r}.")

    fits: list = []
    configs: list = []
    energies: list = []
    bnd_errors: list = []
    n_valids: list = []
    scores: list = []

    for combo in combos:
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
        try:
            _, fit = fit_committor(
                samples, in_A=in_A, in_B=in_B, return_details=True, **cfg, **fixed
            )
        except Exception:
            if on_error == "raise":
                raise
            fits.append(None)
            energies.append(float("inf"))
            bnd_errors.append(float("nan"))
            n_valids.append(0)
            scores.append(float("inf"))
            continue

        energy = float(fit.dirichlet_energy) if fit.dirichlet_energy is not None else float("nan")
        fits.append(fit)
        energies.append(energy)
        bnd_errors.append(_mean_boundary_error(fit.result))
        n_valids.append(int(jnp.sum(fit.result.valid_mask)))

        raw = energy if score_fn is None else float(score_fn(fit))
        scores.append(raw if raw == raw else float("inf"))  # NaN -> inf (sorts last)

    successful = [i for i, f in enumerate(fits) if f is not None]
    if not successful:
        raise RuntimeError(
            f"all {len(combos)} sweep combinations failed; re-run with "
            "on_error='raise' to surface the first error."
        )
    selects_by_energy = select_by == "dirichlet" or callable(select_by)
    if selects_by_energy and len({_energy_basis(fits[i].weights) for i in successful}) > 1:
        warnings.warn(
            "this sweep mixes Gram-family (ebmc/pesb/bmc/full_gram) and diagonal "
            "weight solvers, whose Dirichlet energies are on different scales (the "
            "diagonal approximation drops off-diagonal coupling and is "
            "systematically smaller), so ranking by the Dirichlet energy across "
            "them is not valid and tends to pick the diagonal fit. Sweep one "
            "family at a time, or rank within a family. (If your custom select_by "
            "does not use fit.dirichlet_energy, you can disregard this.)",
            stacklevel=2,
        )
    best_idx = min(successful, key=lambda i: scores[i])
    return SweepResult(
        fits=fits,
        configs=configs,
        dirichlet_energies=energies,
        boundary_errors=bnd_errors,
        n_valid=n_valids,
        best_idx=best_idx,
    )
