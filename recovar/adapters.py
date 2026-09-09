"""Adapters: IDS directions driving external committor fitters.

These adapters let the IDS outer loop (:func:`recovar.ids.ids_fit`) drive the
``sliced_committor`` production pipeline (quantile binning, boundary_quantile,
EBMC with tikhonov='cv') instead of, or alongside, this package's own
halfset-eigen regularized dual solve. Each adapter is a factory returning a
``fit_fn(thetas, pass_idx) -> (qhat, handle)`` closure with the data and the
fitter configuration baked in, matching the ``ids_fit`` contract: ``qhat`` is
the committor at all frames, float64 (N,), clipped to [0, 1]; ``handle`` is
opaque to the loop and comes back as ``result['final']``.

Fits hand JAX float64 arrays around; x64 is enabled by the package
``__init__`` at import (the EBMC solve chain requires it).

Contents:

* :func:`make_lib_fit`: ``sliced_committor.fit_committor`` as an IDS fitter;
  the handle is the library's ``CommittorFit``.
* :func:`make_recovar_fit`: this package's ``_fit_pass_w`` fitter, exactly
  what ``ids_path_exact2`` uses internally; the handle is the sol dict.
* :func:`evaluate_handle`: evaluate either handle kind at arbitrary points,
  chunked, with optional basin clamping.
* :func:`halfset_committor_pair`: two independent half-data fits with shared
  directions, evaluated on the full sample set (feeds the graph-Laplacian
  resolution metric).
* :func:`ids_protocol_metadata`: JSON-safe provenance from an IDS result.
"""

import numpy as np

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Handle evaluation (shared by the adapters and the halfset pair)
# ---------------------------------------------------------------------------

def evaluate_handle(handle, X, *, in_A=None, in_B=None, eval_block=65536):
    """Evaluate a fit handle at the rows of ``X``; float64 (N,) in [0, 1].

    Dispatches on the handle kind: a ``sliced_committor`` ``CommittorFit``
    (anything with a callable ``committor`` attribute) is evaluated through
    its JAX closure, chunked over ``eval_block`` points to bound memory; a
    recovar sol dict (keys ``basis`` / ``w`` / ``c``) goes through
    :func:`recovar.assemble.evaluate`. When basin masks are given the basins
    are clamped to exactly 0 / 1: the ``CommittorFit`` callable does this
    itself (its ``in_A`` / ``in_B`` keywords), the sol-dict path snaps after
    evaluation, so the two kinds agree on the contract.
    """
    from .assemble import clip01, evaluate

    X = np.asarray(X, np.float64)
    if in_A is not None:
        in_A = np.asarray(in_A, bool)
    if in_B is not None:
        in_B = np.asarray(in_B, bool)

    committor = getattr(handle, "committor", None)
    if callable(committor):
        N = X.shape[0]
        out = np.empty(N, np.float64)
        for n0 in range(0, N, eval_block):
            n1 = min(n0 + eval_block, N)
            kw = {}
            if in_A is not None:
                kw["in_A"] = jnp.asarray(in_A[n0:n1])
            if in_B is not None:
                kw["in_B"] = jnp.asarray(in_B[n0:n1])
            out[n0:n1] = np.asarray(
                committor(jnp.asarray(X[n0:n1], jnp.float64), **kw), np.float64
            )
        return clip01(out)

    if isinstance(handle, dict) and {"basis", "w", "c"} <= handle.keys():
        q = clip01(
            np.asarray(
                evaluate(handle["basis"], handle["w"], handle["c"], X, block=eval_block),
                np.float64,
            )
        )
        if in_A is not None:
            q[in_A] = 0.0
        if in_B is not None:
            q[in_B] = 1.0
        return q

    raise TypeError(
        "handle is neither a CommittorFit (callable .committor) nor a recovar "
        f"sol dict with 'basis'/'w'/'c'; got {type(handle).__name__}."
    )


# ---------------------------------------------------------------------------
# Fitter factories
# ---------------------------------------------------------------------------

def make_lib_fit(X, in_A, in_B, *, weights="ebmc", weight_kwargs=None,
                 solver_kwargs=None, seed=42, eval_block=65536):
    """``sliced_committor.fit_committor`` as an ``ids_fit`` fitter.

    Returns ``fit_fn(thetas, pass_idx) -> (qhat, CommittorFit)``: each pass
    fits the production pipeline with the IDS directions supplied verbatim
    (``directions=`` overrides random sampling in the solver; ``n_directions``
    is kept consistent with ``thetas.shape[0]``), then evaluates the fitted
    committor at all rows of ``X`` in chunks of ``eval_block``, with the basin
    masks clamping q to exactly 0 / 1 there.

    Args:
        X: (N, d) samples; the fit and the qhat evaluation both use them.
        in_A, in_B: (N,) bool basin masks.
        weights: weight solver for ``fit_committor`` (default ``'ebmc'``).
        weight_kwargs: forwarded to the weight solver; ``None`` means the
            calibration-free default ``{'tikhonov': 'cv'}``.
        solver_kwargs: forwarded to ``compute_sliced_committor``
            (e.g. ``n_bins``, ``binning_method``, ``boundary_quantile``,
            ``sample_weights``).
        seed: solver seed; only tie-break paths depend on it once directions
            are supplied.
        eval_block: chunk size for the qhat evaluation.
    """
    X = np.asarray(X, np.float64)
    in_A = np.asarray(in_A, bool)
    in_B = np.asarray(in_B, bool)
    weight_kwargs = {"tikhonov": "cv"} if weight_kwargs is None else dict(weight_kwargs)
    solver_kwargs = dict(solver_kwargs or {})
    Xj = jnp.asarray(X, jnp.float64)
    inAj = jnp.asarray(in_A)
    inBj = jnp.asarray(in_B)

    def fit_fn(thetas, pass_idx):
        import sliced_committor

        thetas = np.asarray(thetas, np.float64)
        _q, fit = sliced_committor.fit_committor(
            Xj,
            in_A=inAj,
            in_B=inBj,
            weights=weights,
            weight_kwargs=weight_kwargs,
            n_directions=int(thetas.shape[0]),
            seed=seed,
            return_details=True,
            directions=jnp.asarray(thetas, jnp.float64),
            **solver_kwargs,
        )
        qhat = evaluate_handle(fit, X, in_A=in_A, in_B=in_B, eval_block=eval_block)
        return qhat, fit

    return fit_fn


def make_recovar_fit(X, in_A, in_B, *, basis_kw=None, split=None,
                     sample_weights=None, rng=None):
    """This package's own fitter as an ``ids_fit`` fitter.

    Wraps ``_fit_pass_w`` (recovar basis + halfset-eigen regularized dual
    solve) + ``evaluate`` + ``clip01``: exactly what ``ids_path_exact2`` does
    internally, exposed as a standalone factory so it can be swapped against
    :func:`make_lib_fit` under the same outer loop.

    Returns ``fit_fn(thetas, pass_idx) -> (qhat, sol)`` with ``sol`` the dual
    solve dict (keys ``G_reg`` / ``M_gap`` / ``a`` / ``b`` / ``basis`` / ``c``
    / ``delta`` / ``nu_hat`` / ``w`` / ``w_dual``).

    Args:
        X: (N, d) samples.
        in_A, in_B: (N,) bool basin masks.
        basis_kw: kwargs for ``build_basis``; default
            ``dict(n_bins=200, bw_bins=1.5)`` (the ``ids_path_exact2``
            default).
        split: (i1, i2) index halves for the halfset Grams; ``None`` draws an
            ``iid_split`` from ``rng``.
        sample_weights: optional (N,) per-frame equilibrium weights.
        rng: seed / Generator, used only when ``split`` is ``None``.
    """
    from .ids import _fit_pass_w
    from .assemble import clip01, evaluate
    from .splits import iid_split

    X = np.asarray(X, np.float64)
    in_A = np.asarray(in_A, bool)
    in_B = np.asarray(in_B, bool)
    basis_kw = dict(basis_kw) if basis_kw is not None else dict(n_bins=200, bw_bins=1.5)
    if split is None:
        split = iid_split(X.shape[0], rng=np.random.default_rng(rng))
    w = None if sample_weights is None else np.asarray(sample_weights, np.float64)

    def fit_fn(thetas, pass_idx):
        thetas = np.asarray(thetas, np.float64)
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qhat = clip01(
            np.asarray(evaluate(sol["basis"], sol["w"], sol["c"], X), np.float64)
        )
        return qhat, sol

    return fit_fn


# ---------------------------------------------------------------------------
# Halfset pair (for the graph-Laplacian resolution metric)
# ---------------------------------------------------------------------------

def halfset_committor_pair(X, in_A, in_B, split, make_fit, thetas):
    """Fit each half of ``split`` independently; evaluate both on the full X.

    ``make_fit(X_half, in_A_half, in_B_half)`` is a factory returning a
    ``fit_fn`` (e.g. a partial of :func:`make_lib_fit` or
    :func:`make_recovar_fit` over the config kwargs); both halves use the SAME
    ``thetas``, so the pair differs only in the data. The two committors are
    evaluated at every row of ``X`` (basins clamped with the full masks),
    which is what :func:`recovar.resolution.laplacian_shell_correlation`
    consumes as ``(q1, q2)``.

    Args:
        X: (N, d) full sample set.
        in_A, in_B: (N,) bool full basin masks.
        split: (i1, i2) index halves, as produced by
            ``recovar.splits.iid_split`` / ``block_split``.
        make_fit: factory ``(X_half, in_A_half, in_B_half) -> fit_fn``.
        thetas: (M, d) directions shared by both fits.

    Returns:
        (q1, q2): float64 (N,) committors on the full X, one per half.
    """
    X = np.asarray(X, np.float64)
    in_A = np.asarray(in_A, bool)
    in_B = np.asarray(in_B, bool)
    thetas = np.asarray(thetas, np.float64)
    qs = []
    for idx in split:
        idx = np.asarray(idx)
        fit_fn = make_fit(X[idx], in_A[idx], in_B[idx])
        _qh, handle = fit_fn(thetas, 0)
        qs.append(evaluate_handle(handle, X, in_A=in_A, in_B=in_B))
    q1, q2 = qs
    return q1, q2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def ids_protocol_metadata(ids_result):
    """JSON-safe provenance dict from an ``ids_fit`` / ``ids_path_exact2`` result.

    Extracts per-pass diagnostics (rank ``r``, CG iteration counts, the
    generalised spectrum ``lam``, the Horn rank thresholds when present) as
    plain Python ints / lists of floats, so the returned dict round-trips
    through ``json.dumps`` unchanged. Prefers the ``ids_history`` entry of an
    ``ids_path_exact2`` result (it carries the rank thresholds); falls back to
    ``history``, skipping the appended final-fit entry (it carries no
    spectrum). Tagged ``'sampler': 'ids_path'``.
    """
    hist = ids_result.get("ids_history") or ids_result.get("history") or []
    passes = []
    for entry in hist:
        if entry.get("lam") is None:
            continue
        record = {
            "r": int(entry["r"]),
            "n_cg": int(entry.get("n_cg", 0)),
            "n_cg_floor": int(entry.get("n_cg_floor", 0)),
            "lam": [float(v) for v in np.asarray(entry["lam"], float).ravel()],
        }
        if entry.get("rank_thresholds") is not None:
            record["rank_thresholds"] = [
                float(v) for v in np.asarray(entry["rank_thresholds"], float).ravel()
            ]
        passes.append(record)
    return {"sampler": "ids_path", "n_passes": len(passes), "passes": passes}
