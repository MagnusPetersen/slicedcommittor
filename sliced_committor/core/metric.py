"""Pull-back of a Cartesian diffusion shape onto a feature space.

The sliced committor's variational objective is the Dirichlet energy
``D[q] = INT rho (grad q)^T D (grad q)``, and for the sliced ansatz
``q_bar = sum_j w_j q_j(theta_j . u) + c`` this touches the estimator at exactly
one factor, ``theta_j^T D theta_k``. The library default asserts ``D = I`` in the
FEATURE space ``u``, i.e. that ``J J^T`` is a multiple of the identity for the
feature map ``u = u(x)``. For sin/cos torsions that is false: a torsion peaked at
``phi ~ -60 deg`` has ``<cos^2 phi> ~ 0.25`` against ``<sin^2 phi> ~ 0.75``, so
its sin and cos features carry diffusion in a ~3:1 ratio.

Given a Cartesian diffusion tensor ``D_cart = D_c * M0`` (``M0`` the SHAPE,
``D_c`` an unknown scalar) the correct constant feature-space tensor is the
pull-back

    Mbar = < J(x) M0 J(x)^T >_pi ,      J = du/dx .

Only the SHAPE of ``M0`` matters. The EBMC solve returns
``w = G^-1 (b-a) / (b-a)^T G^-1 (b-a)``, exactly invariant under ``G -> cG``; the
same cancellation holds in the rate bridge (``D_0 = D_s/G_Q``, ``D_q = D_0 G_q``).
Every entry point here therefore returns a ``trace/d = 1`` normalised matrix, and
``M0`` may be supplied up to scale.

Two caveats to that invariance, both measured and pinned in
``tests/test_metric_gram.py``:

* a float ``tikhonov`` (an ABSOLUTE ridge) does not co-scale with G, so a
  literal carried over from a ``D = I`` calibration changes the answer by
  orders of magnitude.
* ``tikhonov='halfset_eigen'`` is
  homogeneous of degree 1 on paper, but it reads a band correlation off the
  EIGENBASIS of the half-set Gram average, whose eigenvectors are ill-determined
  when the spectrum is closely spaced. Rescaling perturbs the rounding and the
  band SSNR moves a few percent (weights ~6%, committor ~5e-3, over six decades).

Hence: always pass the metric trace-normalised, which is what these functions
return by default. Do not switch ``normalize`` off outside tests.

The torsion case in closed form
-------------------------------
With ``B = dphi/dx`` (12 nonzeros per row -- a dihedral touches 4 atoms),
``g(x) = B M0 B^T`` the ``(n, n)`` Wilson-G-like torsion metric, and
``S(x) = [[diag(cos phi)], [-diag(sin phi)]]`` the ``(2n, n)`` sin/cos
differential,

    M_feat(x) = J M0 J^T = S g S^T
        M[sin_i, sin_j] =  cos_i cos_j g_ij
        M[cos_i, cos_j] =  sin_i sin_j g_ij
        M[sin_i, cos_j] = -cos_i sin_j g_ij

in the BLOCK layout ``u = [sin phi_1..n, cos phi_1..n]`` this package uses
throughout. ``M_feat(x)`` has rank <= n = d/2 at every point (the torus
constraint); the AVERAGE is generically full rank.

Nothing of size ``(N, n, n)`` or ``(N, n, 3A)`` is ever materialised. The four
per-pair moments accumulated below are length ``P`` (shared-atom torsion pairs),
which is all four blocks of ``S g S^T`` need, so the whole pass streams.

Sign convention
---------------
:class:`AngleSign` is REQUIRED and has no default. mdtraj's ``compute_phi`` /
``compute_psi`` return ``+phi_IUPAC`` (chignolin's 86-D features); the AIB9
(52-D) and villin (350-D) features of the paper were built with the negated
convention. A
mix-up flips every ``M[sin, cos]`` cross-block while leaving both diagonal blocks
-- hence the trace, the eigenvalue spectrum and the condition number -- untouched,
so no norm-based check can catch it. Only the reflection test can.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AngleSign",
    "MetricResult",
    "TorsionIncidence",
    "dihedral_and_grad",
    "remap_quads",
    "build_torsion_incidence",
    "sincos_pullback_metric",
    "normalize_metric",
    "metric_diagnostics",
]


class AngleSign(IntEnum):
    """Which dihedral convention the FEATURES were built in. No default."""

    IUPAC = 1
    """``+phi_IUPAC``: mdtraj ``compute_phi``/``compute_psi``/... (chignolin)."""

    SRC_NEGATED = -1
    """``-phi_IUPAC``: the convention of the paper's AIB9 (52-D) and villin
    (350-D) features."""


# ===========================================================================
# DIHEDRALS AND THEIR GRADIENTS
# ===========================================================================


def _dihedral_scalar(quad: jnp.ndarray, sign: float) -> jnp.ndarray:
    """Dihedral of one ``(4, 3)`` atom quadruple, in the ``sign`` convention.

    Expression-for-expression the same as ``src.domains.aib9.compute_dihedral``
    (which is ``sign = -1``), so autodiff of THIS function is consistent with the
    features that function produced -- the whole reason the gradient is taken by
    autodiff rather than from the closed form.
    """
    b1 = quad[1] - quad[0]
    b2 = quad[2] - quad[1]
    b3 = quad[3] - quad[2]
    c1 = jnp.cross(b2, b3)
    c2 = jnp.cross(b1, b2)
    p1 = jnp.sum(b1 * c1) * jnp.sum(b2 * b2) ** 0.5
    p2 = jnp.sum(c1 * c2)
    return sign * jnp.arctan2(p1, p2)


@functools.cache
def _dihedral_kernel(sign: float):
    """jitted ``(F, n, 4, 3) -> ((F, n), (F, n, 4, 3))`` value-and-gradient, per sign."""
    vg = jax.value_and_grad(lambda q: _dihedral_scalar(q, sign))
    return jax.jit(jax.vmap(jax.vmap(vg)))


def dihedral_and_grad(quads, *, angle_sign: AngleSign):
    """Per-frame dihedral angles and their local 4-atom gradients.

    Args:
        quads: ``(F, n, 4, 3)`` gathered atom positions, nm.
        angle_sign: :class:`AngleSign` of the convention the features use.

    Returns:
        ``(angles (F, n), grads (F, n, 4, 3))``, with
        ``grads[f, i, k] = d phi_i / d x_{quad[i, k]}``.

    The gradients satisfy ``sum_k grads[f, i, k] = 0`` (translation) and are
    SO(3)-equivariant, so any global rigid motion -- including the Kabsch
    superposition the loaders apply -- leaves the pulled-back metric unchanged.
    The lone singularity is a fully eclipsed geometry, ``|b1 x b2| -> 0``.
    """
    if int(angle_sign) not in (1, -1):
        raise ValueError(
            f"angle_sign must be AngleSign.IUPAC (+1) or AngleSign.SRC_NEGATED (-1); "
            f"got {angle_sign!r}. It has no default on purpose -- see the module docstring."
        )
    q = jnp.asarray(quads)
    if q.ndim != 4 or q.shape[2] != 4 or q.shape[3] != 3:
        raise ValueError(f"quads must be (F, n, 4, 3); got shape {tuple(q.shape)}")
    return _dihedral_kernel(float(int(angle_sign)))(q)


def remap_quads(quad_index):
    """Global atom ids -> ``(local ids (n, 4), the sorted global ids (A_touched,))``.

    ``touched[quad_local] == quad_index`` by construction.
    """
    qi = np.asarray(quad_index, dtype=np.int64)
    if qi.ndim != 2 or qi.shape[1] != 4:
        raise ValueError(f"quad_index must be (n, 4); got shape {qi.shape}")
    touched, local = np.unique(qi.reshape(-1), return_inverse=True)
    return local.reshape(qi.shape).astype(np.int32), touched.astype(np.int64)


# ===========================================================================
# SHARED-ATOM SPARSITY OF g = B M0 B^T
# ===========================================================================


@dataclass(frozen=True)
class TorsionIncidence:
    """Precomputed shared-atom sparsity of ``g``, built once per system.

    ``g_ij = sum_a m0_a (grad_a phi_i . grad_a phi_j)`` is nonzero only when
    torsions ``i`` and ``j`` share at least one atom. Storing that pattern turns
    the contraction into ``O(Q)`` work with ``Q`` a few thousand rows, instead of
    materialising ``B`` of shape ``(F, n, 3A)`` -- 0.9 GB per chunk at villin, and
    25x the flops.

    Attributes:
        pairs: ``(P, 2)`` torsion pairs ``(i, j)``, ``i <= j``, sharing an atom.
        pair_of_slot: ``(Q,)`` index into ``pairs`` for each slot-coincidence row.
        gi_index, gj_index: ``(Q,)`` flat indices into the ``(n*4,)`` quad-slot
            axis, i.e. ``torsion * 4 + slot`` for each side of the coincidence.
        atom: ``(Q,)`` the shared atom, in LOCAL indexing.
        n: number of torsions.
    """

    pairs: np.ndarray
    pair_of_slot: np.ndarray
    gi_index: np.ndarray
    gj_index: np.ndarray
    atom: np.ndarray
    n: int

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.atom.shape[0])


def build_torsion_incidence(quad_local, n_atoms_local: int | None = None) -> TorsionIncidence:
    """Shared-atom pattern of the ``(n, n)`` torsion metric ``g``.

    Args:
        quad_local: ``(n, 4)`` LOCAL atom indices (see :func:`remap_quads`).
        n_atoms_local: optional bound, used only for validation.
    """
    qi = np.asarray(quad_local, dtype=np.int64)
    if qi.ndim != 2 or qi.shape[1] != 4:
        raise ValueError(f"quad_local must be (n, 4); got shape {qi.shape}")
    if n_atoms_local is not None and (qi.min() < 0 or qi.max() >= int(n_atoms_local)):
        raise ValueError(
            f"quad_local indexes atoms [{qi.min()}, {qi.max()}] outside "
            f"[0, {int(n_atoms_local)}); did you forget remap_quads?"
        )
    n = int(qi.shape[0])

    by_atom: dict[int, list[tuple[int, int]]] = {}
    for i in range(n):
        for k in range(4):
            by_atom.setdefault(int(qi[i, k]), []).append((i, k))

    rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for a, slots in by_atom.items():
        for i, ki in slots:
            for j, kj in slots:
                if i <= j:
                    rows.setdefault((i, j), []).append((ki, kj, a))

    pairs = sorted(rows)
    pair_of_slot, gi_index, gj_index, atom = [], [], [], []
    for p, (i, j) in enumerate(pairs):
        for ki, kj, a in rows[(i, j)]:
            pair_of_slot.append(p)
            gi_index.append(i * 4 + ki)
            gj_index.append(j * 4 + kj)
            atom.append(a)

    return TorsionIncidence(
        pairs=np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        pair_of_slot=np.asarray(pair_of_slot, dtype=np.int32),
        gi_index=np.asarray(gi_index, dtype=np.int32),
        gj_index=np.asarray(gj_index, dtype=np.int32),
        atom=np.asarray(atom, dtype=np.int32),
        n=n,
    )


# ===========================================================================
# THE PULL-BACK
# ===========================================================================


@dataclass(frozen=True)
class MetricResult:
    """A pulled-back feature-space metric plus its provenance."""

    M: np.ndarray
    """``(d, d)`` symmetric PSD; ``trace/d = 1`` when ``normalize=True``."""
    M_raw_trace: float
    """``trace(Mbar)`` BEFORE normalisation. Provenance only -- the scale cancels."""
    n_frames: int
    weight_sum: float
    settings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _make_moment_kernel(incidence: TorsionIncidence):
    """jitted per-chunk accumulator of the four ``S g S^T`` pair moments."""
    gi_index = jnp.asarray(incidence.gi_index, dtype=jnp.int32)
    gj_index = jnp.asarray(incidence.gj_index, dtype=jnp.int32)
    atom = jnp.asarray(incidence.atom, dtype=jnp.int32)
    seg = jnp.asarray(incidence.pair_of_slot, dtype=jnp.int32)
    n_pairs = incidence.n_pairs
    pi = jnp.asarray(incidence.pairs[:, 0], dtype=jnp.int32)
    pj = jnp.asarray(incidence.pairs[:, 1], dtype=jnp.int32)
    n = incidence.n

    @jax.jit
    def kernel(grads, angles, w, atom_w):
        F = grads.shape[0]
        gflat = grads.reshape(F, n * 4, 3)
        gi = gflat[:, gi_index, :]
        gj = gflat[:, gj_index, :]
        dots = atom_w[atom][None, :] * jnp.sum(gi * gj, axis=-1)  # (F, Q)
        g_pair = jax.ops.segment_sum(dots.T, seg, num_segments=n_pairs).T  # (F, P)

        s = jnp.sin(angles)
        c = jnp.cos(angles)
        si, sj = s[:, pi], s[:, pj]
        ci, cj = c[:, pi], c[:, pj]
        wg = w[:, None] * g_pair
        return (
            jnp.sum(wg * ci * cj, axis=0),  # -> M[sin_i, sin_j]
            jnp.sum(wg * si * sj, axis=0),  # -> M[cos_i, cos_j]
            jnp.sum(wg * ci * sj, axis=0),  # -> -M[sin_i, cos_j]
            jnp.sum(wg * si * cj, axis=0),  # -> -M[sin_j, cos_i]
        )

    return kernel


def sincos_pullback_metric(
    xyz_chunks: Iterable[np.ndarray],
    quad_local,
    *,
    angle_sign: AngleSign,
    atom_weights=None,
    weight_chunks: Iterable[np.ndarray] | None = None,
    incidence: TorsionIncidence | None = None,
    normalize: bool = True,
    settings: dict | None = None,
) -> MetricResult:
    """``Mbar = < S g S^T >_pi`` for ``u = [sin phi_1..n, cos phi_1..n]``.

    Args:
        xyz_chunks: iterable yielding ``(F_c, A_local, 3)`` coordinate chunks, nm.
            Streamed; only the running moments are retained.
        quad_local: ``(n, 4)`` LOCAL atom indices (:func:`remap_quads`).
        angle_sign: :class:`AngleSign`. Required.
        atom_weights: optional ``(A_local,)`` per-atom diffusion shape
            (``D_cart = D_c diag(atom_weights) (x) I_3``); None means every
            atom diffuses alike, the natural choice in explicit solvent and
            what the paper uses. Only the shape matters (the scale cancels).
        weight_chunks: optional iterable of ``(F_c,)`` unnormalised weights (MBAR).
            ``None`` means uniform. Must be aligned chunk-for-chunk with
            ``xyz_chunks``.
        incidence: prebuilt :class:`TorsionIncidence`; built from ``quad_local``
            when omitted.
        normalize: return ``Mbar * d / trace(Mbar)``. The scale is unphysical
            (it cancels in both the committor and the rate), so leave this on.
        settings: content-addressed provenance recorded on the result.

    Returns:
        :class:`MetricResult` whose ``M`` is ``(2n, 2n)``, symmetric and PSD.
    """
    quad_local = np.asarray(quad_local, dtype=np.int64)
    inc = incidence if incidence is not None else build_torsion_incidence(quad_local)
    if inc.n != int(quad_local.shape[0]):
        raise ValueError(f"incidence.n={inc.n} does not match quad_local n={quad_local.shape[0]}")
    n = inc.n
    n_atoms = int(np.max(quad_local)) + 1
    m0 = (
        jnp.ones(n_atoms, dtype=jnp.float64)
        if atom_weights is None
        else jnp.asarray(np.asarray(atom_weights, dtype=np.float64))
    )
    if m0.ndim != 1:
        raise ValueError(f"atom_weights must be 1-D (A_local,); got shape {tuple(m0.shape)}")
    if not np.all(np.isfinite(np.asarray(m0))) or float(np.min(np.asarray(m0))) <= 0.0:
        raise ValueError("atom_weights must be finite and strictly positive.")

    kernel = _make_moment_kernel(inc)
    quad_j = jnp.asarray(quad_local, dtype=jnp.int32)

    acc = [np.zeros(inc.n_pairs, dtype=np.float64) for _ in range(4)]
    n_frames = 0
    weight_sum = 0.0
    w_iter: Iterator | None = iter(weight_chunks) if weight_chunks is not None else None

    for xyz in xyz_chunks:
        x = jnp.asarray(np.asarray(xyz, dtype=np.float64))
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"each xyz chunk must be (F, A_local, 3); got {tuple(x.shape)}")
        F = int(x.shape[0])
        if F == 0:
            continue
        if w_iter is None:
            w = jnp.ones(F, dtype=x.dtype)
        else:
            w = jnp.asarray(np.asarray(next(w_iter), dtype=np.float64)).reshape(-1)
            if int(w.shape[0]) != F:
                raise ValueError(f"weight chunk has {int(w.shape[0])} entries for {F} frames")

        angles, grads = dihedral_and_grad(x[:, quad_j, :], angle_sign=angle_sign)
        parts = kernel(grads, angles, w, m0)
        for t, p in enumerate(parts):
            acc[t] += np.asarray(p, dtype=np.float64)
        n_frames += F
        weight_sum += float(jnp.sum(w))

    if n_frames == 0:
        raise ValueError("xyz_chunks yielded no frames.")
    if weight_sum <= 0.0:
        raise ValueError(f"total weight is {weight_sum}; weights must be positive.")
    m_cc, m_ss, m_cs, m_sc = (a / weight_sum for a in acc)

    i = inc.pairs[:, 0].astype(np.int64)
    j = inc.pairs[:, 1].astype(np.int64)
    SS = np.zeros((n, n), dtype=np.float64)  # sin-sin block: <cos_i cos_j g_ij>
    CC = np.zeros((n, n), dtype=np.float64)  # cos-cos block: <sin_i sin_j g_ij>
    SC = np.zeros((n, n), dtype=np.float64)  # sin-cos block: -<cos_i sin_j g_ij>
    SS[i, j] = m_cc
    SS[j, i] = m_cc
    CC[i, j] = m_ss
    CC[j, i] = m_ss
    SC[i, j] = -m_cs
    SC[j, i] = -m_sc

    M = np.empty((2 * n, 2 * n), dtype=np.float64)
    M[:n, :n] = SS
    M[n:, n:] = CC
    M[:n, n:] = SC
    M[n:, :n] = SC.T
    M = 0.5 * (M + M.T)

    raw_trace = float(np.trace(M))
    if not np.isfinite(raw_trace) or raw_trace <= 0.0:
        raise ValueError(
            f"pulled-back metric has trace {raw_trace}; expected a positive finite value."
        )
    if normalize:
        M = normalize_metric(M)

    return MetricResult(
        M=M,
        M_raw_trace=raw_trace,
        n_frames=n_frames,
        weight_sum=weight_sum,
        settings=dict(settings or {}),
        diagnostics=metric_diagnostics(M, n_torsions=n),
    )


# ===========================================================================
# POST-PROCESSING AND DIAGNOSTICS
# ===========================================================================


def normalize_metric(M) -> np.ndarray:
    """``M * d / trace(M)`` -- the numerical statement that only the shape matters."""
    A = np.asarray(M, dtype=np.float64)
    tr = float(np.trace(A))
    if not np.isfinite(tr) or tr <= 0.0:
        raise ValueError(f"cannot normalise a metric with trace {tr}.")
    return A * (A.shape[0] / tr)


def metric_diagnostics(M, *, n_torsions: int | None = None) -> dict:
    """Conditioning and block structure of a pulled-back metric.

    ``sincos_diag_ratio`` is the per-torsion ``M[cos_i, cos_i] / M[sin_i, sin_i]``.
    Under ``D = I`` it is 1 by fiat; its spread is the anisotropy the identity
    metric throws away.
    """
    A = np.asarray(M, dtype=np.float64)
    d = A.shape[0]
    lam = np.linalg.eigvalsh(0.5 * (A + A.T))
    lam_max = float(lam[-1])
    out = {
        "dim": int(d),
        "eigval_min": float(lam[0]),
        "eigval_max": lam_max,
        "cond": float(lam_max / lam[0]) if lam[0] > 0 else float("inf"),
        "rank_1e10": int(np.sum(lam > 1e-10 * max(lam_max, 1.0))),
        "n_small": int(np.sum(lam < 1e-6 * max(lam_max, 1.0))),
        "trace_over_d": float(np.trace(A) / d),
        "symmetry_residual": float(np.linalg.norm(A - A.T) / max(np.linalg.norm(A), 1e-300)),
        "frob_rel_to_identity": float(np.linalg.norm(A - np.eye(d)) / np.linalg.norm(np.eye(d))),
        "offdiag_frobenius_fraction": float(
            np.linalg.norm(A - np.diag(np.diag(A))) / max(np.linalg.norm(A), 1e-300)
        ),
    }
    if n_torsions is not None and d == 2 * int(n_torsions):
        n = int(n_torsions)
        dg = np.diag(A)
        ratio = dg[n:] / np.where(dg[:n] > 0, dg[:n], np.nan)
        out["sincos_diag_ratio_min"] = float(np.nanmin(ratio))
        out["sincos_diag_ratio_median"] = float(np.nanmedian(ratio))
        out["sincos_diag_ratio_max"] = float(np.nanmax(ratio))
        nrm = max(np.linalg.norm(A), 1e-300)
        out["block_frobenius"] = {
            "ss": float(np.linalg.norm(A[:n, :n]) / nrm),
            "cc": float(np.linalg.norm(A[n:, n:]) / nrm),
            "sc": float(np.linalg.norm(A[:n, n:]) / nrm),
        }
    return out
