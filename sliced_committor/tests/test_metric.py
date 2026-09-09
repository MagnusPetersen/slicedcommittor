"""Pull-back of a Cartesian diffusion shape onto the sin/cos torsion features.

Correctness of ``Mbar = <J M0 J^T>`` itself: the dihedral gradient, the
streaming accumulator, and the invariances that make the construction well
posed. The Gram-side consequences live in ``test_metric_gram.py``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor.core.metric import (
    AngleSign,
    build_torsion_incidence,
    dihedral_and_grad,
    metric_diagnostics,
    normalize_metric,
    remap_quads,
    sincos_pullback_metric,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def random_system(n_atoms=12, n_tors=5, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, n_atoms, 3)) * 0.15
    quads = np.stack([rng.choice(n_atoms, size=4, replace=False) for _ in range(n_tors)])
    quad_local, touched = remap_quads(quads)
    return xyz[:, touched, :], quad_local, touched, rng


def dihedral_numpy(x, quad):
    """Reference dihedral in the SRC_NEGATED convention, plain numpy."""
    a, b, c, d = x[quad[0]], x[quad[1]], x[quad[2]], x[quad[3]]
    b1, b2, b3 = b - a, c - b, d - c
    c1, c2 = np.cross(b2, b3), np.cross(b1, b2)
    return -np.arctan2(np.dot(b1, c1) * np.dot(b2, b2) ** 0.5, np.dot(c1, c2))


def dense_pullback(x, quad_local, atom_w, angle_sign):
    """Brute-force ``J M0 J^T`` for one frame via an explicit dense Jacobian."""
    n = quad_local.shape[0]
    ang, grad = dihedral_and_grad(x[None][:, quad_local, :], angle_sign=angle_sign)
    ang, grad = np.asarray(ang)[0], np.asarray(grad)[0]
    J = np.zeros((2 * n, x.shape[0] * 3))
    for i in range(n):
        for k in range(4):
            sl = slice(quad_local[i, k] * 3, quad_local[i, k] * 3 + 3)
            J[i, sl] += np.cos(ang[i]) * grad[i, k]
            J[n + i, sl] += -np.sin(ang[i]) * grad[i, k]
    return J @ np.diag(np.repeat(atom_w, 3)) @ J.T


def torsion_g(grad, atom_w=None):
    """``g_ii = sum_a w_a |grad_a phi_i|^2`` for a single torsion, from its four gradients."""
    g = np.asarray(grad)  # (F, 1, 4, 3)
    w = np.ones(4) if atom_w is None else atom_w
    return np.einsum("fk,fkc,fkc->f", np.broadcast_to(w, (*g.shape[:1], 4)), g[:, 0], g[:, 0])


# ---------------------------------------------------------------------------
# the gradient
# ---------------------------------------------------------------------------
def test_dihedral_matches_the_kernel_that_built_the_features():
    """Bit-exact against the frozen output of the research kernel the paper's
    AIB9/villin features were built with (``golden/dihedral_reference.npz``)."""
    ref = np.load(os.path.join(GOLDEN, "dihedral_reference.npz"))
    got, _ = dihedral_and_grad(
        jnp.asarray(ref["xyz"])[:, ref["quads"], :], angle_sign=AngleSign.SRC_NEGATED
    )
    np.testing.assert_array_equal(np.asarray(got), ref["dihedral_src_negated"])


def test_angle_sign_flips_the_angle():
    xyz, ql, _, _ = random_system(seed=1)
    a_neg, _ = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    a_pos, _ = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.IUPAC)
    np.testing.assert_allclose(np.asarray(a_neg), -np.asarray(a_pos), atol=0)


def test_angle_sign_is_required_and_validated():
    xyz, ql, _, _ = random_system(seed=1)
    with pytest.raises(ValueError, match="angle_sign"):
        dihedral_and_grad(xyz[:, ql, :], angle_sign=0)
    with pytest.raises(ValueError, match=r"\(F, n, 4, 3\)"):
        dihedral_and_grad(xyz, angle_sign=AngleSign.IUPAC)


def test_dihedral_gradient_matches_finite_differences():
    xyz, ql, _, _ = random_system(seed=2)
    x = xyz[0]
    _, grad = dihedral_and_grad(x[None][:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    grad = np.asarray(grad)[0]
    h = 1e-6
    worst = 0.0
    for i in range(ql.shape[0]):
        for k in range(4):
            a = ql[i, k]
            for comp in range(3):
                xp, xm = x.copy(), x.copy()
                xp[a, comp] += h
                xm[a, comp] -= h
                fd = (dihedral_numpy(xp, ql[i]) - dihedral_numpy(xm, ql[i])) / (2 * h)
                worst = max(worst, abs(fd - grad[i, k, comp]))
    assert worst < 1e-6, worst


def test_gradient_has_no_translation_component():
    """``sum_k grad_k phi = 0``: a rigid shift cannot change a dihedral."""
    xyz, ql, _, _ = random_system(seed=3)
    _, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    assert float(jnp.max(jnp.abs(jnp.sum(grad, axis=2)))) < 1e-12


def test_gradient_is_rotation_equivariant():
    xyz, ql, _, rng = random_system(seed=5)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = rng.normal(size=3)
    a0, g0 = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    a1, g1 = dihedral_and_grad((xyz @ Q.T + t)[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    assert float(jnp.max(jnp.abs(a1 - a0))) < 1e-12
    assert float(jnp.max(jnp.abs(g1 - g0 @ Q.T))) < 1e-10


def test_rigid_motion_leaves_the_metric_invariant():
    xyz, ql, _, rng = random_system(seed=6)
    w = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    kw = dict(angle_sign=AngleSign.SRC_NEGATED, atom_weights=w)
    base = sincos_pullback_metric([xyz], ql, **kw).M
    moved = sincos_pullback_metric([xyz @ Q.T + rng.normal(size=3)], ql, **kw).M
    np.testing.assert_allclose(moved, base, atol=1e-10)


def test_reflection_flips_only_the_cross_block():
    """The only structural check that catches a sign-convention mix-up."""
    xyz, ql, _, rng = random_system(seed=7)
    n = ql.shape[0]
    kw = dict(
        angle_sign=AngleSign.SRC_NEGATED, atom_weights=rng.uniform(0.5, 2.0, size=xyz.shape[1])
    )
    M = sincos_pullback_metric([xyz], ql, **kw).M
    Mm = sincos_pullback_metric([-xyz], ql, **kw).M
    np.testing.assert_allclose(Mm[:n, :n], M[:n, :n], atol=1e-12)
    np.testing.assert_allclose(Mm[n:, n:], M[n:, n:], atol=1e-12)
    np.testing.assert_allclose(Mm[:n, n:], -M[:n, n:], atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(Mm), np.linalg.eigvalsh(M), atol=1e-10)


# ---------------------------------------------------------------------------
# the pull-back
# ---------------------------------------------------------------------------
def test_accumulator_equals_dense_jacobian_pullback():
    xyz, ql, _, rng = random_system(n_frames=25, seed=8)
    w = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    dense = np.mean(
        [dense_pullback(xyz[f], ql, w, AngleSign.SRC_NEGATED) for f in range(xyz.shape[0])], axis=0
    )
    got = sincos_pullback_metric(
        [xyz], ql, angle_sign=AngleSign.SRC_NEGATED, atom_weights=w, normalize=False
    ).M
    np.testing.assert_allclose(got, dense, rtol=1e-11, atol=1e-13)


def test_default_atom_weights_are_ones():
    xyz, ql, _, _ = random_system(seed=8)
    a = sincos_pullback_metric([xyz], ql, angle_sign=AngleSign.IUPAC).M
    b = sincos_pullback_metric(
        [xyz], ql, angle_sign=AngleSign.IUPAC, atom_weights=np.ones(xyz.shape[1])
    ).M
    np.testing.assert_array_equal(a, b)


def test_per_frame_metric_has_rank_n_but_the_mean_is_full_rank():
    xyz, ql, _, _ = random_system(n_atoms=14, n_tors=6, n_frames=200, seed=10)
    n = ql.shape[0]
    single = dense_pullback(xyz[0], ql, np.ones(xyz.shape[1]), AngleSign.SRC_NEGATED)
    assert np.linalg.matrix_rank(single, tol=1e-9 * np.linalg.norm(single)) == n
    M = sincos_pullback_metric([xyz], ql, angle_sign=AngleSign.SRC_NEGATED).M
    assert np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M)) == 2 * n


def test_rigid_rotor_is_exact():
    """Closed form: one bond with ``b1 _|_ b2 _|_ b3`` gives a constant
    ``g = 2/|b1|^2 + 2/|b3|^2`` and ``Mbar = (g/2) I_2`` exactly."""
    n_grid = 16
    t = 2 * np.pi * np.arange(n_grid) / n_grid
    x0 = np.array([1.0, 0.7, 0.0])
    xyz = np.zeros((n_grid, 4, 3))
    xyz[:, 0] = x0
    xyz[:, 2] = np.array([0.0, 0.0, 1.0])
    xyz[:, 3] = xyz[:, 2] + np.stack([np.cos(t), np.sin(t), np.zeros(n_grid)], axis=1)
    ql = np.array([[0, 1, 2, 3]], dtype=np.int32)
    g_exact = 2.0 / float(x0 @ x0) + 2.0
    ang, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    spacing = np.diff(np.unwrap(np.asarray(ang)[:, 0]))
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-12)
    np.testing.assert_allclose(torsion_g(grad), g_exact, rtol=1e-12)
    res = sincos_pullback_metric([xyz], ql, angle_sign=AngleSign.SRC_NEGATED, normalize=False)
    np.testing.assert_allclose(res.M, 0.5 * g_exact * np.eye(2), atol=1e-12)
    np.testing.assert_allclose(normalize_metric(res.M), np.eye(2), atol=1e-12)


def test_torsion_metric_varies_along_a_generic_bond_rotation():
    """The control for the rigid-rotor case: drop the perpendicularity and g moves."""
    n_grid = 16
    t = 2 * np.pi * np.arange(n_grid) / n_grid
    xyz = np.zeros((n_grid, 4, 3))
    xyz[:, 0] = np.array([1.0, 0.7, 0.0])
    xyz[:, 2] = np.array([0.0, 0.0, 1.0])
    xyz[:, 3] = xyz[:, 2] + np.stack([np.cos(t), np.sin(t), 0.3 * np.ones(n_grid)], axis=1)
    ql = np.array([[0, 1, 2, 3]], dtype=np.int32)
    _, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    g = torsion_g(grad)
    assert np.ptp(g) / g.mean() > 0.1


# ---------------------------------------------------------------------------
# streaming, weights, scale
# ---------------------------------------------------------------------------
def test_chunking_is_exact():
    xyz, ql, _, _ = random_system(n_frames=97, seed=12)
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    one = sincos_pullback_metric([xyz], ql, **kw).M
    many = sincos_pullback_metric([xyz[:13], xyz[13:60], xyz[60:]], ql, **kw).M
    np.testing.assert_allclose(many, one, atol=1e-12)


def test_weights_are_honoured():
    xyz, ql, _, _ = random_system(n_frames=20, seed=13)
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    w = np.zeros(20)
    w[:5] = 1.0
    weighted = sincos_pullback_metric([xyz], ql, weight_chunks=[w], **kw).M
    plain = sincos_pullback_metric([xyz[:5]], ql, **kw).M
    np.testing.assert_allclose(weighted, plain, atol=1e-12)


def test_only_the_shape_of_the_atom_weights_matters():
    xyz, ql, _, rng = random_system(seed=14)
    w = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    base = sincos_pullback_metric([xyz], ql, angle_sign=AngleSign.SRC_NEGATED, atom_weights=w).M
    for c in (1e-6, 1e6):
        got = sincos_pullback_metric(
            [xyz], ql, angle_sign=AngleSign.SRC_NEGATED, atom_weights=c * w
        ).M
        np.testing.assert_allclose(got, base, atol=1e-12)


def test_pullback_input_validation():
    xyz, ql, _, _ = random_system(seed=15)
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    with pytest.raises(ValueError, match="no frames"):
        sincos_pullback_metric([], ql, **kw)
    with pytest.raises(ValueError, match="strictly positive"):
        sincos_pullback_metric([xyz], ql, atom_weights=-np.ones(xyz.shape[1]), **kw)
    with pytest.raises(ValueError, match="1-D"):
        sincos_pullback_metric([xyz], ql, atom_weights=np.eye(xyz.shape[1]), **kw)
    with pytest.raises(ValueError, match=r"\(F, A_local, 3\)"):
        sincos_pullback_metric([xyz[:, :, :2]], ql, **kw)
    with pytest.raises(ValueError, match="entries for"):
        sincos_pullback_metric([xyz], ql, weight_chunks=[np.ones(3)], **kw)


def test_remap_quads_roundtrip():
    rng = np.random.default_rng(16)
    quads = np.stack([rng.choice(40, size=4, replace=False) for _ in range(7)])
    local, touched = remap_quads(quads)
    np.testing.assert_array_equal(touched[local], quads)
    assert local.max() < len(touched)
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        remap_quads(np.zeros((3, 5), dtype=int))


def test_incidence_rejects_unmapped_indices():
    with pytest.raises(ValueError, match="remap_quads"):
        build_torsion_incidence(np.array([[0, 1, 2, 99]]), n_atoms_local=4)


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def test_normalize():
    rng = np.random.default_rng(17)
    B = rng.normal(size=(6, 6))
    assert np.trace(normalize_metric(B @ B.T)) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="trace"):
        normalize_metric(np.zeros((3, 3)))


def test_diagnostics_report_the_sincos_anisotropy():
    xyz, ql, _, _ = random_system(n_tors=6, n_frames=300, seed=18)
    d = sincos_pullback_metric([xyz], ql, angle_sign=AngleSign.SRC_NEGATED).diagnostics
    assert d["trace_over_d"] == pytest.approx(1.0)
    assert d["symmetry_residual"] < 1e-14
    assert d["eigval_min"] > -1e-12 and d["cond"] > 1.0
    assert set(d["block_frobenius"]) == {"ss", "cc", "sc"}
    assert d["sincos_diag_ratio_min"] <= d["sincos_diag_ratio_median"] <= d["sincos_diag_ratio_max"]
    assert d["frob_rel_to_identity"] > 0.05


def test_diagnostics_on_the_identity():
    d = metric_diagnostics(np.eye(8), n_torsions=4)
    assert d["cond"] == pytest.approx(1.0)
    assert d["frob_rel_to_identity"] == pytest.approx(0.0)
    assert d["offdiag_frobenius_fraction"] == pytest.approx(0.0)
    assert d["sincos_diag_ratio_median"] == pytest.approx(1.0)
