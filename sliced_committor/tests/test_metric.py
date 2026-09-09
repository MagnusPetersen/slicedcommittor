"""Pull-back of a Cartesian diffusion shape onto a feature space.

Correctness of ``Mbar = <J M0 J^T>`` itself: the dihedral gradient, the
``S g S^T`` identity, the streaming accumulator, and the invariances that make
the construction well posed. The Gram-side consequences live in
``test_metric_gram.py``.
"""

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
    shrink_metric,
    sincos_pullback_metric,
    torsion_g_matrix,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def random_system(n_atoms=12, n_tors=5, n_frames=60, seed=0):
    """Random coordinates plus a random (but valid) set of dihedral quadruples."""
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n_frames, n_atoms, 3)) * 0.15
    quads = np.stack([rng.choice(n_atoms, size=4, replace=False) for _ in range(n_tors)])
    quad_local, touched = remap_quads(quads)
    return xyz[:, touched, :], quad_local, touched, rng


def dihedral_numpy(x, quad):
    """Reference dihedral in the SRC_NEGATED convention, in plain numpy."""
    a, b, c, d = x[quad[0]], x[quad[1]], x[quad[2]], x[quad[3]]
    b1, b2, b3 = b - a, c - b, d - c
    c1, c2 = np.cross(b2, b3), np.cross(b1, b2)
    return -np.arctan2(np.dot(b1, c1) * np.dot(b2, b2) ** 0.5, np.dot(c1, c2))


def dense_pullback(x, quad_local, m0_atom, angle_sign):
    """Brute-force ``J M0 J^T`` for one frame, via an explicit dense Jacobian."""
    n = quad_local.shape[0]
    ang, grad = dihedral_and_grad(x[None][:, quad_local, :], angle_sign=angle_sign)
    ang, grad = np.asarray(ang)[0], np.asarray(grad)[0]
    J = np.zeros((2 * n, x.shape[0] * 3))
    for i in range(n):
        for k in range(4):
            sl = slice(quad_local[i, k] * 3, quad_local[i, k] * 3 + 3)
            J[i, sl] += np.cos(ang[i]) * grad[i, k]
            J[n + i, sl] += -np.sin(ang[i]) * grad[i, k]
    return J @ np.diag(np.repeat(m0_atom, 3)) @ J.T


# ---------------------------------------------------------------------------
# the gradient
# ---------------------------------------------------------------------------
def test_dihedral_matches_the_kernel_that_built_the_features():
    """Bit-exact against ``src.domains.aib9.compute_dihedral``.

    The whole reason the gradient is taken by autodiff rather than from the
    closed form: J is then guaranteed consistent with the features, sign
    convention included.
    """
    src = pytest.importorskip("src.domains.aib9", reason="repo src/ not importable")
    rng = np.random.default_rng(3)
    xyz = rng.normal(size=(40, 14, 3)) * 0.15
    quads = np.stack([rng.choice(14, size=4, replace=False) for _ in range(6)])
    ref = np.asarray(src.compute_dihedral(jnp.asarray(xyz), jnp.asarray(quads)))
    got, _ = dihedral_and_grad(jnp.asarray(xyz)[:, quads, :], angle_sign=AngleSign.SRC_NEGATED)
    np.testing.assert_array_equal(np.asarray(got), ref)


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
    xyz, ql, _, _ = random_system(seed=4)
    _, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    assert float(jnp.max(jnp.abs(jnp.sum(grad, axis=2)))) < 1e-12


def test_gradient_is_rotation_equivariant():
    """``phi(Rx+t) = phi(x)`` and ``grad phi(Rx+t) = R grad phi(x)``.

    This is why the Kabsch superposition the loaders apply cannot perturb Mbar.
    """
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
    """The end-to-end consequence: Mbar is blind to alignment."""
    xyz, ql, _, rng = random_system(seed=6)
    m0 = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    base = sincos_pullback_metric([xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED).M
    moved = sincos_pullback_metric(
        [xyz @ Q.T + rng.normal(size=3)], ql, m0, angle_sign=AngleSign.SRC_NEGATED
    ).M
    np.testing.assert_allclose(moved, base, atol=1e-10)


def test_reflection_flips_only_the_cross_block():
    """Under ``x -> -x``, ``g`` is invariant and ``M[sin, cos]`` changes sign.

    The ONLY structural check that catches a sign-convention mix-up: the trace,
    the eigenvalue spectrum and the condition number are all blind to it, because
    the mirror leaves both diagonal blocks untouched.
    """
    xyz, ql, _, rng = random_system(seed=7)
    n = ql.shape[0]
    m0 = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    M = sincos_pullback_metric([xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED).M
    Mm = sincos_pullback_metric([-xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED).M
    np.testing.assert_allclose(Mm[:n, :n], M[:n, :n], atol=1e-12)
    np.testing.assert_allclose(Mm[n:, n:], M[n:, n:], atol=1e-12)
    np.testing.assert_allclose(Mm[:n, n:], -M[:n, n:], atol=1e-12)
    # and the blind diagnostics really are blind, so the test above is load-bearing
    np.testing.assert_allclose(
        np.linalg.eigvalsh(Mm), np.linalg.eigvalsh(M), atol=1e-10
    )


# ---------------------------------------------------------------------------
# the pull-back
# ---------------------------------------------------------------------------
def test_accumulator_equals_dense_jacobian_pullback():
    """``<S g S^T>`` against an explicit dense ``J M0 J^T``, frame by frame."""
    xyz, ql, _, rng = random_system(n_frames=25, seed=8)
    m0 = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    dense = np.mean(
        [dense_pullback(xyz[f], ql, m0, AngleSign.SRC_NEGATED) for f in range(xyz.shape[0])],
        axis=0,
    )
    got = sincos_pullback_metric(
        [xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED, normalize=False
    ).M
    np.testing.assert_allclose(got, dense, rtol=1e-11, atol=1e-13)


def test_torsion_g_matrix_equals_dense_B_M0_Bt():
    xyz, ql, _, rng = random_system(n_frames=5, seed=9)
    n, A = ql.shape[0], xyz.shape[1]
    m0 = rng.uniform(0.5, 2.0, size=A)
    inc = build_torsion_incidence(ql, A)
    _, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    got = np.asarray(torsion_g_matrix(grad, inc, m0))
    grad = np.asarray(grad)
    for f in range(xyz.shape[0]):
        B = np.zeros((n, A * 3))
        for i in range(n):
            for k in range(4):
                B[i, ql[i, k] * 3 : ql[i, k] * 3 + 3] += grad[f, i, k]
        np.testing.assert_allclose(got[f], B @ np.diag(np.repeat(m0, 3)) @ B.T, rtol=1e-11)


def test_per_frame_metric_has_rank_n_but_the_mean_is_full_rank():
    """The torus constraint caps the per-frame rank at n = d/2; averaging lifts it."""
    xyz, ql, _, rng = random_system(n_atoms=14, n_tors=6, n_frames=200, seed=10)
    n = ql.shape[0]
    m0 = np.ones(xyz.shape[1])
    single = dense_pullback(xyz[0], ql, m0, AngleSign.SRC_NEGATED)
    assert np.linalg.matrix_rank(single, tol=1e-9 * np.linalg.norm(single)) == n
    M = sincos_pullback_metric([xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED).M
    assert np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M)) == 2 * n


def test_rigid_rotor_is_exact():
    """Closed-form case: one rotatable bond with ``b1 _|_ b2 _|_ b3``.

    In general ``g`` is NOT constant along a bond rotation -- the two central
    gradients are combinations of the terminal ones,

        grad_1 = ((b1.b2)/|b2|^2 - 1) grad_0 - ((b3.b2)/|b2|^2) grad_3

    so ``sum_a |grad_a phi|^2`` picks up a ``grad_0 . grad_3 ~ c1 . c2`` term that
    varies with phi (a ~20% swing for a generic geometry). Choosing
    ``b1 . b2 = b3 . b2 = 0`` collapses that: ``grad_1 = -grad_0``,
    ``grad_2 = -grad_3``, and with ``|grad_0| = 1/|b1|``, ``|grad_3| = 1/|b3|``,

        g = 2/|b1|^2 + 2/|b3|^2      (constant, closed form)

    while rotating ``b3`` in the plane normal to ``b2`` sweeps phi uniformly. Then

        Mbar = g * [[<cos^2>, -<cos sin>], [-<sin cos>, <sin^2>]] = (g/2) I_2

    exactly. This pins the gradients, the ``g`` contraction, both diagonal blocks,
    the cross-block SIGN and the weighted average against a closed form.
    """
    n_grid = 16
    t = 2 * np.pi * np.arange(n_grid) / n_grid
    x0 = np.array([1.0, 0.7, 0.0])
    x1 = np.zeros(3)
    x2 = np.array([0.0, 0.0, 1.0])  # b2 along +z
    xyz = np.zeros((n_grid, 4, 3))
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = x0, x1, x2
    # b3 stays in the xy-plane, so b1 . b2 = b3 . b2 = 0 exactly
    xyz[:, 3] = x2 + np.stack([np.cos(t), np.sin(t), np.zeros(n_grid)], axis=1)
    ql = np.array([[0, 1, 2, 3]], dtype=np.int32)
    m0 = np.ones(4)

    b1_sq, b3_sq = float(x0 @ x0), 1.0
    g_exact = 2.0 / b1_sq + 2.0 / b3_sq

    ang, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    spacing = np.diff(np.unwrap(np.asarray(ang)[:, 0]))
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-12)  # uniform in phi

    inc = build_torsion_incidence(ql, 4)
    g = np.asarray(torsion_g_matrix(grad, inc, m0))[:, 0, 0]
    np.testing.assert_allclose(g, g_exact, rtol=1e-12)

    res = sincos_pullback_metric(
        [xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED, normalize=False
    )
    np.testing.assert_allclose(res.M, 0.5 * g_exact * np.eye(2), atol=1e-12)
    np.testing.assert_allclose(normalize_metric(res.M), np.eye(2), atol=1e-12)


def test_torsion_metric_varies_along_a_generic_bond_rotation():
    """The control for the test above: drop the perpendicularity and g moves.

    Guards against the orthogonal construction silently becoming vacuous (e.g. if
    the central-atom gradients were ever dropped, g would be constant either way).
    """
    n_grid = 16
    t = 2 * np.pi * np.arange(n_grid) / n_grid
    xyz = np.zeros((n_grid, 4, 3))
    xyz[:, 0] = np.array([1.0, 0.7, 0.0])
    xyz[:, 2] = np.array([0.0, 0.0, 1.0])
    xyz[:, 3] = xyz[:, 2] + np.stack(
        [np.cos(t), np.sin(t), 0.3 * np.ones(n_grid)], axis=1
    )  # b3 . b2 = 0.3 != 0
    ql = np.array([[0, 1, 2, 3]], dtype=np.int32)
    inc = build_torsion_incidence(ql, 4)
    _, grad = dihedral_and_grad(xyz[:, ql, :], angle_sign=AngleSign.SRC_NEGATED)
    g = np.asarray(torsion_g_matrix(grad, inc, np.ones(4)))[:, 0, 0]
    assert np.ptp(g) / g.mean() > 0.1


def test_linear_feature_map_recovers_C_M0_Ct():
    """A constant Jacobian is the trig-free control on the accumulator."""
    rng = np.random.default_rng(11)
    A, n = 9, 4
    xyz = rng.normal(size=(30, A, 3)) * 0.2
    quads = np.stack([rng.choice(A, size=4, replace=False) for _ in range(n)])
    ql, touched = remap_quads(quads)
    x = xyz[:, touched, :]
    m0 = rng.uniform(0.5, 2.0, size=len(touched))
    inc = build_torsion_incidence(ql, len(touched))
    _, grad = dihedral_and_grad(x[:, ql, :], angle_sign=AngleSign.IUPAC)
    g = np.asarray(torsion_g_matrix(grad, inc, m0))
    # g is the pull-back of M0 through the ANGLE map alone; check it against the
    # dense form on every frame (no sin/cos anywhere).
    grad = np.asarray(grad)
    for f in (0, 7, 29):
        B = np.zeros((n, len(touched) * 3))
        for i in range(n):
            for k in range(4):
                B[i, ql[i, k] * 3 : ql[i, k] * 3 + 3] += grad[f, i, k]
        np.testing.assert_allclose(g[f], B @ np.diag(np.repeat(m0, 3)) @ B.T, rtol=1e-11)


# ---------------------------------------------------------------------------
# streaming, weights, scale
# ---------------------------------------------------------------------------
def test_chunking_is_exact():
    xyz, ql, _, rng = random_system(n_frames=97, seed=12)
    m0 = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    one = sincos_pullback_metric([xyz], ql, m0, **kw).M
    many = sincos_pullback_metric([xyz[:13], xyz[13:60], xyz[60:]], ql, m0, **kw).M
    np.testing.assert_allclose(many, one, atol=1e-12)


def test_weights_are_honoured():
    """A weighted pass over all frames equals an unweighted pass over duplicates."""
    xyz, ql, _, rng = random_system(n_frames=20, seed=13)
    m0 = np.ones(xyz.shape[1])
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    w = np.zeros(20)
    w[:5] = 1.0
    weighted = sincos_pullback_metric([xyz], ql, m0, weight_chunks=[w], **kw).M
    plain = sincos_pullback_metric([xyz[:5]], ql, m0, **kw).M
    np.testing.assert_allclose(weighted, plain, atol=1e-12)


def test_only_the_shape_of_M0_matters():
    """``Mbar`` is scale-free by construction -- the units question does not arise."""
    xyz, ql, _, rng = random_system(seed=14)
    m0 = rng.uniform(0.5, 2.0, size=xyz.shape[1])
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    base = sincos_pullback_metric([xyz], ql, m0, **kw).M
    for c in (1e-6, 1e6):
        np.testing.assert_allclose(
            sincos_pullback_metric([xyz], ql, c * m0, **kw).M, base, atol=1e-12
        )


def test_pullback_input_validation():
    xyz, ql, _, rng = random_system(seed=15)
    m0 = np.ones(xyz.shape[1])
    kw = dict(angle_sign=AngleSign.SRC_NEGATED)
    with pytest.raises(ValueError, match="no frames"):
        sincos_pullback_metric([], ql, m0, **kw)
    with pytest.raises(ValueError, match="strictly positive"):
        sincos_pullback_metric([xyz], ql, -m0, **kw)
    with pytest.raises(ValueError, match="1-D"):
        sincos_pullback_metric([xyz], ql, np.eye(xyz.shape[1]), **kw)
    with pytest.raises(ValueError, match=r"\(F, A_local, 3\)"):
        sincos_pullback_metric([xyz[:, :, :2]], ql, m0, **kw)
    with pytest.raises(ValueError, match="entries for"):
        sincos_pullback_metric([xyz], ql, m0, weight_chunks=[np.ones(3)], **kw)


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
def test_normalize_and_shrink():
    rng = np.random.default_rng(17)
    B = rng.normal(size=(6, 6))
    M = B @ B.T
    Mn = normalize_metric(M)
    assert np.trace(Mn) == pytest.approx(6.0)
    np.testing.assert_allclose(shrink_metric(M, 1.0), np.eye(6), atol=1e-12)
    np.testing.assert_allclose(shrink_metric(M, 0.0), Mn, atol=1e-12)
    assert np.linalg.eigvalsh(shrink_metric(M, 0.3)).min() > 0
    with pytest.raises(ValueError, match=r"lam must be in"):
        shrink_metric(M, 1.5)
    with pytest.raises(ValueError, match="trace"):
        normalize_metric(np.zeros((3, 3)))


def test_diagnostics_report_the_sincos_anisotropy():
    """The identity metric asserts a sin/cos ratio of 1; the diagnostic measures it."""
    xyz, ql, _, rng = random_system(n_tors=6, n_frames=300, seed=18)
    m0 = np.ones(xyz.shape[1])
    res = sincos_pullback_metric([xyz], ql, m0, angle_sign=AngleSign.SRC_NEGATED)
    d = res.diagnostics
    assert d["trace_over_d"] == pytest.approx(1.0)
    assert d["symmetry_residual"] < 1e-14
    assert d["eigval_min"] > -1e-12
    assert d["cond"] > 1.0
    assert set(d["block_frobenius"]) == {"ss", "cc", "sc"}
    assert d["sincos_diag_ratio_min"] <= d["sincos_diag_ratio_median"] <= d["sincos_diag_ratio_max"]
    # a metric equal to the identity would score exactly 0 here
    assert d["frob_rel_to_identity"] > 0.05


def test_diagnostics_on_the_identity():
    d = metric_diagnostics(np.eye(8), n_torsions=4)
    assert d["cond"] == pytest.approx(1.0)
    assert d["frob_rel_to_identity"] == pytest.approx(0.0)
    assert d["offdiag_frobenius_fraction"] == pytest.approx(0.0)
    assert d["sincos_diag_ratio_median"] == pytest.approx(1.0)
