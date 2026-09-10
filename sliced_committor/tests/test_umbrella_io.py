"""COLVAR parsing, restraint parsing, and COLVAR-trajectory stride alignment (numpy only)."""

import numpy as np

from sliced_committor.umbrella import io

from ._helpers import write_colvar


def test_colvar_1d_q(tmp_path):
    p = write_colvar(
        tmp_path / "COLVAR",
        np.arange(3.0),
        {"q": np.array([0.17, 0.22, 0.19]), "restr.bias": np.array([2.1, 3.0, 1.5])},
        ["time", "q", "restr.bias"],
    )
    d = io.read_colvar(p)
    assert d.fields == ("time", "q", "restr.bias")
    np.testing.assert_allclose(d.column("q"), [0.17, 0.22, 0.19])
    assert d.non_time_fields() == ("q", "restr.bias")
    assert abs(io.colvar_dt(d) - 1.0) < 1e-9


def test_colvar_periodic_2d(tmp_path):
    p = write_colvar(
        tmp_path / "colvar.dat",
        np.arange(2.0),
        {"phi": np.array([-2.5, -0.7]), "psi": np.array([2.4, 2.6]), "bb.bias": np.zeros(2)},
        ["time", "phi", "psi", "bb.bias"],
        periodic={"phi": (-np.pi, np.pi)},
    )
    d = io.read_colvar(p)
    assert "phi" in d.periodic
    np.testing.assert_allclose(d.periodic["phi"], (-np.pi, np.pi))


def test_colvar_pi_token(tmp_path):
    (tmp_path / "COLVAR").write_text(
        "#! FIELDS time phi\n#! SET min_phi -pi\n#! SET max_phi pi\n0.0 1.0\n1.0 -1.0\n"
    )
    d = io.read_colvar(tmp_path / "COLVAR")
    assert d.periodic["phi"] == (-np.pi, np.pi)


def test_parse_restraint_2d(tmp_path):
    (tmp_path / "plumed.dat").write_text(
        "INCLUDE FILE=q.dat\nrestr: RESTRAINT ARG=cv1,cv2 KAPPA=2092,2092 AT=-0.125,0.625\n"
    )
    r = io.parse_restraint(tmp_path / "plumed.dat")
    assert r.arg_names == ("cv1", "cv2")
    np.testing.assert_allclose(r.kappa, [2092, 2092])
    np.testing.assert_allclose(r.at, [-0.125, 0.625])


def test_parse_restraint_scalar_kappa_broadcast(tmp_path):
    (tmp_path / "plumed.dat").write_text("r: RESTRAINT ARG=a,b KAPPA=10 AT=1.0,2.0\n")
    r = io.parse_restraint(tmp_path / "plumed.dat")
    np.testing.assert_allclose(r.kappa, [10, 10])


def test_parse_restraint_none(tmp_path):
    (tmp_path / "plumed.dat").write_text("INCLUDE FILE=q.dat\nPRINT ARG=q FILE=COLVAR\n")
    assert io.parse_restraint(tmp_path / "plumed.dat") is None
    assert io.parse_restraint(tmp_path / "missing.dat") is None


def test_align_same_cadence():
    t = np.arange(10.0)
    ci, ti = io.align_colvar_traj(t, t, 10)
    np.testing.assert_array_equal(ci, np.arange(10))
    np.testing.assert_array_equal(ti, np.arange(10))


def test_align_traj_coarser():
    colvar_t = np.arange(0, 10, 1.0)  # 1 ps cadence
    traj_t = np.arange(0, 10, 2.0)  # 2 ps cadence: coarser, fewer frames
    ci, ti = io.align_colvar_traj(colvar_t, traj_t, traj_t.size)
    assert ci.size == ti.size == traj_t.size
    np.testing.assert_allclose(colvar_t[ci], traj_t, atol=0.5)


def test_align_no_timestamps_falls_back():
    colvar_t = np.arange(20.0)
    ci, ti = io.align_colvar_traj(colvar_t, None, 10)
    assert ci.size == ti.size == 10
