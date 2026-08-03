"""COLVAR parsing and bias parsing (no optional deps)."""

import numpy as np

from sliced_committor.workflows import bias, colvar

from .conftest import write_colvar


def test_colvar_1d_q(tmp_path):
    p = write_colvar(
        tmp_path / "COLVAR",
        np.arange(3.0),
        {"q": np.array([0.17, 0.22, 0.19]), "restr.bias": np.array([2.1, 3.0, 1.5])},
        ["time", "q", "restr.bias"],
    )
    d = colvar.read_colvar(p)
    assert d.fields == ("time", "q", "restr.bias")
    np.testing.assert_allclose(d.column("q"), [0.17, 0.22, 0.19])
    assert d.non_time_fields() == ("q", "restr.bias")
    assert abs(colvar.colvar_dt(d) - 1.0) < 1e-9


def test_colvar_periodic_2d(tmp_path):
    p = write_colvar(
        tmp_path / "colvar.dat",
        np.arange(2.0),
        {"phi": np.array([-2.5, -0.7]), "psi": np.array([2.4, 2.6]), "bb.bias": np.zeros(2)},
        ["time", "phi", "psi", "bb.bias"],
        periodic={"phi": (-np.pi, np.pi)},
    )
    d = colvar.read_colvar(p)
    assert "phi" in d.periodic
    np.testing.assert_allclose(d.periodic["phi"], (-np.pi, np.pi))


def test_colvar_pi_token(tmp_path):
    (tmp_path / "COLVAR").write_text(
        "#! FIELDS time phi\n#! SET min_phi -pi\n#! SET max_phi pi\n0.0 1.0\n1.0 -1.0\n"
    )
    d = colvar.read_colvar(tmp_path / "COLVAR")
    assert d.periodic["phi"] == (-np.pi, np.pi)


def test_parse_restraint_2d(tmp_path):
    (tmp_path / "plumed.dat").write_text(
        "INCLUDE FILE=q.dat\nrestr: RESTRAINT ARG=cv1,cv2 KAPPA=2092,2092 AT=-0.125,0.625\n"
    )
    r = bias.parse_restraint(tmp_path / "plumed.dat")
    assert r.arg_names == ("cv1", "cv2")
    np.testing.assert_allclose(r.kappa, [2092, 2092])
    np.testing.assert_allclose(r.at, [-0.125, 0.625])


def test_parse_restraint_scalar_kappa_broadcast(tmp_path):
    (tmp_path / "plumed.dat").write_text("r: RESTRAINT ARG=a,b KAPPA=10 AT=1.0,2.0\n")
    r = bias.parse_restraint(tmp_path / "plumed.dat")
    np.testing.assert_allclose(r.kappa, [10, 10])


def test_parse_restraint_none(tmp_path):
    (tmp_path / "plumed.dat").write_text("INCLUDE FILE=q.dat\nPRINT ARG=q FILE=COLVAR\n")
    assert bias.parse_restraint(tmp_path / "plumed.dat") is None


def test_window_name_decode():
    np.testing.assert_allclose(bias.center_from_window_name("W_C1m001p00_C2p008p00"), [-0.1, 0.8])
    np.testing.assert_allclose(bias.center_from_window_name("W_C1p000p00_C2p005p25"), [0.0, 0.525])
    assert bias.center_from_window_name("window_07") is None
