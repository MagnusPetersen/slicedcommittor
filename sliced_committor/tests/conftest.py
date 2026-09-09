"""Shared fixtures and JAX setup for the test suite."""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "beta_invariance_golden.npz")


@pytest.fixture(scope="session")
def golden_data():
    """The frozen 500-sample two-basin fixture behind the golden gates (samples, labels,
    and the slice basis of v0.6.0; see ``golden/freeze_1_0_references.py``)."""
    return np.load(GOLDEN_PATH, allow_pickle=False)


@pytest.fixture(scope="session")
def golden_samples(golden_data):
    return jnp.asarray(golden_data["samples"])


@pytest.fixture(scope="session")
def golden_labels(golden_data):
    return jnp.asarray(golden_data["in_A"]), jnp.asarray(golden_data["in_B"])


# ---------------------------------------------------------------------------
# umbrella-subpackage helpers (synthetic only; mdtraj-gated where needed)
# ---------------------------------------------------------------------------
def write_colvar(path, time, columns, fields, periodic=None):
    """Write a minimal PLUMED COLVAR file; ``columns`` maps field name -> array."""
    lines = [f"#! FIELDS {' '.join(fields)}"]
    for name, (lo, hi) in (periodic or {}).items():
        lines.append(f"#! SET min_{name} {lo}")
        lines.append(f"#! SET max_{name} {hi}")
    data = np.column_stack([time] + [columns[f] for f in fields if f != "time"])
    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in data)
    path.write_text("\n".join(lines) + "\n" + body + "\n")
    return path


@pytest.fixture
def synthetic_traj():
    """A tiny alanine-backbone mdtraj Trajectory factory (N, CA, C, O per residue)."""
    md = pytest.importorskip("mdtraj")

    def _make(n_frames=50, n_res=4, seed=0):
        top = md.Topology()
        chain = top.add_chain()
        for i in range(n_res):
            res = top.add_residue("ALA", chain, resSeq=i + 1)
            top.add_atom("N", md.element.nitrogen, res)
            top.add_atom("CA", md.element.carbon, res)
            top.add_atom("C", md.element.carbon, res)
            top.add_atom("O", md.element.oxygen, res)
        rng = np.random.default_rng(seed)
        xyz = rng.normal(scale=0.3, size=(n_frames, n_res * 4, 3)).astype(np.float32)
        traj = md.Trajectory(xyz, top)
        traj.time = np.arange(n_frames, dtype=float)
        return traj

    return _make
