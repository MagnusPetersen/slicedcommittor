"""Docs parity: every public name is documented, and every ``python`` code block
of the README and the user docs executes against small synthetic inputs.

A block whose info string contains ``skip`` (``python skip``) is not run: it
needs files, mdtraj or matplotlib. Everything else runs in one namespace per
document, in order, seeded with ``samples``, ``in_A``, ``in_B``, ``points``,
``q``, ``trajectory``, ``dt``, ``lag``, ``window_ids``, ``s``, ``s_traj``,
``run_ids``, ``D_s``, ``g_s``, ``dataset`` and ``w``.
"""

import pathlib
import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import sliced_committor as sc
from sliced_committor import umbrella

from ._helpers import (
    double_well_samples,
    overdamped_double_well_trajectory,
    umbrella_double_well_dataset,
)

ROOT = pathlib.Path(sc.__file__).resolve().parent.parent
DOCS = ROOT / "docs"
USER_DOCS = [
    "README.md",
    "docs/quickstart.md",
    "docs/rates.md",
    "docs/advanced.md",
    "docs/umbrella.md",
]
BLOCK = re.compile(r"```python([^\n]*)\n(.*?)```", re.S)


def test_every_public_name_appears_in_the_api_docs():
    text = "".join(p.read_text() for p in (DOCS / "api").glob("*.md"))
    missing = [n for n in list(sc.__all__) + list(umbrella.__all__) if n not in text]
    assert not missing, missing


def test_every_public_name_is_documented():
    for name in sc.__all__:
        doc = getattr(sc, name).__doc__
        assert doc and doc.strip(), name
    for name in umbrella.__all__:
        doc = getattr(umbrella, name).__doc__
        assert doc and doc.strip(), name


@pytest.fixture(scope="module")
def namespace():
    samples, in_A, in_B = double_well_samples(n=2000, seed=0)
    X = jnp.asarray(samples)
    q = sc.fit_committor(
        X, in_A=jnp.asarray(in_A), in_B=jnp.asarray(in_B), n_directions=64, n_bins=60, seed=1
    )
    trajectory = overdamped_double_well_trajectory(T=5000, dt=0.01, D0=0.05, seed=1)
    dataset = umbrella_double_well_dataset(n_windows=6, n_per=2500, dt=0.01, D0=0.05, seed=0)
    w = umbrella.reweight(dataset, method="wham", n_bins=60).sample_weights
    return dict(
        samples=X,
        in_A=np.asarray(in_A),
        in_B=np.asarray(in_B),
        points=X[:10],
        q=q,
        trajectory=jnp.asarray(trajectory),
        dt=0.01,
        lag=1,
        window_ids=np.repeat(np.arange(5), 1000),
        s=np.asarray(samples)[:, 0],
        s_traj=trajectory[:, 0],
        run_ids=None,
        D_s=0.05,
        g_s=1.0,
        dataset=dataset,
        w=w,
    )


@pytest.mark.parametrize("doc", USER_DOCS)
def test_doc_snippets_execute(doc, namespace):
    text = (ROOT / doc).read_text()
    blocks = [(info, code) for info, code in BLOCK.findall(text) if "skip" not in info]
    assert blocks, f"{doc} has no runnable python block"
    ns = dict(namespace)
    for i, (_, code) in enumerate(blocks):
        try:
            exec(compile(code, f"{doc}:block{i}", "exec"), ns)
        except Exception as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{doc} block {i} failed: {exc}\n{code}") from exc
