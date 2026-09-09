"""Shared pytest setup for the lib/recovar test suite.

Puts <repo>/lib (the JAX port, importable as ``recovar``) and
<repo>/lib/recovar/_reference (the vendored numpy prototype, importable as
``slicedcv``) on sys.path; both paths are computed from __file__ so the suite
runs from any checkout location. Forces JAX onto CPU with float64 before
anything imports jax, and registers the ``slow`` marker used by the heavier
end-to-end comparisons.
"""
import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_LIB_DIR = _TESTS_DIR.parents[1]                    # <repo>/lib
_REF_DIR = _TESTS_DIR.parents[0] / "_reference"     # <repo>/lib/recovar/_reference

for _p in (str(_LIB_DIR), str(_REF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must be set before jax initialises its backend; harmless if already set by
# the caller (the documented invocation exports JAX_PLATFORMS=cpu itself).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# Importing recovar enables x64 too; doing it here as well keeps direct jax
# use in tests safe even before the package is imported.
jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (> 20s)")
