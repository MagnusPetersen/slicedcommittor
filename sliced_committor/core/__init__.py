"""Core sliced-committor algorithm: solver, weight solvers, directions, and the
callable committor model.

This subpackage groups the internal implementation modules. The public API is
re-exported from the top-level :mod:`sliced_committor` package; import from
there (e.g. ``from sliced_committor import fit_committor``) rather than reaching
into ``sliced_committor.core`` directly.
"""

# This subpackage is internal. Nothing is re-exported via ``*``; use the
# top-level ``sliced_committor`` package for the public API.
__all__: list[str] = []
