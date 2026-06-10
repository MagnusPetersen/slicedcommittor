"""Sphinx configuration for sliced-committor docs."""

from __future__ import annotations

import os
import sys

# Make the package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

import sliced_committor  # noqa: E402

# -- Project information -----------------------------------------------------

project = "sliced-committor"
author = "Magnus Petersen"
copyright = "2026, Magnus Petersen"
release = sliced_committor.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False

myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "dollarmath",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"sliced-committor v{release}"

# Avoid warnings on missing intersphinx targets in CI.
nitpicky = False
