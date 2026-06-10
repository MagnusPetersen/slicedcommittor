# Contributing to sliced-committor

Thanks for your interest. The library is small and welcomes focused improvements.

## Development setup

Clone the monorepo, then install the library in editable mode with all dev extras:

```bash
pip install -e ./lib[dev]
```

This pulls test (pytest, pytest-cov), docs (sphinx, furo, myst-parser,
sphinx-autodoc-typehints), lint (ruff, pre-commit), and example (matplotlib,
jupyter) dependencies.

Activate pre-commit so style is enforced locally:

```bash
cd lib
pre-commit install
```

## Running the test suite

```bash
pytest lib/sliced_committor/tests -v
```

For coverage:

```bash
pytest lib/sliced_committor/tests --cov=sliced_committor --cov-report=term-missing
```

The full suite includes validation tests against the 1D Ornstein-Uhlenbeck
closed form and an inlined 2D Jacobi PDE solver; they take longer than the
smoke tests. Filter with `pytest -k "not validation"` when iterating locally.

## Lint and format

```bash
ruff check lib/sliced_committor
ruff format lib/sliced_committor
```

Configuration lives in `lib/pyproject.toml` under `[tool.ruff]`.

## Building the docs

```bash
cd lib/docs
make html
```

Output goes to `lib/docs/_build/html/`.

## Style rules

A few project-local conventions:

1. **Float64 requirement.** EBMC / PESB / plain BMC require
   `jax.config.update("jax_enable_x64", True)` before computing the result.
   New code that touches the Gram solvers should check this at the public-API
   boundary and raise a clear `ValueError` if missing.

2. **No em dashes in committed text.** Replace `:`, `,`, `;`, or a period
   depending on the cadence. A pre-commit hook rejects U+2014 in staged files.
   The only exception is author-list-style separators (`Name : Affiliation`).

3. **Type hints are mandatory on the public API.** Internal helpers can skip
   them when the surrounding code makes the type obvious.

4. **Docstrings on every public function in `__all__`.** Minimum is a one-line
   summary, `Args`, and `Returns`. Add `Raises` and `Examples` when relevant.

5. **No data files in the library.** The library is usage-focused: users bring
   their own data. Examples must be self-contained (synthetic samples,
   analytical references) and never require external downloads.

6. **Tests should not depend on the main monorepo.** When porting an assertion
   from `tests/` at the monorepo root, copy the math; do not import.

## Submitting changes

The remote does not exist yet; until it does, please share patches via email
or by appending to the local monorepo. Once a public GitHub remote is set up,
issue and PR templates will be added and this section will be expanded.
