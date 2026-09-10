# Contributing to sliced-committor

The library is small and welcomes focused improvements. It keeps one way to
do each thing; a proposed alternative needs the evidence that it is better,
and `docs/design_decisions.md` lists what has already been tried.

## Development setup

```bash
pip install -e .[dev]     # test, docs, lint, examples and umbrella extras
pre-commit install        # ruff, ruff format, and the em-dash check
```

## Layout

```
sliced_committor/
  core/        the committor: solver.py (slices), gram.py, _ebmc.py (the
               weight solve and the bootstrap), _halfset.py (the half-set
               filter, folds, held-out cap), _moments.py, committor.py
               (the callable), directions.py, metric.py
  rates/       the {D_q, pi} pair: quantities.py, diffusion.py,
               formulas.py, _coordinate.py (Profile and its reducers),
               baselines.py, units.py
  umbrella/    the umbrella-sampling estimators: dataset.py, reweight.py,
               io.py, mdtraj_metric.py, rates.py (fit_and_rate)
  tests/       one file per module, plus golden/ (frozen references)
```

The dependency order is `umbrella -> rates -> core`; a test rejects an
import the other way.

## Tests

```bash
JAX_PLATFORMS=cpu pytest sliced_committor/tests -q
```

The golden gates (`test_golden.py`, `test_halfset.py`) check the weights
against the frozen references under `tests/golden/`: in the environment the
references were produced in (JAX 0.5.3, the `numerics` CI job) to the bit
for the half-set filter and to `1e-13` for the scalar ridge, and under any
other JAX version at the measured cross-version drift. A change that moves
them in the frozen environment is a change of the published numbers and
needs a reason in the changelog.
`test_public_api.py` pins `__all__`; add a name there deliberately.
`test_docs.py` executes every `python` code block of the README and the user
docs in a namespace seeded with small synthetic inputs (`samples`, `in_A`,
`in_B`, `points`, `q`, `trajectory`, `dt`, `lag`, `window_ids`, `s`,
`s_traj`, `run_ids`, `D_s`, `g_s`, `dataset`, `w`), so a snippet that goes
stale fails the suite. Mark a block that cannot run there (it needs files,
mdtraj or matplotlib) with the info string ```` ```python skip ````.
The pymbar and mdtraj paths are skipped without those packages; CI runs
them in the `umbrella` job.

## Lint

```bash
ruff check sliced_committor && ruff format --check sliced_committor
```

## Docs

```bash
make -C docs html
```

## Conventions

1. Float64 is required for the weight solve; the public entry points raise
   a clear `ValueError` without it.
2. No em dashes in committed text (a pre-commit hook and a CI job reject
   U+2014).
3. Every public name has a docstring with the physics of what it computes,
   its arguments and its return value, and appears in `docs/api/`.
4. No data files in the library; examples and tests are self-contained.
5. A silent fallback is an error: an argument that would be ignored raises.
