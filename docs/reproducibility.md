# Reproducibility

The library's determinism contract:

| Held fixed | Same result |
|---|---|
| `seed`, JAX minor version, accelerator (CPU/GPU/TPU) | bit-exact |
| `seed`, JAX minor version, CPU only | bit-exact |
| `seed`, different JAX minor versions, same accelerator | up to ~1e-12 relative drift in float64 outputs; semantic outputs (committors, weights, basin moments) are stable |
| `seed`, GPU vs CPU | small numerical drift (~1e-9-1e-7 in float64); driven by non-associative reductions |

The bundled regression tests use a frozen snapshot
(`sliced_committor/tests/golden/beta_invariance_golden.npz`) generated on
JAX 0.4.x running on CPU. The snapshot is reproduced bit-exactly by the
test `test_beta_invariance.py` whenever the library is run against the
same JAX minor and accelerator.

Pinning the JAX version is recommended for downstream regression suites
that want bit-exact reproducibility:

```toml
dependencies = ["slicedcommittor==0.4.0", "jax==0.4.30"]
```

If you observe drift larger than the bands above, please open an issue
with the JAX version, accelerator, and a reproducible script.

## Why no `beta` argument?

Because the committor is $\beta$-invariant given fixed samples (see
[theory.md](theory.md)), the library does not expose an
inverse-temperature argument: feeding $\beta$ in and pulling $\beta$ out
would be cosmetic. Internally the library fixes $\beta = 1$ so
`result.free_energies` stores $-\log \rho$ directly. Multiply by
$1/\beta_{\rm physical}$ to recover physical-units free energies.
