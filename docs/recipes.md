# Recipes

Short snippets for the most common variations from the default pipeline.

## Switching the boundary-error estimator

```python
from sliced_committor import compute_full_gram_weights
from sliced_committor.weights import compute_epsilon_rms

result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
gram = compute_full_gram_weights(result, samples, epsilon_fn=compute_epsilon_rms)
# Equivalent string form:
gram = compute_full_gram_weights(result, samples, epsilon_fn="rms")
```

## Using a custom direction sampler (PCA / gcPCA)

```python
from sliced_committor import DirectionSamplingConfig

result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=256,
    direction_sampling=DirectionSamplingConfig(mode="pca", n_bias_axes=4),
)
```

For label-aware bias use `mode="gcpca"`. To supply your own directions:

```python
import jax.numpy as jnp
my_axes = jnp.asarray(...)  # (M, dim) unit vectors
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=my_axes.shape[0],
    directions=my_axes,
)
```

## Affine label-mean calibration

The sliced aggregator suffers basin-mean shrinkage in high d; the affine
correction pins `E_A[q̂] = 0` and `E_B[q̂] = 1` exactly (pre-clip).

```python
from sliced_committor.calibration import calibrate_weights_affine

raw = compute_weights_multi(result, [corrected_dirichlet_inv_rd])["corrected_dirichlet_inv_rd"]
cal = calibrate_weights_affine(result, raw, mode="global")
q = evaluate_committor(result, points, cal)   # `cal` is a centered-basis dict
```

## 1D recalibration along $\bar q$

```python
from sliced_committor.calibration import evaluate_committor_with_recal

q = evaluate_committor_with_recal(
    result, points, weights, samples,
    in_A_samples=in_A, in_B_samples=in_B,
)
```

## High-dimensional halo mitigation

```python
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=1024,
    boundary_quantile=0.97,   # paired by default with absorption_quantile
)
```

## Memory-light mode

```python
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=4096,
    store_projected_samples=False,   # diagonal RD only; no full-Gram / BMC
)
```

## Inspecting masked slices

```python
from sliced_committor import why_masked
for j, ok in enumerate(result.valid_mask):
    if not ok:
        print(j, why_masked(result, j))
```
