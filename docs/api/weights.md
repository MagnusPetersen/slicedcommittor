# The weights

```{eval-rst}
.. automodule:: sliced_committor.core._ebmc
   :members:
      solve_weights,
      Weights,
      bootstrap_weights,
      Bootstrap,
      RepresentationError
```

## The half-set spectral filter

The regularisation behind the default `tikhonov="halfset_eigen"`, the
contiguous basin-stratified folds, and the held-out Dirichlet cap. Reached
through `solve_weights`; documented here for the record.

```{eval-rst}
.. automodule:: sliced_committor.core._halfset
   :members:
      make_folds,
      pool_folds,
      halfset_grams,
      halfset_eigen_regularize,
      regularized_gram
```
