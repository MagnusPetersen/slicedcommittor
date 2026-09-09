# Rates

The state is the pair `{D_q(q), pi(q)}`; see [rates.md](../rates.md) for the
walkthrough.

## The static ensemble

```{eval-rst}
.. automodule:: sliced_committor.rates.quantities
   :members:
      density,
      basin_populations,
      committor_grad_sq,
      committor_diffusion_from_cv,
      linear_response_grad_sq,
      committor_diffusion_from_cv_reparam
```

## The diffusion, measured

```{eval-rst}
.. automodule:: sliced_committor.rates.diffusion
   :members:
      diffusion_profile,
      lag_scan,
      hummer_diffusion,
      pooled_acf_diffusion,
      PooledDiffusion
```

## The reduction

```{eval-rst}
.. automodule:: sliced_committor.rates.formulas
   :members:
      rate_from_profiles,
      committor_rate
```

## Profiles and their reducers

```{eval-rst}
.. automodule:: sliced_committor.rates._coordinate
   :members:
      Profile,
      value_at,
      flux_flatness,
      find_plateau,
      PlateauWindow
```

## Baselines and units

```{eval-rst}
.. automodule:: sliced_committor.rates.baselines
   :members:
      pmf_kramers_rate

.. automodule:: sliced_committor.rates.units
   :members:
      estimated_to_per_s,
      reference_to_per_s,
      is_reduced
```
