# Rate and transport-coefficient estimation

The `sliced_committor.rates` subpackage turns a callable committor (from
`fit_committor` / `build_committor`) into reaction rates and the transport
coefficients they are built from. Every function takes the committor `q` and your
data; nothing needs the underlying result or weights. See
[Computing rates](../recipes.md#computing-rates) for worked snippets and
`examples/02_rates_wolfe_quapp.py` for an end-to-end comparison against an exact
reference rate.

Two families share one interface. The **feature-space** forms (`dirichlet_rate`,
`tpt_rate`) multiply a length-scale `D` by the autodiff `|∇q̄|²`, so they need a
trusted configurational `D`. The **coordinate-invariant** form (`committor_rate`)
is a functional of the pair `{D_q(q), π(q)}` and needs no length-scale `D`;
`berezhkovskii_szabo_rate` is a thin named alias for its harmonic / local
reductions.

## Quantity primitives

```{eval-rst}
.. automodule:: sliced_committor.rates.quantities
   :members:
      density,
      diffusion_coefficient,
      reactive_flux,
      saddle_bridge_D,
      BridgeD
```

## Rate formulas

```{eval-rst}
.. automodule:: sliced_committor.rates.formulas
   :members:
      committor_rate,
      berezhkovskii_szabo_rate,
      dirichlet_rate,
      tpt_rate,
      kramers_rate
```

## Profiles and the plateau finder

```{eval-rst}
.. automodule:: sliced_committor.rates._coordinate
   :members:
      Profile,
      value_at,
      find_plateau,
      PlateauWindow
```
