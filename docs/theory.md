# Theory primer

## The committor

For a diffusive process $dx = -\nabla V(x)\,dt + \sqrt{2/\beta}\,dW$ with two
metastable basins $A$ and $B$, the committor $q(x)$ is the probability that
a trajectory starting at $x$ reaches $B$ before $A$. It solves

$$
\nabla \cdot \bigl(e^{-\beta V} \nabla q\bigr) = 0,
\qquad q|_A = 0,\ q|_B = 1.
$$

For $d \gtrsim 5$ a direct grid solve is infeasible. The slicedcommittor
method replaces the high-dimensional solve with $M$ one-dimensional solves.

## The sliced approach

1. **Project.** Draw $M$ random unit vectors $\theta_j$ on $S^{d-1}$ and
   project every sample to a scalar $s_{ij} = \theta_j \cdot x_i$.
2. **Solve 1D.** Estimate the projected free energy $F_j(s) = -\log
   \rho_j(s)$ from a histogram of $\{s_{ij}\}$, then solve the 1D
   Smoluchowski committor with absorbing boundaries on the projected
   basins $\theta_j(A)$, $\theta_j(B)$. The result is a 1D committor
   $q_{\theta_j}(s)$ along the slice.
3. **Aggregate.** Combine the slice committors into the d-dimensional
   estimator

   $$
   \hat q(x) = c + \sum_{j=1}^{M} w_j\,q_{\theta_j}(\theta_j \cdot x).
   $$

   The weights $w_j$ and the optional bias $c$ are chosen to satisfy
   basin-mean constraints (`E_A[q̂] = 0`, `E_B[q̂] = 1`) while minimising
   the aggregate Dirichlet energy.

## Weight solvers

| Solver | Constraint | Bias | Closed form |
|---|---|---|---|
| `corrected_dirichlet_inv_rd` | diagonal: $w_j \propto (1-\varepsilon_j)_+/D_j$ | no | yes |
| `compute_full_gram_weights` | simplex $\sum w_j = 1$ | no | self-consistent loop |
| `compute_basin_moment_weights` (BMC) | $\mu_A[\hat q]=0$, $\mu_B[\hat q]=1$ | no | yes (2x2 KKT) |
| `compute_enriched_basin_moment_weights` (EBMC) | $(b-a)^\top w = 1$ | yes | yes |
| `compute_enriched_basin_moment_weights_power` (PESB-EBMC) | EBMC + smoothstep basis | yes | yes |

EBMC is the recommended default: strict Dirichlet-energy improvement over
plain BMC, single Cholesky factor, no self-consistency loop. PESB-EBMC
enriches each slice's basis with $\Psi_n(v) = v^n / (v^n + (1-v)^n)$ for
$n \in \{1, 2\}$ by default; it adds resolving power on heterogeneous
transitions at the cost of a wider Gram matrix.

## Boundary error $\varepsilon$

Per-slice corrections rest on a boundary-error estimator $\varepsilon_j$
that measures how far the 1D solve falls short of `q=0` / `q=1` on the
projected basin edges. Three estimators ship:

- `compute_epsilon_equilibrium`: volume average,
  $\varepsilon^{\rm eq} = \mu_A[q] + (1 - \mu_B[q])$. Default; cached.
- `compute_epsilon_rms`: RMS variant, $\varepsilon^{\rm RMS}
  = \sqrt{\mu_A[q^2]} + \sqrt{\mu_B[(1-q)^2]}$. Cauchy-Schwarz gives
  $\varepsilon^{\rm RMS} \ge \varepsilon^{\rm eq}$ on every direction.
- `compute_epsilon_flux1d`: flux-importance-reweighted variant in 1D.

Pass `epsilon_fn=compute_epsilon_rms` (or the string `"rms"`) to
`compute_full_gram_weights` to override the cached equilibrium estimator.

## Halo mitigation in high d

When `d ≫ k`, where `k` is the number of "active" dimensions, the
projected basin shadows inflate and the absorbing boundaries cover
nuisance-dimension tails. Two paired knobs on `compute_sliced_committor`:

- `boundary_quantile` shrinks the q=0 / q=1 boundary placement toward the
  state centroid (default `1.0`, extreme).
- `absorption_quantile` zeros $\rho_A, \rho_B$ past their inner-edge
  quantile so RD absorption stops covering the inflated tail. `None`
  (default) inherits `boundary_quantile`.

For $d \gtrsim 100$ a typical setting is `boundary_quantile=0.95`-`0.98`.
Lower values give a tighter projected support; too low and the boundary
moves into the basin, which is detectable from the `cond_basin` / `cond_enriched`
diagnostics.

## Why no `beta`?

The committor is $\beta$-invariant given fixed samples: the algorithm
computes $F = -(1/\beta) \log \rho$ from a density estimate, then uses
$\rho = \exp(-\beta(F - F_\min))$ internally. The two $\beta$'s cancel.
The library stores `result.free_energies = -log rho` and you multiply by
$1/\beta_{\rm physical}$ if you want physical units.

## References

The accompanying paper derives the constraint algebra and the
Galerkin-monotonicity property in full. Once available, the DOI and a
formal citation will appear in the [README](https://github.com/MagnusPetersen/slicedcommittor#citation)
and in [CHANGELOG](changelog.md).
