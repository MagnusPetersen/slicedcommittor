# Examples

Three self-contained walkthroughs on the rotated Wolfe-Quapp 2D potential,
the paper's figure-1 benchmark (Bonati, Zhang and Parrinello, PNAS 2019).
No data ships: samples are drawn from the analytic Boltzmann density and the
reference is a converged Jacobi PDE committor, both in `_wolfe_quapp.py`.

| script | what it shows | runtime (CPU) |
|---|---|---|
| [01_wolfe_quapp_2d.py](01_wolfe_quapp_2d.py) | The committor step by step: the input contract, `compute_sliced_committor` and its `SlicedCommittorResult`, `solve_weights` under the half-set filter and the scalar ridge with their Dirichlet energies, `build_committor` and `rescale_transition`, the PDE cross-check, and the paper's figure-1 layout. | 2-3 min |
| [02_rates_wolfe_quapp.py](02_rates_wolfe_quapp.py) | The rate: the pair `{D_q, pi}` built three ways (Kramers-Moyal on the committor after a lag scan, the Jacobian map of a diffusion measured along `x`, and the assumed `D0`), the four reductions of each with the flatness of the flux, the Kramers baseline, all against the exact PDE rate, with self-checks and a figure. | 2-3 min |
| [03_model_selection_wolfe_quapp.py](03_model_selection_wolfe_quapp.py) | Choosing settings without a reference: a grid of `n_directions` x `n_bins` ranked by the held-out Dirichlet cap (with its overfitting gap), then block-bootstrap error bars on the winner. The PDE RMSE is printed as the outcome only. | 1-2 min |

```bash
pip install sliced-committor[examples]   # matplotlib
python 01_wolfe_quapp_2d.py
```
