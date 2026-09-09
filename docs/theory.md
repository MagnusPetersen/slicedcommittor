# Theory primer

## The committor

For a diffusive process $dx = -\nabla V(x)\,dt + \sqrt{2/\beta}\,dW$ with two
metastable basins $A$ and $B$, the committor $q(x)$ is the probability that
a trajectory starting at $x$ reaches $B$ before $A$. It solves

$$
\nabla \cdot \bigl(e^{-\beta V} \nabla q\bigr) = 0,
\qquad q|_A = 0,\ q|_B = 1,
$$

and it is the minimiser of the Dirichlet form $\langle |\nabla q|^2 \rangle_\pi$
over all functions with those boundary values. For $d \gtrsim 5$ a direct grid
solve is infeasible. The sliced committor replaces it with $M$ one-dimensional
solves and one linear solve.

## The sliced approach

1. **Project.** Draw $M$ unit vectors $\theta_j$ on $S^{d-1}$ and project
   every sample to a scalar $s_{ij} = \theta_j \cdot x_i$.
2. **Solve 1D.** Estimate the projected free energy $F_j(s) = -\log
   \rho_j(s)$ from a histogram of $\{s_{ij}\}$ and solve the 1D
   reaction-diffusion committor with absorption on the projected basins
   $\theta_j(A)$, $\theta_j(B)$. The result is a 1D committor
   $q_j(s)$ along the slice.
3. **Aggregate.** Combine the slice committors into

   $$
   \hat q(x) = c + \sum_{j=1}^{M} w_j\,q_j(\theta_j \cdot x).
   $$

The 1D committors are trial functions that violate the boundary conditions:
each is exactly 0 and 1 only on the shadow of the basins along its own
direction. The weights repair this in the aggregate. With the Gram matrix
$G_{jk} = \langle \nabla q_j \cdot \nabla q_k \rangle_\pi$ and the basin moments
$a_j = \mathbb{E}_A[q_j]$, $b_j = \mathbb{E}_B[q_j]$, the weights minimise the
Dirichlet form $w^\top G w$ under the single constraint $(b - a)^\top w = 1$
(the enriched basin-moment constraint, EBMC of the paper). The solution is
closed-form,

$$
w = \frac{G^{-1}(b - a)}{R}, \qquad R = (b - a)^\top G^{-1} (b - a), \qquad c = -a^\top w,
$$

so that $\mathbb{E}_A[\hat q] = 0$ and $\mathbb{E}_B[\hat q] = 1$ hold exactly.
$R$ is the moment gap; $1/R$ is the Dirichlet energy of the solve, and
`Weights.cond`, the ratio of $R$ to the basin moments' own scale, is the
representation gate: it collapses when no combination of the slices can
separate the basins.

## The half-set spectral filter

$G$ is estimated from samples and its condition number reaches $10^{12}$, so
the solve needs regularisation, and a scalar ridge cannot do it: the noise
is not the same across the spectrum. The library's default, the half-set
spectral filter (`tikhonov="halfset_eigen"`), deals the frames into two
halves by contiguous, basin-stratified blocks, assembles a Gram matrix from
each, and compares the two in the eigenbasis of their mean. Bands on which
the halves agree carry signal and are left alone; bands on which they
disagree are noise, and their eigenvalues are inflated by $1 + 1/\mathrm{SNR}$.
The filter has no constant and nothing to select, which is why it is the
paper's universal setting. It needs the frames in time order; a permutation
of the frames destroys the split's independence.

## The boundary error

The 1D committors miss their boundary values by the boundary error
$\varepsilon_j = \mathbb{E}_A[q_j] + (1 - \mathbb{E}_B[q_j])$, reported per
slice in `result.boundary_errors`. It is the paper's central diagnostic: the
aggregate constraint above is what makes the boundary values right in the
mean, and a slice with a large $\varepsilon_j$ is one whose direction
separates the basins badly.

## Halo mitigation in high d

When $d \gg k$, with $k$ the number of active dimensions, the projected basin
shadows inflate and the absorbing boundaries reach into the nuisance
dimensions' tails. `boundary_quantile` on `compute_sliced_committor` moves
the $q = 0$ and $q = 1$ boundaries from the extreme projected basin sample
toward its centroid (`1.0` is the extreme, the default); the paper uses
0.99 on AIB9 and 0.98 on villin. Too low a value moves the boundary into the
basin, which shows up as a growing boundary error.

## Why no `beta`?

The committor is $\beta$-invariant given fixed samples: the algorithm
estimates $F = -\log \rho$ from the samples and uses $\rho = e^{-F}$
internally, so the temperature cancels. `result.free_energies` stores
$-\log \rho$ per slice; multiply by $1/\beta_{\rm physical}$ for physical
units.

## The rate

Projected onto the committor coordinate the dynamics is a one-dimensional
diffusion with density $\pi(q)$ and diffusion $D_q(q)$, and its reactive flux
$\nu_R(q) = D_q(q)\,\pi(q)$ is constant in $q$ for the exact committor. The
rate constants are $k_{AB} = \nu_R / \rho_A$ and $k_{BA} = \nu_R / \rho_B$
with $\rho_B = \mathbb{E}_\pi[q]$. By the co-area formula
$\langle |\nabla q|^2 \rangle_q = \Phi_{D = 1}(q) / \pi(q)$, and a diffusion
measured along a collective variable $s$ maps into committor space as

$$
D_q(q) = D_s \,\frac{\langle |\nabla q|^2 \rangle_q}{\langle |\nabla s|^2 \rangle},
$$

one configurational scale seen through two gradients. [rates.md](rates.md)
takes it from here.
