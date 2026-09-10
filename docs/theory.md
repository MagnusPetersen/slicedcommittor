# Theory primer

## The committor, and why samples suffice

For a diffusive process $dx = -\nabla V(x)\,dt + \sqrt{2/\beta}\,dW$ with
two metastable states $A$ and $B$, the committor $q(x)$ is the probability
that a trajectory started at $x$ reaches $B$ before $A$. It solves

$$
\nabla \cdot \bigl(e^{-\beta V} \nabla q\bigr) = 0,
\qquad q|_A = 0,\ q|_B = 1,
$$

and, equivalently, it minimises the Dirichlet form
$\langle |\nabla q|^2 \rangle_\pi$ over all functions with those boundary
values, with $\pi \propto e^{-\beta V}$ the equilibrium density. The second
form is the one the library uses. It contains the dynamics only through
$\pi$ and the boundary values, so equilibrium samples with state labels
determine the committor: no trajectory, no lag time, no propagator.

Two things are assumed. The dynamics is reversible, which is what makes the
committor a minimiser in the first place. And the diffusion tensor is known
up to a scale: with a tensor $D$ the form reads
$\langle \nabla q^\top D \nabla q \rangle_\pi$, the scale of $D$ drops out
of the minimiser, and only its shape matters. In Cartesian coordinates with
every atom diffusing alike the shape is the identity. In a feature space
$u = u(x)$ it is the pull-back $\bar M = \langle J J^\top \rangle_\pi$ with
$J = \partial u / \partial x$, which the library takes as its
`feature_metric` and which `sincos_pullback_metric` builds for sine/cosine
torsion features ([settings](settings.md)). The metric enters the weight
solve only; the slices do not see it.

For $d \gtrsim 5$ a grid solve is out of reach. The sliced committor
replaces it with $M$ one-dimensional solves and one linear solve.

## The sliced ansatz

1. **Project.** Draw $M$ unit vectors $\theta_j$ on $S^{d-1}$ and project
   every sample to a scalar $s_{ij} = \theta_j \cdot x_i$. Each state
   projects to a stretch of the line, $\theta_j(A)$ and $\theta_j(B)$.
2. **Solve in 1D.** Histogram the projections into `n_bins` bins to
   estimate the projected free energy $F_j(s) = -\log \rho_j(s)$, and solve
   the one-dimensional committor equation along the slice with the two
   projected states as absorbing regions: a reaction term of strength
   `rd_kappa` pins the solution to 0 where $A$ projects and to 1 where $B$
   projects, up to each state's inner edge (the extreme of its projections,
   or the `boundary_quantile` quantile of them). One tridiagonal solve gives
   the slice committor $q_j(s)$.
3. **Aggregate.** Combine the slice committors into

   $$
   \hat q(x) = c + \sum_{j=1}^{M} w_j\,q_j(\theta_j \cdot x).
   $$

The slice committors are trial functions that violate the boundary
conditions: $q_j$ is exactly 0 and 1 only on the shadows of the states along
its own direction, and the shadows of $A$ and $B$ overlap wherever the
direction does not separate them. The weights repair this in the aggregate.

## The weights

With the Gram matrix
$G_{jk} = \langle \nabla q_j \cdot \nabla q_k \rangle_\pi$ (with a metric,
$\theta_j^\top \bar M \theta_k$ times the sample average of the slice
slopes) and the basin moments $a_j = \mathbb{E}_A[q_j]$,
$b_j = \mathbb{E}_B[q_j]$, the weights minimise the Dirichlet form
$w^\top G w$ under the single constraint $(b - a)^\top w = 1$ (the enriched
basin-moment constraint, EBMC of the paper). The solution is closed-form,

$$
w = \frac{G^{-1}(b - a)}{R}, \qquad R = (b - a)^\top G^{-1} (b - a), \qquad c = -a^\top w,
$$

so that $\mathbb{E}_A[\hat q] = 0$ and $\mathbb{E}_B[\hat q] = 1$ hold
exactly; the free bias $c$ is what lets slices whose average over $A$ is
not zero contribute. $R$ is the moment gap and $1/R$ the Dirichlet energy of
the solve. `Weights` reports both, the energy $w^\top G w$ of the committor
actually built (on the unregularised Gram), and `cond`, the ratio of $R$ to
the basin moments' own scale: the representation gate, which collapses when
no combination of the slices separates the states.

## The boundary error

The 1D committors miss their boundary values by the boundary error
$\varepsilon_j = \mathbb{E}_A[q_j] + (1 - \mathbb{E}_B[q_j])$, reported per
slice in `result.boundary_errors`. It is the paper's central diagnostic: the
aggregate constraint above is what makes the boundary values right in the
mean, and a slice with a large $\varepsilon_j$ is one whose direction
separates the states badly. The weights suppress such slices on their own;
the error tells you how much of the basis is doing the work.

## Regularisation: the half-set spectral filter

$G$ is estimated from samples and its condition number reaches $10^{12}$,
so the solve needs regularisation, and a scalar ridge cannot do it: the
noise is not the same across the spectrum. The library's default, the
half-set spectral filter (`tikhonov="halfset_eigen"`), deals the frames
into ten contiguous, basin-stratified folds, assembles a Gram matrix from
the even folds and one from the odd folds, and compares the two in the
eigenbasis of their mean, band by band (twelve bands). The correlation $c$
of a band between the halves is read as a signal-to-noise ratio
$\mathrm{SNR} = 2c / (1 - c)$, and the band's eigenvalues are inflated by
$1 + 1/\mathrm{SNR}$: bands the halves agree on are left alone, bands they
disagree on are damped in proportion. The filter has no constant and
nothing to select, which is why it is the paper's universal setting. It
needs the frames in time order; a permutation of the frames destroys the
independence of the two halves and reports the noise as smaller than it
is. `tikhonov="auto"` (a closed-form scalar ridge) and an absolute ridge
exist for controlled comparisons ([settings](settings.md)).

## The held-out cap

The same objective can be read out of sample. Hold out one fold, solve the
weights on the others, and evaluate
$w^\top G_k w / \bigl((b_k - a_k) \cdot w\bigr)^2$ on the held-out fold
$k$; average over the folds. This held-out Dirichlet cap bounds the
reaction flux for whatever trial space produced the weights, and because it
is read on frames the solve never saw it ranks trial spaces (direction
sets, sampler settings, bin counts) against each other, which the in-sample
energy cannot. Its difference to the in-sample value, the gap, is the
overfitting guard. `fit_committor(..., heldout_cap=True)` returns the cap,
its standard error over the folds, the per-fold values and the gap; the
paper selects the direction cone's `(mu, alpha)` with it.

## Directions

Uniform directions on the sphere need no information about the states.
When the states are known, the Fisher discriminant axis
$v = \Sigma_w^{-1}(\mu_B - \mu_A)$, with $\Sigma_w$ the pooled within-state
covariance (ridged by `lda_shrinkage`), is the direction along which they
separate best, and directions near it carry most of the transition. The
cone draws from the mixture

$$
p(\theta) = \alpha\,p_{\rm unif}(\theta) + (1 - \alpha)\,p_{\rm PS}(\theta;\, v, \kappa),
\qquad p_{\rm PS}(\theta; v, \kappa) \propto (1 + v \cdot \theta)^{\kappa},
$$

with $\kappa$ fixed by the mean cosine $\mu$ to the axis
($\kappa = \mu(d - 1)/(1 - \mu)$, so that $\mu$ means the same in every
dimension) and the uniform part a coverage floor, $p \ge \alpha\,p_{\rm unif}$,
so that no direction is ever excluded. $\mu$ and $\alpha$ are chosen by the
held-out cap.

## Halo mitigation in high d

When $d \gg k$, with $k$ the number of dimensions that take part in the
transition, the projected shadows of the states inflate: the nuisance
dimensions add a halo of random projections around each shadow, and the
absorbing regions of the 1D solves reach into the transition region.
`boundary_quantile` moves the inner edge of each shadow from the extreme
projected state sample toward its centroid (`1.0` is the extreme, the
default); the paper uses 0.99 on AIB9 and 0.98 on villin. Too low a value
moves the boundary into the state itself, which shows up as a growing
boundary error.

## Why no `beta`?

The committor is $\beta$-invariant given fixed samples: the algorithm
estimates $F = -\log \rho$ from the samples and uses $\rho = e^{-F}$
internally, so the temperature cancels. `result.free_energies` stores
$-\log \rho$ per slice; multiply by $1/\beta_{\rm physical}$ for physical
units.

## From the committor to the rate

Projected onto the committor coordinate the dynamics is a one-dimensional
diffusion with density $\pi(q)$ and diffusion $D_q(q)$, and its reactive flux
$\nu_R(q) = D_q(q)\,\pi(q)$ is constant in $q$ for the exact committor. The
rate constants are $k_{AB} = \nu_R / \rho_A$ and $k_{BA} = \nu_R / \rho_B$
with $\rho_B = \mathbb{E}_\pi[q]$. The density is in the samples; $D_q$ is
the one quantity that needs dynamics. By the co-area formula
$\langle |\nabla q|^2 \rangle_q = \Phi_{D = 1}(q) / \pi(q)$, and a diffusion
measured along a collective variable $s$ maps into committor space as

$$
D_q(q) = D_s \,\frac{\langle |\nabla q|^2 \rangle_q}{\langle |\nabla s|^2 \rangle},
$$

one configurational scale seen through two gradients. [rates.md](rates.md)
takes it from here.
