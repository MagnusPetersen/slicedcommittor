# RECOVAR transfers to the sliced committor — implementation and test results

All nine ideas from `claude/NOTE_cryoem_transfers.md` implemented and tested
against a finite-difference PDE reference. Four hold up, two hold up only
after their criterion was replaced, one is refuted, one turns out not to be
needed, and one is a large win with a sharp caveat.

---

## Setup

Self-contained numpy/scipy port of the method (`slicedcv/core.py`): random
directions → RD slice profiles (stiff-limit tridiagonal solve) → Gram matrix
→ basin moments → dual solve. The dual form of `NOTE_dual_identity.md` is
used throughout, `h(w) = 2(b−a)'w − w'Gw`, since it makes held-out evaluation
a plain sample average.

**Test system.** Rotated Wolfe-Quapp, sparse-direct FD committor on a 401²
grid as ground truth, plus a 2D double well and a separable ND version
(`V = V_2d(x[:2]) + k/2|x[2:]|²`, so the 2D PDE solution is exact ground
truth in any dimension).

**Temperature matters, and the project's default is unusable for this.**
At β = 2.0 only **0.02 %** of transition frames have q ∈ (0.1, 0.9) and the
Monte-Carlo estimate of D[q] disagrees with grid quadrature by **1.6×**
(and by 2.7× at N = 30 000): the barrier is unsampled, so every
Dirichlet-energy quantity is meaningless and the RMSE is trivially ~0.002 for
any method. β = 1.0 gives a 3.9 % barrier population and MC/grid = 1.01. All
results below use β = 1.0. *Worth checking which β the paper's Table
`tab:2d_params` uses — if it is cold, the quoted RMSE 0.007 is measuring
almost nothing.*

**Validation.** At β = 1.0 the port gives transition-region RMSE 0.0037–0.0042
(paper: 0.007), κ-insensitive from κ_rel = 10² to 10⁸, and an oracle-LS fit in
the same slice basis reaches 0.0014 — reproducing the ~3× under-determination
gap documented in the Limited-Data note.

---

## Results by section

| § | Idea | Verdict |
|---|---|---|
| 3 | Held-out dual objective | **Works.** Bound violation confirmed and quantified |
| 1 | Halfset regularization of G | **Works via eigen-bands.** The a-priori "shell" version fails |
| 2 | Committor resolution | **Works via Laplacian bands.** Real-space smoothing fails |
| 5 | Adaptive profile bandwidth | **Works** once the criterion is the profile, not the density |
| 9 | Nested-M curve | **Works** |
| 7 | Uncertainty quantification | **Works when variance dominates**; blind to bias |
| 6 | Flux-weighted basin moments | **Not needed** — the approximation is genuinely small |
| 4 | Matrix-free / large M | **Refuted** as a cure for high d |
| 8 | Feature mask | **Largest single win**, but fragile |

---

### §3 Held-out dual objective — works, and the bound really does break

`h_train` rises monotonically with M while `h_test` peaks and falls. At
N = 60 000 the in-sample flux ratio `1/M_gap ÷ ν_AB` runs

    M   =   8      16      32      64     128     256     512
    ratio  1.006   0.999   0.997   0.984   0.969   0.941   0.923

crossing below 1 at M ≈ 16 and degrading monotonically thereafter — the
predicted overfitting bias. At N = 15 000 it reaches 0.251 at M = 512, and
`h_test` goes **negative** by M = 64 (the fitted weights are worse than w = 0
on held-out data). The held-out value never violates the bound.

**Corrected claim.** My note proposed `1/h_train ≤ ν_AB ≤ 1/h_test` as a
bracket. Across 5 seeds × 3 systems × 2 sample sizes it holds only **8/30**
times. The two *biases* are real and have opposite signs, but other biases
(barrier undersampling, regularization) can push both ends the same way — on
the double well both ends sit above ν. So: **the train–test gap is a reliable
overfitting diagnostic; 1/h_test is not a confidence-interval endpoint.**

**Practical payoff.** `argmax h_test` over nested M picks the RMSE-optimal M
exactly in 2 of 3 cases and within 1.7× in the third. It also correctly ranks
regularizers and — see §5 — correctly flags a bad bandwidth rule. It does
*not* select the bandwidth itself (it prefers under-smoothing).

### §10 Block splitting — the caveat is not marginal, it inverts the sign

On Langevin data with equilibrium marginal and τ_int ≈ 55 frames:

| split | reported overfitting at M = 256 |
|---|---|
| random frame | **−2.7 %** (i.e. "no overfitting") |
| contiguous blocks, len 20 | +31 % |
| contiguous blocks, len 100 | +59 % |
| whole trajectories | +64 % |

Random splitting does not merely understate the diagnostic, it reverses it.
The diagnostic is still climbing from block length 100 to whole-trajectory,
so **check convergence in block length**, don't just pick one.

### §1 Halfset regularization — the pragmatic shell index wins, the faithful one fails

The eigen-band version (Wiener ridge in eigen-bands of Ḡ, SSNR from halfset
correlation) is tuning-free and beats a ridge **tuned against the truth**:

| case | ridge 1e-3 (current default) | best ridge (oracle) | halfset-eigen |
|---|---|---|---|
| N=60k, M=512 | 0.0042 | 0.0037 | **0.0036** |
| N=15k, M=512 | 0.0073 | 0.0038 | **0.0033** |
| N=6k, M=256 | 0.0450 | 0.0241 | **0.0120** |

and the flux ratio stays 0.98–1.02 where the default ridge gives 0.70–0.92
(and 0.076 at N = 6000 — a 13× error). Across 5 seeds it beats the oracle
ridge 20/30 times, winning 5/5 on both Wolfe-Quapp temperatures (mean RMSE
ratio 0.49–0.92) and losing 0/5 on the double well (ratio 1.09–1.15), whose
near-1D committor gives a well-conditioned Gram where a ridge suffices. It
wins where the problem is hard.

**The faithful port fails.** Indexing slices a priori by their own profile
bandwidth and regularizing per shell-pair — the literal analogue of RECOVAR's
SI S.C — does nothing: the measured shell SSNRs come out at 10³–10⁵, so the
Wiener factor is ≈ 1 and no shrinkage happens. The two SSNR estimators
(correlation-based and noise-power-based) disagree at corr ≈ −0.02 to 0.37 in
that grouping, which is itself the signal that the grouping is wrong. **The
sampling noise is not diagonal in any a-priori slice-property basis; it is
diagonal in the eigenbasis of G.** In eigen-bands the picture is stark: for
M = 512 in d = 2, only 1–3 of 12 bands carry SSNR > 0 at all.

### §2 Committor resolution — needs a genuine shell decomposition

Correlating *smoothed* half-committors saturates at 0.9998–1.0000 at every
scale: the committor's 0→1 sweep across the domain dominates every band. So
does a real-space band-pass (successive smoothing differences), at 0.93–0.99
flat. Only the **graph-Laplacian eigenband** version produces an FSC-shaped
falloff, and it moves out with data volume:

    N =  3 000 : 1.00  0.97  0.24  0.36 -0.05 ...   crosses 0.5 at band ~22
    N = 15 000 : 1.00  0.99  0.79  0.62  0.62  0.25 ...            band ~68
    N = 60 000 : 1.00  1.00  0.95  0.95  0.69  0.86  0.73 ...      band ~128

corr(resolution band, −log RMSE) = 0.67 across five sample sizes — it tracks
the true error, which it never sees, but noisily (10 bands on a 2500-point
subsample; N = 6000 is an outlier). **This is the most promising unfinished
item.** It needs more eigenvectors, a larger subsample, averaging over
splits, and a calibrated threshold before it is quotable.

### §5 Adaptive bandwidth — works once the objective is the profile

A density-ISE (Rudemo/Bowman) criterion is the wrong objective and is
1.36–2.8× *worse* than the best global bandwidth. Replacing it with a halfset
CV risk on the profile — score each candidate against the *other* half's
finest-bandwidth profile, per s-region — fixes it. Scoring the derivative
q′ (what actually enters the Gram) is best:

| N | best global (tuned on truth) | profile-CV | derivative-CV |
|---|---|---|---|
| 6 000 | 0.0113 | 0.0104 (0.92×) | **0.0092 (0.81×)** |
| 15 000 | 0.0054 | 0.0052 (0.95×) | **0.0051 (0.93×)** |
| 60 000 | 0.0039 | 0.0038 (0.97×) | **0.0033 (0.84×)** |

It beats the best hand-tuned global bandwidth without seeing the truth, and
removes `n_bins`/bandwidth as a tuned parameter. Bandwidth matters most where
data is thin (RMSE varies 4× across the grid at N = 3000).

### §6 Flux-weighted basin moments — the paper's approximation is fine

Measuring the true flux-weighted φ_j from the PDE gradient: mean relative
bias of φ̂ = b − a is **+1.5 %**, correlation 0.974. Solving with the *true*
φ gives RMSE 0.0034 vs 0.0033 with φ̂ and an identical flux ratio. The
self-consistent iteration works as designed — it drives corr(δ_flux, φ) from
0.974 to 0.997 — and changes nothing downstream. **The paper's claim that
this error is small is confirmed; the deconvolution fix is unnecessary.**
(2D, well-separated basins; the high-d case is untested.)

### §4 Matrix-free / Nyström — correct, but refutes its own motivation

The matvec identity `Gv = Σ_a Y_a ∘ (P(P'(Y_a ∘ v)))` verifies to 5×10⁻⁸
relative error, and streaming (no O(NM) storage) matches. But:

- **CG is not viable**: 470 iterations for M = 64. The redundancy that makes
  G low-rank is exactly what makes it ill-conditioned (λ₁/λ_M = 8×10¹² at
  M = 512).
- **Nyström is the right route** and confirms the redundancy: rank 64 of
  M = 512 reproduces the full solve (0.0068 vs 0.0064); 99 % of the trace sits
  in 30 directions. **512 random slices in d = 2 carry ~64 dimensions of
  information.**
- **No timing win** at these sizes (0.7–0.8× vs explicit BLAS-3 assembly);
  the asymptotic advantage is swamped by constants below M ≈ 2000.

**The strategic hypothesis is refuted.** Large-M isotropic does *not* retire
the LDA sampler once d ≳ 12:

| d | isotropic M=256 | isotropic M=2048 | LDA M=256 |
|---|---|---|---|
| 6 | 0.0490 | **0.0168** | 0.0178 |
| 12 | 0.1182 | 0.0695 | **0.0246** |
| 24 | 0.1424 | 0.1294 | **0.0569** |

At d = 6 brute force matches the LDA sampler (8× the directions, 9× the
time). At d ≥ 12 it does not close, and barely improves from M = 64 to 2048.
You cannot buy your way out of the concentration of measure.

### §8 Feature mask — the largest win, and the most fragile

One isotropic pass, read the per-feature contribution to b − a, mask to the
top 2, run a second isotropic pass inside the mask. It identifies features
{0,1} correctly at d = 12, 24, 48 and then:

| d | one-pass isotropic | LDA M=256 | two-pass mask |
|---|---|---|---|
| 12 | 0.0960 | 0.0277 | **0.0036 (26.7×)** |
| 24 | 0.1273 | 0.0576 | **0.0036 (35.5×)** |
| 48 | 0.1548 | 0.0945 | **0.0036 (43.1×)** |

RMSE 0.0036 at d = 48 is the *d = 2* result — the mask removes the
dimensionality penalty entirely, and recovers ν to 0.5 %.

**But it only works if the feature basis is meaningful.** Apply a random
rotation to feature space (same physics, same committor, no sparse feature
set) and masking becomes 2–4× *worse* than doing nothing:

| d | one-pass | mask, axis-aligned | mask, rotated |
|---|---|---|---|
| 12 | 0.1217 | 0.0036 | **0.2686** |
| 24 | 0.1458 | 0.0036 | **0.5679** |

The LDA sampler is rotation-equivariant and barely moves (0.0277 → 0.0299).
So the mask is a big win for genuinely sparse feature sets — torsions and
contacts plausibly qualify — and a liability otherwise. **The obvious fix is
to mask a learned *subspace* (leading LDA/PCA directions) rather than raw
coordinates**, which would keep the gain and the rotation-equivariance. The
mask size also matters (top-8 was as bad as no mask) and needs a selection
rule; whether held-out h detects the rotated failure is untested.

### §7 Uncertainty quantification, §9 nested-M

Delta-method and block-bootstrap standard errors on ν agree with each other
to ~10 % and match the empirical spread over 20 independent datasets at
N = 30 000 (3.07e-4 vs 3.07e-4, 65 % coverage against a 68 % target). At
N = 6000 they are **6× too small** — because there the error is dominated by
*bias* (mean ν̂/ν = 0.74), which a variance estimate cannot see. Same
bias/variance split as everywhere else: CV and bootstrap catch variance;
nothing here catches barrier undersampling.

Nested-M is free and works (see §3).

---

## A correction found by verifying

The first verification pass failed all three headline claims. Root cause: a
degeneracy guard I had added to `rd_profile` scaled its ridge by
`max|diag|`, which the stiff absorption term dominates — swamping the
diffusion-scale entries and corrupting every profile. Fixed (ridge applied
only in the genuinely singular absorption-free case, scaled to the diffusion
scale) and confirmed by exact regression against the pre-bug numbers. Every
result above is post-fix. Worth noting that a plausible-looking numerical
guard silently degraded the committor by 20× without raising anything.

---

## What I would do next, in order

1. **Adopt halfset-eigen regularization and the held-out h diagnostic.** Both
   are small, tuning-free, and immediately change reported numbers. Re-check
   the chignolin rate: in-sample `1/M_gap` is biased low, and the paper's M
   was not selected by a held-out criterion.
2. **Fix the block-splitting rule before anything else uses splits.** On MD
   data a random frame split reports the wrong sign.
3. **Mask a learned subspace, not raw features** — the §8 gain is too large
   (26–43×) to leave on the table, and the rotation failure has an obvious fix.
4. **Finish the Laplacian resolution metric.** It is the only item here that
   could stand as its own contribution.
5. **Drop §6 (flux-weighted moments) and the §4 large-M argument.** The first
   is unnecessary; the second is refuted. Keep Nyström only as the statement
   that the slice basis is ~64-dimensional, which is a useful diagnostic.
6. Check what β the paper's 2D figure uses.

---

## Files

    slicedcv/systems.py    potentials, sampling, sparse FD committor reference
    slicedcv/core.py       the method (basis, Gram, dual solve, evaluation)
    slicedcv/recovar.py    all nine transfers
    t01..t09*.py           one test per section; logs in out/
    mkfig.py, figs/summary.png
