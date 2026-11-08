# Feature forecasts from the proposed parameter step

Frozen for task #188 on 2026-11-08, before measuring forecast outcomes.
Extend tasks #186/#187 on all 24 original cases, all original input/PTX/state
guards and both complete splits. Preserve the old-margin bottom deciles.
Step-10000 probes remain outside the original training budget.

## Information boundary

The old parameters and the proposed rounded parameter increment are known
from the restored state and the next scheduled training minibatch. Construct
the candidate update in memory, then freeze the entire forecast before any
forward evaluation at the candidate parameters. No new features, logits,
losses or measured execution discrepancies enter a forecast or its radius.
Ordinary source train_step must still reproduce the candidate state bitwise.

Propagate the old feature value a, its first-order directional change v,
and a Euclidean error radius r around a+v. The embedding gather is affine,
so its initial radius is zero. All matrices use their Frobenius norm as a
conservative operator bound. No fitted constants, outcome-based tuning,
new training, or architecture selection.

## Propagation rules

For the linear layer with proposed matrix A+D, propagate tangent Av+Da
and radius `||A+D||_F*r + ||Dv||`. The mixed term Dv is known from the old
tangent and parameter step. Residual addition adds error radii.

For a nonlinear map f, evaluate its old derivative on v. Compute its local
defect `||f(a+v)-f(a)-Df(a)v||` at the known nominal input. If f obeys an
anchored Lipschitz bound L at a+v, the output radius is `L*r+defect`.
Nominal nonlinear evaluations supply the scalar defect only: their values
are never propagated as the next layer's feature. Thus the forecast remains
the old-network tangent, rather than an exact updated-network forward pass.

Use the exact finite RMS bound `L=2/s(a+v)`, where
`s(z)=sqrt(mean(z^2)+epsilon)` and epsilon is the source FP32 epsilon.
The regularized pair radius sqrt(a^2+b^2+epsilon) is 1-Lipschitz; sigmoid
is 1/4-Lipschitz. Their local defects are computed directly, without a
global small-denominator Hessian estimate. All original normalization
sites and the radial gate are retained.

For coordinate or complex bilinear products, keep the old directional
derivative as tangent. Bound the remaining error by the norm of the known
tangent-tangent product, plus nominal block maximum norms times input
radii and their product. Complex multiplication keeps its source 1/sqrt(2)
factor; the diagonal and real products retain their own source equations.

## From feature balls to outcomes

Compute predicted logits with the proposed rounded readout and the frozen
feature center. Each competitor gap radius is the readout-contrast norm
times the feature radius, plus the declared numerical logit allowance.
Enclose every class gap, then worst-competitor margin and actual CE via
log-sum-exp. Certify strict correctness, margin-change and loss-change
directions only with positive slack of 1e-8*(1+largest relevant magnitude).
Retain all inconclusive rows and old errors. Compare with #187's retrospective
joint intervals, explicitly distinguishing their information sets.

## Numerical checks and limits

The real-arithmetic forecast has a separately declared, fixed FP32 feature
allowance of **1e-3 in Euclidean norm per query** and an additional executed
readout-logit allowance of **1e-3 per class** (2e-3 per contrast). These are
audited numerical assumptions, not a formal verification of CUDA arithmetic.
They must not be increased after observing results. Check all old/new feature
allowances against an independent FP64 forward evaluation, and all executed
logit allowances against dot products at the actual GPU features. A violation
invalidates that case's numerical certificates and remains in the report.

Use 1e-8 scaled tolerance for FP64 enclosures. Check the forecast tangent by
central differences on the fixed first eight training rows, and test nonzero
mixed terms, nonlinear defects, cancellation and arbitrary directions in
native fixtures. Keep task #186's derivative and task #187's query checks.
Runtime cap is 600 seconds excluding compilation; retain capped/failed cases.

Report all 24 cases, both full splits and fixed difficult subsets: actual
versus bounded feature forecast error, old/new losses and margins, certified
directions, classification coverage, interval widths, all violations and
inconclusive counts. Preserve per-query forecasts/radii and generated evidence
outside the worktree, untracked. INDEX.md remains the generated-file exception.
Tight coverage would establish a checked local forecast given a proposed
step, not convergence, optimizer discovery or statistical generalization.
