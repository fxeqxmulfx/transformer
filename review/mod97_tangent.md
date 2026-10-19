# Frozen MLP Jacobians and their GD control

Tasks #75 and #76 of M9.4 connect an exploratory derivative measurement to
an exact-real statement about its explicitly frozen prediction model.
The source is `landscape_grokking_plan.md`, Mechanisms to test, and
`experiments/mod97_representation_controls/access.py`. This is intermediate
progress toward the actual-network attraction question.

## Statement and source comparison

`Neural.Tangent.Basic.loss J r n x` is `||J x + r||^2 / (2 n)`, where `x`
is the parameter displacement, `J` is a fixed continuous linear prediction
derivative, and `r` is the initial prediction minus target. The identity
`loss_eq_mean_squared` unfolds the Euclidean output norm into the sum of
row errors, with no extra division by output width. The diagnostic uses
512 training rows for this frozen model. Its separate interaction-energy
calculation uses all 9,312 equations; those two calculations must not be
identified with one another or with the actual 4,656-row training split.

The Frechet derivative of the norm objective gives parameter gradient
`J.adjoint (J x + r) / n`. `gradient_loss_of_hasFDerivAt` explicitly
identifies the kernel with a certified derivative of an actual prediction
map at its stated base weights, and uses that map's value in the residual.
It does not assume that the nonlinear map retains this derivative elsewhere.

The existing Landscape `gradientStep` applied to that actual quadratic
therefore produces residual update `r_k - eta * K r_k`, with
`K = J J.adjoint / n`. Adjoint identities prove symmetry, and a positive
example count proves positivity and nonnegative unit-mode eigenvalues.
The derivative identities do not require a positive count; the loss bounds
do. Empty-data division conventions are not used in the measurements or
the nonempty witnesses.

`inner_residual_iterate` proves exact mode amplitude
`(1 - eta * mu)^k * inner(v, r_0)` for an eigenvector `K v = mu v`.
It follows by induction on genuine parameter GD, rather than defining an
optimizer to have the desired residual recurrence. Cauchy--Schwarz turns
unit-mode amplitude into a lower bound on the full loss. An exact null
mode retains its entire initial contribution at every update.

For `eta * mu <= 1` and `k * eta * mu <= 1/2`, Bernoulli's inequality
gives `(1 - eta * mu)^k >= 1/2`. The loss is consequently at least
`inner(v, r_0)^2 / (8 n)`. No sign assumption on eta is needed for this
lower bound; the diagnostic chooses a positive step and the positive-kernel
theorem supplies nonnegative eigenvalues. This is a finite-budget bound,
not a proof of an asymptotic obstruction.

The concrete feature is `J x = x / 10`, with one example, initial residual
one, unit mode one, kernel eigenvalue `1/100` and step one. After fifty
updates the loss is at least `1/8`, while displacement `-10` attains zero
loss in that very same model. Explicit examples realize every conditional
theorem's premises, including the derivative certificate, positive count,
unit direction, exact null direction and positive slow mode.

## Numerical scope

The diagnostic differentiates the executable scalar MLP with every matrix
and bias free, at the original initialization and final checkpoint of all
twelve scalar-output controls. It promotes the exact float32 coordinates
to CPU float64 for these measurements. Full-table double centering removes
additive row/column effects from the measured Jacobian and target, without
dropping zero numerators or changing training. Raw and centered energies
and both target and actual-residual contractions remain distinct fields.

The frozen kernel uses the first 512 training and first 512 held-out rows
of each original source permutation. It follows full-batch GD at
`1/lambda_max` by spectral powers, not the experiments' minibatch AdamW.
Eigenvalues at most `1e-12 * lambda_max` are treated as zero numerically;
every measured eigenvalue is retained in the artifact. That cutoff does
not establish an exact nullspace or instantiate the null-mode theorem.
Long extrapolations of the linear model do not certify a Taylor
approximation of the nonlinear model at large parameter displacements.

Independent per-example autograd checks every batched Jacobian entry on
small scalar and one-hot networks. Parameter finite differences check its
centered response, and direct loss differentiation checks the target
contraction without constructing that Jacobian. A separate linear example
with three rows, two parameters and a genuine nullspace checks the kernel
powers against explicit full-batch parameter GD updates.

The resulting geometry is a local measurement. It does not establish
AdamW convergence, a random-start attraction basin, a scalar-MLP
impossibility, or the original Rotate90 network's generalization mechanism.

## Validation

The full tree builds successfully (12,780 jobs); the new modules have no
warnings or `sorry`. All thirteen new theorems use only `propext`,
`Classical.choice` and `Quot.sound` in `#print axioms`. The global audit is
`39 sorry · 0 resting on a sorry · 0 extra axioms`. The regenerated index
has 3,683 modules and 18,919 theorems, with zero vacuous statements and
zero placeholders. The forbidden and whitespace checks pass.

The existing full Python gate checks 339 tests in 2,357.907 seconds, with
three skipped; all others pass. All four new `MLPAccess` tests pass in a
separate focused run. The complete diagnostic exits successfully for all
twelve cases, and an independent file audit verifies every checkpoint hash
and its analysis-source hash. The 36-case training grid is still running;
this milestone does not claim its final comparison results.
