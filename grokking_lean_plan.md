# Grokking: an active experiment-to-Lean research cycle

Started 2026-10-07 at the user's explicit request. Status: active. Work solo.
The [Basis/convex cycle](plan.md) remains paused and unfinished; ANSR stays
stopped. Continue the [measurement study](grokking_plan.md) with ordinary
softmax and unchanged native AdamW. Every declared run retains 150,000
updates. The research goal has no requested token or time limit.

## Objective and limits of the claims

Explain delayed generalization through precise, competing mathematical
formulations and formalize the defensible statements in Lean. Systematically
expand the formulations below as new evidence warrants. A finite list is
not a claim to exhaust every possible explanation. An empirical transition,
a simplified model and a theorem about the actual GPTMini training system
are separate results; the transfer must be proved, not hidden in a definition.

## Formulations to investigate

| Route | First mathematical question | Evidence / source | Status |
| --- | --- | --- | --- |
| Operational delayed generalization | Define train fit, a sustained held-out plateau, later generalization, and finite-budget censoring without future information entering a detector | Power et al., arXiv:2201.02177v1, sections 1 and 3.1; pinned causal histories | Twenty-seven proved threshold, prefix, confirmation, sustained-window, delay-bound and bounded-continuation laws/counterexamples; the full Python heuristic remains open |
| Confidence versus decisions | Positive logit scaling preserves every ordering but can strictly decrease cross-entropy; quantify the missing conditions and counterexamples | Prieto et al., arXiv:2501.04697v1, section 4.2; measured endpoint projections | Six proved theorems in `Transformer.Grokking.NaiveLoss.Section4_LogitScaling`; no sorry |
| Spectral optimization dynamics | Derive slow modes and exact delayed test-boundary crossing from an actual gradient flow or update recurrence, including convex toy models | Liu et al., arXiv:2205.10343v2, effective embedding dynamics; Žunkovič/Ilievski, arXiv:2210.15435v1, section 3 | Nineteen perceptron theorems; fifty-seven effective-model laws/counterexamples, including noncollapse and effective-loss convergence from initial ground data |
| Rule learning versus memorization | State when a reusable rule component wins over an example-specific component under the same training objective | Nanda et al., arXiv:2301.05217v1, sections 4–5; Varma et al., arXiv:2309.02390v1, appendices C–D | Sixty-nine allocation/CE/gradient/update laws and counterexamples: both budget regimes, attained actual CE minima, correct fixed-table decisions, actual subweight gradients and finite GD rates/absence obstruction; native AdamW rule selection remains open |
| Compositional circuit formation | Distinguish zero coordinate gradients from a local minimum when a useful computation needs multiple learned components | Nanda et al., appendix Further speculations on grokking, Hypothesis: Phase Transitions are inherent to composition | Seventy-four proved actual-CE component, saddle, stationary-state, class-centered output and decision-transfer results/counterexamples; 644 real head-pair loss observations and 644 stage/output observations reject interaction-only detection; causal multi-step formation remains open |
| Geometry of representations | Prove orbit projection identities, scale/bias invariances and counterexamples to symmetry-only success; connect train-fitted probes to held-out decoding | Division common-scaling observer, actual checkpoint features | Fifty-five proved mean, energy, margin, cleanup and integer-encoding laws/counterexamples; exact certificates measured on all 238 preserved snapshots; 266 Python tests passed |
| Division-task symmetry | Derive the diagnostic cells from the actual numeric task rather than assuming their labels or orbit interpretation | Power et al., section 3.1; author-code corpus and GrokkingObserver at 43d4d66 | Nineteen proved generator, valid-domain, orbit-equivalence, disjointness and finite-cardinality laws/counterexamples; token/Python implementation bridges remain open |
| Gradient coherence and implicit bias | State what batch gradient agreement can and cannot imply; distinguish loss descent, task structure and optimizer-specific bias | Fixed-batch gradients; ordinary AdamW | Eighteen proved actual-CE derivative/decomposition/counterexample theorems; shared targets can yield arbitrarily high full alignment despite opposed answers |
| Regularization and norms | Relate an explicitly stated penalty or decay update to competing solutions; reject norm-only success claims | Golechha, arXiv:2405.12755v1, section 3; current grouped norms | Whole-model norm barely changes across the observed transition |
| Phase transitions | Specify an order parameter, control parameter, asymptotic regime and distribution before claiming a thermodynamic transition | Liu effective theory; Žunkovič/Ilievski solvable models | Analogy only for current finite GPTMini; finite-size scaling not established |
| Numerical precision and softmax collapse | Compare exact-real loss gradients with floating-point zeros and prove only the quantization model actually used | Prieto et al., section 3; local CUDA/CPU execution | Observed-logit integer certificate bridge and eight conditional native first-step gradient-error bounds/counterexamples proved; actual floating-point forward, autograd error generation and softmax collapse remain open |
| Actual transformer and AdamW transfer | Identify which premises about the real forward map, token task and optimizer are verified, and which remain unproved | Existing GPTMini List Int semantics; native lab checkpoints | Eighty-seven update/curve laws/counterexamples: state-dependent descent, finite overshoot, conditional gradient-error bounds, reachable momentum ascent, compositional CE growth, interval derivative bounds and initial-Hessian counterexamples; repeated-step rule convergence remains unproved |

## Cycle

1. Read each local manuscript and pin its version, section, definitions,
   hypotheses and quantifiers. Search library lemmas with DT.
2. State the claim as a Lean theorem. Prove a corrected source statement
   when fixable; prove a counterexample when false. Add a satisfying
   example for every hypothesis list. Never use an axiom or move the
   requested result into a definition.
3. Find an observable consequence and a control that can reject it. Run
   the ordinary transformer or read immutable weights; probes and
   ablations never alter training or supply a teacher to the model.
4. Compare with all available seeds and negative controls. Retain failures
   and distinguish a causal signal from a retrospectively selected explanation.
5. Build/audit/index/check Lean changes; run all Python tests for a Python
   change. Commit each checked logical result immediately. Update this
   plan with proved declarations, measurements and remaining gaps.
6. Continue with the next unsolved route or a revised formulation. Do
   not mark the research direction complete merely because a toy
   example, a metric or a long run succeeds.

## Current evidence and immediate work

The original GPTMini seed 1 completed its full 150,000-update budget with
100% held-out answer accuracy. The checkpoint-preserving repeat also
completed 150,000 updates: all 601 canonical train/held-out observations
match the original exactly, including the zero-step observation.
Frozen probes, spectra, neuron profiles, gradients, head/subspace ablations
and endpoint logit changes have been measured around the actual
33,000–36,000 transition. These are associations and named intervention
effects, not a universal detector. The reference control completed all
150,000 updates with about
1.50% held-out answer accuracy; absence of a later transition is not proved.
Seed 2 and seed 3 both complete 150,000 updates with 100% accuracy, first
exceeding 99% at 1,250 and 750 compared with 35,500 for seed 1. These are
early learning comparisons, not second delayed-transition replications.
All declared training runs, checkpoint archives and six-measurement
workers have completed. No training worker is left running.

The six-probe implementation and frozen results are committed as `4436290`.
All 249 Python tests passed, including CPU/CUDA noninterference checks.
The first Lean module proves standard-CE equivalence, common-shift
invariance, positive-scale decision invariance, strict loss reduction under
explicit margin conditions, their combination, and the all-tied countercase.
The complete Lean build, audit, index generation and forbidden check pass:
the existing 157 sorry declarations are unchanged and the new module adds none.

`Transformer.Grokking.Perceptron.Section3_SymmetricMSE` derives the actual
two-example MSE, its exact strong Jensen remainder, joint convexity, partial
derivatives, an exponential negative-gradient flow, and strict training-loss
decrease. `Section3_DelayedDecisions` derives an arbitrarily delayed finite
test-accuracy jump with perfect train classification and differentiable
scores. The delay is varied through the test task's margin; the theorem is
not arbitrary delay on one fixed task and is not about AdamW. A zero-bias
case also disproves automatic delayed generalization from separability alone.
These modules add nineteen theorems and no sorry declarations.

`Transformer.Grokking.EffectiveTheory.Section3_QuotientGradient` verifies
actual partial derivatives of a normalized parallelogram loss. The
conservation module proves the radial identity, complete norm conservation
for classical flows on the nonzero domain, and the corrected centroid
derivative. The unqualified centroid-conservation claim fails without a
centering condition. A centered, nonzero-loss stationary point also refutes
replacing the quotient gradient by the numerator gradient divided by the
norm. Constancy of a denominator along a flow does not justify discarding
its partial derivatives. These modules add sixteen theorems and no sorry.

`Section3_ScalarMode` proves scalar-ODE uniqueness, sign and decay,
threshold behavior, the factor-exp(-1) characteristic time, energy and
time-shift identities. `Section3_RelativeModes` recovers the spectral
mechanism from actual quotient partial derivatives: the radial correction
cancels in `(E0 + E2 - 2 E1) / (E0 - E2)`, giving rate `12 / Z0` and exact
exponential decay when the ground component stays nonzero. This is a
one-constraint specialization with constant averaging factors suppressed;
the coefficient is not claimed for an arbitrary dataset or GPTMini.
These modules add sixteen theorems and no sorry, bringing the new grokking
formalization to 57 proved theorems.

`Section3_BoundedGrowth` and `Section3_GroundDynamics` derive the previously
assumed all-time ground-component nonvanishing from a nonzero initial
component. The actual quotient gradient gives coefficient `2 * numerator /
Z0^2`; the residual bound `numerator <= 6 * Z0` and conserved norm bound
it uniformly. Squared amplitude and weighted squared amplitude prove
nonvanishing forward and backward in time, with an explicit interval bound.
The strengthened spectral formula now has only the initial ground-mode
premise, while the nonzero representation domain and classical-flow
existence remain explicit. These modules add twelve proved theorems and
no sorry, bringing the grokking formalization to 69 theorems.

`Section3_InitialDomain` proves that a differentiable path obeying the
quotient-gradient equation only off the origin has constant squared norm
everywhere. At the origin the squared-norm derivative is zero directly;
elsewhere the actual radial-gradient identity applies. Nonzero initial norm
therefore excludes the origin without an all-time domain hypothesis.
`Section3_InitialModes` then derives ground nonvanishing and exact relative
mode decay from initial ground data alone, bounds the normalized loss by
`2 * initial_relative_mode^2 * exp(-24 * t / initial_norm)`, and proves
convergence to zero effective loss. Existence of a global differentiable
ODE path remains an input. A classifier, discrete optimizer and arbitrary
dataset are not identified with this one-parallelogram model. These modules
add thirteen proved theorems and no sorry, bringing the total to 82.

`Transformer.Grokking.AdamW.FirstStep` derives the first bias-corrected
direction from zero moment buffers and the derivative of the actual finite
update's squared norm. At the nonzero quotient-loss point `(1, 2, 0)`,
the true Euclidean gradient is tangent but the adaptive direction is not.
Its norm derivative is strictly negative for every positive epsilon and
nonnegative decay, including zero decay. Finite updates with betas
`(0.9, 0.98)`, rate `0.001`, epsilon `1e-8`, with and without decay `0.1`,
also lower the norm. These six theorems are exact-real counterexamples to
an unrestricted conservation transfer, not to grokking or to this actual
GPTMini trajectory. No sorry is added; the total is now 88.

The checked objective decomposition is committed as `d5bf4b4`. Across all
three initialized GPTMinis, EOS supplies most of the full train/held-out
gradient dot product. Seed 1 has full cosine 0.9524, answer cosine 0.3154,
and EOS–EOS contribution 0.9297 to the full cosine. By 1,000 updates EOS
is negligible and answer alignment is still 0.8567 before generalization.
At the delayed transition answer alignment decreases. Both shared targets
and answer-only false signals must therefore enter the formalization.

`Transformer.Grokking.GradientEvidence.BinaryObjectives` specializes
ordinary finite-class softmax CE to two independent readout coordinates,
averages answer and common-target losses in the actual protocol's
proportions, and proves every partial derivative. `SharedTarget` proves
the signed energy and four-term dot decompositions and the exact cosine
`(R^2 - 1) / (R^2 + 1)`. For every positive tolerance there is a positive
shared feature scale with full alignment exceeding one minus that
tolerance, while the answer-only gradients remain opposed and both
initial losses equal `log 2`. The loss is actual CE, not a surrogate;
the two-coordinate readout remains an explicit deviation from GPTMini.
These eighteen theorems add no sorry, bringing the total to 106.

`Transformer.Grokking.Operational.Basic` separates a first observed
threshold crossing from later confirmation and proves their finite-prefix
invariance. The clock identity explains the study's 1,000-step confirmation
latency for five observations spaced by 250; it is measurement latency,
not a law of learning. `Censoring` constructs bounded success and failure
continuations of every valid below-threshold prefix. Their recorded lists
are exactly equal, so no finite accuracy-only observer can decide eventual
crossing soundly and completely over all bounded traces. No optimizer
dynamics or full deterministic state is imposed; the impossibility is
model-free, not an impossibility for AdamW under additional premises.
These twelve theorems add no sorry, bringing the total to 118.

`Transformer.Grokking.Geometry.Basic` computes the cell mean as the
actual finite sum divided by cardinality. It derives zero residual sum,
orthogonality, exact energy decomposition, the best constant fit and its
uniqueness on nonempty cells, then proves linearity and idempotence.
Empty cells are handled explicitly and supply no coverage evidence.
These are logit-space identities, with no assumed orthogonality and no
claim of convexity in transformer parameters. They add eleven proved
theorems and no sorry, bringing the total to 129.

`Transformer.Grokking.Geometry.Cells` lifts the actual means to all
selected quotient cells and classes, preserving their different mask
sizes. It proves the exact three-energy identity, contractivity, unit
interval bounds under positive total energy and equivalence of fraction
one with zero residual energy. A locality theorem excludes logits outside
the selected cell from its mean. The quantities are exact-real sums;
Python's energy floor, coverage requirement, None value and roundoff
clipping remain additional numerical policies. These eight theorems add
no sorry, bringing the total to 137.

`Geometry.Decisions` proves a finite-class correctness certificate:
a correct reference margin `m > 0` survives residual squared energy `R`
when `2 * R < m^2`. Failure requires at least that much energy; a binary
tie proves the strict boundary sharp. `Cleanup` links pointwise error to
the actual masked cell residual, restores global class bias, discards
only harmless row shifts, and derives the equivalent criterion using
`R = (1 - structural_fraction) * total_energy`. Mean energies need the
retained-coordinate count before use in the global sum criterion.
The reference's correct margin remains an explicit, observable premise;
no optimizer is assumed to learn it merely from symmetry.

`Geometry.SymmetryCounterexample` constructs two balanced cells with both
labels swapped. Row shifts and global class bias are zero, every cell
has two points, total energy is positive and the structural fraction is
one, yet every prediction is strictly wrong with target gap `-2`.
It therefore fails the missing positive-margin premise. These eighteen
theorems add no sorry, bringing the total to 155.

`Geometry.RowAlignment` derives the actual error measured after harmless
row alignment, preserves class bias in the reference, and proves row-shift
and residual class-bias invariance. `IntegerEncoding` derives the reader's
integer residual numerator from finite means, including every class and
observation factor. `IntegerCertificate` proves exact equivalence with
the real cleanup test and certifies the original observed raw row. The
integer matrix and positive common scale are inputs; Lean does not yet
verify Python's float conversion or the transformer forward program.
These eighteen theorems add no sorry, bringing the total to **173**.

Both read-only readers are complete. All 238 preserved snapshots meet
the nonzero-cell coverage and energy protocol; 759,256 positive
certificates across 1,159,032 current-input evaluations include zero
incorrect answers. The pinned 23-snapshot subset agrees exactly with
the full reader. No training run or budget changes. Seed 1's cell mean
first reaches 99% at a retained 34,000 snapshot, before the canonical
raw 35,500 crossing, while certificate coverage first reaches 99% at
38,000. Coverage later falls at 40,000 despite high accuracy; seed 3
finishes at 100% accuracy with 83.72% coverage. The sufficient criterion
confirms current robustness, not future generalization or monotone progress.
The conservative global-residual version certifies no current answers.
See [the detailed scope and figure](experiments/grokking_internals/README.md)
and [the complete observations](experiments/grokking_internals/geometry_all_results.json).
Ten new focused tests and the full 266-test Python suite pass.

`Operational.Windows` gives positive-width score predicates for fit,
low held-out accuracy and their simultaneous memorization window. Actual
score bounds imply the train/test gap and disjointness from a successful
window. Neither event order is built into those predicates. Prefix
invariance requires every point through confirmation to be observed.
`Sustained` formalizes the first consecutive-success window, proves its
uniqueness and confirmation-prefix locality, and distinguishes its onset
from the first isolated crossing with a bounded counterexample.

`DelayedWindows` proves that a first held-out crossing lies outside a
memorization window. An additional observed below-threshold prefix
excludes earlier success and implies that the full window ends before
the crossing. Together with the actual first train fit, this derives
`window_width <= held_out_crossing - train_fit` in observation intervals.
Identical train/held-out scores cannot supply a separated memory window.
A bounded trace with early sustained success, later memorization and
recovery refutes treating every recovery as first grokking. These are
conditional trace laws, not optimizer dynamics; Python's finite-list
scan, longest-stretch selection, medians and full phase heuristic remain
additional implementation bridges. These fifteen proved theorems add no
sorry, bringing the total to **188**.

`DivisionOrbits.Basic` derives the generator's answer from actual field
division, recovers every valid operand pair and proves that equal answers
are exactly common nonzero scaling of both operands. The scale factor
is unique. A zero-denominator counterexample refutes dropping the task's
domain restriction; Python rejects those inputs rather than using Lean's
total division convention. `FiniteCells` constructs the finite generator
image, proves actual-answer membership, nonemptiness, disjointness and
cardinality. It derives 96 points in every full modulo-97 cell and 9,312
valid inputs. Those counts include quotient zero; they do not certify
held-out mask coverage or a learned model's correct margins. Integer
inverse code, token wrappers and the NumPy split are not verified by
these field identities. These nineteen proved theorems add no sorry,
bringing the total to **207**.

`AdamW.LossDirection` differentiates the actual effective quotient along
the native finite first-update path. Decoupled decay cancels from its
learning-rate derivative at zero by the true gradient's radial identity;
bias correction gives minus an explicit weighted gradient-square sum.
`LossDescent` proves this energy is positive precisely away from stationary
gradients when epsilon is positive, then derives a genuine positive
interval of finite learning rates that strictly lower the effective loss.
The interval depends on the initial representation and hyperparameters;
the experimental numerical rate is not assumed to lie in it. Scaling
the gradient by a positive factor changes effective epsilon to epsilon
divided by that factor. A two-coordinate counterexample refutes absorbing
this change by one time factor. Thus the source's suppressed averaging
constants, harmless for its Euclidean time argument, need separate
attention for AdamW. These thirteen proved theorems add no sorry, bringing
the total to **220** on 2026-10-08. This still concerns one scalar effective
constraint, exact reals and zero initial moment buffers, not GPTMini CE,
floating-point execution or later momentum states.

`AdamW.CenteredFamily` derives the actual quotient gradient and first
update on `(t, -t/2, -t/2)` for every positive scale. The initial norm
is nonzero and the normalized loss is always 3/2. `FiniteThreshold`
computes the updated norm and loss difference, proves that the actual
updated family remains centered and nonzero, and derives the exact
improvement threshold `rate < 3*t / (2*A + 3*decay*t)`, where
`A = 3 / (3 + epsilon*t)` and decay is nonnegative.

`Overshoot` proves a complementary increase criterion. For every fixed
positive rate and epsilon and nonnegative decay, the explicit positive
scale `t = rate / (4 + epsilon*rate)` increases the actual loss despite
a strictly negative derivative at rate zero. Thus no uniform positive
rate guarantees nonincrease over all such centered scales. A separate
concrete counterexample retains all experimental optimizer parameters:
betas `(0.9, 0.98)`, epsilon `1e-8`, decay `0.1`, rate `0.001`, at
embedding scale `1e-6`. These nineteen proved theorems add no sorry,
bringing the total to **239**. Neither these exact-real counterexamples
nor the source's flow model identify actual GPTMini CE, its gradient,
floating-point execution or later momentum states.

A native CPU float64 sanity check with the same hyperparameters also
raises the scalar loss, from about 1.5 to 4.35856. Unlike the exact-real
calculation, autograd produces a first-coordinate gradient of
`-4.656612873077393e-10` where the true derivative is zero. First-step
normalization turns this into a centroid shift of about `4.44942e-5`.
This is a single numerical check, not a formal kernel bridge or a new
GPTMini training result. The exact centering theorem must not be reported
as a floating-point invariant; add a quantified gradient-error transfer.

`AdamW.GradientError` now derives that conditional transfer from the two
actual coordinate updates, with correct zero gradient versus supplied
error `delta`. Their absolute discrepancy is exactly
`abs(rate)*abs(delta)/(abs(delta)+epsilon)`, independently of the current
parameter and common decay. It is bounded by one step and by
`abs(rate)*abs(delta)/epsilon`. A relative error budget
`abs(delta) <= rho*epsilon` gives the sharper fraction `rho/(rho+1)`.
Errors at least epsilon move at least half a step; an explicit
counterexample makes the absolute error arbitrarily small while varying
epsilon with it. Epsilon is not fixed in that last statement. Supplying
the observed error `-2^-31` at the experimental parameters yields an
exact-real coordinate discrepancy strictly between `4e-5` and `5e-5`.
No theorem verifies that PyTorch generated the supplied error or bounds
its full gradient. These eight proved theorems add no sorry, bringing
the total to **247**.

`AdamW.SecondStep` derives the native bias-corrected direction from the
previous and current gradients, starting from zero buffers. Differentiating
the actual finite effective-loss update gives minus its current
gradient/direction alignment. Positive alignment suffices for local
decrease, but no weighted gradient-square identity supplies its sign
after momentum accumulates. The current representation and prior gradient
remain inputs to this conditional formula; arbitrary states are not
asserted reachable.

`MomentumCounterexample` closes that reachability issue for a separate
strictly convex scalar quadratic with its actual derivative. Starting at
`0.00075` with zero buffers and all experimental optimizer parameters,
the actual first point is `-750195003 / 3000040000000` and its loss is
strictly lower. The retained moment gives a positive second direction
while the true current gradient is negative. Including decay, the full
second-update loss has a strictly positive rate derivative at zero and
increases for every positive second rate. The example uses the same
objective at both steps and no artificial moment input. It refutes
unconditional repeated-step descent, not grokking or convergence of
GPTMini. These fourteen proved theorems add no sorry, bringing the total
to **261**. A native CPU float64 two-step sanity check also gives loss
`2.8125e-7 -> 3.12654e-8 -> 2.12373e-7` at the rate `0.001`.

`Composition.Basic`, `Curvature` and `StationarySaddle` now specialize the
source's explicitly speculative composition hypothesis to ordinary binary
softmax CE with a bilinear score `x*y`. Deriving both actual partial
derivatives distinguishes flat coordinate restrictions from stationary
regions: the axes have constant loss, but the only zero full gradient is
the origin. Every neighborhood of that origin contains both better and
worse states; it is not a local minimum, and the actual parameter loss is
not jointly convex. A four-corner removal contrast is negative for aligned
components and positive for opposed ones. Adding the explicitly stated
coupled L2 penalty `lambda*(x^2+y^2)/2` gives the actual origin Hessian
`[[lambda,-1/2],[-1/2,lambda]]`. Its aligned direction has negative
curvature for `0 <= lambda < 1/2`; above that threshold the origin's
quadratic form is nonnegative. This is local curvature, not global
convexity, a thermodynamic limit, or a theorem about decoupled AdamW.
The CE has unit weight, so this numeric threshold is not silently
transferred to the answer/EOS averaged training objective. These
twenty-nine proved results add no sorry, bringing the total to **290**.
No theorem makes exactly zero deterministic initialization escape, gives
a probability for random initialization, or identifies this score with
a GPTMini head circuit. Increasing an already positive product can be
confidence alone, without any new held-out decision. Next measure all
pairwise head-removal contrasts on immutable real GPTMini checkpoints,
with individual-removal effects, all four losses, nonzero-quotient
controls and early-generalization seeds. A contrast sign by itself will
not count as a unique algorithm or a forecast of grokking.

`Composition.StationaryStates` now classifies the stationary equations
of that same actual penalized loss, assuming nonnegative lambda. Every
stationary pair has `x=y`, with either `x=0` or
`lambda=1/(exp(x^2)+1)`. Nonzero states exist exactly for
`0 < lambda < 1/2`; the ordinary expression
`sqrt(log((1-lambda)/lambda))` gives positive and negative aligned
witnesses whose actual gradients are proved zero. At or above `1/2`
only the origin is stationary. These eight new proved results bring
the total to **298**, without new sorry. They establish a finite
equilibrium threshold, not global-minimum classification, dynamical
attraction or a thermodynamic phase transition. Native AdamW uses
decoupled decay and coordinate normalization; transferring this L2
coefficient to that algorithm remains unsupported.

The real pair-removal reader has now completed all 644 head-pair
observations at 23 immutable checkpoints of the three seeds and the
reference. Four intermediate reference weights remain missing and are
recorded rather than reconstructed. All four actual answer-only losses
and decision changes are retained on train/held-out splits with separate
nonzero-quotient scopes. Head removal follows the original intervention,
not the scalar model. Counting negative interactions for which both
single removals hurt gives seed 1 held-out counts `4,20,0,3,7,3,9,7,13`
at steps `0,1k,30k,33k,34k,35k,36k,40k,150k`. The large early count
and later decline reject a monotone or current-count-only detector.
At 150k the failed reference has 28 counted pairs on training examples
and zero on held-out examples, so training interactions alone do not
certify a reusable rule. Successful seeds have different final counts
`13,14,4`. Contrast signs use an explicit heuristic tolerance, not a
formal floating-point certificate. The read-only result and controls
are in the grokking_internals README and pair_interaction_results.json.
Keep unique-circuit identification and causal multi-step formation open.

`Composition.AdditiveContrast` provides a stronger exact diagnostic
counterexample. Two examples use ordinary binary CE of additive scores
`2*a-b` and `2*b-a`; at `(1,1)` both individual removals increase the
mean loss and its four-corner contrast is negative. Both score functions
have zero actual mixed partial derivatives and zero four-corner score
contrasts for every component state. Nonlinear CE and aggregation can
produce the pattern without any product in the logits. This is an
explicit fixed two-example task, not a theorem about the division data
or the measured head algorithm. The bilinear score contrast, by contrast,
equals `a*b`. These eleven proved identities/counterexamples bring the
total to **309**, without new sorry. Next measure four-corner **logit**
contrasts before applying loss, removing softmax-invisible common row
shifts. This can distinguish additive output scores from nonlinear
output interaction, but downstream normalization/FFN nonlinearities
still prevent unique attention-circuit identification. Preserve the
original CE reader/results under their frozen source identity.
Validation: the full Python suite passes all 275 tests; experiment check,
full Lean build, audit, regenerated index and forbidden checks pass.

`Composition.OutputInteractions.Basic` now derives class centering of
the four output corners before CE. It commutes with the contrast and
removes arbitrary independent common row shifts at the four states,
assuming a nonempty measured class set. The actual squared centered
energy vanishes exactly when the raw contrast is a common class offset
on that set. This is a finite-output condition, not a trained-parameter
convexity or a numerical certificate. `NonIdentification` proves that
affine class scores with arbitrary nonlinear common row shifts have
zero energy; the stated binary product score has actual energy
`(a*b)^2/2`. A polynomial gate score `a*b*(1-a)*(1-b)` has zero energy
at the complete binary removal endpoints, while its interior energy
at `(1/2,1/2)` is exactly `1/512`. Thus a zero endpoint measurement
does not establish global additivity. These seventeen proved results
bring the total to **326**, with no new sorry. Next observe centered
logit contrasts together with uncentered attention-projection, FFN
output and residual contrasts, and report the hypothetical additive
output reconstruction's accuracy. Same-layer head outputs pass
through an affine projection; downstream nonlinearities can create
final-output interaction independently of a learned cross-layer
attention algorithm. That implementation-stage transfer must be
checked on actual saved weights rather than inferred from the toy map.

The stage reader has completed all 644 pair observations at the same
23 immutable snapshots, retaining the original CE reader's identity.
It captures attention projections, FFN projections, residuals, final
normalization and answer logits at equals; only class logits are centered.
Same-block head pairs have their own attention-projection RMS contrast
between `2.52e-9` and `9.00e-8` in the observed float32 outputs, while
their own downstream FFN contrast is positive in all 276 observations,
with RMS between `0.001217` and `1.09648`. The real-block float64 control
also isolates the affine projection from downstream nonlinearities.
This distinguishes computation stages; it does not identify an attention
algorithm from final-output interaction.

The median relative logit interaction over 28 pairs is `0.2963` at seed
1's step 1,000 and `0.1476` at 34,000, so it does not grow monotonically
into grokking. At 150,000 the failed reference has ratio `0.3212` with
0.49% held-out nonzero accuracy, compared with seed 1's ratio `0.3163`
and 100% accuracy. High interaction is therefore insufficient to identify
rule learning. Hypothetical additive reconstruction at seed 1's final
state yields pair-dependent accuracies from 27.51% to 100%; the other
successful seeds range from 0% to 100% and 9.56% to 100%. These are
current-output sensitivities, not prospective training interventions.
Every contrast and ratio remains an uncertified float64 statistic of
observed model outputs; Lean's energy is a per-row coordinate sum,
whereas the reported energy is a coordinate/example mean.

`OutputInteractions.Decisions` derives the actual three-corner
reconstruction and its error, preserves fixed class bias, and proves
that zero centered energy preserves CE and every strict target decision.
Its current-margin bound `2 * interactionEnergy < margin^2` preserves
the reconstruction's correct decision after removing the computed
common row offset. A positive-interaction example preserves its decision
as well, rejecting interaction-as-decision-change. `Basic` also proves
that fixed class bias cancels from the contrast, even though it may
change predictions. These nine proved results bring the total to **335**
without new sorry. No aggregate mean or unverified floating-point bound
supplies the theorem's per-row margin hypothesis. The implementation,
controls, checkpoint hashes and full findings are in
`experiments/grokking_internals/output_interaction_results.json` and its
README. The full Python suite passes **285 tests in 888.098 seconds**;
the earlier frozen readers and results remain unchanged.

`AdamW.CoupledFirstStep` and `CoupledThreshold` apply the verified native
first-update algorithm to the actual two-component CE gradient. At a
positive aligned state `x=y=s`, both updated amplitudes equal
`s + rate*s*(1/(s + epsilon*(exp(s^2)+1)) - decay)`. Growth is exactly
`decay*(s + epsilon*(exp(s^2)+1)) < 1`. A growing positive state exists
iff `decay < 1/(2*epsilon)`; continuity gives an actual interval of
small positive states with growth and strictly smaller CE. At or above
that threshold no positive aligned state grows. Exactly absent components
remain fixed, and the growing states retain their already correct binary
decisions. This can be confidence growth, not delayed generalization.

At the experimental epsilon/betas/rate, decay one-half still permits
small positive growing states. Thus the earlier coupled L2 curvature
threshold one-half cannot be transferred to the native decoupled
algorithm. The CE is unit-weight binary CE, unlike the lab's shared
answer/EOS averaged objective; neither this coefficient nor zero initial
moment buffers is silently assigned to GPTMini's later checkpoints.
These fourteen proved results bring the total to **349** with no new
sorry. Persistent momentum, minibatch histories and learned rule decisions
remain open beyond this specialization.

The archived-momentum reader has now completed 19 noninitial snapshots.
It restores the actual next minibatch on one disposable CPU model and
native optimizer copy, preserving the original CUDA training and budget.
Four initial weight files lack archived optimizer/sampler states and are
excluded; four intermediate reference weights remain missing. Exhaustive
sample-weighted train/held-out answer, EOS and original averaged gradients
are diagnostics, not gradients supplied to the update. All named clocks
and parameter registrations match the archived native builder.

All 19 measured total directions have negative estimated derivatives
for the exhaustive training objective and its next minibatch, yet finite
full train CE increases in three probes. Seed 1's held-out nonzero answer
slopes at 33k and 34k are positive, with observed finite CE increases
`0.0025951` and `0.0370853`, before the later canonical transition. At
36k the estimated slope is negative but the finite CE change positive.
The failed reference's final probe reduces full train CE and increases
held-out answer CE, despite its retained-buffer direction predicting
held-out decrease. The inserted stochastic gradient and updated adaptive
denominator can change that conclusion. One-step alignment, old momentum
and decay alone therefore do not provide a causal grokking forecast.

These are float64 diagnostics of float32 CPU autograd/optimizer outputs;
they are not the visited CUDA next step or exact derivative certificates.
Very small loss changes and sign disagreements need rounding/curvature
bounds before causal attribution. The completed
`momentum_direction_results.json` and grokking_internals README preserve
every checkpoint, source and next-batch hash. Eight new independent
controls cover native update/sampler agreement, full-batch gradients,
weighted tails, invalid states, zero-quotient scopes and noninterference.
The full Python suite passes **293 tests in 904.206 seconds**; experiment
check and the full Lean build, audit, generated index and forbidden
checks pass. Existing readers and training implementations stay frozen.

`AdamW.CurvatureBound` now derives a finite quadratic loss bound from
actual derivative hypotheses and the interval envelope
`d(t) <= d(0) + curvature*t`. Subtracting the linear and quadratic terms
gives an antitone remainder by the real mean-value theorem. Therefore
`f(rate) <= f(0) + d(0)*rate + curvature*rate^2/2`; the strict condition
`curvature*rate < -2*d(0)` supplies finite descent for a positive rate.
A concrete quadratic proves this coefficient/strict boundary sharp.
The interval premise does not require global second continuous
differentiability across activation boundaries, but it is not supplied
by one measured Hessian or finite samples.

`CurvatureCounterexample` uses actual binary CE of the explicit score
`t - coefficient*t^4`. Every coefficient gives the same initial loss
`log 2`, derivative `-1/2` and second derivative `1/4`. For every proposed
positive rate, coefficient `2/rate^3` makes the endpoint score negative
and increases actual positive CE. This varies the score family with the
rate; it does not say that one fixed curve has no improving rate. The
family is not identified with GPTMini's learned score. These thirteen
proved results bring the total to **362** without new sorry.

The new curvature reader now covers the same 19 actual noninitial
moment/sampler states. Seven fractions of the fixed observed native CPU
displacement retain exhaustive full train CE and held-out nonzero answer
CE, with autograd HVP at five fractions. Endpoint losses agree with the
frozen momentum observer to `1.78e-15` and initial slopes to `1.60e-14`;
accuracy and next-batch hashes agree exactly. No earlier observer,
training source or budget changes. The summed elapsed observation time is
396.64 seconds on four CPU threads. The curves are rounded float32 diagnostics, not exact-real
interval certificates or visited CUDA next steps.

For seed 1 36k, the negative answer slope predicts `-1.14e-7`, while
the initial quadratic predicts `+2.87e-6`, close to the observed `+2.77e-6`.
The large primary train increases at 30k and 34k also match the quadratic
sign and approximate size. But seed 2 35k reduces full train CE by
`0.16926` while its initial quadratic predicts an increase of `0.18296`;
held-out answer CE also has the wrong quadratic sign. Curvature varies
substantially along the sampled interval. Overall initial-quadratic signs
agree in 15/19 train and 16/19 held-out cases; very small-loss mismatches
remain unresolved between interval variation and numerical error. These
are retrospective finite-step comparisons, not a success forecast. The
complete reader and scientific figure are in grokking_internals. Seven
new controls and the full **300-test Python suite in 915.584 seconds** pass;
experiment and Lean build/audit/index/forbidden checks pass. Next improve
multi-step rule-selection formulations rather than treating local
quadratic agreement as a complete explanation of grokking.

`CircuitEfficiency.SectionD_EfficientAllocation` begins the rule-versus-memory
route from Varma et al., arXiv:2309.02390v1, appendix D, Theorem case 1.
For two fixed circuits with identical training logits, transferring weight
preserves the actual logits. For exponent `r = p / scalingExp` in `(0,1]`,
the reduced objective admits a transfer whose gain is at least
`multiplier * (expensiveCost - cheapCost) * weight^r`. A positive penalty
and a strictly larger normalized cost therefore exclude the costly
circuit at every global minimum. Global optimality is a hypothesis;
neither minimizer existence nor arrival of AdamW is inferred. A smooth
bounded MSE counterexample shows why omitting the positive-penalty premise
fails. At equal cost and exponent one, the objective cannot distinguish
allocations with opposite held-out margins in the paper's lookup tables.
These seven proved results bring the total to **369** without new sorry.
The fixed tables, norm-scaling model and coupled penalty are not identified
with GPTMini's learned circuits or decoupled AdamW decay.

`SectionD_PowerDerivative` checks the local estimate used in the source's
proof of the other allocation regime. The printed statement multiplies
the tangent lower bound by a fixed neighborhood radius `delta`; it is
false even for `x=1`, `r=2`, `c=1`, both at increment zero and with all
increments restricted to be positive. The corrected statement multiplies
by the actual increment `epsilon`, requires positive tolerance, and
restricts `0 < epsilon < delta`. Actual real-power differentiation proves
the corrected lower and upper estimates for all real bases and `r >= 1`.
A positive active weight gives a strictly positive linear transfer budget.
The quadratic difference below its tangent also refutes retaining zero
tolerance in the strict corrected estimate. These seven results bring
the total to **376** without new sorry; they repair the local ingredient,
not the entire printed case-2 proof or its optimizer transfer.

`SectionD_TransferDirections` and `SectionD_SuperlinearAllocation` now
derive the second regime for every exponent `r > 1`, beyond just a
quadratic specialization. The actual objective on `(x+epsilon,y-epsilon)`
has derivative `multiplier*r*(cost0*x^(r-1)-cost1*y^(r-1))`. Training loss
is constant along this curve because the stated training logits agree;
no derivative of an arbitrary reduced training loss is assumed. At an
absent-circuit boundary, the actual negative derivative gives a strictly
feasible finite improvement, so every nonzero nonnegative global minimum
with positive multiplier/costs has both weights positive. Fermat's theorem
then derives cost balance and the ratio
`x/y = (cost1/cost0)^(1/(r-1))`. A concrete smooth bounded MSE example
proves that all global-minimum hypotheses are jointly attainable.

When the generalizing lookup table has strictly smaller cost, that balance
gives `x-y > 0`, its actual binary held-out margin. The current transfer
still assumes the fixed Gen/Mem tables, normalized norm-scaling objective,
nonzero attained minimum and coupled penalty; it does not show that native
AdamW learns those tables or reaches the minimum. These twelve results
bring the total to **388** without new sorry. Next remove the nonzero-
minimum premise where the actual CE allows it, then investigate multi-step
learning and rule formation with preserved moments. The two-circuit
case analysis does not exhaust arbitrary circuit families or all possible
explanations of grokking.

`SectionC_TableLoss`, `SectionD_CEBudgetBasic`, `SectionD_CEExistence`
and `SectionD_CEMinima` now use the actual finite-class train and test
tables from Varma et al., appendices C–D. With q=remaining+2, training CE
is `log(exp(x+y)+q-1)-(x+y)` and held-out CE is
`log(exp(x)+exp(y)+q-2)-x`. These formulas are derived from standard
finite-class cross-entropy. The train derivative is
`-(q-1)/(exp(score)+q-1)`, strictly negative at every finite score; for
the source's q=113 its initial value is exactly `-112/113`.

For exponent r>1 the actual CE budget has an improving feasible direction
at the origin, independent of its finite cost coefficients. Every
nonnegative global minimum consequently has positive total score. With
positive multiplier and both costs positive, a compact-rectangle argument
proves that a global minimum is attained for r>=1. In the superlinear
regime both weights are positive, cost balance and the inverse-cost
ratio follow from the actual objective, and a strictly cheaper Gen table
gives correct train and held-out decisions among every one of the q
classes. Thus neither attainment nor a nonzero minimum is merely an
assumption for this stated actual-CE model. Satisfying examples use
its proved q=113 minimum, rather than an MSE substitute.

The source-specific application uses normalized norms one/two, p=2,
scalingExp=1.2 and appendix D's multiplier alpha/p=0.005/2=1/400.
Appendix C's displayed LossWD omits that factor, so an exact simulator
normalization is not asserted. These twenty-nine checked theorems bring
the total to **417** without new sorry. Fixed circuits, their norm-scaling
assumption and coupled penalty remain explicit; the result still does
not establish delayed arrival, discovery of Gen or native AdamW
convergence. Next derive dynamics of the actual product subweights and
compare their rates/obstructions with preserved native moments, keeping
all full-budget runs and earlier observers unchanged.

`SectionC_SubweightGradient` derives both actual product-coordinate CE
derivatives and all four partial derivatives in the source's independent
subweights, valid also at zero products for r>=1. The candidate gradient
is proved equal to those derivatives; it is not assigned as a success
premise. `SectionC_SubweightDynamics` applies simultaneous finite GD to
them and defines its actual repeated iteration. A product `a*b` changes
to `a*b*(1+rate^2*marginal^2)-rate*marginal*(a^2+b^2)`. The difference
of factor squares changes by multiplier `1-rate^2*marginal^2`, so the
continuous-flow conservation is not silently transferred to finite GD.

For r>1, initial factors `(0,genSeed)` and `(0,memSeed)` give product
weights `rate*((q-1)/q)*seed^2` after one step. Unequal nonnegative seeds
therefore yield correct training and wrong held-out decisions. The
source's q=113, rate=0.01 and seeds 0.005/1 give exact products
`7/28250000` and `28/2825`, a factor 40000. This does not assert loss
descent for every prescribed rate or a later transition. A completely
absent Gen factor pair remains absent at every actual defined GD
iteration, independently of the other circuit and penalty coefficients.
Thus the proved effective optimum and a feasible product-space
improvement do not establish discovery of Gen. Fourteen checked results
bring the total to **431** without new sorry. Next check how native
adaptive normalization changes the seed-dependent rate; persistent
moments, circuit discovery and delayed crossing remain open.

Next investigate dynamics that select the correct reference rather than
assuming a learned margin. Derive which descent and noncollapse
properties survive the native adaptive first update even though the
Euclidean norm conservation law does not. Keep the now-derived
state-dependent finite-step interval; do not replace it by a universal
prescribed rate. The reachable momentum counterexample and archived
direction study now identify the missing direction-alignment and
finite-rate conditions. Next connect a multi-step objective/geometry
change to persistent native moments, rather than inferring it from one
stochastic update. Extend the now-derived
error certificate to nonzero gradients and actual numerical kernels only
with verified premises. Continue rule/memorization competition and
coupled-component formation, plus control-parameter/spectral-gap
formulations rather than treating these
optimizer obstructions as a complete grokking mechanism. Keep those claims distinct
from multi-step momentum dynamics and from the actual GPTMini loss. Preserve
the distinction between an inverse-rate characteristic time and the time
to cross a task-dependent generalization threshold. Extend phase-transition
formulations only with stated control parameters and asymptotic regimes.
Keep the actual-transformer/AdamW transfer open. New papers may introduce
open theorem statements under AGENTS.md's explicit allowance; existing
sorry counts must not increase.
