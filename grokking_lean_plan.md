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
| Rule learning versus memorization | State when a reusable rule component wins over an example-specific component under the same training objective | Nanda et al., arXiv:2301.05217v1, sections 4–5; circuit-efficiency literature to investigate | Research pending |
| Geometry of representations | Prove orbit projection identities, scale/bias invariances and counterexamples to symmetry-only success; connect train-fitted probes to held-out decoding | Division common-scaling observer, actual checkpoint features | Fifty-five proved mean, energy, margin, cleanup and integer-encoding laws/counterexamples; exact certificates measured on all 238 preserved snapshots; 266 Python tests passed |
| Division-task symmetry | Derive the diagnostic cells from the actual numeric task rather than assuming their labels or orbit interpretation | Power et al., section 3.1; author-code corpus and GrokkingObserver at 43d4d66 | Nineteen proved generator, valid-domain, orbit-equivalence, disjointness and finite-cardinality laws/counterexamples; token/Python implementation bridges remain open |
| Gradient coherence and implicit bias | State what batch gradient agreement can and cannot imply; distinguish loss descent, task structure and optimizer-specific bias | Fixed-batch gradients; ordinary AdamW | Eighteen proved actual-CE derivative/decomposition/counterexample theorems; shared targets can yield arbitrarily high full alignment despite opposed answers |
| Regularization and norms | Relate an explicitly stated penalty or decay update to competing solutions; reject norm-only success claims | Golechha, arXiv:2405.12755v1, section 3; current grouped norms | Whole-model norm barely changes across the observed transition |
| Phase transitions | Specify an order parameter, control parameter, asymptotic regime and distribution before claiming a thermodynamic transition | Liu effective theory; Žunkovič/Ilievski solvable models | Analogy only for current finite GPTMini; finite-size scaling not established |
| Numerical precision and softmax collapse | Compare exact-real loss gradients with floating-point zeros and prove only the quantization model actually used | Prieto et al., section 3; local CUDA/CPU execution | Observed-logit integer certificate bridge proved; actual floating-point forward, softmax collapse and optimizer-precision analysis remain open |
| Actual transformer and AdamW transfer | Identify which premises about the real forward map, token task and optimizer are verified, and which remain unproved | Existing GPTMini List Int semantics; native lab checkpoints | Nineteen native first-step laws/counterexamples, including actual effective-loss descent on a state-dependent rate interval and the failure of uniform gradient/time rescaling; repeated-step rule convergence remains unproved |

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

Next investigate dynamics that select the correct reference rather than
assuming a learned margin. Derive which descent and noncollapse
properties survive the native adaptive first update even though the
Euclidean norm conservation law does not. First derive a finite-step
threshold and test overshoot on a centered family; do not replace the
state-dependent descent interval by a universal prescribed rate.
Keep first-step claims distinct
from multi-step momentum dynamics and from the actual GPTMini loss. Preserve
the distinction between an inverse-rate characteristic time and the time
to cross a task-dependent generalization threshold. Extend phase-transition
formulations only with stated control parameters and asymptotic regimes.
Keep the actual-transformer/AdamW transfer open. New papers may introduce
open theorem statements under AGENTS.md's explicit allowance; existing
sorry counts must not increase.
