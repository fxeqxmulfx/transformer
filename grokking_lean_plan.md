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
| Confidence versus decisions | Positive logit scaling preserves every ordering but can strictly decrease cross-entropy; quantify the missing conditions and counterexamples | Prieto et al., arXiv:2501.04697v1, section 4.2; measured endpoint projections | Six proved theorems in `Transformer.Grokking.NaiveLoss.Section4_LogitScaling`; no sorry; CircuitEfficiency additionally proves actual delayed-accuracy paths with zero margin and uniform confidence/CE limits, including original legal beta1=0.9/beta2=0.98 retained memory and permanent accuracy below every fixed above-uniform confidence threshold |
| Spectral optimization dynamics | Derive slow modes and exact delayed test-boundary crossing from an actual gradient flow or update recurrence, including convex toy models | Liu et al., arXiv:2205.10343v2, effective embedding dynamics; Žunkovič/Ilievski, arXiv:2210.15435v1, section 3 | Nineteen perceptron theorems; fifty-seven effective-model laws/counterexamples, including noncollapse and effective-loss convergence from initial ground data |
| Rule learning versus memorization | State when a reusable rule component wins over an example-specific component under the same training objective | Nanda et al., arXiv:2301.05217v1, sections 4–5; Varma et al., arXiv:2309.02390v1, appendices C–D | 630 allocation/CE/gradient/update laws and counterexamples: attained actual CE minima, actual subweight derivatives, finite GD rates, closed clipped native parameter/moment feedback, positive-seed formation, zero-pair invariance, corrected generated-history bounds, zero train-CE limits with permanent held-out obstruction, equal positive interior native balance, actual nonzero retained trajectories, closed positive-limit allocation/margin/test-CE laws, attained physical forward-efficiency norm budgets, all true gained-forward CE coordinate derivatives, their closed uniform-decay native feedback, actual gained input/loss limits, necessary native balance with derived retained-buffer limits, efficient-circuit positive-interior parameter/logit selection, actual retained trajectories at gained balanced points, eventual all-class held-out correctness on actual positive finite-limit feedback, nonzero positive-decay Mem-only boundary trajectories with permanently unique wrong test decisions, gained-feedback numerical sign preservation, finite-clock persistence of positive seeded coordinates, actual CE generation of the retained pair envelope with derived scale limits, exclusion of zero Gen limiting mass above the actual decay/epsilon threshold, nonnegative pair classification, derivation of that threshold from a positive Mem factor and greater Gen efficiency, positive Gen limits and eventual all-class correctness from initial positive Gen-partner data rather than a positive Gen reference premise, actual cold feedback computation, explicit nonempty decay/epsilon regimes, positive Gen mass/margin under a static cold regime without a live-Mem-limit premise, actual margin limits/eventual all-class correctness under that initial-data regime, a derived finite-step actual Gen mass/moment growth envelope, its full finite-clock path/logit ceilings, coordinate pure-decay floors, train-correct uniquely wrong-test prefixes from numerical initial-data bounds, arbitrarily long such prefixes under fixed task/native constants despite immediate positive Gen formation and bounded physical parameters/both retained buffers on those same actual trajectories, legal zero-beta native constants with actual unclipped amplitude swaps, and an actual bounded period-two physical-parameter path with proved nonconvergence and increasing retained clocks, plus actual raw coordinate/norm/logit ceilings and a shared clipping lower bound on a finite physical box, a positive full-CE scale floor derived on actual initialized bounded native paths, and a quantitative actual coordinate-input partner floor, full within-pair state preservation for gained feedback, and a derived actual seeded balanced zero-beta amplitude recurrence, and an explicit positive relative-rate advantage while Gen is weaker from the derived CE floor/parameter box and a uniform greater-than-one ratio increment for the actual current native step while Gen is weaker, finite Gen/Mem factor crossing from positive balanced zero-beta initialization without parameter convergence, and actual strictly correct train/held-out decisions at a generated finite update, plus permanent actual balanced factor dominance after that finite crossing without parameter convergence, a positive quantitative current physical score gap, strict train correctness at every finite clock, and permanent all-class train/held-out correctness after a generated finite start on the same initialized path, arbitrarily delayed permanent accuracy generalization on one bounded actual path with a fixed task/native configuration, and a derived first strictly correct held-out clock beyond the prescribed budget, exact unbalanced zero-beta secant identities/intervals, and actual small-rate physical pair-difference contraction, initialized finite-clock power ceilings and absolute balance limits without individual parameter convergence, unequal-pair normalized input/mass floors and gain monotonicity, plus an actual common relative-difference ratio in [0,1) and both actual pointwise relative-asymmetry contractions, initialized full-clock positive-mass/relative-asymmetry power bounds, relative-balance limits without individual parameter convergence, and both actual normalized products tending to their current fixed-sum maximum, an explicit positive vanishing Gen-seed family with actual prefixes through its index, its long-time/seed decision limits, lack of a uniform success start, and noncommuting actual iterated decision limits, plus exact same-current-CE unequal-pair balanced proxies, nonnegative quadratic correction bounds and actual initialized relative-error/residual limits, plus exact unequal mass multipliers, positive remaining decay under the sufficient small rate, native mass positivity/ceilings and a quantitative half-gap actual ratio increment while Gen is weaker, generated finite error/growth tails and actual unequal Gen mass crossing beyond every budget including source-style zero-first-factor seeds, quantitative ordered proxy differences, positive current error tolerances, strict actual unequal order preservation, initialized permanent Gen mass dominance and positive actual factors at every successor clock, actual product/score lower bounds from relative balance, successor training correctness, and initialized permanent all-class train/held-out correctness for unequal positive-sum seeds without parameter convergence, and arbitrarily delayed permanent accuracy generalization on one bounded first-zero-Gen native path with its least successful clock beyond the budget, plus actual unequal pair/coordinate mass ceilings, initialized power ceilings and zero absolute mass/all-parameter limits under static strong decay, actual zero physical score/margin and uniform full CE/softmax limits, and arbitrary delayed permanent strict accuracy with those collapse limits on the same native path, disproving automatic zero held-out CE convergence, plus bounded target-only multiclass counterperturbations, actual eventual perturbability at every positive radius, exclusion of a positive tail margin, and arbitrary delayed permanent accuracy without a fixed positive tail robustness radius, plus total retained parameter/moment mass laws, exact computed memory coefficients, static contraction factors, initialized power ceilings and parameter/applied-input collapse with legal nonzero betas, and actual initialized zero retained-buffer limits, complete denominator tending to epsilon, diverging completed clocks and positive computed cold CE/clipping feedback limit, plus exact native moment/partner coefficients, their generated legal-beta limits and a positive asymptotic/eventual current Gen partner-coefficient advantage, plus exact retained pair measurements and positive-coordinate interval comparisons for the original native step, plus generated initialized weighted coefficient limits/tails and distinct positive rational Gen/Mem comparison rates, plus generated complete original pair rate tails with nonzero native memory, plus current Gen full-memory decay floors, initialized positivity and separated complete tail power bounds, plus actual initialized geometric Mem/Gen measurement ratio ceilings/zero limits and arbitrary relative multiples despite complete absolute collapse, plus generated own-coordinate moment/physical coverage and arbitrary true Gen/Mem parameter-sum dominance, plus generated positive partner mixing and a uniform actual Gen factor cone without full pair balance, plus actual generated positive gained product margins/permanent train and held-out decisions with legal nonzero memory and vanishing margin, plus arbitrary delayed permanent accuracy/least true clocks on the same fixed legal-beta path and no uniform positive-seed success start, plus generated full score/margin, true train/held-out CE and every finite-class softmax uniform limits for the legal retained-beta strong-decay path; broader attraction and learned transfer remain open |
| Compositional circuit formation | Distinguish zero coordinate gradients from a local minimum when a useful computation needs multiple learned components | Nanda et al., appendix Further speculations on grokking, Hypothesis: Phase Transitions are inherent to composition | Seventy-four proved actual-CE component, saddle, stationary-state, class-centered output and decision-transfer results/counterexamples; 644 real head-pair loss observations and 644 stage/output observations reject interaction-only detection; seeded fixed-table native formation is checked in CircuitEfficiency, learned-head multi-step formation remains open |
| Geometry of representations | Prove orbit projection identities, scale/bias invariances and counterexamples to symmetry-only success; connect train-fitted probes to held-out decoding | Division common-scaling observer, actual checkpoint features | Fifty-five proved mean, energy, margin, cleanup and integer-encoding laws/counterexamples; exact certificates measured on all 238 preserved snapshots; 266 Python tests passed |
| Division-task symmetry | Derive the diagnostic cells from the actual numeric task rather than assuming their labels or orbit interpretation | Power et al., section 3.1; author-code corpus and GrokkingObserver at 43d4d66 | Nineteen proved generator, valid-domain, orbit-equivalence, disjointness and finite-cardinality laws/counterexamples; token/Python implementation bridges remain open |
| Gradient coherence and implicit bias | State what batch gradient agreement can and cannot imply; distinguish loss descent, task structure and optimizer-specific bias | Fixed-batch gradients; ordinary AdamW | Eighteen proved actual-CE derivative/decomposition/counterexample theorems; shared targets can yield arbitrarily high full alignment despite opposed answers |
| Regularization and norms | Relate an explicitly stated penalty or decay update to competing solutions; reject norm-only success claims | Golechha, arXiv:2405.12755v1, section 3; current grouped norms | Whole-model norm barely changes across the observed transition |
| Phase transitions | Specify an order parameter, control parameter, asymptotic regime and distribution before claiming a thermodynamic transition | Liu effective theory; Žunkovič/Ilievski solvable models | Actual fixed native two-circuit decision limits do not commute between explicit vanishing positive Gen initialization families and long time, for both the earlier balanced zero-beta path and original legal beta1=0.9/beta2=0.98 retained memory with source-style double-zero first factors. Each member succeeds, but no uniform finite success start exists. Width/task remain fixed and actual full feedback generates every limit; no thermodynamic size limit, temperature or Gibbs distribution is supplied. Analogy only for finite GPTMini; finite-size scaling and learned/numerical transfer remain open |
| Numerical precision and softmax collapse | Compare exact-real loss gradients with floating-point zeros and prove only the quantization model actually used | Prieto et al., section 3; local CUDA/CPU execution | Observed-logit integer certificate bridge and eight conditional native first-step gradient-error bounds/counterexamples proved; actual floating-point forward, autograd error generation and softmax collapse remain open |
| Actual transformer and AdamW transfer | Identify which premises about the real forward map, token task and optimizer are verified, and which remain unproved | Existing GPTMini List Int semantics; native lab checkpoints | 232 update/history/curve laws and counterexamples: state-dependent descent, finite overshoot, gradient-error bounds, reachable momentum ascent, compositional CE growth, interval curvature bounds, causal corrected-moment bounds, direct buffer forgetting, distinct-gradient feedback, partial resets, retained scalar states, exact-real native clipping, derived moment/direction limits, necessary finite-limit parameter balance, actual corrected-denominator identities, positivity, parameter lower bounds, denominator limits from actual retained inputs, explicit two-factor retained-moment envelope growth/noncollapse laws, the full first-clock epsilon denominator floor/parameter increment ceiling and the legal zero-beta normalized-input identity at arbitrary retained clocks; repeated-step rule convergence remains unproved |

The latest checked cold-route closure includes exact critical equality with
original retained betas: initialized physical collapse, native input/buffer/
direction limits, and uniform full CE/softmax limits are derived from static
initial data. The absolute-mass threshold is exact for positive Gen-seeded
paths; small epsilon can exclude that collapse under every positive remaining
decay. This fixed-table classification leaves weak-decay relative attraction
and learned-head formation open, while the other routes above stay active.

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

`SectionC_AdaptiveFirstStep` and `SectionC_SeedRateComparison` transfer
the same actual four CE partials to the documented native AdamW first
update from zero moments. Plain CE and decoupled parameter decay are
used explicitly; the coupled-penalty minimum is not declared a native
AdamW equilibrium. With `mu=(q-1)/q`, the actual first product is
`rate*(1-rate*decay)*mu*seed^2/(mu*seed+epsilon)`. Positive epsilon,
positive rate and a positive remaining decay factor make it positive.
For `0<genSeed<memSeed`, the native product ratio lies strictly between
the squared seed ratio and the unsquared seed ratio. The rate and decay
factor cancel in that ratio. Both correct training and incorrect test
table decisions are retained on the first update, and a fully absent
Gen pair remains zero on that first step.

Combining the source's q=113 and 0.005/1 seeds with the lab's beta
0.9/0.98, epsilon 1e-8, decay 0.1 and rate 0.001 gives a native product
ratio strictly between `1/201` and `1/200`, versus GD's `1/40000`.
These twelve checked results bring the total to **443** without new
sorry. This is a proved difference in initialization-based growth,
not a visited GPTMini circuit decomposition or later grokking time.
The preserved q=97 runs, full budgets and earlier observers remain
unchanged. Next derive persistent-moment subweight dynamics, positive-
seed multi-step emergence and feasible margin crossing rather than
inferring them from first-step normalization or an attained static
optimum. Zero seed, overshoot and momentum ascent remain relevant
countercases to unrestricted transfers.

`AdamW.MomentRecurrence`, `MomentBounds`, `MomentMemory` and `PartialReset`
derive the documented PyTorch 2.14.1 zero-initialized exponential moments,
positive-clock corrections, continued histories and compatibility with the
previous first/second-step formulas. A bounded causal gradient prefix gives
corrected first-moment magnitude at most its coordinate bound and corrected
second moment between zero and the squared bound. The adaptive direction is
bounded by the coordinate bound divided by epsilon; no alignment with the
current gradient follows from that bound.

Under exactly the same subsequent gradient inputs, the difference caused by
an old moment is exactly beta^age times its old difference. It tends to zero,
and erasing an actual first/second buffer has the proved causal-prefix bounds.
For the lab's beta1=0.9 and beta2=0.98, raw contributions are attenuated below
1e-4 after 100 and 500 shared-gradient insertions, respectively, and remain
so afterwards. These are raw buffer contributions, not a training-time
explanation: adaptive normalization and parameter-induced changes in future
gradients remain outside that shared-input estimate.

The counterfactual partial-reset formulas keep the original completed-update
clock. Erasing a physical second moment cannot reduce absolute adaptive
direction at a fixed numerator and current gradient. At zero current gradient
it exposes a stale first moment divided by epsilon. Erasing the first moment
instead restores scalar adaptive-current-gradient alignment. Even for a
constant already aligned history it also multiplies the original direction
by `(1-beta1)/(1-beta1^(clock+1))`; removing misalignment is therefore not an
isolated explanation of a reset's observed benefit. Neither reset theorem
includes finite-step descent, decoupled decay or a grokking guarantee.

These thirty-three checked results bring the total to **476** without new
sorry. Native persistent-moment formulas are now derived; next bound how
different parameter trajectories change their gradient inputs and connect
that feedback to actual circuit development. The positive-seed product
trajectory and later task-margin crossing remain open. Preserve all prior
full-budget runs, observers and numerical results.

The same-gradient moment-reset observer now covers all 19 preserved native
noninitial states, with seven new controls and all **307 Python tests passing
in 943.004 seconds**. Four disposable CPU branches retain actual buffers,
reset both buffers and clock, reset only the first moment, or reset only
variance. Their retained branch agrees exactly with the frozen momentum
reader on both exhaustive populations: starting/ending CE, answer accuracy,
direction norms/slopes and restored minibatches. All checkpoint/source hashes
are verified; no training run, budget or earlier observer/result changes.

At the primary 34k checkpoint, variance-only erasure gives held-out nonzero
answer CE 352.40 versus 1.2292 with retention; total direction norm grows from
145.30 to 384010.22. At seed 2's successful 150k state, a fresh optimizer
lowers the next-copy accuracy from 100% to 71.49%; seed 3 variance erasure
lowers it to 1.17%. First-moment erasure helps the temporary seed 2 35k
regression but also changes effective magnitude, and the failed reference
remains near chance under every branch. Thus next-step dependence on moments
is observed and formally explained in its stated scalar cases, while reset
sensitivity does not single out initial grokking. These are single-step CPU
counterfactuals, not evidence of multi-step recovery or loss of the represented
rule before the intervention. Full results and all-state plots are linked in
the experiment README. Continue the feedback and circuit-development proofs.

`AdamW.MomentFeedback` derives the missing different-input comparison. For
two initial buffers and two gradient streams with causal prefix disagreement
at most D, their moment difference is bounded by
`beta^n*abs(initialDifference) + (1-beta^n)*D`. The later-gradient term is
attained exactly by constant different streams and cannot be removed in
general. Constant different inputs also leave their exact difference in
the corrected first moments at every positive clock, despite identical
zero initial buffers. These are input-stream countercases, not asserted
visited GPTMini trajectories.

For coordinate magnitudes bounded by C and gradient disagreement at most D,
the two actual zero-initialized second moments differ by at most
`(1-beta2^n)*2*C*D`. For supplied scalar parameter paths, a local bound
on their loss-derivative difference by L times parameter distance, together
with distance at most R on the visited prefix, gives the first-moment
bound `(1-beta1^n)*L*R`. The theorem uses objective derivative values and
explicit pathwise assumptions; it does not establish those bounds for
GPTMini, control the two parameter paths, or certify its numerical gradients.

These eight checked results bring the total to **484** without new sorry.
The direct memory law now has an explicit gradient-feedback extension.
Next close the parameter/gradient coupling for the actual fixed-circuit CE
subweights under persistent native AdamW, retaining its zero-seed and
finite-rate obstructions. Positive-seed emergence, circuit competition,
later margin crossing and transfer to learned GPTMini circuits remain open;
do not infer them from buffer contraction or the attained coupled optimum.

`AdamW.ScalarRecurrence` and `ScalarStability` keep the parameter, both
retained buffers and completed clock in one numerical state. The update
agrees with the previously checked first step at zero initialization and
retains old momentum even at a zero current gradient. Nonpositive current
and retained first-moment signs give a nonpositive adaptive direction;
a strictly negative current derivative makes it negative. With positive
remaining decay factor, seeded positive coordinates cannot collapse in
that sign region, and a negative derivative activates a zero coordinate.
These are derived update facts, not sign or success claims built into the
state. Current-gradient signs remain hypotheses until the CE callback
derives them on the actual coupled path.

`AdamW.GradientClipping` formalizes the documented PyTorch 2.14.1 flattened
norm-two clipping coefficient `min(1, bound/(norm+1e-6))`. A positive finite
bound makes it positive and at most one. The actual clipped norm and every
coordinate magnitude are bounded by it; negative signs persist strictly.
This derives the exact-real coordinate premise for the earlier moment
bounds without assuming bounded raw derivatives. The source doc's wording
that scaling only occurs above max_norm is corrected: no scaling requires
`norm+1e-6<=bound`, and a unit gradient at unit max_norm has multiplier
`1000000/1000001`. This is an exact-real coefficient counterexample, not a
claim that a numerical kernel or the training recipe has changed.

These twenty-seven checked results bring the total to **511** without new
sorry. Next instantiate the complete retained state and clipping wrapper
with each of the four actual product-CE partial derivatives at its current
factors. Close initialization, parameter feedback, noncollapse, zero-seed
invariance and competition symmetries on that actual iteration before
claiming any later rule-selection or generalization threshold crossing.

`CircuitEfficiency.SectionC_NativeRecurrence`, `NativeSigns`,
`NativePositive` and `NativeSymmetry` close the actual four-factor
parameter/gradient/moment/clock loop. All four derivatives are proved to
be actual finite-class product-CE partials, evaluated at the old parameter
state, passed through shared native clipping, and updated simultaneously.
The source q=113, seeds 0.005/1 initialization has raw norm below 0.999,
so its first unit-bound step agrees with the earlier unclipped formula.
Clipping now supplies every coordinate bound on every visited state.

Each derivative is its present factor partner times the common strictly
negative training CE slope. Nonnegative factors and retained nonpositive
moments stay in their numerical sign region under valid beta1/beta2,
positive epsilon and nonnegative remaining decay factor. With positive
second-factor seeds, positive rate and positive remaining decay factor,
both zero first factors activate after one actual native step; all four
parameters remain positive at every later finite step. Consequently all
training classes are strictly correctly ranked after that first update.
This is a decision result, not proof that training CE tends to zero.

Equal full Gen/Mem factor states, including both buffers and clocks, are
invariant under actual CE, shared clipping and uniform native decay.
Equal seeds therefore keep both product logits equal forever, and their
true multiclass held-out CE is at least log two at every finite clock.
Positive component formation and perfect training decisions alone do not
imply eventual rule selection. This obstructs a naive transfer to native
AdamW; it does not refute the source's GD with a cost-asymmetric coupled
circuit-norm penalty. Fixed reference tables and exact-real arithmetic
remain explicit, and no general argmax-tie error claim is made.

These twenty-nine checked results bring the total to **540** without new
sorry. Next connect the generated clipped gradient sequence to the actual
retained buffer histories and their corrected bounds. Prove zero-pair
invariance on this closed trajectory, then identify an actual symmetry-
breaking mechanism before asserting late held-out margin crossing or
transfer to learned GPTMini circuits.

`CircuitEfficiency.SectionC_NativeHistory` identifies both retained
buffers and the completed clock with the causal history of each actual
generated clipped CE derivative. The next actual parameter is its pure
decay contribution minus the native bias-corrected direction of that same
history. The wrapper supplies all visited coordinate bounds, so at every
positive clock the corrected first moment has magnitude at most bound and
the physical corrected variance lies between zero and bound squared. The
next adaptive direction has the explicit bound/epsilon estimate. These
are actual closed-path consequences; no arbitrary successful future
gradient stream is supplied, and loss descent or generalization does not
follow from the loose magnitude estimate.

These eight checked results bring the total to **548** without new sorry.
Next prove the fully zero pair's invariance under actual CE feedback and
retained native moments. Then investigate a mechanism that selects Gen
over Mem despite training-table symmetry; uniform decoupled parameter
decay cannot be identified with the source's cost-asymmetric penalty.

`CircuitEfficiency.SectionC_NativeZero` proves that a fully absent Gen
pair has zero actual CE coordinate derivatives while the product-weight
CE slope remains negative. Shared clipping cannot insert derivatives;
zero parameters and actual buffers remain zero at every native update,
with clocks still advancing. The entire closed path from two zero Gen
seeds preserves the absent pair, regardless of the other circuit's seeds.
Nonzero stale first moments are explicitly excluded by the initial-buffer
conditions; a zero current gradient alone is not enough for the result.

With nonnegative Mem seeds and the checked valid-beta/epsilon/remaining-
decay conditions, true held-out CE stays at least log q for every finite
step. This is its uniform-initialization loss, stronger than the earlier
log-two equal-circuit floor. It is not a claim about argmax tie-breaking or
the source's positive-second-factor Gen simulation. Clipped native
training cannot discover this entirely absent fixed product circuit;
discovery of learned GPTMini features remains a different open problem.

These eight checked results bring the total to **556** without new sorry.
The closed native trajectory now distinguishes positive-seed formation,
complete absence, and full-state Gen/Mem symmetry. Next identify a
symmetry-breaking mechanism or its necessary state/geometry conditions;
the source's circuit-efficiency allocation theorem cannot be substituted
for native adaptive dynamics under uniform parameter-wise decay.

`CircuitEfficiency.SectionC_NativeDescent` derives actual zero-decay
training-CE descent on the formed positive-factor/nonpositive-moment
region. Every current clipped derivative is strictly negative, so retained
native adaptation increases all four factors at any positive rate; both
products and their total training score strictly increase. The actual
multiclass CE has a globally negative derivative and therefore strictly
decreases. Along positive source seeds, this holds between any two finite
clocks after formation, with admissible beta2 on the actual closed path.

The joint equal-seed countercase proves correct training decisions after
the first step, strict subsequent training-CE decrease and held-out CE at
least log two at every finite clock. The held-out loss may improve towards
that floor; neither no held-out improvement nor universal argmax error is
asserted. Training-CE convergence to zero is not yet established, and
positive native decay or learned GPTMini features require new analysis.
The source's coupled cost-asymmetric penalty is not transferred by naming
uniform native decay as circuit efficiency.

These seven checked results bring the total to **563** without new sorry;
CircuitEfficiency now has 133 checked results and AdamW 155. The actual
closed fixed-table native feedback loop has formation and nonformation,
retained history estimates, symmetry obstruction and conditional continued
loss improvement. Next derive a genuine symmetry-breaking or allocation
mechanism with physical parameters, and analyze positive decay and native
stationary limits before predicting a late held-out margin crossing.
Continue the other spectral, geometry, operational and phase routes;
these fixed-table results are not a completed grokking mechanism for GPTMini.
For the next native-dynamics iteration, first derive necessary stationary
conditions from the actual recurrence under an explicit convergent-state
premise; do not presume convergence from bounded moments. Investigate
efficiency encoded in the physical forward parameterization under the
same uniform native decay, separately from introducing unequal decay
groups or a coupled objective. Any simpler zero-decay convergence result
must retain the symmetry test-loss obstruction and its fixed-table scope.

`AdamW.MomentLimits` derives actual retained-buffer limits from convergent
input gradients. A shifted tail bound keeps the entire earlier buffer as
an exponentially decaying term. An epsilon-tail argument then proves the
first moment tends to the gradient limit and the second to its square,
without assuming buffer/current-input agreement or instantaneous resets.
Completed-clock bias corrections tend to one, including any retained
clock tending to infinity; no finite limiting clock is assumed. The actual
corrected-history direction tends to gradient/(abs(gradient)+epsilon).

These eight checked results bring the total to **571** without new sorry;
AdamW now has 163 checked results. Gradient convergence remains a premise
for this generic result. Bounded clipping alone does not supply it, and
stochastic minibatch inputs need not converge at finite parameter limits.
Next derive the actual scalar recurrence's necessary parameter balance
from convergent parameters and inputs, using these automatic moment limits.
For the fixed-table CE path, derive input convergence from its actual
parameter feedback before applying that balance. Then analyze whether a
positive coexistence limit can select a rule under uniform native decay.

`AdamW.ScalarLimits` derives both retained histories and the unbounded
completed clock from the actual scalar native recurrence. Convergent input
gradients force the actual next direction to the normalized gradient limit;
no moment convergence, instantaneous matching or clock reset is assumed.
If parameters also converge at a positive constant rate, the actual update
forces decay*parameter + gradient/(abs(gradient)+epsilon) = 0. Finite
parameter convergence is explicit and is not deduced from bounded inputs.
The hypotheses have a nonzero retained native stationary instance with
valid zero betas, epsilon one, gradient minus one, parameter one and decay
one half; the growing clock remains a separate natural-number coordinate.

These four checked results bring the total to **575** without new sorry;
AdamW now has 167 checked results. Next derive convergence of the actual
clipped fixed-table CE callback from convergent parameters and instantiate
this native limit balance. Then check which positive Gen/Mem limit
allocations it permits, preserving the distinction from the source's
coupled circuit-norm objective and from learned stochastic GPTMini states.

`CircuitEfficiency.SectionC_NativeGradientLimits` derives the actual
fixed-table callback limits from convergence of all four current
parameters. Products determine the train-score limit; actual CE partials,
their shared squared norm, the norm-plus-1e-6 native clip coefficient and
every applied coordinate then converge to their actual finite-point
values. Raw zero gradients cause no clip-denominator singularity. The
actual multiclass train and held-out CE also have their finite-point
limits, retaining both competing products and every remaining class.

These seven checked results bring the total to **582** without new sorry;
CircuitEfficiency now has 140 checked results. The reference state encodes
only parameters, not an assumed limiting zero buffer or finite clock.
Its examples preserve an unbounded clock with nonzero constant parameter
coordinates. Next instantiate the scalar native balance on the actual
closed CE update: no separate convergence assumption on gradients or
moments is now needed for this deterministic fixed-table application.
Check the permitted finite allocations and actual nonzero initial paths
before transferring an asymptotic statement to learned minibatch GPTMini.

`CircuitEfficiency.SectionC_NativeLimitBalance` now instantiates the
native scalar limit law on the actual closed product-CE feedback loop.
Every finite parameter limit at constant positive learning rate must
balance its uniform decay against its actual normalized clipped partial;
gradient and moment convergence are consequences, not extra hypotheses.
Without decay, positive clip bound forces every finite parameter limit
to be the all-zero point, even without a parameter-sign premise. The
common actual CE slope never vanishes at a finite score, and clipping
cannot erase it. A fully cold actual path with growing clocks satisfies
all convergence hypotheses, so the conditional statements are nonempty.

These six checked results bring the total to **588** without new sorry;
CircuitEfficiency now has 146 checked results. They do not assume or
prove finite convergence on nonzero seeds. Next combine native positive
formation and monotonicity with this finite-limit obstruction to decide
the zero-decay train-CE limit. Then inspect the positive-decay allocations
and genuine physical circuit-efficiency asymmetry, without identifying
the fixed-table model with stochastic learned GPTMini features.

`CircuitEfficiency.SectionC_NativeScoreGrowth` proves unbounded training
score on the actual no-decay trajectory from any two positive second
seeds. Every formed coordinate strictly increases. A bounded total score
would bound each coordinate through its positive first-formed partner;
monotone coordinates would then have finite positive limits. The closed
native limit law would force these limits to zero, a contradiction.
Consequently the full score path tends to positive infinity, and the
four parameters cannot have a finite joint limit. No convergence premise
on future parameters, gradients, buffers or logits enters this dynamics
result, and all native moments and completed clocks are retained.

These five checked results bring the total to **593** without new sorry;
CircuitEfficiency now has 151 checked results. Next transfer the derived
score growth to the actual CE limit and combine it with the exact
Gen/Mem symmetry floor. The no-decay scope is essential; positive native
decay, physical efficiency asymmetry, learned features, stochastic inputs
and finite-precision implementation are still separate obligations.

`CircuitEfficiency.SectionC_NativeCELimits` closes the actual no-decay
train-loss limit. The source's true multiclass CE equals
`log (1 + (q-1) * exp (-score))`; the score limit is derived from native
feedback, so train CE tends to zero from any two positive second seeds.
For equal Gen/Mem seeds, actual retained updates preserve equal product
logits: true held-out CE never falls below log two and cannot tend to
zero, and the distinct wrong Mem class stays tied with the correct
class, ruling out a unique correct argmax. The actual all-class softmax
target probability at equal test weights is at most one half. No
arbitrary argmax tie-breaking accuracy is assigned.

These seven checked results bring the total to **600** without new sorry;
CircuitEfficiency now has 158 checked results. This is a full
actual-feedback counterexample to inferring rule selection from
arbitrarily good training confidence, formation and continuing native
updates, with retained moments and clipping. Its zero-decay fixed-table
scope is explicit. Next characterize the finite positive allocations
permitted by uniform native decay, then introduce physical forward
efficiency asymmetry rather than the source's coupled penalty or an
assumed successful margin. Learned stochastic GPTMini and numerical
kernel transfer still require separate proofs/evidence.

`CircuitEfficiency.SectionC_NativePositiveBalance` characterizes the
strictly positive interior of the native finite-limit balance equations.
Actual applied partials factor into one shared positive clipped CE scale
and their present partners. Positive native normalized pair balance
forces equal pair factors and positive decay; comparing both pairs then
forces all four parameters equal. Actual held-out CE at every such
balanced point is at least log two, without a symmetric-buffer premise.
A nonzero unit-parameter actual-CE witness with epsilon equal to its
clipped CE scale and uniform decay one half jointly satisfies the point
balance hypotheses. It is not declared an attracting or convergent
trajectory merely by satisfying these point equations.

These six checked results bring the total to **606** without new sorry;
CircuitEfficiency now has 164 checked results. Next build an actual
retained native trajectory witnessing the nonzero balance, then state
the positive-limit allocation corollaries directly on the closed update.
This equal-coefficient fixed-table result is separate from physical
forward efficiency asymmetry, boundary allocations, global convergence,
stochastic learned GPTMini features and numerical implementation.

`CircuitEfficiency.SectionC_NativeUnitPath` identifies the nonzero
unit-parameter balance with an actual retained closed CE trajectory.
Current equal parameters imply equal actual callbacks independently of
buffers/clocks. With epsilon equal to the unit point's positive clipped
CE scale and uniform decay one half, each actual step keeps parameters
at one while advancing the true first/second-moment histories and
completed clocks. This is proved for every valid beta1/beta2, including
0.9/0.98, starting with the actual native zero buffers. The constant
inputs are consequences of current-parameter feedback, not supplied
future gradients, reset moments or assumed instantaneous matching.
Every coordinate's finite positive parameter limit is derived.

These five checked results bring the total to **611** without new sorry;
CircuitEfficiency now has 169 checked results. The chosen epsilon and
unit seeds are a mathematical witness, not the frozen GPTMini run's
configuration or an attraction result for the paper's source seeds.
Next state positive-limit allocation/margin/test-CE laws directly on
actual closed paths, using this nonzero valid native path to witness all
joint hypotheses, before inspecting physical forward efficiency.

`CircuitEfficiency.SectionC_NativePositiveLimits` states the positive
interior consequences directly on actual closed clipped CE updates.
At constant positive learning rate and valid native betas/epsilon, finite
convergence of all four parameters to positive limits implies positive
uniform decay and equal limit factors within and across Gen/Mem. Actual
input and buffer convergence are derived, without a symmetric-buffer,
supplied-gradient-limit or instantaneous matching premise. The actual
Gen-minus-Mem product margin tends to zero; actual multiclass held-out
CE converges to its actual reference-point value, at least log two, and
cannot tend to zero. A zero limiting margin is explicitly not assigned
a finite-clock accuracy, since a margin may stay positive at every
finite clock while tending to zero.

These three checked results bring the total to **614** without new sorry;
CircuitEfficiency now has 172 checked results. Every joint-hypothesis
example uses the actual nonzero native unit path at beta1=0.9/beta2=0.98,
with evolved buffers, growing clocks and its explicitly chosen epsilon.
Next inspect boundary allocations and physical forward efficiency:
different logit production per trainable parameter must enter the actual
forward and CE derivatives, preserving uniform AdamW rather than
silently inserting unequal decay groups or the source's coupled norm
objective. Global positive-limit convergence from unequal source seeds,
stochastic learned GPTMini and numerical-kernel transfer remain open.

`CircuitEfficiency.SectionC_PhysicalGain` implements efficiency in the
actual forward: a fixed readout gain multiplies two trainable factors,
whose actual squared norm is the sum of their coordinate squares. For
every nonnegative target logit, the minimum squared trainable norm is
twice that logit divided by the positive gain. Explicit square-root
factors attain it, and equality forces balanced factors. A larger gain
therefore achieves the same positive training logit at strictly smaller
actual trainable norm. The fixed gain is not a trainable parameter, an
optimizer group or a norm penalty inserted into cross-entropy. Both
output and norm scaling laws follow from the physical parameters, and
unit gains recover the previous source-table forward.

These eight checked results bring the total to **622** without new sorry;
CircuitEfficiency now has 180 checked results. The degree-two physical
specialization differs from the source simulation's exponent 1.2 and
assumed circuit costs. It does not establish that native AdamW selects
the cheaper circuit. Next derive the true gained-forward CE partials,
their shared clipping and actual retained feedback recurrence, then
analyze the necessary positive finite-limit allocations under uniform
decay. No successful held-out margin, supplied gradient stream or
learned GPTMini feature-discovery premise is hidden in this norm result.

`CircuitEfficiency.SectionC_GainGradient` derives all four true CE
coordinate derivatives of the physical gained forward and identifies
the entire native raw callback vector with them. Each gain contributes
both to the actual training score and to its coordinate chain factor.
The source's one-zero-factor seeds produce nonzero first-coordinate
gradients proportional to gain times seed; the two second-coordinate
gradients vanish initially. Unit gains recover both the source's plain
CE and the earlier actual callback. There is no assigned gradient stream
or gain-dependent regularization term in the objective.

These eight checked results bring the total to **630** without new sorry;
CircuitEfficiency now has 188 checked results. Next clip the complete
actual gained derivative vector, insert it into the retained native
recurrence with one uniform decay, and derive necessary finite-limit
allocations. Cheaper physical logits alone still do not prove path
convergence, selection, a grokking delay or learned GPTMini transfer.

`CircuitEfficiency.SectionC_GainRecurrence` closes the physically gained
CE callback on the retained native optimizer. The complete true gradient
vector receives one norm-two coefficient with the actual 1e-6 strip;
all four scalar updates use identical decay, rate and epsilon. Every
applied derivative factors into one shared positive CE magnitude, its
fixed physical gain and its present partner. Positive gains/partners
therefore retain strict negative inputs at finite points, and the actual
clipping wrapper enforces coordinate bounds. Parameters determine the
callback independently of stale moments and clocks, without equating
their resulting optimizer steps. Actual clock growth, the fully cold
closed trace and exact unit-gain update specialization are checked.

These ten checked results bring the total to **640** without new sorry;
CircuitEfficiency now has 198 checked results. Next derive actual
gained-gradient/clipping limits from convergent current parameters,
then instantiate the retained scalar native balance with no supplied
input or moment convergence. Analyze whether physically efficient Gen
has a stronger positive interior limit, and provide an actual retained
nonzero path satisfying every convergence premise. Attraction, slow
formation, late margin crossing and learned GPTMini transfer remain open.

`CircuitEfficiency.SectionC_GainGradientLimits` derives the actual
physical-gain score, raw derivative, shared norm, clip coefficient and
applied input limits from all four parameter limits. Both gains remain
in the score and chain factors; the shared native clip retains its
positive epsilon strip even at raw zero gradients. Actual train/test
multiclass CE have their finite-point limits with both competing
products and every remaining class. Reference moment/clock fields
are not limiting optimizer-state premises.

These seven checked results bring the total to **647** without new sorry;
CircuitEfficiency now has 205 checked results. Next derive necessary
native finite-limit balance directly from the closed gained recurrence,
then characterize positive physical-gain allocations. Gradient limits
are now feedback consequences, while parameter convergence, learned
stochastic GPTMini and numerical-kernel transfer remain explicit gaps.

`CircuitEfficiency.SectionC_GainLimitBalance` derives necessary finite
parameter balance directly on the actual closed gained native step.
The true clipped derivative at the reference point determines the
normalized limiting direction. Both retained moment limits are also
derived from parameter feedback and their entire causal histories,
with arbitrary finite initial buffers and unbounded clocks. Positive
readouts and cap force zero parameters when all actual inputs vanish;
without decay, every possible finite parameter limit is therefore zero.
Actual cold traces jointly satisfy the convergence and recurrence
hypotheses at ordinary beta1=0.9/beta2=0.98.

These four checked results bring the total to **651** without new sorry;
CircuitEfficiency now has 209 checked results. Parameter convergence is
still conditional and clipping alone does not establish it. Next solve
the strictly positive physical-gain balance equations under one uniform
decay and test whether efficiency forces a strictly stronger Gen logit.
Then construct a nonzero retained actual path witnessing every joint
positive-limit premise before deriving eventual held-out correctness.
Boundary allocations, source-seed attraction, delay and GPTMini transfer
remain separate obligations.

`CircuitEfficiency.SectionC_GainAllocation` solves the strictly positive
actual normalized balance with different physical gains and uniform
decay. Both factor pairs must be balanced and decay must be positive.
When Gen's gain is greater, both its balanced physical coordinate and
its actual gained product logit are strictly greater than Mem's.
The correct held-out class then strictly beats the wrong Mem class
and every zero-logit competitor. The successful margin is derived
from true CE/native balance, not inserted as a premise. Gains 3/2,
Gen factors 1, Mem factors 1/2, uniform decay 1/2 and epsilon three
times the actual clipped CE scale give a nonzero joint point witness.

These three checked results bring the total to **654** without new sorry;
CircuitEfficiency now has 212 checked results. This is positive-interior
allocation, not proof that arbitrary source seeds converge there or
that a parameter-point solution is an actual native trajectory. Next
construct its retained zero-buffer native path at valid ordinary betas,
then prove eventual held-out correctness directly on actual positive
finite-limit feedback paths. Source-seed attraction, boundary failures,
delayed crossing and learned GPTMini transfer remain open.

`CircuitEfficiency.SectionC_GainBalancedPath` identifies every actual
normalized gained-CE balanced point with a full closed zero-buffer
native trajectory at valid betas. Candidate numerical fields contain
the actual point inputs and their entire first/second-moment histories;
parameter-only actual feedback then proves these inputs are generated
by the current point at every step. The actual update advances both
buffers and growing clocks while preserving the physical coordinates.
The candidate equals the entire actual native path, so its finite
parameter limits are derived rather than assumed. The positive
unequal-gain point jointly witnesses the hypotheses at beta1=0.9 and
beta2=0.98, without instantaneous matching or buffer resets.

These four checked results bring the total to **658** without new sorry;
CircuitEfficiency now has 216 checked results. The selected positive
epsilon and balanced initialization are a mathematical witness, not
the preserved GPTMini configuration or attraction from source seeds.
Next transfer positive-interior allocation directly to convergent
closed feedback paths and derive eventual held-out correctness from
their positive margin limits. Boundary allocations, source-seed
convergence, arbitrarily delayed formation and GPTMini transfer remain
open; point stationarity alone has not been promoted to attraction.

`CircuitEfficiency.SectionC_GainPositiveLimits` proves efficient Gen
selection directly on actual retained uniform-decay CE trajectories,
conditional on all four current parameters converging to positive
finite limits. Native input/buffer limits and normalized balance are
feedback consequences. Positive uniform decay, balanced limit pairs,
eventually stronger Gen coordinates and a strictly positive actual
Gen-minus-Mem limiting logit margin are derived. Both actual gained
products converge; eventually Gen strictly beats the wrong Mem class
and every remaining held-out competitor. No successful margin or
supplied future input is assumed. Nonzero actual balanced zero-buffer
paths at beta1=0.9/beta2=0.98 jointly witness all hypotheses.

These three checked results bring the total to **661** without new sorry;
CircuitEfficiency now has 219 checked results. Physical forward
efficiency can select a rule under the same uniform native optimizer
in this explicit fixed-table positive-interior model. This is not
global attraction, source-seed convergence, a delayed transition or
learned GPTMini generalization. Next analyze boundary allocations and
how much of the positive-limit premise can be derived from actual
source-seed dynamics. Formulate early Mem dominance and finite delay
without placing future success in a definition or discarding retained
moments. The other spectral, geometry, operational and phase routes
remain active and their GPTMini transfer remains open.

`CircuitEfficiency.SectionC_GainBoundary` checks why the strictly
positive-limit premise cannot simply be deleted. An entirely cold
Gen pair and positive equal Mem factors admit actual normalized balance
with Gen's larger physical gain 3 versus Mem's 2, positive uniform decay
and explicitly chosen positive epsilon. The full retained zero-buffer
native path is derived at every valid beta, including 0.9/0.98, and
all physical coordinates keep their seeded values. Every native clock
uniquely fits the train class while uniquely selecting the wrong Mem
test class; no tie-breaking accuracy is assigned. The absent efficient
pair is not recreated by greater norm efficiency alone.

These four checked results bring the total to **665** without new sorry;
CircuitEfficiency now has 223 checked results. The counterfamily varies
its amplitude and chosen decay/epsilon and is not a frozen-run setting
or attraction claim from positive Gen seeds. Next quantify its actual
train confidence separately from its permanently wrong decisions, and
analyze positive perturbations/source formation before inferring a
late transition. In particular, investigate whether actual positive
Gen seeds can approach a Mem-only boundary under finite convergence;
retained moment and variance dynamics must remain in that argument.
Continue the other formulations and keep learned stochastic GPTMini
and numerical-kernel bridges open.

`CircuitEfficiency.SectionC_GainSigns` derives the actual gained CE input
signs from current nonnegative partners and positive physical gains.
Induction closes all four parameter/moment/variance signs on the actual
retained path. Each actual parameter remains above its pure-decay
contribution, and a positive seeded coordinate stays positive at every
finite clock when `1 - rate * decay > 0`. Finite parameter limits inherit
only nonnegativity; finite-clock positivity is not promoted to a positive
limiting value. The source's one-zero-factor, small-positive-partner seeds
jointly witness the optimizer and sign hypotheses.

These seven checked results bring the total to **672** without new sorry;
CircuitEfficiency now has 230 checked results. The next boundary argument
must include retained first moments and the true corrected denominators:
near a zero Gen pair with a nonzero Mem limit, efficiency may amplify a
positive perturbation against decay. Prove that implication from actual
feedback before deleting the positive-Gen-limit premise. Finite parameter
convergence, eventual formation delay and learned GPTMini transfer remain
open.

`AdamW.ScalarDenominator` rewrites the actual retained step using the
new first moment divided by the complete denominator, including its
first bias correction. Positive epsilon and valid beta1 ensure positivity.
An actual denominator ceiling yields a parameter-increment lower bound
with nonpositive current/historical inputs. Under the native recurrence,
convergent inputs force this denominator to `abs(input_limit) + epsilon`;
vanishing inputs therefore eventually fall below any strict epsilon
ceiling. Both variance history and the diverging completed clock are
derived, without moment resets or an instantaneous-gradient substitution.

These six checked results bring the total to **678** without new sorry;
AdamW now has 173 checked results. They supply the true denominator
estimate for the positive-Gen boundary argument. The application still
must derive vanishing inputs and a feedback gain exceeding decay from
its current CE parameters, then rule out collapse of the retained pair.
No parameter convergence or actual GPTMini generalization is supplied
by the optimizer helper.

`AdamW.PairEnvelope` controls the numerical quantity
`(1 - beta1) * ceiling * parameter_mass + beta1 * rate * negative_moment_mass`.
Explicit moment/parameter recurrence lower bounds imply a weighted
increment of at least
`rate * (1 - beta1) * (coefficient - decay * ceiling) * parameter_mass`.
Above the critical threshold it grows strictly; at or above it, the
retained tail stays above its positive starting mass. Conditional on
finite parameter-mass convergence and vanishing retained moment, the
limiting parameter mass is therefore positive, including beta1=0.
All hypothesis lists have joint witnesses, including critical constant
paths. Neither the coefficient nor ceiling is defined as success.

These five checked results bring the total to **683** without new sorry;
AdamW now has 178 checked results. The next step is the closed gained-CE
bridge: derive the moment/parameter envelope bounds from present partners,
actual shared clipping and retained denominators. Until that bridge is
proved, these supplied lower bounds are not asserted for an arbitrary
CE path. Source-seed convergence, a delayed crossing and learned GPTMini
transfer remain open.

`CircuitEfficiency.SectionC_GainEnvelope` closes the supplied pair
envelope to the actual gained-forward CE/native update. Parameter
convergence gives the true shared clipped-CE scale limit. Numerical
signs give nonnegative physical parameter and negative retained-moment
masses. The true product derivatives yield the exact native first-moment
mass recurrence. A present coefficient lower bound and upper bounds on
both actual corrected denominators give the two envelope inequalities,
with the same uniform decay and no moment resets. Positive source-style
seeds jointly witness the parameter-bound hypotheses with a ceiling
constructed from their actual two positive denominators.

These five checked results bring the total to **688** without new sorry;
CircuitEfficiency now has 235 checked results. Next derive those ceilings
and the coefficient threshold on a hypothetical zero-Gen finite limit,
then use the retained weighted floor to exclude that limit from positive
finite-clock mass. Convergence remains explicit, and the actual source's
GD/coupled cost, learned GPTMini features and stochastic/numerical bridges
remain distinct from this fixed-table native model.

`CircuitEfficiency.SectionC_GainNoncollapse` proves positive limiting
Gen mass on an actual closed gained-CE/native trajectory with positive
finite-clock Gen mass and nonnegative numerical state. All physical
parameters must converge finitely. The actual reference coefficient
must exceed `decay * epsilon`, with positive decay and epsilon. Under
a hypothetical zero Gen mass, both factors and their actual CE inputs
vanish. Retained first moments then converge to zero, and the actual
corrected denominators converge to epsilon. A derived eventual ceiling
and coefficient lower bound enter the already checked weighted tail
floor, contradicting zero limiting mass. No positive future Gen mass,
external input sequence, buffer agreement or successful margin is assumed.

This checked result brings the total to **689** without new sorry;
CircuitEfficiency now has 236 checked results. Nonzero actual retained
balanced paths at beta1=0.9/beta2=0.98 jointly witness every hypothesis.
Next derive the reference coefficient threshold from positive Mem
balance and Gen's larger physical gain, then replace finite-clock mass
premises by actual source-seed persistence and derive eventual all-class
selection. Finite convergence, global source-seed attraction, delay and
learned stochastic/numerical GPTMini transfer remain open.

`CircuitEfficiency.SectionC_GainReferenceBalance` classifies each
nonnegative normalized two-factor pair as fully zero or positive,
equal and balanced at positive decay. It retains the actual cold
boundary. At a true gained-CE reference, a positive Mem factor
forces its partner positive, positive decay, and
`decay * epsilon < actual_CE_scale * memGain`.
Gen's larger physical gain therefore derives the threshold required
by the preceding noncollapse proof, without assuming positive Gen
reference factors. A positive Gen mass and one positive Mem factor
then imply positivity of all four actual balanced coordinates.
Reference buffers and clocks are not parameter-limit hypotheses.

These four checked results bring the total to **693** without new sorry;
CircuitEfficiency now has 240 checked results. Next combine reference
balance, retained noncollapse and initial positive-partner persistence
on the actual path. This should replace the four positive-limit
premises by source numerical signs, one positive initial Gen factor,
finite parameter convergence and a nonzero Mem limit. Global
convergence, the all-cold boundary, finite formation delay and learned
stochastic/numerical GPTMini transfer remain open.

`CircuitEfficiency.SectionC_GainSourceLimits` removes the four positive
reference-factor hypotheses from eventual fixed-table Gen selection.
Initial numerical signs and one positive Gen partner give positive Gen
mass at every finite native clock when `1 - rate * decay > 0`.
Under actual finite parameter convergence and one positive Mem reference
factor, necessary native balance supplies the feedback threshold. The
retained noncollapse proof then supplies positive Gen reference mass;
nonnegative pair classification supplies all four positive reference
coordinates, and the actual output limits yield eventual strict
correctness against every held-out competitor. No future Gen amplitude,
supplied input stream, optimizer reset or successful-margin premise is
used. The source's first Gen factor may start at zero with a small
positive second factor.

These three checked results bring the total to **696** without new sorry;
CircuitEfficiency now has 243 checked results and AdamW 178. Every joint
hypothesis list is witnessed by numerical seeds or actual retained
nonzero balanced paths with derived convergence. These witnesses do not
establish attraction from the source's small Gen/large Mem initialization.
Next remove the remaining live-Mem-limit condition where possible by
classifying the all-cold boundary with an explicit decay/epsilon regime.
Then attack actual parameter convergence and finite-delay bounds. A
separate finite-budget route should compare small positive perturbations
with the already checked permanently wrong Mem-only path while retaining
the same hyperparameters and all moment clocks; delayed failure alone
must not be promoted to eventual grokking. Learned stochastic/numerical
GPTMini transfer and the other formulations remain active and open.

`CircuitEfficiency.SectionC_GainColdRegime` computes the actual clipped
CE coefficient at zero physical parameters, even with stale buffers
and retained clocks:
`min(1, bound / 1e-6) * (remaining + 1) / (remaining + 2)`.
It is positive at positive cap; unit cap removes the zero-norm clip.
Every finite positive-cap actual CE coefficient lies strictly between
zero and one. A nonzero actual balanced point jointly witnesses a
positive decay/epsilon regime below cold Gen feedback. Unit cap,
physical Gen gain three, decay 0.1 and epsilon 1e-8 also meet that
regime for every finite class count; this is a toy gain certificate,
not a statement about learned GPTMini gains or convergence.

These six checked results bring the total to **702** without new sorry;
CircuitEfficiency now has 249 checked results. Next exclude the fully
cold reference under this static regime and combine it with the already
excluded Mem-only boundary. That should eliminate the positive-Mem-limit
premise while retaining explicit finite parameter convergence. A stale
zero-parameter point is not called an invariant trajectory: the formula
only concerns its current callback.

`CircuitEfficiency.SectionC_GainColdLimits` replaces the remaining
live-Mem reference premise by the static condition
`decay * epsilon < cold_CE_scale * genGain` at positive decay.
Actual finite parameter convergence, source numerical signs and one
positive initial Gen partner then imply positive Gen limiting mass.
A hypothetical zero Gen pair either accompanies a live Mem reference,
whose native balance derives the feedback threshold, or accompanies
zero Mem, where the exact all-cold coefficient and static regime derive
it. Retained noncollapse excludes both cases. No future nonzero Mem or
Gen factor is supplied. Separately, any actual nonnegative balanced
reference with positive Gen mass has positive Gen score and a strict
Gen-minus-Mem margin, allowing zero Mem instead of requiring an interior.

These two checked results bring the total to **704** without new sorry;
CircuitEfficiency now has 251 checked results. Nonzero actual balanced
paths with derived convergence jointly witness every hypothesis list.
Next derive the actual margin limit and eventual all-class correctness
under the static regime, then address convergence and formation time.
Neither boundary exclusion nor reference balance alone proves global
attraction, a delayed transition or learned GPTMini transfer.

`CircuitEfficiency.SectionC_GainColdSelection` derives actual positive
Gen score, positive Gen-minus-Mem limiting margin, its physical output
limit and eventual strict held-out correctness under the static regime.
It uses initial numerical signs, one positive Gen partner, positive
rate/decay/epsilon/cap, valid betas, positive remaining decay factor and
actual finite parameter convergence. No positive future Gen or Mem
factor, independently supplied input history, buffer match or successful
margin is assumed. Zero Mem references remain admitted. All hypotheses
are jointly realized by nonzero actual retained balanced paths at
beta1=0.9/beta2=0.98 with derived parameter convergence.

These two checked results bring the total to **706** without new sorry;
CircuitEfficiency now has 253 checked results and AdamW 178. Efficient
fixed-table rule selection is therefore conditional on finite native
parameter convergence and an explicit initial-data regime, rather than
on future rule presence. The new theorem does not establish global
convergence, a delayed train/test transition, learned circuit formation
or stochastic/numerical GPTMini transfer.

On 2026-10-08, `AdamW.ScalarDenominator` derives the complete numerical
denominator floor `(1 - beta1) * epsilon` at every retained clock.
It follows from the completed first bias correction and nonnegative
actual square root; no supplied variance/current-gradient agreement
is needed. Nonpositive old moments and current inputs then cap the
actual parameter increment by its newly inserted retained moment
divided by that floor. Valid first beta and positive epsilon/rate
conditions are explicit and jointly satisfiable at beta1=0.9.

These two checked results bring the total to **708** without new sorry;
AdamW now has 180 checked results. This is a finite-step numerical
bound, not a convergence or generalization certificate. Next close
the gained-CE Gen upper envelope and derive a same-hyperparameter
wrong-decision prefix from small positive Gen seeds and the Mem
pure-decay floor.

`CircuitEfficiency.SectionC_GainGrowthEnvelope` introduces the numerical
quantity `u + t * w`, with actual Gen parameter mass `u`, retained
negative first-moment mass `w`, and `t = rate / ((1 - beta1) * epsilon)`.
It covers `u` in the already derived numerical sign region. Actual
current CE insertion gives `w_next <= w + genGain * u`; the corrected
denominator floor gives `u_next <= (1 - rate * decay) * u + t * w_next`.
At nonnegative decay these imply the actual one-step upper factor
`2 + 2 * t * genGain`, which is at least two. No independent input
stream, future buffer bound or successful margin is a premise.

These five checked results bring the total to **713** without new sorry;
CircuitEfficiency now has 258 checked results. Every joint hypothesis
list is realized by numerical positive-partner seeds at beta1=0.9.
The bound is loose and may be large at small epsilon. Next derive the
full actual finite-clock upper bound and Mem pure-decay lower bound,
then choose positive seeds for an arbitrarily long wrong-test prefix
under fixed hyperparameters. Later success and convergence stay open.

`CircuitEfficiency.SectionC_GainGrowthBounds` iterates that retained
upper envelope on the actual closed gained-CE path. At every finite
clock, its growth weight is at most `R^n * initial_weight`, where
`R = 2 + 2 * rate * genGain / ((1 - beta1) * epsilon)`.
Derived nonnegative signs and the true product forward then bound
the Gen score by `genGain * (R^n * initial_weight)^2`.
Separately, every physical coordinate is at least
`(1 - rate * decay)^n * initial_parameter`; retained CE adaptation
only adds to that pure-decay contribution in this numerical region.
No future score, gradient or parameter convergence is presumed.

These four checked results bring the total to **717** without new sorry;
CircuitEfficiency now has 262 checked results. Next compare these
bounds uniformly through a fixed budget, with initially positive
Mem factors and a positive Gen partner whose other factor/buffers
start at zero. That setup isolates a train-fitted plateau; it is
not the source's entirely untrained initialization and does not
establish subsequent success.

`CircuitEfficiency.SectionC_GainDelayPrefix` compares those bounds at
every clock through a fixed budget. If initial Gen growth weight is
small enough that `genGain * (R^budget * initial_weight)^2` is below
`memGain * ((1 - rate * decay)^budget * amplitude)^2`, and both initial
Mem factors are at least positive `amplitude`, the actual Mem score is
positive and strictly dominates Gen throughout the prefix. This derives
strict correct train classification and strict unique selection of the
wrong Mem held-out class against every competitor. The condition is
entirely on numerical initial data, not on observed future performance.
The hypotheses are jointly witnessed at a nonzero three-step budget,
beta1=0.9/beta2=0.98, epsilon one and a positive Gen partner of 0.005.

These three checked results bring the total to **720** without new sorry;
CircuitEfficiency now has 265 checked results. Next construct a positive
partner seed meeting the inequality for every finite budget under the
same hyperparameters and task. A long wrong-decision prefix alone
proves neither later generalization nor convergence. Small epsilon may
require extremely small seeds under this loose envelope; this is not a
quantitative delay explanation for the preserved GPTMini runs.

`CircuitEfficiency.SectionC_GainDelaySeeds` constructs a strictly
positive partner seed for every finite budget by dividing the square
root of the positive Mem budget floor by twice the actual Gen upper
factor to that budget. Task, gains, betas, cap, epsilon, decay, rate
and Mem seed stay fixed. The actual initially zero first Gen factor
activates at step one and both Gen factors remain positive at every
later finite clock. An efficient Gen gain above Mem can nevertheless
yield strict correct train and unique wrong Mem test decisions at
every clock through the chosen budget. No future plateau, success,
moment reset or independently supplied gradient history is assumed.

The fixed configuration uses gains three/two, cap one, beta1=0.9,
beta2=0.98, epsilon 1e-8, decay 0.1 and rate 0.001. It meets the
already checked static cold regime, but finite parameter convergence
from its small Gen/positive Mem seeds remains unproved. Positive
finite-clock formation alone does not establish a positive limit.
The loose exponential envelope may require seeds below machine
range; its real-arithmetic possibility is not a quantitative account
of the preserved delayed GPTMini seed. Mem begins already train-fitted,
rather than at the source's entirely untrained initialization.

These four checked results bring the total to **724** without new sorry;
CircuitEfficiency now has 269 checked results and AdamW 180. All joint
hypothesis lists have concrete nonzero valid-beta/rate examples.
Next derive bounded actual parameter/buffer paths at positive decay,
then identify a justified convergence or small-rate attraction argument.
Do not join the delayed prefixes to conditional eventual selection
without proving convergence on those same actual trajectories.

`CircuitEfficiency.SectionC_GainBounds` proves numerical bounds for
the entire actual gained-CE path. Clipping derives each present
input magnitude at most the cap; actual convex moment insertions
keep `-moment <= cap` and `variance <= cap^2`. With valid betas,
positive epsilon and nonnegative remaining decay factor, a constant
parameter ceiling satisfying
`cap <= decay * (1 - beta1) * epsilon * ceiling`
is invariant. For nonnegative zero-buffer seeds at positive decay,
the explicit ceiling is their total initial parameter mass plus
`cap / (decay * (1 - beta1) * epsilon)`. No parameter limit, future
input sequence or buffer matching is presumed. Clocks remain unbounded.

The added `fixed_native_bounded_wrong_test_prefix` connects that result
to the same fixed-native small-seed paths above: every budget admits
a positive Gen seed whose physical parameters and both buffers stay
bounded at all clocks, whose Gen activates immediately, and whose
train decisions are correct/test decisions uniquely wrong through
the budget. These are jointly checked facts about one trajectory,
not separate existence witnesses joined by a convergence assumption.

These three checked results bring the total to **727** without new sorry;
CircuitEfficiency now has 272 checked results and AdamW 180. The finite
ceiling is loose (cap/decay/epsilon can be large). Boundedness does not
prove convergence. Next identify justified rate-dependent attraction
conditions and test whether the current unrestricted native hypotheses
admit actual periodic gained-CE trajectories before attempting to remove
the finite-convergence premise.

`CircuitEfficiency.SectionC_GainCycleFeedback` evaluates the actual
binary gained-CE inputs for equal Gen factors and zero Mem. On Gen
amplitudes in [0, 2], the true raw full-gradient norm is at most nine,
so native cap ten (including norm+1e-6) leaves it unchanged. The
actual coordinate magnitudes are
`v(a) = 3*a / (exp(3*a^2) + 1)`. Kernel-checked exponential bounds give
`v(1) > 0`, `v(2) > 0`, `3*v(2) < v(1)` and `v(1) < 3`.
No future callback is independently supplied.

These four checked results bring the total to **731** without new sorry;
CircuitEfficiency now has 276 checked results. An exploratory arithmetic
calculation proposes a period-two native beta1=beta2=0 path with
epsilon `v(1)`, high-amplitude direction `B = v(2)/(v(2)+v(1))`,
decay `(1/2+B)/3` and rate `2/(decay+1/2-B)`. It numerically alternates
Gen amplitudes one/two, at positive remaining decay factor and below
the cold feedback threshold. Next derive the configuration's regime,
actual clock/buffer-preserving updates and infinite path before claiming
a periodic or nonconvergence theorem. This large-rate legal zero-beta
specialization is not a claim about the preserved beta1=0.9/beta2=0.98 run.

`AdamW.scalar_zero_betas_parameter` follows the actual insertions and
completed-clock bias corrections at beta1=beta2=0, for arbitrary retained
buffers and clock. `CircuitEfficiency.SectionC_GainCycleStep` derives
`0 < B < 1/4`, strictly positive epsilon/decay/rate and remaining decay,
and the static cold Gen inequality for the proposed constants. It proves
the actual amplitude step from the present parameters, with no independently
supplied callback, and both exact numerical swaps one to two and two to one.
The satisfying retained-state example has nonzero buffers and clock 37.
These five checked results bring the total to **736** without new sorry;
CircuitEfficiency now has 280 and AdamW 181 checked results. Infinite
physical-parameter periodicity and nonconvergence still need a path proof.
Completed clocks themselves are not periodic. The fixed legal zero-beta
configuration differs from the preserved nonzero-beta GPTMini settings.

`CircuitEfficiency.SectionC_GainCyclePath` proves that the actual native
recurrence alternates both full physical parameter vectors
Gen=(1,1)/Gen=(2,2), with Mem=(0,0), indefinitely. The step lemmas apply
to arbitrary retained buffers/clocks at the current parameter point.
Even/odd actual subsequences force contradictory finite Gen limits,
while every physical parameter stays in [0,2]. All completed clocks
equal the actual update count. These six checked results bring the total
to **742** without new sorry; CircuitEfficiency now has 286 checked results.
Together with the proved configuration regime, this is a counterexample
to automatic parameter convergence from boundedness, positive remaining
decay and static cold feedback alone. It is not a counterexample to
held-out correctness: this Gen-only path is already correctly ordered.
Its large chosen rate and zero betas do not characterize the preserved
GPTMini settings. Next derive observable correctness/buffer bounds on
the same path and seek a rate-dependent attraction condition; retain
formation, delayed-prefix and competing spectral/phase routes.

`CircuitEfficiency.SectionC_GainFeedbackBounds` derives actual raw inputs
in `[-genGain*C, 0]`, full raw norm at most `2*genGain*C`, actual train
score at most `(genGain+memGain)*C^2`, and a shared clipping lower bound
`min(1, cap/(2*genGain*C+1e-6))`. Hypotheses constrain present physical
parameters in [0,C] and positive gains with Gen at least as efficient;
retained buffers/clocks are unrestricted and no future callback is
supplied. Four checked results bring the total to **746** without new
sorry; CircuitEfficiency now has 290 checked results. Next combine
these actual bounds into a positive full-CE feedback floor on the
already-bounded native path, then analyze relative Gen/Mem growth
for balanced zero-beta pairs without a finite convergence premise.
This specialized decision-selection route does not replace the open
nonzero-beta attraction and learned-transformer transfer obligations.

`CircuitEfficiency.SectionC_GainFeedbackFloor` combines those true
current bounds into the explicit positive floor
`min(1, cap/(2*genGain*C+1e-6)) * (r+1)/(exp((genGain+memGain)*C^2)+r+1)`.
It proves the actual shared scale dominates this number and derives
one uniform strictly positive scale on each initialized nonnegative
positive-decay native path, with any legal betas and positive epsilon.
The physical box is derived from initialization/decay/clipping, without
future scale, boundedness or convergence hypotheses. Actual coordinate
inputs also dominate the floor times their own gain/current partner.
These four checked results bring the total to **750** without new sorry;
CircuitEfficiency now has 294 checked results. This is qualitative in
exact reals: the loose ceiling inside the exponential can produce a
very small floor, not a measured time bound or a numerical-kernel
certificate. Next follow the actual balanced zero-beta recurrence and
prove a finite relative-growth escape, then compare with momentum and
source-style unbalanced initialization. Eventual margin selection for
nonzero betas still requires the explicit convergence premise.

`CircuitEfficiency.SectionC_GainBalancedStep` follows the actual native
step through zero-beta normalization and true current CE inputs.
It proves full within-pair parameter/buffer/clock equality is retained
for arbitrary fixed betas and at every finite clock from equal-factor
seeds. With legal zero betas, nonnegative signs give the exact two-factor
amplitude recurrence using the shared actual CE multiplier, derived
from initialization without a future callback or margin assumption.
These five checked results bring the total to **755** without new sorry;
CircuitEfficiency now has 299 checked results. Equal positive factors
within each pair specialize the source's unbalanced zero-first-factor
initialization; zero betas differ from preserved GPTMini settings.
Next establish an explicit relative growth advantage while Gen is
weaker and show that the generated amplitudes cross in finite time.

`CircuitEfficiency.SectionC_GainRelativeRate` factors the already-derived
actual zero-beta step into its amplitude times a relative multiplier.
At a shared current scale, it proves the exact same-amplitude rate gap,
then derives the positive lower gap
`lower*epsilon*(genGain-memGain)/((genGain*C+epsilon)*(memGain*C+epsilon))`
whenever Gen's factor amplitude is no larger than Mem's. Scale bounds
and the physical box are supplied by the checked actual native estimates;
no future parameter limit or prescribed callback is needed. The Mem
relative rate is nonnegative and at most `memGain/epsilon`.
These five checked results bring the total to **760** without new sorry;
CircuitEfficiency now has 304 checked results. Next use the gap to
prove finite-time escape and persistent actual held-out correctness
for positive balanced zero-beta initialization, then join this with
an arbitrarily long wrong-test prefix. Nonzero-beta attraction and
source-style unbalanced initialization remain distinct obligations.

`CircuitEfficiency.SectionC_GainRatioGrowth` converts the checked gap
into `q = 1 + rate*gap/(1-rate*decay+rate*memGain/epsilon) > 1`.
The positive actual Mem multiplier is bounded by that fixed denominator.
As long as Gen's factor amplitude is no greater than Mem's, the next
Gen/Mem ratio is at least q times its current ratio. The last theorem
applies this to the actual native step, deriving the shared current
CE floor/unit ceiling from the present physical box and true callback.
No independently supplied gradient, future scale, parameter convergence
or successful margin is assumed. Four checked results bring the total
to **764** without new sorry; CircuitEfficiency now has 308 checked
results. Next iterate on the actual initialized bounded path and use
unbounded powers of q to force finite crossing, then prove persistent
held-out correctness on that same trajectory. Positive balanced seeds
and zero betas remain explicit specializations.

`CircuitEfficiency.SectionC_GainBalancedEscape` proves that positive
balanced zero-beta initialized native paths have positive physical
factors at every finite clock and must reach Gen-factor > Mem-factor
at some finite update. Greater Gen gain, positive cap/epsilon/decay/rate
and positive remaining decay suffice. The actual physical box, shared
CE floor and pair-state equality are derived from initialization.
Noncrossing would force the actual ratio above `q^n*(a/b)` for q>1
while keeping it below one; the checked unbounded-powers argument
contradicts this. There is no convergence, successful reference or
future callback premise. A concrete fixed binary native configuration
works for every positive balanced Gen seed against unit Mem. True
physical product scores at the generated crossing then give strictly
correct train and held-out logits against every class.
These four checked results bring the total to **768** without new sorry;
CircuitEfficiency now has 312 checked results. This proves one actual
finite correct update; next preserve ordering at every subsequent step
and combine persistent success with arbitrarily long initial wrong-test
prefixes. Exact reals, fixed gained tables, balanced positive seeds and
zero betas remain explicit specializations, not a GPTMini transfer.

`CircuitEfficiency.SectionC_GainBalancedOrder` proves strict native
normalization monotonicity at positive epsilon and derives Gen update
> Mem update from already-ordered positive balanced factors, greater
Gen gain, a shared positive current CE scale, positive rate and
nonnegative remaining decay. Its actual-step theorem derives all
physical signs and the current CE scale from the present point.
Combining this with the generated finite crossing proves that the
same initialized native path has Gen-factor > Mem-factor at every
clock after some finite start, without parameter convergence.
These four checked results bring the total to **772** without new sorry;
CircuitEfficiency now has 316 checked results. Next transfer the
entire tail to actual train/held-out logits and combine it with a
fixed-task/fixed-hyperparameter arbitrarily long wrong-test prefix.
Equal positive pair factors, zero betas and fixed gained tables
remain explicit limits; source unbalanced seeds, general nonzero-beta
attraction, confidence/margin floors and learned GPTMini remain open.


`CircuitEfficiency.SectionC_GainBalancedSelection` translates the actual
persistent factor order into the positive current score-gap lower bound
`(genGain-memGain)*memFactor^2` and strict correctness against every class.
Positive balanced native initialization also gives strict training decisions
at every finite clock, independent of factor dominance. The generated
finite crossing and preserved ordering give correct train and held-out
decisions at all clocks after a derived finite start, without parameter
convergence or a supplied future successful reference/margin.
These four checked results bring the total to **776** without new sorry;
CircuitEfficiency now has 320 checked results. The current score lower
bound is not a uniform positive limiting margin or CE-loss convergence.
Exact-real fixed gained tables, positive equal-factor pairs and zero betas
remain explicit limits. Next join the arbitrary finite wrong-test prefix
to this permanent success on one fixed task/native configuration by
choosing a small positive balanced Gen seed. Source-style unbalanced
initialization, general nonzero-beta attraction and learned GPTMini
transfer remain separate open obligations.


`CircuitEfficiency.SectionC_GainBalancedDelay` completes an actual-path
accuracy formulation of grokking without a parameter-convergence premise.
For every finite budget, the same fixed task/gains/native constants and
Mem amplitude admit a positive balanced Gen seed whose actual path is
bounded in all physical parameters/both buffers, is strictly train-correct
at every finite clock, uniquely chooses the wrong held-out Mem class
through that budget, and permanently chooses the true held-out class
after a derived finite start. A concrete configuration uses gains 3/2,
cap 1, epsilon 1, decay 0.1, rate 0.001 and legal zero betas. The first
strictly correct held-out native update also exists and lies beyond
the budget; it is distinguished from the derived permanent-success start.
These three checked results bring the total to **779** without new sorry;
CircuitEfficiency now has 323 checked results. Initial positive balanced
Gen amplitude controls delay, with no future successful margin/reference,
parameter limit or independently prescribed callback. These are exact-real
fixed-table decisions, not CE convergence, a machine-range seed guarantee
or learned GPTMini transfer. Next relax equal-factor initialization:
derive the actual zero-beta pair difference recurrence and small-rate
contraction, then compare asymmetry decay with positive pair-mass growth
before addressing nonzero momentum and learned computations.


`CircuitEfficiency.SectionC_GainPairDifference` proves the exact unequal
partner-update difference identity with secant slope
`scale*gain*epsilon/((scale*gain*a+epsilon)*(scale*gain*b+epsilon))`.
Nonnegative factors and an actual-form scale in [0,1] put this slope
in [0,gain/epsilon]. The sufficient fixed restriction
`rate*(decay+gain/epsilon) <= 1` puts the difference coefficient in
[0,1-rate*decay] and yields absolute difference contraction.
`SectionC_GainPairContraction` identifies both actual gained native
pairs with these updates, derives the shared CE scale/sign bounds,
and proves both contractions at an actual point and the full
`kappa^n * initial_difference` bound on initialized nonnegative
retained paths. Positive rate/decay make both physical differences
tend to zero without convergence of any individual parameter.
Eight checked results bring the total to **787** without new sorry;
CircuitEfficiency now has 331 checked results. The joint hypothesis
examples include the source's zero-first-factor initialization.
This removes equal-factor initialization for absolute balance only:
both factors could shrink together, so positive mass, relative balance,
efficient competition and eventual held-out success remain subsequent
obligations. The sufficient small-rate region is nonempty for gains
3/2, epsilon 1, rate 0.001, decay 0.1; it is not asserted for the
preserved GPTMini settings. Next derive a positive native pair-mass
lower multiplier from the actual initialized box/CE floor, then compare
it with the difference contraction to get relative balance.


`CircuitEfficiency.SectionC_GainPairMass` proves a normalized partner
input floor and the lower pair-sum multiplier
`lambda = kappa + rate*lower*gain/(gain*C+epsilon)`.
It exceeds pure decay kappa, stays positive when kappa is nonnegative,
and puts kappa/lambda in [0,1). Greater gain increases lambda on a
common box. Combining the pair-sum floor and difference contraction
bounds the next relative asymmetry by (kappa/lambda) times its present
value, including initially zero first factors with positive pair sum.
`SectionC_GainRelativeStep` applies these estimates to the actual full
clipped CE, deriving its floor from the current physical box. Both
actual pair sums share the Mem-based lambda; both relative differences
contract by the same explicit actual-box ratio in [0,1) under the
sufficient fixed small-rate condition. Eight checked results bring
the total to **795** without new sorry; CircuitEfficiency now has 339
checked results. These pointwise estimates require a present finite
box, not a future callback, balance, parameter limit or success margin.
Next generate that box and positive pair masses from nonnegative
zero-buffer initialization, prove full relative-asymmetry power bounds
and limits, then study efficient Gen/Mem competition. Lambda need not
exceed one, so shrinking total mass is compatible with relative
balance. Fixed tables, exact reals, zero betas and a sufficient small
rate remain explicit limits of the transfer to learned GPTMini.


`CircuitEfficiency.SectionC_GainRelativePath` generates a finite physical
and retained-buffer box from nonnegative zero-buffer initialization,
positive decay and the sufficient fixed small rate. Its true CE floor
gives one explicit q in [0,1). Actual pair-mass floors keep both sums
strictly positive at every finite clock, and induction bounds both
relative differences by q^n times their initial values. Both relative
differences tend to zero without finite limits for individual parameters.
The exact identity `4*a*b/(a+b)^2 = 1-relative_difference^2` then proves
that both actual normalized products tend to their current fixed-sum
maximum. Positive limiting product is not asserted by this normalization.
Four checked results bring the total to **799** without new sorry;
CircuitEfficiency now has 343 checked results. Joint hypothesis examples
include the source's zero-first-factor seeds. No future box, gradient
callback, lower mass, balance or success margin is supplied. Next compare
true unequal-pair mass updates to their balanced proxies with a derived
error controlled by relative asymmetry, then prove finite efficient Gen
crossing and persistent held-out correctness for unbalanced seeds.
The quantitative CE floor is loose and can be extremely small; these
exact-real fixed-table, zero-beta/small-rate statements are qualitative,
not a measured or floating-point GPTMini clock bound. Nonzero-momentum
attraction, learned circuit formation and loss/margin limits stay open.
A separate phase-formulation route can now examine noncommuting small-
positive-Gen-seed and long-time accuracy limits on the already-checked
balanced actual model; this would not establish a thermodynamic limit.


`CircuitEfficiency.SectionC_GainSeedSequence` constructs the explicit
positive balanced Gen family
`sqrt(memGain/genGain)*memAmplitude/4 * (kappa/R)^index`, where R is
the checked native epsilon-floor envelope factor. Under fixed legal
constants its ratio lies in (0,1), every member is positive and the
family tends to zero. The initial Gen envelope score at its own
budget is exactly one quarter of the Mem pure-decay score, so the
same actual initialized path is strictly train-correct and uniquely
held-out-wrong through that index. This five-result family construction
changes only initialization, leaving task/native constants fixed.
`SectionC_GainDecisionLimits` uses gains 3/2, cap/epsilon 1, decay 0.1,
rate 0.001, zero betas and unit Mem to define the actual physical
forward and its strict true-target success indicator. For each fixed
clock the seed-family trace tends to zero; for every fixed positive
member its actual clock trace tends to one by derived permanent
success. All inner limits are proved before using `Filter.limUnder`.
The actual two iterated limits are 0 and 1 and therefore do not commute.
No common finite permanent-success start covers the entire family.
Seven decision results plus five family results bring the total to
**811** without new sorry; CircuitEfficiency now has 355 checked results.
The two indices represent actual native clock and decreasing initial
Gen amplitude, not varying width, dataset size or optimizer constants.
This is a proved nonuniform long-time/small-initialization singularity
in the closed exact-real fixed-table model, without parameter convergence
or a prescribed staircase/future-success input. It is not a Gibbs,
temperature or thermodynamic-limit claim, nor ordinary GPTMini transfer.
The indicator counts ties as zero under an explicit convention; both
proved early and late phases are strict, so the result uses no tie artifact.
Next return to unequal-seed competition, derive its balanced-proxy error
from proved relative balance, and characterize which positive-mass/margin
or loss limits accompany accuracy selection. Nonzero-beta attraction,
learned-circuit formation, actual floating-point seeds and finite-size
scaling remain independent obligations.

Next attack actual parameter bounds, convergence and formation time.
For a finite-budget delay route, derive the full corrected denominator
floor `(1 - beta1) * epsilon`, an actual Gen mass/retained-moment upper
envelope using the checked `actual_CE_scale < 1`, and the Mem pure-decay
lower floor. From zero Gen buffers and a sufficiently small positive
partner seed, compare these bounds at every clock through a fixed budget
with the same hyperparameters. Such a prolonged wrong-decision prefix
alone proves neither later success nor a grokking transition; finite
convergence remains a separate obligation. Keep spectral, geometry,
operational, component-formation and phase formulations active.

Next derive source-seed attraction and delayed selection in the actual
physical forward under the same uniform native AdamW, and characterize
boundary limits. Retained closed CE feedback has checked formation,
nonformation, zero-decay score/loss asymptotics, equal-gain allocation
obstructions and efficient physical-gain positive-limit selection.
The last result derives its successful fixed-table margin; global
positive convergence and a learned GPTMini generalizing margin remain
unproved. Boundary allocations and changing learned features need
separate analysis.
Keep the derived state-dependent finite-step interval and reachable
momentum counterexample; do not substitute a universal prescribed rate
or infer multi-step objective/geometry progress from one stochastic step.
Extend the numerical error certificate to nonzero gradients and actual
kernels only with verified premises. Continue coupled-component,
control-parameter and spectral-gap routes as competing formulations,
rather than treating optimizer obstructions as a complete mechanism.
Keep fixed-table dynamics distinct from actual GPTMini loss. Preserve
the distinction between an inverse-rate characteristic time and the time
to cross a task-dependent generalization threshold. Extend phase-transition
formulations only with stated control parameters and asymptotic regimes.
Keep the actual-transformer/AdamW transfer open. New papers may introduce
open theorem statements under AGENTS.md's explicit allowance; existing
sorry counts must not increase.


`CircuitEfficiency.SectionC_GainPairProxy` identifies the exact positive-
epsilon unequal-factor correction to the normalized native pair sum.
The next sum equals twice the balanced update at the pair mean, minus
rate times an explicit nonnegative error. CE and clipping remain at the
original actual point; this is a current proxy, not another trajectory.
The error is at most `gain^2 * difference^2 / (2 * epsilon^2)`. On the
derived finite box, its ratio to current positive pair mass is bounded
by `gain^2 * ceiling / epsilon^2` times squared relative asymmetry.
`SectionC_GainProxyLimits` instantiates both exact Gen/Mem laws on actual
retained native steps and proves that both normalized correction errors
and next-mass proxy residuals tend to zero along initialized unequal-
pair paths. The earlier full-clock mass and relative-balance theorems
generate every sign, box and balance premise, including zero-first-
factor seeds. No parameter convergence, future mass floor, successful
reference or altered CE scale is supplied. Five scalar and three actual
path results bring Grokking to **819** and CircuitEfficiency to 363, with
no new sorry. Target modules compile without warnings.
These exact-real fixed-table laws require zero betas and the sufficient
small-rate condition `rate * (decay + genGain / epsilon) <= 1`; they do
not cover the preserved nonzero-beta GPTMini configuration. Next use
the vanishing relative correction to prove late efficient Gen/Mem mass
ratio growth, finite crossing and persistent actual held-out correctness
from unequal seeds without individual parameter limits. Loss convergence,
nonzero-beta attraction and learned-circuit transfer remain open.


`CircuitEfficiency.SectionC_GainMassRates` derives strict positive
remaining decay from the sufficient small-rate inequality, rewrites the
unequal actual-form pair mass as its balanced relative multiplier minus
the explicit error, and proves mass positivity, an epsilon multiplier
ceiling and a lower multiplier from a current relative-error budget.
`SectionC_GainUnequalGap` proves that when Gen mass is no greater than
Mem, a Gen error per mass at most half the derived balanced relative gap
leaves a positive actual relative-mass advantage. Mem's nonnegative
correction only helps. General current-number identities transfer a
quantified relative multiplier advantage to a greater mass ratio.
`SectionC_GainUnequalRatio` converts this half-gap into the fixed factor
`1 + rate * gap / (2 * MemMultiplierCeiling) > 1`, including the actual
retained native step and its original closed clipped CE. The scale floor
and current signs are derived from the physical box. A strict current
ratio increase is also proved. Six mass results, four gap results and
three ratio results bring Grokking to **832** and CircuitEfficiency to
376 with no new sorry. All three target modules compile without warnings.
The error premise is explicitly current and numerical, not a future
trajectory or success assumption. Next generate its finite tail from the
proved initialized relative-error limit, iterate the actual ratio bounds
to finite efficient Gen/Mem crossing for unequal seeds, and then derive
persistent physical task correctness. Individual parameter convergence,
positive limiting margin, nonzero-beta attraction, floating-point transfer
and learned GPTMini circuit selection remain separate open obligations.


`CircuitEfficiency.SectionC_GainUnequalGrowthTail` derives a finite
start meeting every positive actual normalized-error tolerance from
initialized nonnegative positive-sum seeds. It then chooses half the
efficiency gap from the same derived physical box and actual clipped CE
floor, producing a fixed ratio multiplier greater than one at all late
clocks where Gen mass is no greater than Mem. The static small rate
generates positive remaining decay. The pinned zero-first-factor family
with gains 3/2, cap/epsilon 1, decay 0.1, rate 0.001 and zero betas also
has this derived growth tail for every positive Gen partner seed.
`SectionC_GainUnequalEscape` proves that after every prescribed budget
there is an actual finite clock with strict Gen pair-mass dominance.
Otherwise the positive ratio has an unbounded q^k lower bound while
remaining at most one. Both general and pinned families cannot have
a permanently Gen-weaker tail. These three tail and four crossing
results bring Grokking to **839** and CircuitEfficiency to 383 with no
new sorry. Both target modules compile without warnings. There is no
parameter-convergence, future error bound, successful reference or
prescribed gradient-stream hypothesis: all bounds are initialized
consequences of the original actual clipped native CE recurrence.
Mass crossing is not yet physical product/decision selection. Next
control the unequal Gen correction after crossing, prove persistent
mass dominance, and combine it with relative balance to obtain permanent
held-out correctness. Positive limiting mass/margin, CE convergence,
nonzero-beta attraction, finite precision and learned GPTMini transfer
remain separate open obligations.


`CircuitEfficiency.SectionC_GainMassOrder` proves monotonicity of the
balanced pair input in current mass and a quantitative ordered-proxy
difference: remaining decay times the mass difference plus rate times
the efficiency gap times Mem mass. It derives a positive numerical
tolerance bounded by half the gap and, when rate-weighted, by half
remaining decay. `SectionC_GainUnequalOrderStep` subtracts the exact Gen
correction and keeps Mem's helpful nonnegative correction, proving that
this current tolerance preserves strict next Gen mass dominance on both
actual-form pairs and the original retained native step. Positive current
pair mass also gives two positive factors after one native-form update,
including either initial zero factor. `SectionC_GainUnequalOrderTail`
generates the required error tail from actual initialized relative limits,
uses the crossing beyond that start and iterates the actual order step.
Both the general nonnegative positive-sum seeds and the pinned first-zero-
factor family have permanent strict Gen mass dominance after a generated
finite clock, without individual parameter convergence or a future success
premise. All actual factors are positive at every successor clock. Four
proxy/tolerance, three step/formation and three initialized-tail results
bring Grokking to **849** and CircuitEfficiency to 393, with no new sorry
and no warnings in the three new target modules.
The pair sums can remain small or shrink; positive limiting mass and
physical product/margin order are not asserted. Next combine permanent
mass dominance with actual relative balance to obtain a positive current
physical score gap and permanent all-class train/held-out correctness for
unequal seeds, then join the arbitrary wrong prefix on that same path.
Nonzero-beta attraction, CE convergence, finite precision and learned
GPTMini transfer remain separate obligations.


`CircuitEfficiency.SectionC_GainMassBalanceScore` turns a squared
relative-asymmetry budget into a lower product bound and derives positive
product when that budget is below one, without separately assuming factor
signs. If Gen mass is at least Mem mass and its squared relative difference
is at most `(genGain - memGain) / (2 * genGain)`, the true physical score
gap is at least `(genGain - memGain) * MemMass^2 / 8 > 0`. Applying this
to the actual current state gives strict all-class train/held-out decisions.
`SectionC_GainUnequalSelection` derives strictly correct actual training
decisions at every successor clock and generates the squared-balance
threshold from the initialized relative limit. Combined with permanent
mass dominance, the same actual native path has a positive current physical
margin and permanent all-class train/held-out correctness after a generated
finite clock. This includes general nonnegative positive-sum unequal seeds
and the pinned source-style `(0, seed)` Gen / `(0, 1)` Mem family. There is
no equal-factor, individual parameter-convergence, future margin, successful
reference or prescribed gradient-stream hypothesis. Four scalar/current
score results and four initialized selection results bring Grokking to
**857** and CircuitEfficiency to 401, without new sorry or target warnings.
The sufficient small rate and zero betas remain explicit; all physical CE,
clipping, retained clocks and buffers are the original native recurrence.
The margin bound depends on current Mem mass and can vanish. This is
permanent exact-real accuracy selection, not positive limiting confidence,
CE-loss convergence, finite precision or learned nonzero-beta GPTMini
transfer. Next join the arbitrary finite wrong prefix and permanent success
on one unequal Gen path, then investigate the fully source-style initial
zero products, confidence limits and nonzero-beta attraction separately.


`CircuitEfficiency.SectionC_GainUnequalDelay` joins the arbitrary finite
wrong-test prefix with initialized permanent success on one actual path.
The task, gains, clipping cap, epsilon, decay, rate and positive equal Mem
factors are fixed. Only the positive Gen partner seed varies; its first
factor is exactly zero. The same initialized path has bounded physical
parameters/both retained buffers, strictly correct train decisions at every
finite clock, a positive Gen product at every successor clock, uniquely
wrong held-out Mem decisions through the prescribed budget, and permanent
true held-out decisions after a generated later start. The pinned legal
configuration is instantiated. The least strictly correct actual held-out
clock is derived beyond the budget; it is not claimed to be the permanent-
success start. Three results bring Grokking to **860** and CircuitEfficiency
to 404 without new sorry or target warnings. No equal Gen factors, finite
individual weight limits, future gradient stream, teacher or successful
reference is supplied. The sufficient small rate derives positive remaining
decay and every subsequent path premise follows from initial data.
Mem is already train-fitted here, unlike the source's first-zero Mem factor;
permanent source-style first-zero Gen/Mem selection is proved separately,
but its arbitrary wrong prefix after initial formation still needs a bound.
These are exact-real fixed-table zero-beta accuracy theorems. Seeds may be
below machine range and the margin may vanish; CE-loss convergence and
learned nonzero-beta GPTMini transfer are not asserted. Next derive delayed
selection from both initially zero products, characterize confidence/CE
limits (including strong-decay counterexamples), and extend justified
attraction beyond zero betas. Keep spectral, geometry and phase formulations
active and retain the preserved ordinary-transformer evidence unchanged.


`CircuitEfficiency.SectionC_GainMassDecay` derives the actual native
pair-mass ceiling `1 - rate * decay + rate * gain / epsilon`, including
zero-mass pairs. The shared clipped CE coefficient is the actual current
one and lies below one. Larger Gen gain gives a common ceiling for both
pairs; retained state signs also bound each next coordinate by its pair
ceiling. Decay greater than Gen gain divided by epsilon, with nonnegative
remaining decay and positive rate, puts the ceiling strictly between zero
and one. `SectionC_GainStrongDecayPath` then iterates that ceiling from
any nonnegative retained initial state, derives both pair-mass limits zero,
and squeezes all four actual physical parameter limits to zero. Initial
masses can vanish, buffers and clocks can be retained, and no future box,
parameter convergence, reference or positive mass floor is assumed. Five
current-step and three initialized-path results bring Grokking to **868**
and CircuitEfficiency to 412, with no new sorry or target warnings.
For gains 3 and 2, epsilon/clipping 1, decay 10 and rate 1/1000, the
ceiling is 993/1000 and the previously proved small-rate accuracy selection
condition also holds. Combining that actual delayed-accuracy result with
these absolute limits, then deriving logits, confidence and actual train/
held-out CE limits, is next. The current limits alone do not assert task
success. All claims remain exact-real fixed-table zero-beta native laws;
nonzero-beta attraction and learned GPTMini/numerical transfer remain open.


`CircuitEfficiency.SectionC_GainStrongDecayLoss` derives both original
physical score limits and their margin limit zero from initialized strong
decay. Actual train and held-out CE tend to `log(remaining+2)`, and every
held-out class softmax probability tends to `1/(remaining+2)` using the full
finite-class exponential sum. No parameter convergence, future reference,
loss/confidence limit or successful margin is supplied.
`SectionC_GainAccuracyCollapse` combines these results with arbitrary
delayed permanent accuracy selection on the same actual bounded first-zero
Gen path, under compatible static constants. It retains perfect training
decisions, a positive Gen product at every successor clock, an arbitrarily
long uniquely wrong Mem test prefix, and a generated permanent true-class
tail, while all physical weights/margin vanish and both CE limits are
uniform-class loss. The pinned gains 3/2, cap/epsilon 1, decay 10 and
rate 1/1000 give a formal nonzero-held-out-CE counterexample despite delayed
permanent strict accuracy. Three limit and two joined-path/counterexample
results bring Grokking to **873** and CircuitEfficiency to 417, with no new
sorry or target warnings. This is exact-real accuracy selection, with
vanishing confidence rather than improved confidence. It is not a source
coupled-cost GD or learned nonzero-beta GPTMini counterexample. Next make
positive tail-margin/confidence and perturbation robustness failures precise,
then continue the source-style initially zero products and nonzero-beta
attraction obligations. Preserve all measured transformer evidence.


`CircuitEfficiency.SectionC_GainCollapseRobustness` constructs an actual
multiclass counterperturbation: subtract a positive radius from the target
logit only. Every coordinate error is bounded by that radius; if the actual
Gen-minus-Mem margin is smaller, the wrong Mem coordinate beats the
perturbed target. Initialized strong decay derives eventual perturbability
at every fixed positive radius and excludes every positive uniform margin
lower bound on a tail. The pinned actual configuration joins arbitrary
wrong-test prefixes, perfect train decisions, permanent later exact-real
test correctness, and eventual counterperturbability at every positive
radius on that same original path. Four results bring Grokking to **877**
and CircuitEfficiency to 421, without new sorry or target warnings.
The counterperturbations are mathematical controls; they are not claimed
to be the errors produced by a particular floating-point kernel and do not
change training. This proves a robustness obstruction for this exact-real
accuracy formulation, not numerical failure of preserved GPTMini. Next
continue confidence thresholds and initialized attraction/convergence for
legal nonzero native betas; source-style initial zero products, spectral/
geometry/phase transfer and actual transformer kernel bridges remain open.


`CircuitEfficiency.SectionC_GainMemoryMass` keeps the complete physical
parameter mass and negative retained first-moment mass. It proves their
numerical signs/coordinate coverage, the exact actual moment insertion,
a beta-dependent moment ceiling, the original corrected-denominator
parameter ceiling, and an actual applied-input cap by greater gain times
current total parameter mass. `SectionC_GainMemoryContraction` uses the
explicit weight `rate/((1-beta1)^2*epsilon)` and derives both coefficient
identities and the common ceiling
`max(1-rate*decay+rate*genGain*(2-beta1)/((1-beta1)*epsilon), beta1*(2-beta1))`.
Decay greater than `genGain*(2-beta1)/((1-beta1)*epsilon)`, with nonnegative
remaining decay and positive rate, makes that computed factor strictly
between zero and one. The threshold is sufficient, not claimed sharp.
`SectionC_GainMemoryCollapse` iterates the actual state with both legal
betas and arbitrary nonnegative retained initial buffers/clocks. It derives
all-clock weighted-mass/coordinate power ceilings, zero mass and physical
parameter limits, and zero limits of every actual applied CE input. No
future parameter/input/sign bound, supplied reference, zero-beta or reset
premise is used. Five current-mass, four coefficient/current-ceiling and
four initialized-collapse results bring Grokking to **890** and
CircuitEfficiency to 434, with no new sorry or target warnings.
The nonempty example uses betas 0.9/0.98, gains 3/2, cap/epsilon 1, rate
1/1000 and decay 100; the contraction factor is 99/100. This proves
initialized attraction to zero under a static strong-decay regime, not
relative circuit selection or preserved GPTMini attraction. Its epsilon/
decay differ from the measured model. Next derive retained buffer and
actual denominator/feedback limits from generated inputs, then investigate
relative competition with nonzero-beta memory. Positive-confidence
attraction in a weaker-decay regime, source-style zero initial products,
confidence detectors, stochastic/numerical kernels and learned transformer
transfer remain open.


`CircuitEfficiency.SectionC_GainMemoryLimits` derives both original
retained buffer limits zero at every coordinate from the generated applied
inputs and their actual causal histories. The complete newly corrected
denominator tends to the original positive epsilon, while every completed
clock diverges. Physical limits generate the true shared CE/clipping scale's
computed positive cold limit, including the finite class count and the
norm-plus-1e-6 clipping strip. No input/parameter convergence, future
reference, instantaneous variance match or reset is supplied. A finite
all-zero numerical reference only evaluates the continuous feedback; the
optimizer state is not declared to have a finite clock limit. Three
results bring Grokking to **893** and CircuitEfficiency to 437, without
new sorry or target warnings. Coordinate inputs vanish through learned
partner factors although the limiting multiclass CE score multiplier stays
positive. This closes the static strong-decay retained-state limits for
legal nonzero betas; it does not yet prove relative circuit selection or
positive-confidence attraction. Next compute the exact memory/partner
recurrence coefficients and their generated asymptotic gain gap, then
analyze the actual coupled competition with nonzero-beta memory. Keep
source-style zero products, confidence detectors and learned stochastic/
numerical GPTMini transfer open.


`CircuitEfficiency.SectionC_GainMemoryCoefficients` rewrites the original
actual parameter step into remaining decay times its present parameter,
a computed coefficient times its negative retained moment, and a computed
coefficient times its current partner. Each coefficient retains the full
actual denominator, current shared CE/clipping and physical gain. The
initialized legal-beta strong-decay path gives their limits
`rate*beta1/epsilon` and `rate*(1-beta1)*coldCE*gain/epsilon` without an
independent convergence or future-coefficient hypothesis. Greater Gen gain
generates a positive limiting partner-coefficient difference and a strictly
greater actual Gen partner coefficient on a derived tail. Three results
bring Grokking to **896** and CircuitEfficiency to 440, with no new sorry
or target warnings. This is an exact current recurrence with generated
coefficient limits, not a derivative at an invented infinite-clock state.
A component coefficient advantage does not by itself prove coupled
parameter/moment dominance or within-pair balance. Next bound actual
positive pair mass/moment dynamics by ordered two-dimensional recurrences,
control the signed within-pair parameter/moment modes, and derive relative
Gen selection with retained nonzero-beta memory. Keep positive-confidence
attraction at weaker decay, the source-style arbitrary prefix after two
initial zero products, confidence detectors, spectral/geometry/phase
formulations and learned stochastic/numerical transformer transfer open.
All preserved full-budget ordinary-transformer evidence remains unchanged.


`CircuitEfficiency.SectionC_GainMemoryWeights` adds a numerical pair
measurement: its current parameter sum plus a nonnegative multiple of
its retained negative first-moment sum. The exact original next measurement
is a positive-coordinate combination with computed current coefficients,
including both actual denominators, shared CE/clipping and all retained
buffers/clocks. Current coefficient intervals therefore bound the actual
next pair measurement from above and below. Four results bring Grokking
to **900** and CircuitEfficiency to 444, with no new sorry or target
warnings. Numerical signs and intervals are current premises at this
level; next generate separated Gen/Mem intervals on the legal-beta
initialized strong-decay path and compare their full retained weights.
A pair measurement advantage is not a product margin: signed within-pair
parameter/moment balance, confidence, weaker-decay attraction and learned
stochastic/numerical transformer transfer remain open. All preserved
ordinary-transformer full-budget evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryIntervals` derives both actual
retained pair coefficients' computed cold limits from initialized legal-
beta strong decay, and generates finite coefficient interval tails from
strict inequalities on those explicit static numbers. The binary fixed
configuration has beta1=0.9, beta2=0.98, gains 3 and 2, epsilon/cap 1,
decay 100 and learning rate 0.001. Weighting negative retained moments
by 0.08 for Gen and 0.095 for Mem gives numerical intervals with Gen
lower rate 0.911 and Mem upper rate 0.910. Three results bring Grokking
to **903** and CircuitEfficiency to 447, without new sorry or target
warnings. These intervals do not assume future parameter, buffer or
coefficient convergence, and do not replace the original optimizer by
a fixed matrix. Next combine the generated coordinate tails into actual
pair weight bounds and initialized relative competition. For eventual
product selection, investigate positive partner mixing and coverage of
retained moments by physical parameters as an alternative to full signed
pair balance; whether those bounds suffice must be proved. Strong decay
permits absolute collapse, so confidence and exact decisions stay separate.
Weaker-decay attraction, learned stochastic/numerical GPTMini transfer,
spectral/geometry/phase formulations and confidence detectors stay open.


`CircuitEfficiency.SectionC_GainMemoryCompetition` combines generated
coordinate intervals with the original exact retained pair law on the
pinned binary configuration. From arbitrary nonnegative parameter,
nonpositive first-moment and nonnegative second-moment initial data,
the full original path keeps those numerical signs and generates a
finite tail on which the Gen measurement is nonincreasing but at least
0.911 times its preceding value; the Mem measurement is at most 0.910
times its preceding value. Beta1=0.9 and beta2=0.98 are retained at all
clocks, including corrections and current shared clipped CE. Three
results bring Grokking to **906** and CircuitEfficiency to 450, without
new sorry or target warnings. The rates now apply to the actual original
nonlinear trajectory, not just computed limiting numbers or a supplied
constant matrix. Next derive initialized positive Gen measurements,
tail power bounds and vanishing Mem/Gen measurement ratios. Physical
product selection still needs coverage of retained moments by physical
parameters and positive partner mixing, or a full pair-balance theorem.
Confidence, weaker-decay attraction and learned stochastic/numerical
GPTMini transfer remain open; frozen transformer evidence is unchanged.


`CircuitEfficiency.SectionC_GainMemoryPowers` derives the unconditional
current Gen parameter/moment measurement floor 0.9 from original
nonpositive CE inputs and the pinned equality beta1=remaining decay.
Its initialized full-clock power bound preserves every positive initial
numerical Gen measurement at finite clocks, including nonzero retained
buffers/clocks. The generated common tail separately gives Gen lower
powers 0.911^k and Mem upper powers 0.910^k from the same actual start
state. Four results bring Grokking to **910** and CircuitEfficiency to
454, without new sorry or target warnings. No future signs, parameter
limits, coefficient intervals, gradients or success conditions are
independently supplied. Next use the strict ratio 910/911 to derive a
vanishing actual Mem/Gen measurement ratio and compare arbitrary finite
amplitude multiples. Product selection still requires physical coverage
and partner mixing, or pair balance. Positive confidence, weaker-decay
attraction and learned stochastic/numerical GPTMini transfer remain open.
All frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryRatio` proves a geometric ceiling
(910/911)^k for the actual Mem/Gen retained pair measurement ratio from
a generated common start, and its full-clock limit zero. Initial signs
and positive present Gen numerical mass suffice; no future convergence,
coefficient interval or task success is supplied. Every fixed positive
multiple of actual Gen mass eventually exceeds Mem permanently. On the
same original beta1=0.9/beta2=0.98 path, all physical parameters, both
retained buffers and both pair measurements tend to zero. Four results
bring Grokking to **914** and CircuitEfficiency to 458, without new sorry
or target warnings. Relative selection and absolute collapse coexist;
zero component limits alone would not determine this quotient limit.
Next derive a physical-parameter ceiling for retained negative moments
from generated full-denominator bounds, then transfer the ratio to true
parameter sums. Investigate positive partner mixing to obtain product
selection without requiring complete signed pair balance. Confidence,
weaker-decay attraction, learned stochastic/numerical GPTMini transfer,
other detector/geometry/spectral/phase routes remain open. Frozen full-
budget ordinary-transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryCoverage` uses the original full
denominator's generated tail ceiling 2 to bound each newly retained
negative first moment by 2000 times its own new physical parameter.
The actual 0.08-weighted Gen measurement is at most 161 times the
true Gen parameter sum on that tail; no initial moment-size constraint
or future denominator/moment hypothesis is supplied. Original relative
measurement selection then implies that every fixed positive multiple
of the actual Gen parameter sum eventually exceeds Mem's parameter
sum permanently. Four results bring Grokking to **918** and
CircuitEfficiency to 462, without new sorry or target warnings. This
closes the transfer to true physical sums for the pinned original
beta1=0.9/beta2=0.98 path, while all parameters/buffers still collapse
absolutely. Next derive generated positive partner-coefficient mixing
and a uniform lower bound on both Gen factors relative to their current
sum; combine with arbitrary small Mem/Gen sum ratios to prove actual
product/held-out selection without complete signed pair balance.
Positive confidence, weaker-decay attraction, source-style arbitrary
delay with nonzero-beta memory and learned stochastic/numerical GPTMini
transfer remain open. Frozen full-budget transformer evidence is unchanged.


`CircuitEfficiency.SectionC_GainMemoryMixing` generates a common positive
actual partner coefficient floor 1/20000 on the original legal-beta path.
Current numerical signs and its actual denominator then make each new
physical factor at least 1/20000 times its preceding partner-pair sum,
with the retained moment contribution kept. Generated Gen measurement
monotonicity and coverage give the next true Gen sum ceiling 161 times
the previous sum. Consequently both new Gen factors occupy at least
1/3220000 of their own new sum on a derived tail. Four results bring
Grokking to **922** and CircuitEfficiency to 466, without new sorry or
target warnings. No future coefficient, factor ratio, equal-factor or
parameter limit is supplied. This weak uniform positive cone may suffice
for product selection; full signed relative-balance convergence is not
claimed. Next combine the cone with arbitrarily small actual Mem/Gen
parameter-sum multiples to derive a positive gained product margin and
permanent true held-out decisions with nonzero-beta memory. Absolute
collapse, confidence limits, source-style arbitrary delay and learned
stochastic/numerical GPTMini transfer remain distinct obligations.
All frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemorySelection` closes actual product
and decision selection on the original fixed binary native path with
beta1=0.9 and beta2=0.98. Initial full signs and positive present Gen
parameter/moment measurement suffice. The generated Gen factor cone
and arbitrarily small actual Mem/Gen sum multiples imply a positive
gained Gen product strictly above Mem, then permanent strict correct
train/held-out decisions after a generated finite clock. No future
convergence, interval, factor ratio, margin, successful reference or
replacement gradient stream is supplied; full relative pair balance
is unnecessary for this result. The held-out gained score margin on
the same selected path nevertheless tends to zero. Four results bring
Grokking to **926** and CircuitEfficiency to 470, without new sorry or
target warnings. This advances from earlier zero-beta unconditional
selection and nonzero-beta statements requiring finite parameter limits
to an initialized legal nonzero-memory result under the sufficient
static strong-decay configuration. Next combine original arbitrary
wrong prefixes with this permanent selection on the very same fixed
native constants, and derive the first true held-out clock beyond any
budget. Confidence/robustness, weaker-decay attraction and learned
stochastic/numerical GPTMini transfer remain open. Frozen transformer
evidence and unchanged full-budget optimizers are preserved.


`CircuitEfficiency.SectionC_GainMemoryDelay` closes arbitrary delayed
permanent accuracy on one fixed original native configuration with
beta1=0.9, beta2=0.98, gained binary tables 3/2, epsilon/cap 1, decay
100 and rate 0.001. Every prescribed finite budget admits a positive
Gen partner seed with first Gen factor zero and initial Mem (1,1).
On the same actual clipped-CE path, training decisions are strictly
correct at every clock, Gen's product forms at step one and stays
positive, yet the uniquely wrong held-out Mem class is selected through
the entire budget. A generated later start gives permanent strict true
held-out answers; the least true clock also exceeds the budget and has
no earlier strict true answer. There is no uniform finite success start
over positive Gen partner seeds. Three results bring Grokking to
**929** and CircuitEfficiency to 473, without new sorry or target
warnings. No future convergence, coefficient interval, factor ratio,
margin, successful reference or replacement gradient is supplied. This
advances the previously unconditional zero-beta delayed-accuracy path
to original legal nonzero-beta retained memory. It concerns exact-real
accuracy, not positive limiting confidence or stable floating-point
answers; all physical parameters/buffers and selected margins can vanish.
The next initial-data refinement is source-style zero first factors in
both Gen and Mem: derive a uniform actual post-formation Mem amplitude
and a shifted retained-state prefix, then keep later selection on that
same original path. Also derive complete CE/confidence/robustness limits
for the selected nonzero-beta delay family. Weaker-decay attraction,
learned stochastic/numerical GPTMini transfer and broader detector/phase/
spectral/geometry formulations remain open. Frozen full-budget ordinary-
transformer evidence and unchanged optimizers are preserved.


`CircuitEfficiency.SectionC_GainMemoryLoss` derives complete actual score,
margin, train/held-out CE and held-out softmax probability limits on the
original initialized legal-beta strong-decay path. Both retained buffers
and full correction clocks are preserved. All classes remain in the true
CE/exponential denominator: both CE limits are log(remaining+2) and every
class probability tends to 1/(remaining+2). No future parameter, input,
loss, confidence or successful-reference premise is supplied. The static
sufficient decay bound generates physical collapse; a zero numerical
reference only evaluates the continuous forward. Three results bring
Grokking to **932** and CircuitEfficiency to 476, without new sorry or
target warnings. Next combine these limits with the proved arbitrarily
delayed permanent accuracy on the same beta1=0.9/beta2=0.98 configuration,
and derive the lack of a fixed positive confidence/robustness radius.
Source-style double zero-first-factor initialization, weaker-decay
attraction, full signed pair balance and learned stochastic/numerical
GPTMini transfer remain open. Frozen full-budget transformer evidence
and unchanged optimizers are preserved.


`CircuitEfficiency.SectionC_GainMemoryConfidence` combines actual
nonzero-memory arbitrary delay, formation and permanent strict accuracy
with complete train/held-out CE limits log 2 and both held-out class
probabilities tending to 1/2 on the same original initialized path.
Only the positive Gen partner seed varies with the prescribed budget;
all native/task constants and initial Mem factors stay fixed. The true
held-out CE does not tend to zero. Initial numerical signs also exclude
any fixed confidence threshold above one half on a tail; for each such
threshold the same delayed permanently correct path eventually stays
below it. Five results bring Grokking to **937** and CircuitEfficiency
to 481, without new sorry or target warnings. No future limits, inputs,
successful references or confidence premises are supplied. This is an
exact-real accuracy counterexample under sufficient strong decay, not
a claim of low training CE or numerical stability. Next prove native
nonzero-memory counterperturbability and refine to source-style zero
first factors in both pairs. Weaker-decay positive-confidence attraction,
full signed pair balance, other detector/geometry/spectral/phase routes
and learned stochastic/floating-point GPTMini transfer remain open.
Frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryFormation` derives a uniform
actual post-formation Mem amplitude from source-style zero first factors
in both Gen and Mem. Initial Gen is (0,seed), Mem (0,1), with
0 <= seed <= 1. A complete CE/clipping box floor is positive and below
one. The actual native first-step normalization and partner decay floor
make both Mem factors at clock one at least that floor/2000, independent
of the Gen seed. The actual retained Gen parameter/moment growth weight
is at most 103/50 times the seed. A full-state path shift by one retains
actual moments, variances and completed clock, without restarting fresh
buffers. Five results bring Grokking to **942** and CircuitEfficiency
to 486, without new sorry or target warnings. Initial logits are zero,
so strict training correctness must start at clock one. Next choose a
positive bounded seed using the uniform formed amplitude and prove an
arbitrarily long wrong-test prefix after actual formation, then combine
with permanent selection on that same original nonzero-beta path.
Native robustness, weaker-decay attraction, complete signed pair balance
and learned stochastic/numerical GPTMini transfer remain open. All
frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryFormedPrefix` closes arbitrary
delayed permanent accuracy from source-style zero first factors in
both pairs on the original fixed beta1=0.9/beta2=0.98 configuration.
Gen starts (0,seed) and Mem (0,1); every budget admits a positive seed
at most one. The generated uniform formed Mem amplitude and retained
Gen envelope force uniquely wrong test answers at actual clocks one
through budget+1. Gen forms immediately and stays positive at every
successor clock, and train answers are strictly correct from clock one.
Permanent true test answers are generated later on the same original
path. The prefix uses the full actual clock-one state, keeping both
retained buffers, variance and completed bias clock; there is no reset
to fresh seed data. Three results bring Grokking to **945** and
CircuitEfficiency to 489, without new sorry or target warnings. Neither
Mem formation, future weak margins, convergence nor later success is
an independent premise. Initial logits are zero, so strict correctness
at clock zero is excluded. Next derive least success beyond budget+1
and complete CE/confidence collapse on that same double-zero delay
family, then the legal-memory robustness obstruction. Weaker-decay
positive-confidence attraction, full signed pair balance and learned
stochastic/floating-point GPTMini transfer remain open; all frozen
full-budget ordinary-transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryFormedOutcomes` derives the least
strictly true held-out clock beyond budget+1 for the source-style
double-zero original native delay family. Zero-clock actual logit ties
exclude that clock as a success; least success is distinct from a later
permanent correctness start. The same initialized delayed path has
true train/held-out CE limits log 2 and full softmax class limits 1/2,
so held-out CE does not tend to zero and every fixed above-uniform
confidence threshold is eventually missed forever despite permanent
strict accuracy. Positive Gen seeds bounded by one admit no uniform
finite true held-out start. Five results bring Grokking to **950** and
CircuitEfficiency to 494, without new sorry or target warnings. Initial
Mem formation, future convergence, weak margins and later success are
all derived rather than independently supplied. Fixed width/task/native
constants make this a time/initial-seed family, not thermodynamic
finite-size scaling. Next derive a native nonzero-memory positive-radius
robustness obstruction on that same original family, then investigate
weaker-decay positive-confidence attraction and relative signed pair
balance. Broader detector/geometry/spectral/phase formulations and
learned stochastic/numerical GPTMini transfer remain open. All preserved
full-budget transformer evidence and unchanged optimizers are retained.


`CircuitEfficiency.SectionC_GainMemoryRobustness` derives vanishing
actual target-minus-Mem held-out margins on the original legal-beta
path from initial numerical signs. Every fixed positive radius admits
bounded full-logit counterperturbations at every clock beyond a generated
start; only the target is changed. No positive tail margin lower bound
or radius preserving strict correctness against every bounded logit
perturbation exists. On the same source-style double-zero initialized
path, every prescribed wrong-test prefix is followed by permanent
exact-real accuracy, yet each positive radius has an eventual
counterperturbation tail after success. A complementary current binary
margin law proves that coordinate errors at most one third of a positive
current margin preserve strict target correctness. Six results bring
Grokking to **956** and CircuitEfficiency to 500, without new sorry or
target warnings. Current local robustness and lack of a fixed positive
uniform tail radius coexist. The perturbation is a mathematical control,
not a training intervention or a claim that floating-point kernels
generate such an error. No future input/parameter/margin limit or success
reference is independently supplied. Next investigate the exact binary
coordinate-error radius and its generated collapse, then weaker-decay
positive-confidence attraction, complete signed pair balance and legal-
memory time/initial-seed phase formulations. Broader detector/geometry/
spectral directions and learned stochastic/numerical GPTMini transfer
remain open. All frozen full-budget transformer evidence and unchanged
optimizers are preserved.


`CircuitEfficiency.SectionC_GainMemoryExactRadius` gives the exact
current binary coordinate-error threshold: for any nonnegative radius,
every bounded full-logit perturbation preserves strict target correctness
if and only if twice the radius is strictly below the target/competitor
margin. A symmetric target-down/competitor-up control creates a tie at
equality, so the endpoint is open. The original legal-beta half-margin
observer is a numerical measurement, which may be negative before
selection. Initial full signs and positive Gen measurement generate
a tail where this observer is positive and exactly characterizes all
strict coordinate-error guarantees. Its full-clock limit is zero. The
same source-style double-zero arbitrarily delayed permanent-accuracy
path has positive exact thresholds after success that shrink to zero.
Six results bring Grokking to **962** and CircuitEfficiency to 506,
without new sorry or target warnings. No future gap, convergence,
successful reference or gradient stream is independently supplied.
The threshold concerns actual binary logits and makes no assertion
about an actual floating-point kernel's error. Next investigate an
explicit vanishing bounded seed family with legal nonzero memory and
its time/seed decision limits, then weaker-decay positive-confidence
attraction and complete signed pair balance. Broader detector/geometry/
spectral/phase formulations and learned stochastic/numerical GPTMini
transfer remain open. Frozen full-budget transformer evidence and
unchanged optimizers are preserved.


`CircuitEfficiency.SectionC_GainMemoryDelayFamily` constructs an explicit
numerical Gen partner family F/16480*(45/103)^budget, with F the generated
complete current CE/clipping box floor. Every seed is strictly positive,
below one, strictly decreases with the budget and tends to zero. The
original native Gen envelope ratio R=103/50 cancels the geometric seed
into one quarter of the actual uniform formed Mem pure-decay amplitude:
R^budget*(R*seed)=0.9^budget*(F/2000)/4. Gains 3/2 give the strict initial
comparison for every budget. Actual full clock-one state signs, formed
Mem amplitude, retained Gen growth ceiling and the needed complete
formed-state comparison are generated for each family member. Seven
results bring Grokking to **969** and CircuitEfficiency to 513, without
new sorry or target warnings. The family is computed from numerical
initial data, not chosen by a future-success predicate. The next step
connects it to actual retained wrong prefixes and permanent selection,
then proves time/initial-seed decision-limit statements with legal
nonzero memory. The geometric scales can lie below floating-point range
and do not predict the measured GPTMini transition time. No thermodynamic
size scaling or learned stochastic/numerical transfer is asserted.
Weaker-decay positive-confidence attraction, complete signed pair balance
and broader detector/geometry/spectral/phase formulations remain open.
All frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainMemoryDecisionLimits` connects the
explicit positive bounded vanishing family to original retained native
wrong prefixes and generated permanent true held-out selection. Every
member is strictly train-correct and uniquely wrong-test at clocks one
through index+1; the untrained zero-logit clock has strict decision zero.
Each fixed member later permanently has true-target decision one. At
each fixed actual clock, the vanishing-seed family decision limit is
zero; at each fixed family member, its long-time limit is one. The two
actual iterated real limits are therefore zero and one and do not
commute; there is no uniform finite success start across the family.
Eight results bring Grokking to **977** and CircuitEfficiency to 521,
without new sorry or target warnings. All prefixes and limits follow
from original actual clipped-CE feedback, legal beta1=0.9/beta2=0.98,
both retained buffers and full correction clocks; no prescribed staircase,
future parameter/margin limit or successful reference is supplied. This
advances the earlier balanced zero-beta singularity to original nonzero
memory and source-style double-zero formation. It is time/initial-seed
nonuniformity at fixed task/width, not thermodynamic system-size scaling
or measured stochastic/floating-point GPTMini convergence. Next examine
weaker-decay positive-confidence attraction and complete signed pair
balance, and connect actual internal observer limits to the empirical
pre-transition measurements. Other detector/geometry/spectral/phase
formulations remain open. All frozen full-budget ordinary-transformer
evidence and unchanged optimizers are preserved.


`CircuitEfficiency.SectionC_GainColdInstability` excludes actual all-zero
physical parameter collapse under the static cold condition
decay*epsilon < coldCE*GenGain, initial full numerical signs and one
positive Gen partner. A hypothetical zero limit generates its own
actual CE, moment and denominator tails; retained noncollapse contradicts
it. The result itself assumes no future parameter convergence. Total
nonnegative physical parameter mass cannot tend to zero either, because
that would squeeze all four actual parameters to zero. Source-style
double-zero Gen (0,seed), Mem (0,1) with every positive seed and every
finite class count meet the condition at beta1=0.9, beta2=0.98, cap=1,
rate=0.001, decay=0.1 and epsilon=1e-8. Both retained buffers and completed
clocks evolve without resets. Four results bring Grokking to **981** and
CircuitEfficiency to 525, without new sorry or target warnings. This
excludes the strong-decay cold-collapse mechanism at these smaller
native constants; it does not prove positive confidence, attraction,
delayed success, parameter convergence or learned GPTMini transfer.
Next turn noncollapse into quantitative recurrent internal measurements,
then examine positive-confidence attraction and full signed pair balance.
Broader detector/geometry/spectral/phase formulations and learned
stochastic/numerical transfer remain open. All frozen full-budget
ordinary-transformer evidence and unchanged optimizers are preserved.


`CircuitEfficiency.SectionC_GainColdReturns` translates actual small-decay
noncollapse into recurrent internal observations. A nonnegative real
measurement without a zero limit reaches a fixed positive level beyond
every prescribed clock. On the same source-style retained native path
this gives arbitrarily late total physical-mass excursions. The actual
squared Euclidean parameter norm is a separate numerical observer,
nonnegative at every state, with totalMass^2 <= 4*squaredNorm. A positive
squared-norm level is therefore reached arbitrarily late, excluding its
zero limit as well. Six results bring Grokking to **987** and
CircuitEfficiency to 531, without new sorry or target warnings. These
levels are existential, rather than measured transition estimates;
recurrent excursions allow oscillation and do not imply a positive
uniform tail floor, positive confidence or correct test selection.
Native small decay/epsilon and legal retained beta memory stay fixed.
Next derive a uniform actual partner-mixing floor from initialized
parameter/buffer bounds, then connect recurrent internal mass to actual
product logits and nonuniform-confidence observations. Positive-confidence
attraction, full signed pair balance and learned stochastic/numerical
GPTMini transfer remain open. Frozen full-budget transformer evidence
and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayMixing` derives uniform actual
partner mixing on the source-style small-decay retained path. Clipping
bounds every new input and completed second bias correction is at least
1-beta2. Retained variance at most one therefore gives complete actual
denominator at most eight. Initialized bounds generate that variance
ceiling and a finite physical box at every clock. Its full actual
CE/clipping floor supplies a fixed strictly positive current partner
coefficient below 1/40000 for every coordinate and clock. The original
native parameter law keeps nonnegative old negative-moment and pure-
decay contributions, so each new factor covers this fraction of its
preceding complete pair sum. Three results bring Grokking to **990** and
CircuitEfficiency to 534, without new sorry or target warnings. No
future variance match, denominator/feedback tail, convergence, mass
ratio or successful reference is supplied. The coefficient can be
extremely small under the loose exponential box estimate; it is not
a GPTMini transition prediction. Next connect recurrent physical mass
to actual product logits and nonuniform confidence. Positive-confidence
attraction, complete relative pair balance and learned stochastic/
floating-point GPTMini transfer remain open. Frozen transformer evidence
and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayScores` connects recurrent
internal mass to actual original product training logits. A current
pair-mixing fraction q makes gained next trainScore >= q^2*totalMass^2,
using all four factors and gains 3/2 without pair symmetry. Initialized
retained native mixing generates that positive fraction. On every
positive-Gen-seed path, a fixed positive actual training-score level is
reached arbitrarily late; actual trainScore cannot tend to zero. Train
answers are uniquely correct at every successor clock even with a zero
Gen seed, because the positive initial Mem partner persists and forms
its product. Clock zero remains tied. Five results bring Grokking to
**995** and CircuitEfficiency to 539, without new sorry or target
warnings. No future parameter/input limit, successful reference or
held-out margin is supplied. Training correctness and recurrent logits
do not identify a generalizing rule: both fixed circuits fit training.
The recurrent level may be tiny and is not a uniform tail gap or an
empirical transition-time estimate. Next derive actual full finite-class
training confidence recurrence above uniform and the corresponding CE
obstruction; stable positive-confidence attraction and true held-out
selection at small decay remain open. Full signed pair balance and
learned stochastic/floating-point GPTMini transfer also remain open.
Frozen full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_TrainProbability` derives the actual complete
finite-class train target confidence formula from the exponential sum.
It lies strictly between zero and one at every finite score, equals
1/(remaining+2) at zero and increases strictly with score. The original
full CE equals minus log of that actual probability and decreases
strictly with score. Above-uniform train probability and below-uniform
class CE each correspond exactly to positive total train score. The
full two-circuit probability equals the same total-score observation,
exposing the training-only indistinguishability of Gen/Mem allocations.
Ten results bring Grokking to **1005** and CircuitEfficiency to 549,
without new sorry or target warnings. These are pointwise full-forward
measurement laws, without a supplied trajectory, optimizer clock trend
or future reference. Exact confidence one is absent at finite scores.
Next transfer original small-decay score excursions to joint confidence/
CE observations at the same actual clocks, excluding their uniform
limits without parameter convergence. Stable confidence, true held-out
selection at small decay, complete signed pair balance and learned
stochastic/floating-point GPTMini transfer remain open. Frozen full-
budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayConfidence` measures original
train logits from both actual gained products, probability from their
complete finite-class sum and ordinary CE on the same logits. Their
identities with total-score readouts are proved before any path claim.
Double-zero initialization gives uniform probability and CE at clock
zero for every numerical Gen partner seed. Every positive seed on the
original small-decay retained path has a fixed above-uniform confidence
level and below-uniform CE level jointly reached beyond every budget,
using the same generated score-excursion clocks. Neither actual train
probability nor train CE can tend to its uniform value. At all successor
clocks train confidence is above uniform and CE below uniform-class
loss even when Gen seed is zero; Mem alone can fit training. Seven
results bring Grokking to **1012** and CircuitEfficiency to 556, without
new sorry or target warnings. No future input/weight limit, supplied
confidence level or successful reference is assumed. Recurrence does
not prove stable positive confidence, a uniform tail gap or true held-
out selection. Next formalize the original small-decay absent-Gen
control through actual full retained histories and its permanent wrong
held-out decisions, exposing training-only detector limitations. Full
signed pair balance, stable confidence and learned stochastic/numerical
GPTMini transfer remain open. Frozen full-budget transformer evidence
and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayAbsent` fixes both initial Gen
factors and their initialized buffers to zero on the same original
small-decay retained path. Their entire actual native states stay zero
with increasing completed clocks. Mem forms a positive gained product
at every successor, uniquely chooses the wrong held-out label and makes
true held-out CE strictly worse than uniform initialization. The true
held-out target is never uniquely correct, including the initial tie.
On this same path every successor has a uniquely correct train answer,
above-uniform train confidence and below-uniform train CE. Seven results
bring Grokking to **1019** and CircuitEfficiency to 563, without new sorry
or target warnings. The negative control follows actual retained
histories with no reset, future zero-gradient premise or successful
reference. It excludes training-only signals as sufficient grokking
certificates. Its absent positive Gen seed intentionally violates the
source's slow-formation initialization; positive-seed selection is not
refuted. Next investigate positive-seed small-decay relative selection
and confidence stability, and whether a component-specific internal
signal distinguishes formation from Mem-only training. Complete signed
pair balance, learned stochastic/floating-point GPTMini transfer and
thermodynamic system-size scaling remain open. Frozen full-budget
transformer evidence and optimizers are unchanged.


`AdamW.PairComparison` keeps Gen/Mem parameter masses and their
negative retained first-moment masses as two separate coordinates.
A shared current coefficient inserts gainRatio times the Mem feedback
into Gen. When the Gen denominator ceiling is gainRatio times the Mem
floor, a numerical cone pGen >= factor*pMem and
mGen >= factor*gainRatio*mMem is preserved by the native mass envelopes.
A positive efficient-pair parameter and moment mass generates a
positive cone factor against any nonnegative Mem pair. Induction
preserves the cone through a varying shared-feedback tail. Three
results bring Grokking to **1022** and AdamW to 184, without new sorry or
target warnings. The abstract envelopes and bounds are explicit
premises with joint satisfying examples, not assumed properties of a
trained transformer. Next derive them under hypothetical actual
Gen-only collapse: its inputs/buffers/denominators vanish toward the
epsilon regime, while Mem denominators have a generated near-epsilon
floor. A preserved positive comparison would force Mem to collapse
with Gen and contradict the already proved original small-decay total
noncollapse. This route avoids assuming Mem convergence. It does not
yet prove winning relative dominance or positive confidence stability.
Learned stochastic/floating-point GPTMini transfer and thermodynamic
system-size scaling remain open; frozen transformer runs are unchanged.


`CircuitEfficiency.SectionC_GainPairEnvelopes` adds Mem's actual
parameter and negative first-moment masses. Numerical signs make
both nonnegative. The exact Mem insertion uses the original shared
clipped-CE coefficient and Mem gain. Current complete Mem denominator
floors give its retained parameter ceiling; current Gen denominator
ceilings give the opposing divided parameter floor. Four results
bring Grokking to **1026** and CircuitEfficiency to 567, without new
sorry or target warnings. These current original-update laws retain
both buffers and completed clocks and have joint satisfying examples
for every external hypothesis group. Bounds have not been assumed as
future path properties. Next generate near-epsilon Mem denominator
floors and hypothetical collapsing-Gen ceilings, then connect them to
the already checked retained comparison cone. The intended boundary
obstruction does not assume Mem convergence. Positive stable confidence,
winning small-decay selection, learned stochastic/numerical GPTMini
transfer and system-size phase-transition scaling remain open. Frozen
full-budget transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayBoundary` derives positive
actual Gen parameter and negative retained-moment masses at every
successor from the positive initial partner. Generated completed
clocks give every actual denominator an eventual floor 4/5*epsilon,
without input/parameter convergence. If Gen mass collapses, its
actual applied inputs are squeezed by three times that mass and tend
to zero without a Mem limit premise. Retained native history then
generates both Gen denominator ceilings 6/5*epsilon. Their ratio to
the common floor equals the actual gain ratio 3/2. Five results bring
Grokking to **1031** and CircuitEfficiency to 572, without new sorry or
target warnings. The hypothetical-collapse results allow seed zero;
the actual absent-Gen history supplies joint satisfying examples,
instead of making contradictory positive-seed collapse a theorem
hypothesis. Next combine these generated tails with a positive current
comparison factor, forcing hypothetical Gen collapse to drag Mem with
it and contradict original total-mass noncollapse. This would strengthen
an internal formation signal without assuming Mem convergence. Winning
small-decay relative selection, stable confidence, learned stochastic/
floating-point GPTMini transfer and thermodynamic system-size scaling
remain open. Frozen transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainSmallDecayGen` closes the stronger
original weak-decay boundary obstruction: for every positive source
Gen seed and finite class count, actual Gen parameter mass cannot
tend to zero. No Mem limit, successful reference or future output
premise is used. Hypothetical Gen collapse generates its near-epsilon
denominator ceiling; generated Mem floors and actual positive Gen
parameter/moment masses choose a positive retained comparison cone.
That cone would squeeze Mem mass to zero too, contradicting the
initialized total-mass obstruction. Gen-specific mass has a fixed
positive level at arbitrarily late clocks. Actual pair mixing then
generates recurrent positive levels of the true gained Gen product
logit. Three results bring Grokking to **1034** and CircuitEfficiency
to 575, without new sorry or target warnings. Mem-only training cannot
explain these component observers as it can train accuracy/confidence/
CE or total norm. This is not a uniform positive tail floor or a proof
of winning Gen/Mem dominance or true held-out selection. Next tighten
the cold-feedback upper bound and retained contraction condition to
compare collapse/noncollapse regimes, and investigate stable small-
decay confidence and relative selection. Learned circuit discovery,
stochastic/floating-point GPTMini transfer and thermodynamic system-size
scaling remain open. Frozen full-budget transformer runs and optimizers
are unchanged.


`CircuitEfficiency.SectionC_GainColdCeiling` proves that nonnegative
physical gains/factors give nonnegative actual total train score. Its
complete finite-class CE slope is at most the uniform initial slope;
its actual native clipping multiplier is at most the zero-norm value,
including the additive 1e-6 strip. The computed cold coefficient is
therefore a global actual-feedback ceiling in this numerical region.
The true total retained first-moment insertion and each negative
applied coordinate input have sharper ceilings using cold coefficient
times Gen gain instead of the old gain-only constant. Four results
bring Grokking to **1038** and CircuitEfficiency to 579, without new
sorry or target warnings. All external hypothesis groups have joint
satisfying examples. No future input stream, cold limit or optimizer
replacement is inserted. Next derive a positive parameter/moment
weight with strict contraction whenever this coefficient is below
decay times an actual denominator floor, then generate a near-epsilon
floor from native clocks. This may close the strict collapse/noncollapse
threshold rather than leaving the old conservative sufficient-decay
gap. Critical equality, stable weak-decay confidence/relative selection,
learned stochastic/numerical GPTMini transfer and thermodynamic
system-size scaling remain open. Frozen transformer evidence and
optimizers are unchanged.


`AdamW.PairContraction` derives positive parameter and moment weights
and a common strict contraction factor from coefficient < decay*floor,
valid beta, positive floor/rate/coefficient and nonnegative remaining
decay. Its construction augments the retained beta*rate moment weight
by rate*(decay*floor-coefficient)/(2*coefficient); both normalized row
sums become strictly below one. The two current retained upper
envelopes contract the numerical weighted sum, and induction derives
tail power ceilings from the complete retained starting mass. Three
results bring Grokking to **1041** and AdamW to 187, without new sorry
or target warnings. Every external hypothesis group has a joint
satisfying example. Bounds remain explicit abstract inputs here;
no actual optimizer or moment history has been changed. Next generate
a general eventual native denominator floor arbitrarily near epsilon
and instantiate the envelopes using the actual cold CE ceiling. This
would close the strict collapse threshold opposite the initialized
noncollapse condition. Critical equality, stable weak-decay confidence/
relative selection, learned stochastic/numerical GPTMini transfer and
thermodynamic system-size scaling remain open. Frozen transformer
evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdBounds` bounds the true next
total parameter mass using any supplied positive floor for all four
complete corrected denominators. More importantly, actual growing
clocks generate a common tail for every floor strictly below epsilon,
without gradient, parameter or variance convergence. Combining this
with initialized sign invariance and the actual cold CE ceiling
generates both retained mass envelopes on the original path. Three
results bring Grokking to **1044** and CircuitEfficiency to 582,
without new sorry or target warnings. Every external hypothesis group
has a joint satisfying example, including nonzero source-style data.
Next use the generated tail and positive contraction weights to prove
actual total parameter and retained moment collapse whenever
coldGainCEGradientScale*Gen_gain < decay*epsilon. The opposite strict
initialized noncollapse condition is already proved; critical equality
and stable weak-decay relative circuit selection remain open. Learned
stochastic/numerical GPTMini transfer and thermodynamic system-size
scaling are separate unfinished routes. Frozen transformer evidence
and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdContraction` closes the strict
actual cold-collapse direction: cold CE coefficient times greater
physical gain < decay*epsilon generates a positive sub-epsilon floor,
a common completed-clock tail, positive parameter/moment weights and
a strict factor. The original retained path has a geometric weighted
ceiling after this derived start; both total physical parameter mass
and negative retained first-moment mass tend to zero. No future
gradient, variance, mass or denominator limit is a premise. Two results
bring Grokking to **1046** and CircuitEfficiency to 584, without new
sorry or target warnings. All hypothesis groups have joint nonzero
source-style examples with legal beta1=0.9/beta2=0.98; the binary
epsilon=1/decay=2 example was outside the old conservative decay>33
bound. This matches the already proved opposite strict initialized
noncollapse threshold. Next derive complete cold state/output limits
and a single strict threshold classification, keeping equality open.
Stable weak-decay relative circuit selection, learned stochastic/
numerical GPTMini transfer and thermodynamic system-size scaling remain
unfinished. Frozen transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdCollapse` derives every physical
parameter and original applied CE input tending to zero above the
strict cold threshold. It also joins both strict directions into a
single exact classification: with positive initialized Gen partner,
greater Gen gain and legal retained native data, away from explicit
critical equality, total physical mass tends to zero if and only if
coldGainCEGradientScale*Gen_gain < decay*epsilon. Both below- and
above-threshold source-style examples jointly satisfy the classification
hypotheses. Three results bring Grokking to **1049** and CircuitEfficiency
to 587, without new sorry or target warnings. The positive Gen condition
excludes the actual absent-Gen negative control; noncollapse alone is
not a stable positive limit or held-out success. Next derive the full
retained buffer/denominator, circuit score and original CE/confidence
limits under this sharp strict condition. Critical equality, stable
weak-decay relative selection, learned stochastic/numerical GPTMini
transfer and thermodynamic system-size scaling remain open. Frozen
transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdNativeLimits` transfers the generated
strict-threshold parameter/input collapse through the original causal
native histories. Both retained buffers tend to zero, every complete
next-input denominator tends to epsilon, and every completed clock
diverges. The actual shared CE/clipping coefficient approaches its
strictly positive attained cold value: the vanishing coordinate inputs
come from vanishing partner parameters. No input, buffer, variance or
denominator convergence is supplied separately. This ports the earlier
conservative native limits at 1880c50 to the sharp strict condition;
the source-style beta1=0.9/beta2=0.98 binary example uses decay=2 and
epsilon=1. Three results bring Grokking to **1052** and CircuitEfficiency
to 590, without new sorry or target warnings. All external hypothesis
groups have joint nonzero satisfying examples. Next derive original
output/CE/confidence limits at the same condition, then examine what
critical equality permits without presuming attraction. Stable weak-
decay relative selection, learned stochastic/numerical GPTMini transfer
and thermodynamic system-size scaling remain open. Frozen transformer
evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdLoss` ports the actual retained
output limits at 2c2a85b to the sharp strict cold threshold. Both physical
circuit scores and the Gen-minus-Mem margin tend to zero; original train
and held-out CE approach log(class count); every full held-out softmax
class probability approaches its uniform value. All actual classes enter
the exponential sum, and no successful decision, future confidence or
loss is an independent premise. Three results bring Grokking to **1055**
and CircuitEfficiency to 593, without new sorry or target warnings.
Joint nonzero source-style examples use legal retained beta memory and
the decay=2/epsilon=1 binary configuration. Absolute confidence limits
do not classify exact finite-clock accuracy or relative Gen/Mem decay.
Next examine balanced finite limits at or above the cold threshold,
including equality, before attempting actual critical convergence.
Stable weak-decay relative selection, learned stochastic/numerical
GPTMini transfer and thermodynamic system-size scaling remain open.
Frozen transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdCriticalLimits` proves that a
nonnegative normalized native pair at or above its coefficient threshold
has only the zero balanced parameter point. Applied to the actual
clipped CE Gen/Mem coefficients, the cold greater-gain ceiling excludes
every nonzero nonnegative balanced physical point at or above the
threshold, including critical equality. Consequently any supplied
finite physical parameter limit on the original legal initialized path
there must be zero: input and normalized balance follow from the actual
feedback and retained histories. Three results bring Grokking to **1058**
and CircuitEfficiency to 596, without new sorry or target warnings.
All external hypothesis groups have joint satisfying examples; the
exact critical zero native path realizes the conditional convergence
premise. The point example permits nonzero retained buffers because
normalized parameter balance is not a fixed whole-optimizer-state
claim. Actual positive-seed critical convergence remains open, as do
stable weak-decay relative selection, learned stochastic/numerical
GPTMini transfer and thermodynamic system-size scaling. Next examine
the true critical mass dissipation, beginning with a legal zero-first-
beta control before transferring any argument to retained beta1=0.9.
Frozen transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdMassMonotone` begins the actual
critical-convergence route with explicit legal first beta zero while
retaining the second buffer, its beta and completed clocks. At or above
the actual cold threshold one true native CE step cannot increase total
physical mass. Initialized signs generate an antitone nonnegative mass
sequence, a uniform physical coordinate box from its initial total
mass, and a finite nonnegative mass limit by order completeness. None
of those future bounds or limits is an input. Four results bring
Grokking to **1062** and CircuitEfficiency to 600, without new sorry or
target warnings. Every hypothesis group has a jointly satisfying
nonzero source-style example at exact binary critical decay=1.5,
epsilon=1, beta2=0.98 and rate=0.001. Nonnegative rate remains allowed
in these observer laws, so they do not yet force a zero limit. Next use
positive rate, the derived physical box and true CE floor: each nonzero
input inserts a squared-gradient contribution into retained variance,
making its actual denominator strictly exceed epsilon. A uniform input
floor should then force a mass decrement and exclude a positive limit
without a partner-mixing or variance-ceiling premise. This control does not
replace or reset beta1=0.9/beta2=0.98 in the frozen transformer runs;
nonzero-first-beta critical convergence still needs its own weighted
dissipation argument. Stable weak-decay relative selection, learned
stochastic/numerical GPTMini transfer and thermodynamic system-size
scaling remain open. Frozen transformer evidence and optimizers are
unchanged.


`AdamW.ScalarInputDissipation` derives a strict native denominator
floor from the inserted input square with legal first beta zero and
arbitrary retained nonnegative variance, completed clock and beta2<1.
A positive negative-input floor generates an explicit positive loss
from the linear epsilon adaptive ceiling; positive rate makes that
current scalar ceiling strict. Four results bring Grokking to **1066**
and AdamW to 191, without new sorry or target warnings. All hypothesis
groups have joint satisfying examples with retained beta2=0.98. The
application must still derive the input floor from actual CE feedback;
these scalar results do not presume future gradients or prove a
transformer grokking event. Next combine this gap with true cold mass
envelopes and the generated physical box to exclude a positive mass
limit at critical equality. Transfer to retained nonzero first beta,
stable weak-decay relative selection, learned stochastic/numerical
GPTMini transfer and thermodynamic system-size scaling remain open.
Frozen transformer evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdMassDissipation` sums the true
native scalar bounds: at or above the cold threshold, a positive
floor for one actual negative CE input forces total physical mass
to drop by rate times the explicit adaptive gap. A current physical
box and positive lower mass generate such a positive floor for at
least one coordinate from the true CE multiplier and partner gains.
Two results bring Grokking to **1068** and CircuitEfficiency to 602,
without new sorry or target warnings. Joint satisfying examples use
nonzero source-style factors and retained beta2=0.98; the current
input-floor example is computed from its actual CE callback. No
gradient stream or future parameter limit is prescribed. Next combine
this generated decrement with the actual initialized antitone mass
limit to prove zero-first-beta convergence at critical equality.
Nonzero-first-beta critical convergence, stable weak-decay relative
selection, learned stochastic/numerical GPTMini transfer and
thermodynamic system-size scaling remain open. Frozen transformer
evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdCriticalCollapse` closes actual
critical convergence for the legal first-beta-zero control with its
second beta, variance and completed clocks retained. At or above the
actual cold threshold, initialized nonnegative signs, positive rate
and nonnegative remaining decay force total physical mass, each
physical coordinate and every true clipped CE input to tend to zero.
A positive generated mass limit would preserve an actual input floor
and fixed positive mass decrement, contradicting that same convergence.
No future parameter/input limit, variance ceiling, partner mixing or
successful classifier is an independent premise. Three results bring
Grokking to **1071** and CircuitEfficiency to 605, without new sorry or
target warnings. Joint nonzero source-style examples include exact
binary critical decay=1.5/epsilon=1/beta2=0.98/rate=0.001. This control
does not alter frozen beta1=0.9 experiments or prove their critical
convergence. Next derive complete native/output limits for this
critical branch and transfer the strict adaptive decrement to a
weighted parameter/moment observer with summable clock-correction
errors at nonzero first beta. Stable weak-decay relative selection,
learned stochastic/numerical GPTMini transfer and thermodynamic
system-size scaling remain open. Frozen evidence and optimizers are
unchanged.


`AdamW.RetainedInputDissipation` transfers the native inserted-input
denominator floor and strict adaptive gap to arbitrary legal first
and second betas. Every strict subfloor eventually holds uniformly
over all nonnegative current variances and inputs meeting the fixed
input floor. A nonpositive retained first moment cannot cancel the
new negative input; its true next magnitude covers (1-beta1) times
that floor. The resulting adaptive gap keeps the complete numerator,
variance and completed clock. Three results bring Grokking to **1074**
and AdamW to 194, without new sorry or target warnings. Joint examples
include beta1=0.9/beta2=0.98 and nonzero retained moments at a positive
clock. The uniform tail is conditional on the current input magnitude,
not an independently supplied future gradient trajectory. Critical
native convergence at nonzero first beta still requires deriving a
weighted mass decrement and absorbing other coordinates' geometric
clock errors. Next formalize that correction and apply it to the
actual initialized feedback. Stable weak-decay relative selection,
learned stochastic/numerical GPTMini transfer and thermodynamic
system-size scaling remain open. Frozen evidence and optimizers are
unchanged.


`AdamW.GeometricError` formalizes the numerical correction needed for
nonzero first-beta clock errors. An observer with next-step increase
bounded by budget*beta^n becomes antitone after adding its explicit
remaining geometric budget. Legal beta and nonnegative budget generate
a finite initialized observer ceiling; a nonnegative observer then has
a finite nonnegative limit by order completeness and vanishing
correction. A closed positive-constant counterexample proves that this
law alone cannot force a zero limit. Five results bring Grokking to
**1079** and AdamW to 199, without new sorry or target warnings; every
external hypothesis group has a joint satisfying example. The actual
CE path must still generate the one-step error budget and a strict
input-dependent decrement. Next derive its clock error from the
complete epsilon floor and newly retained moment bounds, then use
the weighted physical parameter/moment observer. Nonzero-first-beta
critical convergence, stable weak-decay relative selection, learned
stochastic/numerical GPTMini transfer and thermodynamic system-size
scaling remain open. Frozen evidence and optimizers are unchanged.


`AdamW.ScalarBiasError` derives the geometric clock error from the
complete actual scalar native denominator. The corrected epsilon
reciprocal exceeds its baseline by at most the completed first-beta
power times the newly retained magnitude over the first-clock epsilon
floor. A current generated moment ceiling supplies the fixed error
budget, indexed by the actual retained old clock. Multiplying by
(1-beta1)*epsilon and adding rate*beta1 times the new negative moment
cancels that epsilon baseline in the weighted parameter/moment law,
leaving rate*bound*beta1^clock. Four results bring Grokking to **1083**
and AdamW to 203, without new sorry or target warnings. Joint examples
use beta1=0.9/beta2=0.98 and nonzero retained buffers at a positive
clock. The squared-gradient denominator is bounded, not replaced;
no future gradient or buffer limit is supplied. Next sum the true
four-coordinate CE laws, derive initialized moment bounds/clocks,
and apply the geometric correction to obtain an actual weighted
observer limit before forcing physical mass collapse by input-induced
dissipation. Nonzero-first-beta critical convergence, stable weak-
decay relative selection, learned stochastic/numerical GPTMini and
thermodynamic system-size transfer remain open. Frozen evidence and
optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdWeightedLimit` derives an actual
retained weighted observer limit at or above the cold threshold for
arbitrary legal first/second betas, including beta1=0.9/beta2=0.98.
The numerical observer is (1-beta1)*epsilon times total physical mass
plus rate*beta1 times total negative first-moment mass. The true
four-coordinate CE update admits a 4*rate*clip_bound*beta1^clock error
budget. Standard nonnegative seeds generate the required moment
ceilings, signs and common clocks; the geometric correction then gives
a finite nonnegative observer limit. No future physical parameter,
input, variance or denominator limit is supplied. Three results bring
Grokking to **1086** and CircuitEfficiency to 608, without new sorry or
target warnings. Joint nonzero source-style examples include exact
binary critical decay=1.5/epsilon=1 with both original betas and
rate=0.001. Nonnegative rate remains allowed here, so zero observer
limit or physical collapse is not claimed. Next use positive rate,
the generated physical box, actual CE input floor and late corrected
denominator dissipation to exclude recurrent positive physical mass
from this actual convergent weighted path. Original nonzero-first-
beta critical physical collapse, stable weak-decay relative selection,
learned stochastic/numerical GPTMini and thermodynamic system-size
transfer remain open. Frozen evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdWeightedDissipation` retains both
native moments while summing a strict input-induced decrement over
the true four-coordinate CE step. A current large actual input and
its denominator above epsilon give a positive gap in the weighted
mass law: the original 4*rate*clip_bound*beta1^clock error remains,
and (1-beta1)*epsilon*rate times the adaptive gap is subtracted.
Other coordinates need no input floors. One result brings Grokking
to **1087** and CircuitEfficiency to 609, without new sorry or target
warnings. The joint current-hypothesis example uses its actual CE
callback; no future input stream or parameter limit is prescribed.
Next derive these current premises on initialized legal nonzero-beta
paths whenever physical mass exceeds an arbitrary positive level.
Convergence of the generated weighted observer and vanishing clock
error should then exclude such late levels and prove critical
physical collapse. Stable weak-decay relative selection, learned
stochastic/numerical GPTMini and thermodynamic system-size transfer
remain open. Frozen evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdSeededCollapse` closes actual
initialized critical physical collapse for arbitrary legal retained
first/second betas, including beta1=0.9/beta2=0.98. At or above the
greater-gain cold threshold, standard nonnegative zero-buffer seeds
and positive rate give total physical mass tending to zero, and
every physical coordinate tends to zero. Clipping and positive decay
generate the box and moment bounds. Whenever mass exceeds a fixed
positive level, the true CE box floor generates an actual input;
late completed clocks force its denominator above epsilon, yielding
a fixed weighted decrement. Weighted convergence and vanishing
clock errors exclude recurrent positive levels without assuming
physical monotonicity or future parameter/input/buffer limits.
Two results bring Grokking to **1089** and CircuitEfficiency to 611,
without new sorry or target warnings. Joint source-style examples
use exact binary critical decay=1.5/epsilon=1, original retained
betas, rate=0.001 and Gen=(0,0.005)/Mem=(0,1). Remaining decay may
be zero. Next propagate actual critical collapse through native
inputs, moments, denominators, scores and losses, and close the
initialized threshold classification including equality. Relative
Gen/Mem selection at weak decay, learned stochastic/numerical
GPTMini and thermodynamic system-size transfer remain open. The
six frozen measurements, runs, optimizers and checkpoints are unchanged.


`CircuitEfficiency.SectionC_GainColdSeededNativeLimits` propagates
generated initialized critical collapse through the original native
optimizer. Every actual clipped CE input and both retained moments
tend to zero; the completed-clock denominator tends to epsilon,
completed clocks diverge, and the actual corrected adaptive direction
tends to zero. All limits include critical equality with arbitrary
legal retained betas and require no future input/buffer/denominator
premise. The actual physical mass cap supplies input convergence;
the scalar native port retains the complete causal histories.
Four results bring Grokking to **1093** and CircuitEfficiency to 615,
without new sorry or target warnings. Each joint nonzero source-style
example uses original beta1=0.9/beta2=0.98, exact binary critical
decay=1.5/epsilon=1, rate=0.001 and Gen=(0,0.005)/Mem=(0,1).
Next close original CE coefficient, score/confidence limits and the
initialized threshold classification including equality. Stable
weak-decay relative selection, learned stochastic/numerical GPTMini
and thermodynamic system-size transfer remain open. Frozen evidence,
optimizers and checkpoints are unchanged.


`CircuitEfficiency.SectionC_GainColdSeededThreshold` closes the exact
initialized absolute-collapse boundary, including critical equality.
With standard nonnegative zero-buffer seeds, a positive Gen partner,
greater Gen gain and legal retained hyperparameters with positive
decay/rate/epsilon/remaining decay, total mass tends to zero iff cold
CE coefficient times Gen gain is at most decay times epsilon. The
same iff holds for zero limits of all four physical coordinates.
No future convergence or held-out success is a premise. A binary
original-beta epsilon=1 control has the exact boundary decay=1.5.
If rate times the cold Gen coefficient is at least epsilon, positive
remaining decay excludes absolute collapse for every legal decay:
the required threshold would force rate*decay at least one. The joint
small-epsilon example uses beta1=0.9/beta2=0.98, epsilon=1e-8,
rate=0.001, decay=0.1 and the same nonzero source-style physical seeds.
Four results bring Grokking to **1097** and CircuitEfficiency to 619,
without new sorry or target warnings. This classification concerns
absolute physical mass, not stable relative Gen/Mem selection or a
thermodynamic size limit. Next finish actual critical CE coefficient,
score/confidence limits, then return to weak-decay relative selection
and learned-head mechanisms. Frozen evidence and optimizers are unchanged.


`CircuitEfficiency.SectionC_GainColdSeededReadouts` determines the
actual retained observer limit as zero at or above the threshold,
including equality. Generated native buffer limits give zero total
negative first-moment mass; this and physical mass collapse force
the same weighted observer used by the critical proof to vanish.
The true shared clipped CE coefficient approaches its strictly
positive attained cold value, while both physical circuit scores
and their held-out margin vanish. Every input vanishes through its
partner parameter, not through a zero shared CE coefficient.
Four results bring Grokking to **1101** and CircuitEfficiency to 623,
without new sorry or target warnings. Joint source-style examples
use original retained betas, exact binary critical decay=1.5,
epsilon=1, rate=0.001 and Gen=(0,0.005)/Mem=(0,1). No future
observer/coefficient/score/parameter limit is a premise. Next derive
complete multiclass confidence and original train/held-out CE limits
at equality; relative ordering and exact decisions must stay distinct.
Weak-decay stable rule selection, learned stochastic/numerical GPTMini
and thermodynamic system-size transfer remain open. Frozen evidence,
optimizers and checkpoints are unchanged.


`CircuitEfficiency.SectionC_GainColdSeededConfidence` closes complete
initialized confidence and original CE limits at or above the cold
threshold, including exact critical equality with retained original
betas. Actual train and held-out CE approach log(remaining+2), every
full held-out softmax probability approaches 1/(remaining+2), and
the true-versus-Mem probability gap tends to zero. All finite classes
remain in the actual exponential sum; no future parameter/score/loss/
probability limit or successful decision is assumed. Three results
bring Grokking to **1104** and CircuitEfficiency to 626, without new
sorry or target warnings. Joint nonzero source-style examples use
the same original betas, exact binary critical decay=1.5/epsilon=1,
rate=0.001 and Gen=(0,0.005)/Mem=(0,1). Absolute uniform confidence
does not decide finite-clock argmax history. The initialized absolute
cold classification now includes equality and is propagated through
parameters, inputs, retained buffers, directions, CE coefficient,
scores and full probabilities. Next return to relative Gen/Mem
selection at weak decay and genuine learned-head formation; the
earlier delayed-generalization, metric, geometric, spectral and phase
routes remain part of the active full transformer objective. Stable
weak-decay attraction, learned stochastic/numerical GPTMini and a
thermodynamic system-size limit remain open. Six frozen measurements,
full-budget runs, optimizers and checkpoints are unchanged.


`AdamW.MomentSquare` derives native first/second-buffer coupling from
same-input insertions, without sign, clipping or convergence premises.
When beta1 squared is below beta2, the squared first moment is bounded
by an explicit positive scale times retained variance. A nonnegative
square remainder proves preservation through each actual insertion
and through every actual scalar path satisfying the initial guard.
The original beta1=0.9/beta2=0.98 scale is exactly 49/17. A second
insertion law propagates a positive present numerical square budget;
it prepares the finite-clock correction rather than prescribing a
future history bound. An actual original-beta first insertion satisfies
the guard, while resetting only variance to zero violates it. This is
a proved numerical counterfactual; preserved optimizer states are not
reset. Six results bring Grokking to **1110** and native AdamW to 209,
with CircuitEfficiency unchanged at 626 and no new sorry or target
warnings. Next derive initialized finite-history budgets and corrected
direction bounds independent of epsilon and gradient magnitude, then
use them for weak-decay relative selection and learned-coordinate
transfer. No initialized corrected direction ceiling, attraction or
task generalization follows from the current buffer guard alone. The
full delayed-generalization, confidence, geometry, spectral and phase
routes stay active; all six frozen measurements and runs are unchanged.


`AdamW.SquareHistory` generates the sharper finite-clock square budget
from zero native buffers and arbitrary shared inputs. Its exact first
value and successor recurrence match actual moment insertions. The
initialized history and actual scalar-path estimates assume neither
future buffer inequalities nor bounded/convergent gradients. The two
native bias-correction factors satisfy the needed square inequality
at every finite clock, by the product identity and nonnegative
squared difference of their powers. No correction is approximated by
one and no buffer or clock is reset. Six results bring Grokking to
**1116** and native AdamW to 215, with CircuitEfficiency still 626
and no new sorry or target warnings. Next combine those actual
history and correction estimates into a uniform corrected direction
bound and an epsilon-independent physical parameter box. Weak-decay
attraction and genuine learned-head generalization remain open;
the broader operational, confidence, geometric, spectral and phase
formulations stay active. Frozen experiments and optimizer states
are unchanged.


`AdamW.SquareDirection` combines generated zero-buffer histories and
both exact finite-clock bias corrections into a uniform corrected
moment-square bound. It derives the true normalized direction ceiling
sqrt(K) for arbitrary changing gradients, without clipping, sign,
gradient magnitude or convergence premises. Positive epsilon stays
in the actual denominator but its magnitude does not enter the
ceiling. For the original beta1=0.9/beta2=0.98, sqrt(K)=sqrt(49/17)
is strictly below the rational 17/10. Actual scalar paths inherit
both the general and original-beta bounds from their update law
and initial zero moments/clock, not future optimizer inequalities.
Six results bring Grokking to **1122** and native AdamW to 221,
with CircuitEfficiency still 626 and no new sorry or target warnings.
Next obtain an epsilon-independent physical parameter box under
positive decay and nonnegative remaining decay, and apply it to the
actual closed CE feedback. These numerical bounds also admit real
learned-coordinate gradients when the exact recurrence bridge is
supplied; that transformer/rounding bridge and generalization are
not supplied here. Weak-decay Gen/Mem attraction remains open.
All broader operational, confidence, geometric, spectral and phase
routes stay active; frozen runs and optimizer states are unchanged.


`AdamW.SquareParameterBox` derives a physical absolute parameter box
from actual zero-buffer/zero-clock native paths, positive decay and
nonnegative remaining decay. The computed bound is
max(abs(initial),sqrt(K)/decay), independent of epsilon magnitude,
clipping and all changing input magnitudes. No future parameter or
direction bound is supplied. The original retained betas give the
rational max(abs(initial),(17/10)/decay) box. At rate=0.001 and
decay=0.1, every initial coordinate of absolute size at most 17
stays within 17. The coordinate-indexed theorem allows arbitrary
globally coupled learned inputs, subject to the explicit exact-real
native recurrence bridge; it does not supply the real GPTMini or
floating-point bridge. Five results bring Grokking to **1127** and
native AdamW to 226, with CircuitEfficiency unchanged at 626 and
no new sorry or target warnings. Next transfer the sharper box to
actual gained full-CE feedback and its positive feedback floors.
Stable weak-decay Gen/Mem attraction and transformer generalization
remain open; the full operational, confidence, geometric, spectral
and phase formulations stay active. Frozen runs, checkpoints and
optimizer states are unchanged.


`CircuitEfficiency.SectionC_GainSquareBounds` transfers the generated
moment-square physical box to the closed gained full-CE native
trajectory. Actual CE partials and shared clipping generate every
input; no independent stream or future state/input/buffer bound is
supplied. At the original retained beta1=0.9/beta2=0.98, rate=0.001
and decay=0.1, initial absolute physical coordinates at most 17 stay
within 17 for every positive epsilon. From nonnegative seeds and
positive ordered gains it derives the explicit positive complete
CE floor at that ceiling, the unit scale ceiling and every actual
partner-input floor at every clock. All finite classes, both gained
product scores and the actual added 1e-6 clipping denominator stay
in those expressions. Four results bring Grokking to **1131** and
CircuitEfficiency to 630, with native AdamW unchanged at 226 and
no new sorry or target warnings. The floor is still qualitative and
can be extremely small; it supplies neither a transition time nor
Gen/Mem attraction. Next investigate the original-beta weak-decay
relative dynamics without a future convergence premise, while
removing the constant-rate restriction for the actual ten-update
warmup and keeping the learned attention/FFN, rounding and stochastic
bridges explicit. All other delayed-generalization, confidence,
representation geometry, spectral and phase formulations remain
active. No frozen run, measurement or optimizer state changed.


`AdamW.ScheduledHistory` removes the constant-rate restriction from
native moment/clock histories, the generated finite-history square
budget and the epsilon-independent direction bound. Both betas,
epsilon and decay remain fixed; each actual update uses its current
supplied rate, and every input may change through the learned model.
Original-beta directions remain below 17/10 even during a zero-rate
update. The actual completed-update warmup starts at rate zero,
but a nonzero input still generates nonzero first/second buffers
and clock one. A checked second numerical insertion uses those
retained buffers and clock two at rate=0.0001. No buffer, clock or
bias correction is restarted. Six results bring Grokking to **1137**
and native AdamW to 232, with CircuitEfficiency still 630 and no
new sorry or target warnings. Next extend the physical parameter
box to nonnegative changing rates with nonnegative remaining decay,
and verify the actual ten-update schedule in exact reals. Warmup
float evaluation and actual learned GPTMini/numerical bridges remain
explicit obligations. The broad operational, confidence, geometry,
Gen/Mem, spectral and phase routes remain active; frozen experiments,
measurements and optimizer states are unchanged.
