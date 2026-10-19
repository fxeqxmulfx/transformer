module

public import Transformer.ScheduleFree.Section2_SGD
public import Transformer.ScheduleFree.SectionB_Momentum
public import Transformer.ScheduleFree.SectionC_ExpectationTransfer
public import Transformer.ScheduleFree.SectionD_Acceleration
public import Transformer.ScheduleFree.SectionD_Optimistic
public import Transformer.ScheduleFree.SectionE_StrongConvex
public import Transformer.ScheduleFree.SectionF_LargeSteps
public import Transformer.ScheduleFree.Section4_AdamW
public import Transformer.ScheduleFree.AMSGrad

/-!
# The Road Less Scheduled

Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky,
arXiv:2405.15682v4. Sources and PDF are downloaded locally into
`papers/arXiv-2405.15682/`; `main.tex` includes `gen_proof.tex`,
`recover.tex`, `bregman.tex`, `acceleration.tex`, and `strongconvex.tex`.

## Mathematical coverage

* §2, Theorem 1: the sharp stochastic Schedule-Free SGD bound `DG/√T`,
  uniformly over interpolation choices, including both endpoints and
  the zero-distance/zero-gradient cases. The actual base updates are
  used in the regret proof. The algorithm's time horizon is used only
  in this optimal constant learning-rate choice, as in the source.
* §2.1, Theorem 2 / Appendix A: weighted online-to-batch conversion,
  first pathwise and then in expectation. Supporting subgradients cover
  nonsmooth convex losses; `convex_of_subgradients` verifies that this
  interface enforces ordinary convexity. Oracle conditions are scalar
  unbiased pairings, with integrability stated explicitly.
* §2.2, Theorem 3 / Appendix F: monitored large-step regret, with the
  hidden constant made explicit as `4`. The maximum-gradient norm is
  bounded by the root sum of squares, as required by the source proof.
* Appendix B, Theorem 4: the weighted momentum reformulation, its
  constant-momentum specialization, and exact recovery of weighted
  linear decay, including equality of the final query and average.
* Appendix C, Theorem 5: the exact Bregman identity, pathwise and in
  expectation. The division-free interpolation remainder agrees with
  the displayed rational expression and covers its singular endpoint.
* Appendix D, Corollary 1: the retained negative gradient-variation
  term, pathwise and in expectation. Its proof includes the smooth
  Bregman lower bound and telescoping endpoint mismatches.
* Appendix D, Corollary 2: the exact accelerated rate with constants
  `14` and `2`, under the corrected interpolation condition and an
  expected optimistic-regret bound. `optimistic_expectation` derives
  that oracle bound from pathwise regret and variance bounds by Jensen.
* Appendix D.1, Lemma 1: the full projected clipped optimistic algorithm
  and its corrected bounded-comparator regret bound are proved, including
  zero radius and zero gradient prefixes. The source constant `7` follows
  from a stronger intermediate scale bound with constant `6`. The
  unrestricted printed claim has a proved counterexample.
* Appendix E, Theorem 6: the strong convexity conversion and the actual
  projected stochastic method's `2(G+2μD)²/(μ(T+1))` rate. Projection,
  ball invariance of the average/query, surrogate gradients, and the
  weighted strongly convex regret are all proved.
* §4.4, Algorithm 1: Schedule-Free AdamW, including both moments' roles,
  warmup, bias correction, decay at the query, squared-step averaging,
  and oracle evaluation at the correct location. Second moments and
  cumulative weights remain nonnegative; positive epsilon removes
  denominator singularities. No AdamW convergence result is inferred
  from its empirical evaluation.

## Source corrections and scope

Theorem 2 permits weights depending on earlier samples, but pulls a
random total weight outside expectation. The corresponding expected
ratio equality in Theorem 5 is false without further assumptions.
`random_weights_counterexample` uses independent uniform signs, a zero
population loss and predictable weights in `[0,1]`: expected normalized
regret is `-1/4`, while the loss gap and all population Bregman terms
are zero. Stochastic results here therefore use deterministic weights.
The first weight is positive, making every prefix average well-defined.
Later weights may be zero except in the momentum formulas that divide
by them; those formulas have explicit nonzero-weight hypotheses.

The conditional-expectation notation in Appendix A is replaced by the
unbiased pairing actually required by its proof. Independence and a
fresh-sample stochastic subgradient oracle are not separately formalized;
their moment and gradient-norm consequences are the stated interface.

Appendix C divides by `α` while allowing `α=0`. The division-free
`mixGap` extends this endpoint by zero and is proved to agree with the
source away from it. Appendix D first prints the incorrect smoothness
coefficient `2L`; its calculation uses `1/(2L)`, which is the proved one.
For weights `w_t=t`, Corollary 1 requires `α_t≤1/(5(t+1))`.
Corollary 2 instead prints `1/(5(t-1))`. The accelerated theorem here
uses Corollary 1's sufficient condition and documents that correction.

Lemma 1 omits `‖x_star‖≤D`, although its proof uses it. Already at one
step, `D=1`, `g_1=1`, `z_1=0`, `x_star=-11` gives regret `11>7√2`.
The corrected lemma adds that radius condition and follows the displayed
algorithm's step size; the earlier conflicting eta formula is not used.
Theorem 6 also needs its initial base point in the projection ball.

The exact accelerated and strongly convex finite-horizon bounds are
the mathematical claims formalized here. The asymptotic rates in the
paper abbreviate these bounds under uniform noise bounds. The optimal
SGD schedule, arbitrary fixed-step large-rate statement, and practical
AdamW procedure retain their distinct hypotheses.

The comparative performance, Pareto frontiers, gradient collapse,
BatchNorm advice, sensitivity plots, and timing/accuracy claims in
§§3--7 and the experimental appendices are empirical observations.
They are not theorems about all training runs. The proposed explanation
in terms of asymptotically quadratic losses specifies no quantitative
conjecture, so no arbitrary formal model is introduced for it.
-/

@[expose] public section
