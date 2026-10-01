/-
# Magnitude--Direction Decoupling

Hägele, Hernández-Cano, Kosson, Jaggi, arXiv:2606.25971v2,
"Improving Neural Network Training by Decoupling the Magnitude and
Direction of Weight Vectors". The downloaded TeX source is
`papers/arXiv-2606.25971v2/main.tex`.

## Mathematical coverage

* §2: exact radial/tangential norm accounting, the strict growth of a
  nonzero perpendicular update, the necessary negative radial signal,
  inverse dependence of relative update on magnitude, and the vanishing
  radial derivative of a differentiable scale-invariant loss.
* §3 and §3.1: the diagonal-matrix factorization, exact fused/unfused
  storage for nonzero gains, scalar specialization, representational
  scope, and the actual loss chain rule for simultaneous direction and
  row/column perturbations. The raw-softplus chain rule verifies all
  three gradient formulas of Algorithm 2.
* Algorithms 1–2: the complete scalar and row/column wrappers with
  separate learning rates and optimizer callbacks, including fused
  reconstruction and the Frobenius-sphere invariant for any callback
  producing a nonzero direction candidate. A callback is the current
  step of the base optimizer, including its externally managed state;
  no Adam/Muon convergence property is assumed by the wrapper.
* User-requested training extension of Appendix A: actual fixed-loss
  training with stateful direction and gain optimizers, a check of the
  complete proposed fused displacement, and a nonsingular true-gradient
  fallback. The corrected weights equal the stored fused weights at
  every step; positive softplus gains and the direction sphere are
  preserved. For a smooth lower-bounded loss the full gradient vanishes
  and loss converges. Strong convexity gives weight convergence to the
  global minimum and a geometric objective-gap bound.
* Concrete AMSGradMD uses all three actual AMSGrad moment histories for
  the direction and both raw-gain vectors. The callback agrees with the
  existing AMSGrad update, and every running maximum stays monotone even
  after rejection. The same corrected training convergence is proved.
* §3.1 and §4.1.5: exact pre-projection relative-update control for
  norm-matched updates; for tangential updates the retracted cosine is
  `1/sqrt(1+eta²)` and the squared relative chord is
  `2-2/sqrt(1+eta²)`, independent of magnitude and dimension.
* §4.1.1 and Appendix D: exact-polar Muon shape scaling, the required
  positive matrix dimensions, initialization second moments, the
  `min(m,n)=d` regime, and the explicit GQA exception.
* §4.1.3: positivity, differentiability, derivative bounds, and the
  inverse/initialization of softplus; positivity of exponential gains.
* Appendix C: arithmetic of the printed downstream scores, including
  exact unrounded averages and the four winning tasks.
* Appendix G: equality of fused and explicit linear-layer outputs;
  parameter/token operation-count algebra and the quadratic/cubic
  comparison. These are not measured-throughput guarantees.
* Appendix H: the row/column outer product's rank bound, the all-ones
  offset, initialization at `B=0`, representation of every row/column
  gain at every factor width at least two, and strict expressivity.

## Corrections and exact scope

The source's unconditional "always increases" needs a nonzero update.
Projection onto a positive sphere requires a nonzero candidate, and
recovering a direction requires nonzero gains. The zero-candidate and
zero-direct-gain counterexamples make the singular cases explicit.
Softplus prevents zero gains, but it does not prevent zero candidates.

Appendix A's instruction to tie both gains to one scalar produces
`gamma² D`, rather than Algorithm 1's `gamma D`. Fixing the other side
to one recovers the scalar algorithm; the literal prescription is
refuted at `gamma=2`.

Normalized additive update size is exactly controlled before projection.
An arbitrary radial normalized update can cause no directional change;
this remains true for arbitrarily small relative updates, so an angular
approximation additionally needs control of the radial component.
even a tangential update has the nonlinear chord formula above. Thus
the source's exact-LR wording is not an identity between LR and the
post-projection displacement. Learning-rate transfer of an optimum,
training stability without warmup/decay, and convergence do not follow
from the sphere invariant alone.

Positive gains on a positive sphere represent every nonzero matrix but
exclude zero. The factorization does not uniquely identify the two gain
vectors; fine-grained gains can also change the fused weight's direction.
Conversely, with anisotropic gains fixed, moving the direction on its
sphere can change the fused magnitude. Only a scalar gain determines
that magnitude independently of direction.
The paper appropriately says the coordinates only "resemble" polar ones.

The training convergence theorems are explicit additions to the paper,
for a corrected algorithm. A proposal is accepted after its full fused
displacement passes gradient-alignment, length and nonzero checks.
Rejected proposals use a true-gradient step of size `sigma/L`, halved
if it would hit zero. Optimizer memories are updated in both cases.
A common positive row-gain rescaling restores the direction sphere,
preserving the fused weight, row-gain ratios and column gains. A valid
accepted proposal is preserved including its raw gain parameters.

The loss has a differentiable quadratic smooth upper model with `L>0`
and a finite lower bound; `0<sigma<=1/2`. The minimum theorem additionally
requires a positive strong-convexity constant, a stationary minimizer
and `sigma²*strong<=L`. These are conditions on the fixed loss, not
assumptions of convergence or alignment along training. Minibatch noise,
floating-point rounding and convergence of redundant gains are not covered.
Finite weights stay nonzero, but a zero limiting minimum is allowed.

Original MD need not inherit convergence: for a one-entry matrix on
`(w+1)²/2`, normalized direction LR `1/4` keeps positive scalar weights
positive under every finite raw-gain optimizer. The sphere invariant
holds forever while the true full-gradient norm stays at least one.
The new fused-step correction converges to the negative minimizer in
this same example. This refutes an unconditional optimizer-agnostic
guarantee, not a theorem claimed by the empirical paper. The concrete
AMSGradMD extension chooses AMSGrad for gains in place of the paper's
experimental Adam callbacks and states that choice explicitly.
The same opposite-quadratic obstruction is also proved for the actual
unguarded AMSGradMD buffers at epsilon one, zero moment coefficients
and positive LRs `1/4`. The safeguarded AMSGradMD converges on that case.

The initialization norm formula is a second-moment radius, not an exact
random-sample norm. The `sqrt(max)` target additionally requires the
shape condition recorded in Appendix D. The Muon norm identity requires
exact full-rank polar orthogonalization, not finite Newton--Schulz steps.
Appendix H's rank `k` bounds `ABᵀ`; `ones+ABᵀ` can exceed that rank, as
the nonsingular 3-by-3 two-factor example proves.

## Empirical and open scope

The reported loss improvements, optimizer comparisons, optimal-LR
transfer, warmup-free/continual-training stability, scaling-law fit and
bootstrap confidence intervals, throughput measurements, and gain
trajectories in §4 and Appendices B–I are empirical observations.
The literal evaluation-table arithmetic is checked, but the experiment
that produced it is not a theorem about arbitrary training runs.
The ablations in Appendices E–F and the qualitative related-work
comparisons in §5/Appendix J likewise provide no quantified general
training guarantee.

§4.1.5, §6 and Appendices H–I ask about optimal schedules, scaling
exponents, interpretation of gains, sharpness, RL, and low precision.
No objective, admissible schedule class, quantitative conjecture, or
conclusion is specified for these research questions. They therefore
cannot be faithfully restated as mathematical theorems without adding
a new model. This development introduces no `sorry` or placeholder
claims for those unspecified questions. Every theorem above is proved.
-/

import Transformer.MagnitudeDirection.Section2_Interference
import Transformer.MagnitudeDirection.Section3_Factorization
import Transformer.MagnitudeDirection.Section3_Representation
import Transformer.MagnitudeDirection.Section3_Sphere
import Transformer.MagnitudeDirection.Section3_TangentialUpdates
import Transformer.MagnitudeDirection.Section3_Gradients
import Transformer.MagnitudeDirection.Section3_LossChainRule
import Transformer.MagnitudeDirection.Section3_ScalarAlgorithm
import Transformer.MagnitudeDirection.Section4_PositiveGains
import Transformer.MagnitudeDirection.Section4_MuonScaling
import Transformer.MagnitudeDirection.SectionA_FullAlgorithm
import Transformer.MagnitudeDirection.SectionC_ReportedEvaluations
import Transformer.MagnitudeDirection.SectionD_Initialization
import Transformer.MagnitudeDirection.SectionG_OperationCounts
import Transformer.MagnitudeDirection.SectionH_HigherRankGains
import Transformer.MagnitudeDirection.SectionA_TrainingStorage
import Transformer.MagnitudeDirection.SectionA_StatefulAlgorithm
import Transformer.MagnitudeDirection.SectionA_TrainingGuard
import Transformer.MagnitudeDirection.SectionA_TrainingModels
import Transformer.MagnitudeDirection.SectionA_TrainingInvariants
import Transformer.MagnitudeDirection.SectionA_TrainingConvergence
import Transformer.MagnitudeDirection.SectionA_TrainingMinimum
import Transformer.MagnitudeDirection.SectionA_TrainingQuadratic
import Transformer.MagnitudeDirection.SectionA_TrainingCounterexampleStep
import Transformer.MagnitudeDirection.SectionA_TrainingCounterexample
import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
import Transformer.MagnitudeDirection.SectionA_AMSGradConvergence
import Transformer.MagnitudeDirection.SectionA_AMSGradCounterexampleStep
import Transformer.MagnitudeDirection.SectionA_AMSGradCounterexample
