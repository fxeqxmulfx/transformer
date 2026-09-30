/-
# Muon is Scalable for LLM Training

Liu et al., arXiv:2502.16982. The source archive is available locally in
`papers/arXiv-2502.16982/`, with `template.tex` as its entry point.

## Mathematical coverage

* §2.1, `eq:Ot` and `eq:iteration`: gradient and Nesterov momentum,
  Frobenius normalization, finite Newton–Schulz iteration, its action on
  singular coordinates, and exact polar orthogonalization. The negative
  polar factor minimizes the first-order objective over the Euclidean
  induced-operator-norm unit ball. With momentum this is the linear
  objective determined by the orthogonalized input, not necessarily the
  current loss gradient.
* §2.2, `lemma:updaterms`, and Appendix A: the general rank-`r` formula
  `RMS(U Vᵀ) = √(r/(a b))`, the full-rank matrix-level theorem
  `muon_update_rms`, shape scaling to RMS one, the final factor `0.2`,
  the original shape-factor equivalence, and decoupled weight decay.
* §2.3, `alg:distribmuon`: lossless scatter/gather, gradient reduction,
  and equality of distributed and centralized momentum and weight
  updates. The separate byte-count calculation proves the source's
  `(1,1.25]` communication ratio without the additional TP gather and
  the factor-of-two saving for equal-precision momentum buffers.
* §3.1: direct RMS normalization and the `[H,4H]` versus `[H,H]`
  adjustment factors. §3.2, `tab:fit`, and Appendix B,
  `tab:dense_scaling_param`: algebra of the literal reported fit functions,
  including equal-loss compute and the product of model size and tokens.
* §3.4: the SVD entropy formula, probability normalization, bounds,
  flat-spectrum value, and invariance under nonzero global scaling.
* Appendix C: centering the expert bias preserves every score ordering
  and top-k selection, and preserves the sum of biases. The gate code's
  sigmoid ordering, normalized probabilities, factor bounds, and energy
  normalization are verified. Equal uncorrelated expert-output variances
  are explicit hypotheses of the variance interpretation.

## Source corrections and domains

The “full-rank matrix parameter” in Lemma 1 must mean the momentum input
being orthogonalized. Full-rank weights may have zero momentum; the
counterexample `fullRank_weight_zero_update` records this difference.
The exact RMS identity requires exact polar orthogonalization, whereas
the printed five-step polynomial is an approximation. Its coefficients
give `f(1)=0.701`, so no scalar trajectory converges to exactly one.
The actual five-step identity input has update RMS between 0.69 and 0.70,
as recorded in `five_step_identity_rms_bounds`.
The ordinary left-Gram inverse square root written in §2.1 fails even
for the tall matrix `[1;0]`; the spectral pseudoinverse gives the stated
polar identity on the active singular subspace. Appendix A's omitted
cross terms cancel by orthonormality, as the trace proof shows.

Positive dimensions are required for the full-rank RMS calculation.
Direct normalization requires positive input RMS; zero updates stay
zero. The entropy's probability interpretation requires positive energy
and at least two singular values. A shared master-weight allocation
prevents a claim that total optimizer memory is halved. Distributed
equality is in exact real arithmetic; bf16 rounding is outside this model.
The literal fit coefficients imply a budget-dependent FLOPs fraction,
and their rounding prevents an exact identity `C=6ND`.

## Additional training convergence results

The original five-step optimizer cycles on the strongly convex quadratic
`‖W‖²/2` for a nonzero scalar weight, zero momentum coefficient and decay,
and constant learning rate `η=1/L=1`. This is a counterexample to an
unconditional convergence interpretation, not to a theorem in the paper.
The new `safeguardedMuonRun` computes the complete paper candidate and
updates momentum, checks its alignment and Frobenius length against the
current full gradient, and uses that gradient when the check fails.
With `η=σ/L` and `0<σ≤1`, its gradient norms vanish and loss values converge
for any fixed lower-bounded differentiable loss with the stated smooth
quadratic upper model. Under the additional strong-convexity model and
an existing stationary point, weights converge to the global minimum
with a geometric loss-gap bound. The actual five printed matrix iterations
remain fixed. These are explicitly documented algorithm corrections and
additional objective assumptions, not source training guarantees.

## Experimental scope

Training quality and stability, measured checkpoint spectra, the
reported approximately 52% compute requirement, reuse of tuned AdamW
hyperparameters, the choice of five steps and momentum 0.95, latency
measurements, benchmark and SFT comparisons, and the sampled gate factor
2.446 are empirical results, not consequences of the algebraic formulas.
This includes the training observations in §2.2–2.3, §3, and Appendices
A, B, D, E, and F. Their data and experimental validity are not proved
here. Theorems about reported fit functions prove their algebra, not
their accuracy or optimality. The qualitative research directions in §4
do not specify quantified mathematical conjectures.
-/

import Transformer.Muon.Section2_Models
import Transformer.Muon.Section2_TrainingConvergence
import Transformer.Muon.Section2_TrainingCycleStep
import Transformer.Muon.Section2_TrainingCounterexample
import Transformer.Muon.Section2_Spectral
import Transformer.Muon.AppendixA_RMS
import Transformer.Muon.Section2_Scaling
import Transformer.Muon.Section2_NewtonSchulz
import Transformer.Muon.Section2_Approximation
import Transformer.Muon.Section2_Euclidean
import Transformer.Muon.Section2_SteepestDescent
import Transformer.Muon.Section2_Distributed
import Transformer.Muon.Section2_Costs
import Transformer.Muon.Section2_PolarIdentities
import Transformer.Muon.Section3_Entropy
import Transformer.Muon.AppendixC_Routing
import Transformer.Muon.AppendixC_GateScaling
import Transformer.Muon.AppendixC_GateModel
import Transformer.Muon.Section3_ScalingLaws
import Transformer.Muon.Section2_Rank
