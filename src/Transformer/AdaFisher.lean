/-
# AdaFisher: Adaptive Second Order Optimization via Fisher Information

Gomes, Zhang, Belilovsky, Wolf, Hosseini, arXiv:2405.16397v3 (ICLR 2025).
The PDF, source archive, extracted TeX, text, and download hashes are in
`papers/arXiv-2405.16397v3/`. All numbered mathematical statements are
accounted for below, including the repeated propositions in Appendix A.2.

## Mathematical coverage

* §2: likelihood maximization versus negative-log-likelihood minimization;
  augmented affine layers, the nonlinear chain rule, transpose backpropagation,
  the weight-gradient outer product and its column-first vectorization;
  derivatives of log probabilities and centered finite probabilistic scores;
  empirical Fisher symmetry and positive semidefiniteness; the exact
  Kronecker outer-product identity and product-sampling factorization.
  Equation (1), `eq:fishermatrix`, is stated with the cross-score-zero
  condition needed for Fisher-of-sum = sum-of-Fishers. K-FAC's empirical
  independence and block-diagonal approximations are modeling assumptions,
  not identities for every trained network.
* §3.1 and Appendix A.1, Theorem A.1: the complex Gershgorin theorem, with
  a nonzero eigenvector and the complete off-diagonal row radius. The DFT
  and SNR formulas are defined, DFT additivity is proved, and off-diagonal
  noise preserves diagonal signal energy. The all-ones Gram example shows
  why Gershgorin localization alone cannot prove diagonal dominance.
* §3.2, Proposition 3.1 / Appendix A.2, Proposition A.1: the proposed
  normalization-layer factors, actual scale/shift directional derivatives and exact
  Fisher expansions including all cross-site terms. A one-channel,
  two-site counterexample refutes the proof's scale and bias factorizations.
* §3.2, Proposition 3.2 / Appendix A.2, Proposition A.2: min-max extrema,
  normalized entry and Loewner bounds, the square-root sandwich formula,
  ordering and distance scaling; the damped
  diagonal Kronecker representation; positive definiteness, eigenvalue
  and reciprocal bounds; the ordinary matrix inverse and unique solve.
  Exact natural-gradient tangent invariance is proved for invertible
  parameter Jacobians. Appendix A.3 covers FC, convolutional patch and
  normalization factors and the identity fallback.
* §3.3, Algorithm 1 and Table 1: KF EMA, raw and bias-corrected gradient
  momentum, its finite weighted-sum formula, AdaFisher/AdaFisherW parameter
  updates and decay. A full block state transition includes both KF buffers,
  normalization, damping, raw momentum, correction and parameter update.
  The first batch uses its actual gradient, and positive damping bounds
  the actual preconditioner constructed by the state transition.
* §3.4, Proposition 3.3 / Appendix A.2, Proposition A.3: a smooth strongly
  convex quadratic on a genuine nonconstant-factor four-parameter block
  refutes both the stated fixed-step convergence and the printed 1/k rate.
  The corrected theorem uses separate strong-convexity
  μ, smoothness L and Fisher bounds [δ,B], with η L≤δ and η μ≤B. Its objective
  rate is (1-ημ/B)^t, and both objective gap and squared Euclidean distance
  to the stationary minimizer converge to zero. `eq:upper_bound1` is
  replaced by the proved sufficient decrease η/(2B) times gradient norm².
  Its diverging allowed-step example also refutes an unconditional reading
  of Table 1's O(log(T) sqrt(T)) regret guarantee.
* §3.4, Proposition 3.4 / Appendix A.2, Proposition A.4: the printed β≤1
  endpoint is refuted by a bounded smooth sine objective and a genuine
  nonconstant-factor four-dimensional efficient Fisher block. Gradients
  are the derivatives of that objective, zero noise is unbiased/independent,
  the effective-preconditioner monotonicity holds, and no finite constants
  independent of the horizon can provide the printed vanishing bound.
  The harmonic/squared-update and C2/C3 variation estimates are proved with
  damping factors included. An explicit weighted-energy budget implies the
  corrected final extraction L*K/(η*sqrt(T)); the absent η in `eq:bound_2`
  also has a numerical counterexample. These conditional estimates do not
  assert the source's unestablished stochastic energy budget for β<1.
* Added deterministic training extension of §3.3–3.4: the full layer-block
  state transition computes true current loss gradients, both measured KF
  EMAs, min-max scaling, damping and positive-time bias-corrected momentum.
  With a fixed smooth lower-bounded loss, Lipschitz gradient, η,δ,L>0,
  0≤β<1 and 2Lη≤δ(1-β), every gradient norm tends to zero and losses have
  a finite limit. Under strong convexity and an existing stationary point,
  actual last weights converge to a global minimizer. The proof derives
  Fisher bounds [δ,1+δ] and the momentum-error Lyapunov budget; it assumes
  no metric monotonicity, bounded gradients, bounded weights or alignment.
  Bias correction is retained and no direction guard is added. A concrete
  four-parameter witness checks varying source-form Gram measurements,
  both actual EMA buffers, Fisher entries, parameter motion and convergence.
  These theorems concern full deterministic gradients, zero decoupled
  weight decay and a constant sufficiently small step, not the printed
  stochastic decreasing-step rate or AdaFisherW's general convergence.
* Appendix A.4, `eq:distributedkf`: averaging integrable unbiased local
  factors preserves unbiasedness without requiring GPU independence.
  Averaging commutes with EMA in exact arithmetic. A two-GPU example shows
  why averaging factors does not commute with multiplying them.
* §4, Tables 2–5, and Appendix A.4, Table 6: the literal CIFAR and transfer
  accuracy means, ImageNet Top-1 baseline scores, language-model PPL and
  GPU times are recorded with their finite comparisons verified. Missing
  measurements remain absent. Appendix C's 200-point grid has the stated
  endpoints/range, and a two-layer example records why first-layer weights
  alone do not generally determine the true full-network loss.

## Source corrections

Uncentered second moments are not covariance matrices: independent
nonzero-mean constants have nonzero off-diagonal second moments. The
normalization proof changes a C×C Fisher block into a C²×C² Kronecker block
and drops cross-site terms without justification. The factors can be used
as approximations, but the printed equalities are refuted.

Min-max normalization needs a nonzero range. We explicitly extend a
constant factor to zero, documenting the convention at its definition;
the identity fallback therefore leaves only damping. The claimed relative
inverse error O(epsilon+lambda) is refuted on exactly diagonal, nonconstant
factors: their normalized minimum becomes zero, and the inverse error
grows as 1/lambda even at epsilon=0.

Algorithm 1 uses an undefined h in place of its computed g, a first bias
denominator 1-β⁰=0, inconsistent moment indices, and unspecified initial KF
EMA buffers. Our transition uses g, stores raw momentum, corrects at the
new positive batch count, and initializes factor buffers to zero.
The γ coefficient weights the fresh factor, as in equation (3).
The convex proof conflates Fisher and loss-Hessian bounds. The stochastic
statement must exclude β=1 (as Algorithm 1 already does); its proof also
confuses a norm lower bound with an entrywise one, repeats/omits time
indices, and loses η in the last inequality. None of these corrections is
silently presented under the unchanged paper statement.

## Experimental scope

The diagonal concentration/noise-robustness observations of §3.1 and A.1,
the flatter-minima/generalization interpretation in §3.3, measured training
quality and resource use in §4, all ablation observations in Appendix B,
the PCA/cubic-interpolation pipeline and logged trajectories in Appendix C,
and experiment protocols/results in Appendix D concern measured runs.
Their data validity and optimizer superiority on unobserved tasks are not
proved by Lean. The reported-score theorems establish only arithmetic
comparisons of the literal means, without asserting statistical significance
or equivalence of different literature training setups. The future-research
directions in §6 are proposals rather than quantified conjectures.
-/

import Transformer.AdaFisher.Section2_Fisher
import Transformer.AdaFisher.Section2_Likelihood
import Transformer.AdaFisher.Section2_Score
import Transformer.AdaFisher.Section2_FisherIdentities
import Transformer.AdaFisher.Section2_Backprop
import Transformer.AdaFisher.Section3_Normalization
import Transformer.AdaFisher.Section3_NormalizationGradient
import Transformer.AdaFisher.Section3_MinMax
import Transformer.AdaFisher.Section3_Efficient
import Transformer.AdaFisher.Section3_Spectrum
import Transformer.AdaFisher.Section3_MinMaxMatrix
import Transformer.AdaFisher.Section3_NaturalGradient
import Transformer.AdaFisher.Section3_InverseError
import Transformer.AdaFisher.Section3_Algorithm
import Transformer.AdaFisher.Section3_State
import Transformer.AdaFisher.Section3_TrainingModels
import Transformer.AdaFisher.Section3_TrainingBias
import Transformer.AdaFisher.Section3_TrainingCoefficients
import Transformer.AdaFisher.Section3_TrainingError
import Transformer.AdaFisher.Section3_TrainingInner
import Transformer.AdaFisher.Section3_TrainingLossDescent
import Transformer.AdaFisher.Section3_TrainingEnergy
import Transformer.AdaFisher.Section3_TrainingBudget
import Transformer.AdaFisher.Section3_TrainingGradientEnergy
import Transformer.AdaFisher.Section3_TrainingErrorLimit
import Transformer.AdaFisher.Section3_TrainingStationarity
import Transformer.AdaFisher.Section3_TrainingLossLimit
import Transformer.AdaFisher.Section3_TrainingMinimum
import Transformer.AdaFisher.Section3_TrainingWitness
import Transformer.AdaFisher.Section3_ConvexFalse
import Transformer.AdaFisher.Section3_ConvexWitness
import Transformer.AdaFisher.Section3_ConvexCorrected
import Transformer.AdaFisher.Section3_ConvexLimit
import Transformer.AdaFisher.Section3_RegretFalse
import Transformer.AdaFisher.Section3_NonconvexFalse
import Transformer.AdaFisher.Section3_NonconvexWitness
import Transformer.AdaFisher.SectionA_Descent
import Transformer.AdaFisher.SectionA_Variation
import Transformer.AdaFisher.SectionA_StochasticTerms
import Transformer.AdaFisher.SectionA_GradientBound
import Transformer.AdaFisher.SectionA_Gershgorin
import Transformer.AdaFisher.SectionA_SpectralAnalysis
import Transformer.AdaFisher.SectionA_Factors
import Transformer.AdaFisher.SectionA_Distributed
import Transformer.AdaFisher.Section4_ReportedResults
import Transformer.AdaFisher.SectionC_Visualization
