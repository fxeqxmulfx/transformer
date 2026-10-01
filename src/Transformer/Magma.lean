/-
# Magma: masking updates in adaptive optimizers

Formalization of arXiv:2602.15322v1, On Surprising Effectiveness of Masking
Updates in Adaptive Optimizers. The source is preserved locally under
papers/arXiv-2602.15322v1. Definitions follow the manuscript's sections.
Corrections and counterexamples are identified in individual docstrings;
no universal optimizer ranking follows from the empirical comparisons.

Coverage of the four numbered results:
* Section 2 / Appendix A.1: actual Frechet derivatives, independent mask
  moments, and a corrected explicit cubic Taylor remainder bound.
* Section 5 / Appendix A.2: counterexamples to joint smoothness and the
  published descent lemma, plus global-smooth corrected mask descent.
* Section 5 / Appendix A.3: counterexample to the intended current-gradient
  alignment estimate, a sign-error witness, and valid noise-coupling bounds.
* Section 5 / Appendix A.4: a first-step counterexample to the published
  theorem and corrected finite-horizon stationarity for actual dense-state
  Magma with an SGD base direction, finite unbiased sampling, globally
  smooth loss, and invariant scale initialization.

Algorithm 1 and Section 5 use different mask normalization; the factor at
p=1/2 is proved explicitly. Reported C4 and ablation scores are audited as
finite rational data. The Gaussian alignment formula has an explicit model;
the square-root Gamma benchmark's strict heavy-tail label is refuted.
The qualitative granularity conjecture is qualified by a counterexample
to universal equality and exact checks of the reported near-equivalence.
The open stable-unbiased-masking direction is addressed by a precise
quadratic contraction result; neural-network stability remains empirical.
-/

import Transformer.Magma.Section2_MaskLaw
import Transformer.Magma.Section2_Moments
import Transformer.Magma.Section2_Expectations
import Transformer.Magma.Section2_BlockExpansion
import Transformer.Magma.Section2_Taylor
import Transformer.Magma.Section2_Granularity
import Transformer.Magma.Section3_Algorithm
import Transformer.Magma.Section3_DampingBounds
import Transformer.Magma.Section3_MomentumVariance
import Transformer.Magma.Section3_GaussianAlignment
import Transformer.Magma.Section3_UnbiasedAlternative
import Transformer.Magma.Section4_C4Results
import Transformer.Magma.Section4_Ablations
import Transformer.Magma.Section4_GammaDensity
import Transformer.Magma.Section4_GammaRefutation
import Transformer.Magma.Section5_SmoothnessRefutation
import Transformer.Magma.Section5_AlignmentRefutation
import Transformer.Magma.Section5_MainRefutation
import Transformer.Magma.Section5_AlignmentBound
import Transformer.Magma.Section5_MaskDescent
import Transformer.Magma.Section5_BlockOperators
import Transformer.Magma.Section5_EffectiveCurvature
import Transformer.Magma.Section5_EfficiencyCoefficients
import Transformer.Magma.Section5_StochasticDescent
import Transformer.Magma.Section5_Transition
import Transformer.Magma.Section5_Convergence
import Transformer.Magma.Section5_ConvergenceWitness
import Transformer.Magma.Section5_MagmaTransition
import Transformer.Magma.Section5_MagmaConvergence
