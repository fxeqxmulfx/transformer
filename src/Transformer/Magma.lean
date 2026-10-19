module

public import Transformer.Magma.Section2_MaskLaw
public import Transformer.Magma.Section2_Moments
public import Transformer.Magma.Section2_Expectations
public import Transformer.Magma.Section2_BlockExpansion
public import Transformer.Magma.Section2_Taylor
public import Transformer.Magma.Section2_Granularity
public import Transformer.Magma.Section3_Algorithm
public import Transformer.Magma.Section3_DampingBounds
public import Transformer.Magma.Section3_MomentumVariance
public import Transformer.Magma.Section3_GaussianAlignment
public import Transformer.Magma.Section3_UnbiasedAlternative
public import Transformer.Magma.Section4_C4Results
public import Transformer.Magma.Section4_Ablations
public import Transformer.Magma.Section4_GammaDensity
public import Transformer.Magma.Section4_GammaRefutation
public import Transformer.Magma.Section5_SmoothnessRefutation
public import Transformer.Magma.Section5_AlignmentRefutation
public import Transformer.Magma.Section5_MainRefutation
public import Transformer.Magma.Section5_AlignmentBound
public import Transformer.Magma.Section5_MaskDescent
public import Transformer.Magma.Section5_BlockOperators
public import Transformer.Magma.Section5_EffectiveCurvature
public import Transformer.Magma.Section5_EfficiencyCoefficients
public import Transformer.Magma.Section5_StochasticDescent
public import Transformer.Magma.Section5_Transition
public import Transformer.Magma.Section5_Convergence
public import Transformer.Magma.Section5_ConvergenceWitness
public import Transformer.Magma.Section5_MagmaTransition
public import Transformer.Magma.Section5_MagmaConvergence

/-!
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

@[expose] public section
