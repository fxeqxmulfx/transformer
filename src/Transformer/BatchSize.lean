module

public import Transformer.BatchSize.Section3_Taylor
public import Transformer.BatchSize.Section3_Design
public import Transformer.BatchSize.Section4_Algorithms
public import Transformer.BatchSize.Section4_BatchLimits
public import Transformer.BatchSize.Section4_Minibatch
public import Transformer.BatchSize.Section4_DiscreteLaw
public import Transformer.BatchSize.Section4_ConditionalWeakError
public import Transformer.BatchSize.Section4_CoefficientRegularity
public import Transformer.BatchSize.Section4_UniformCovariance
public import Transformer.BatchSize.Section4_BoundedCoefficients
public import Transformer.BatchSize.Section4_SmoothCoefficients
public import Transformer.BatchSize.Section4_NoiseLipschitz
public import Transformer.BatchSize.Section4_BrownianPaths
public import Transformer.BatchSize.Section4_EulerGlobalStability
public import Transformer.BatchSize.Section4_EulerFrozenStability
public import Transformer.BatchSize.Section4_EulerLimitMoments
public import Transformer.BatchSize.Section4_EulerClipping
public import Transformer.BatchSize.Section4_EulerWindow
public import Transformer.BatchSize.Section4_EulerIncrementMoments
public import Transformer.BatchSize.Section4_EulerLimitIncrements
public import Transformer.BatchSize.Section4_EulerTimeChange
public import Transformer.BatchSize.Section4_KolmogorovCriterion
public import Transformer.BatchSize.Section4_EulerContinuousLimit
public import Transformer.BatchSize.Section4_EulerLimitLaw
public import Transformer.BatchSize.Section4_EulerLimitRate
public import Transformer.BatchSize.Section4_VectorGaussianIntegration
public import Transformer.BatchSize.Section4_DiagonalGaussianState
public import Transformer.BatchSize.Section4_GaussianDirectionalIntegration
public import Transformer.BatchSize.Section4_GaussianTestIntegration
public import Transformer.BatchSize.Section4_VectorGaussianFlow
public import Transformer.BatchSize.Section4_VectorGaussianDerivative
public import Transformer.BatchSize.Section4_FrozenGaussianGenerator
public import Transformer.BatchSize.Section4_VectorGaussianGenerator
public import Transformer.BatchSize.Section4_VectorGaussianIdentity
public import Transformer.BatchSize.Section4_BrownianVectorIncrement
public import Transformer.BatchSize.Section4_BrownianVectorExpectation
public import Transformer.BatchSize.Section4_BoundedObservableLimits
public import Transformer.BatchSize.Section4_GaussianParameters
public import Transformer.BatchSize.Section4_RandomGaussianIdentity
public import Transformer.BatchSize.Section4_FrozenBrownianState
public import Transformer.BatchSize.Section4_FrozenBrownianExpectation
public import Transformer.BatchSize.Section4_FrozenBrownianGenerator
public import Transformer.BatchSize.Section4_EulerIntervals
public import Transformer.BatchSize.Section4_EulerWindowSecondMoment
public import Transformer.BatchSize.Section4_DyadicObservationTimes
public import Transformer.BatchSize.Section4_EulerObservationNeighbors
public import Transformer.BatchSize.Section4_EulerIntervalGenerator
public import Transformer.BatchSize.Section4_DyadicIntervals
public import Transformer.BatchSize.Section4_DyadicGenerator
public import Transformer.BatchSize.Section4_DyadicGeneratorIdentity
public import Transformer.BatchSize.Section4_DyadicNeighborLimits
public import Transformer.BatchSize.Section4_BoundedIntervalLimits
public import Transformer.BatchSize.Section4_DyadicGeneratorLimit
public import Transformer.BatchSize.Section4_WeightedTestLimits
public import Transformer.BatchSize.Section4_EulerLimitGenerator
public import Transformer.BatchSize.Section4_PathCompensation
public import Transformer.BatchSize.Section4_OptimizerGenerator
public import Transformer.BatchSize.Section4_OptimizerMartingale
public import Transformer.BatchSize.Section4_PathObservables
public import Transformer.BatchSize.Section4_DiffusionLaw
public import Transformer.BatchSize.Section4_CompactDerivativeBounds
public import Transformer.BatchSize.Section4_ModelParameterBounds
public import Transformer.BatchSize.Section4_SignParameters
public import Transformer.BatchSize.Section4_DriftDerivativeBounds
public import Transformer.BatchSize.Section4_ComposedDerivativeLipschitz
public import Transformer.BatchSize.Section4_DeterministicEulerMap
public import Transformer.BatchSize.Section4_DeterministicTestBounds
public import Transformer.BatchSize.Section4_DeterministicHorizonBounds
public import Transformer.BatchSize.Section4_C2DiscreteError
public import Transformer.BatchSize.Section4_OptimizerSmallIncrements
public import Transformer.BatchSize.Section4_DiscreteDeterministicComparison
public import Transformer.BatchSize.Section4_ScaledGeneratorIdentity
public import Transformer.BatchSize.Section4_C2DriftObservable
public import Transformer.BatchSize.Section4_C2GeneratorBounds
public import Transformer.BatchSize.Section4_PathExpectationBounds
public import Transformer.BatchSize.Section4_ExpectedGeneratorDrift
public import Transformer.BatchSize.Section4_OptimizerLocalWeakDrift
public import Transformer.BatchSize.Section4_OptimizerDeterministicLocalError
public import Transformer.BatchSize.Section4_OptimizerDeterministicComparison
public import Transformer.BatchSize.Section4_EulerLaws
public import Transformer.BatchSize.Section4_BrownianHigherMoments
public import Transformer.BatchSize.Section4_VectorFourthMoment
public import Transformer.BatchSize.Section4_ItoMartingale
public import Transformer.BatchSize.Section4_GaussianComparison
public import Transformer.BatchSize.Section4_WeakApproximation

/-!
# Batch size and the Adam-SGD gap

Formalization of arXiv:2506.12543v1, Sreckovic, Geiping, Orvieto,
"Is your batch size the problem? Revisiting the Adam-SGD gap in language
modeling". The PDF and TeX are in papers/arXiv-2506.12543v1.
Source corrections are recorded in the affected declarations.

Coverage: the directional Taylor expansion (Section 3.2); the rotated
quadratic Hessians and their positive design factor (Section 3.3, Appendix D);
grafting and adaptive momentum clipping (Sections 4.1--4.2, Appendix C);
minibatch variance, Gaussian sign moments, exact drift/covariance matrices,
local batch-size scaling and saturation (Section 4.3). The discrete SGD and
SignSGD samplers and the SDE generator are explicit probability models.
Uniform one-step weak consistency, including rescaled derivative bounds,
is proved for both optimizers. Their
Markov expectations are contractions and coincide with the actual n-step
sampler; a proved telescope gives the conditional finite-horizon weak bound.
Both drift and diffusion amplitude have global state Lipschitz bounds;
the amplitude's bound is proportional to sqrt(eta). Uniform covariance bounds
and seven state derivatives of both SDEs' drift and diffusion amplitude
are also proved. For constant scalar Brownian coefficients, the actual
Gaussian expectation operator is constructed: its semigroup law, backward
heat equation, integrated generator identity, local expansion, contraction
and propagation of finite smooth derivative bounds are proved.

An actual continuous vector Brownian path law is constructed, with independent
Gaussian coordinates and increments independent of their entire joint past.
Finite adapted stochastic sums satisfy scalar and vector Ito isometries.
The actual optimizer Euler chains are adapted and square integrable; their
finite-horizon mean-square stability is uniform in grid resolution and
eta in [0,1]. Their continuous Brownian interpolations have normalized path
laws, start at the prescribed state and have the computed endpoint marginal.
Conditional noise means and second, third and fourth moment kernels are proved.
Coarse Brownian Euler chains are coupled with their dyadic refinements;
their mean-square differences decay geometrically. Completeness constructs
adapted square-integrable optimizer limits at every nonnegative time.
Bounded scalar and vector stochastic sums satisfy fourth-moment estimates
uniformly over the number of grid steps. Consistent Brownian interpolations
give fourth-moment increment estimates at arbitrary times; these pass to
the actual mean-square limits with the eta-squared noise factor intact.
Kolmogorov's criterion constructs a continuous modification. Its actual
continuous-path probability law has the prescribed initial condition and
the computed Euler-limit marginals at every nonnegative time.
The vector Gaussian generator identity and its conditional Brownian form
give the finite Euler martingale identity. Bounded observable convergence
passes it to the continuous Euler limit. Its actual path law satisfies the
full bounded-cylinder generator martingale problem, so it is a genuine
diffusion law with exactly the paper's drift and covariance.

The general first-order weak approximation (Theorem 1) is proved with
explicit sufficient regularity conditions. Uniform second drift derivatives
give finite-horizon C2 bounds for the genuine deterministic Euler backward
tests. The actual discrete and continuous optimizer expectations each differ
from this common comparison by O(eta); their difference therefore has the
claimed first-order weak bound. The diffusion law is the constructed Brownian
Euler-limit path law, with the paper's exact drift and covariance.
Optimizer rankings on the reported training
runs are empirical observations, not universal convergence theorems. The
quadratic spectral results hold for every orthogonal block rotation; the
source's unspecified eigenvector sign convention is not treated as a proof
that its rotation sampler has Haar law.
-/

@[expose] public section
