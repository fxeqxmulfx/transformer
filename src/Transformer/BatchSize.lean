/-
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

import Transformer.BatchSize.Section3_Taylor
import Transformer.BatchSize.Section3_Design
import Transformer.BatchSize.Section4_Algorithms
import Transformer.BatchSize.Section4_BatchLimits
import Transformer.BatchSize.Section4_Minibatch
import Transformer.BatchSize.Section4_DiscreteLaw
import Transformer.BatchSize.Section4_ConditionalWeakError
import Transformer.BatchSize.Section4_CoefficientRegularity
import Transformer.BatchSize.Section4_UniformCovariance
import Transformer.BatchSize.Section4_BoundedCoefficients
import Transformer.BatchSize.Section4_SmoothCoefficients
import Transformer.BatchSize.Section4_NoiseLipschitz
import Transformer.BatchSize.Section4_BrownianPaths
import Transformer.BatchSize.Section4_EulerGlobalStability
import Transformer.BatchSize.Section4_EulerFrozenStability
import Transformer.BatchSize.Section4_EulerLimitMoments
import Transformer.BatchSize.Section4_EulerClipping
import Transformer.BatchSize.Section4_EulerWindow
import Transformer.BatchSize.Section4_EulerIncrementMoments
import Transformer.BatchSize.Section4_EulerLimitIncrements
import Transformer.BatchSize.Section4_EulerTimeChange
import Transformer.BatchSize.Section4_KolmogorovCriterion
import Transformer.BatchSize.Section4_EulerContinuousLimit
import Transformer.BatchSize.Section4_EulerLimitLaw
import Transformer.BatchSize.Section4_EulerLimitRate
import Transformer.BatchSize.Section4_VectorGaussianIntegration
import Transformer.BatchSize.Section4_DiagonalGaussianState
import Transformer.BatchSize.Section4_GaussianDirectionalIntegration
import Transformer.BatchSize.Section4_GaussianTestIntegration
import Transformer.BatchSize.Section4_VectorGaussianFlow
import Transformer.BatchSize.Section4_VectorGaussianDerivative
import Transformer.BatchSize.Section4_FrozenGaussianGenerator
import Transformer.BatchSize.Section4_VectorGaussianGenerator
import Transformer.BatchSize.Section4_VectorGaussianIdentity
import Transformer.BatchSize.Section4_BrownianVectorIncrement
import Transformer.BatchSize.Section4_BrownianVectorExpectation
import Transformer.BatchSize.Section4_BoundedObservableLimits
import Transformer.BatchSize.Section4_GaussianParameters
import Transformer.BatchSize.Section4_RandomGaussianIdentity
import Transformer.BatchSize.Section4_FrozenBrownianState
import Transformer.BatchSize.Section4_FrozenBrownianExpectation
import Transformer.BatchSize.Section4_FrozenBrownianGenerator
import Transformer.BatchSize.Section4_EulerIntervals
import Transformer.BatchSize.Section4_EulerWindowSecondMoment
import Transformer.BatchSize.Section4_DyadicObservationTimes
import Transformer.BatchSize.Section4_EulerObservationNeighbors
import Transformer.BatchSize.Section4_EulerIntervalGenerator
import Transformer.BatchSize.Section4_DyadicIntervals
import Transformer.BatchSize.Section4_DyadicGenerator
import Transformer.BatchSize.Section4_DyadicGeneratorIdentity
import Transformer.BatchSize.Section4_DyadicNeighborLimits
import Transformer.BatchSize.Section4_BoundedIntervalLimits
import Transformer.BatchSize.Section4_DyadicGeneratorLimit
import Transformer.BatchSize.Section4_WeightedTestLimits
import Transformer.BatchSize.Section4_EulerLimitGenerator
import Transformer.BatchSize.Section4_PathCompensation
import Transformer.BatchSize.Section4_OptimizerGenerator
import Transformer.BatchSize.Section4_OptimizerMartingale
import Transformer.BatchSize.Section4_PathObservables
import Transformer.BatchSize.Section4_DiffusionLaw
import Transformer.BatchSize.Section4_CompactDerivativeBounds
import Transformer.BatchSize.Section4_ModelParameterBounds
import Transformer.BatchSize.Section4_SignParameters
import Transformer.BatchSize.Section4_DriftDerivativeBounds
import Transformer.BatchSize.Section4_ComposedDerivativeLipschitz
import Transformer.BatchSize.Section4_DeterministicEulerMap
import Transformer.BatchSize.Section4_DeterministicTestBounds
import Transformer.BatchSize.Section4_DeterministicHorizonBounds
import Transformer.BatchSize.Section4_C2DiscreteError
import Transformer.BatchSize.Section4_OptimizerSmallIncrements
import Transformer.BatchSize.Section4_DiscreteDeterministicComparison
import Transformer.BatchSize.Section4_ScaledGeneratorIdentity
import Transformer.BatchSize.Section4_C2DriftObservable
import Transformer.BatchSize.Section4_C2GeneratorBounds
import Transformer.BatchSize.Section4_PathExpectationBounds
import Transformer.BatchSize.Section4_ExpectedGeneratorDrift
import Transformer.BatchSize.Section4_OptimizerLocalWeakDrift
import Transformer.BatchSize.Section4_OptimizerDeterministicLocalError
import Transformer.BatchSize.Section4_OptimizerDeterministicComparison
import Transformer.BatchSize.Section4_EulerLaws
import Transformer.BatchSize.Section4_BrownianHigherMoments
import Transformer.BatchSize.Section4_VectorFourthMoment
import Transformer.BatchSize.Section4_ItoMartingale
import Transformer.BatchSize.Section4_GaussianComparison
import Transformer.BatchSize.Section4_WeakApproximation
