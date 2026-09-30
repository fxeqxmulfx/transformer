/-
# DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers

Modoranu et al., arXiv:2602.02016v2. Source: `papers/arXiv-2602.02016/dash.tex`.
Formalization of the exact real-arithmetic algorithms and their mathematical
properties. Empirical training quality, timing, and floating-point experiments
are distinguished from their mathematical specifications.

This is a partial formalization: the modules cover Shampoo EMAs, regularized
grafting and exact EVD inverse roots; greedy layer allocation; scalar and
matrix CN/NDB convergence, uniform convergence on compact spectral domains,
finite numerical chains for every fixed dyadic root depth, the algebraic
restriction to signed dyadic powers and operation counts; finite-iterate
stopping certificates in operator and Frobenius norms; Euclidean operator-norm
criteria, spectral scaling, Rayleigh equality,
deterministic multi-PI convergence and finite
independent-start probability estimates; ordinary and optimized Clenshaw,
product counts, continuous/discrete Chebyshev orthogonality, exact recovery
by the cosine fit, sample stability, geometric reference error bounds and
uniform convergence of the actual fitted scalar/matrix arrays with corrected
output scaling; dampening and threshold rank; lossless full/residual block
layouts, normalization/vision reshapes and cached inverse-root histories;
explicit batched Newton/Clenshaw recurrences, histories, grafted parameter
updates and product-call counts; corrected time-dependent EMA rank bounds;
automatic spectral certificates with checked PI acceptance and a safe
fallback; complete guarded CN/NDB inverse-root pipelines, batched product
counts and convergence from positive-definite inputs alone; automatically
repaired Chebyshev sample counts with uniform accuracy on original PSD
inputs; and convergence of the complete multi-PI/NDB/grafting temporal
Shampoo parameter update as its root budget increases at each fixed
training time, from its EMA and regularization assumptions.
They do not
prove the measured speedups or training quality, floating-point stability,
PI probability guarantees beyond the stated sampling model, or a numerical
error certificate for the experimental Chebyshev degree and sample choices.
Corrections to the printed CN endpoints and unit comparison scaling, spectral
norm definition, PI scaling guarantee, Clenshaw input/target and fitting
sample count, dampening summary, ABS positivity and EMA rank bound are
documented beside the relevant theorems and counterexamples.

The additional training modules prove a separate limit in training time.
The actual finite NDB/grafting update can cycle on a strongly convex
quadratic even after numerical scaling is repaired and with `η=1/L=1`.
The new `safeguardedDashRun` updates the actual zero-initialized histories,
computes finite PI/NDB roots and Adam grafting from the current full
gradient, then checks the resulting direction's alignment and Frobenius
length. With `η=σ/L` and `0<σ≤1`, a fixed lower-bounded smooth loss has
vanishing full-gradient norms and convergent loss values. Adding the
strong-convexity model and an existing stationary point yields convergence
of weights to the global minimum with a geometric loss-gap bound.
Both finite solver budgets remain fixed as training time increases.
The guard and loss assumptions are explicit user-requested extensions;
they are not presented as printed manuscript claims or as guarantees
for stochastic minibatches or floating-point training.
-/

import Transformer.DASH.Section2_Models
import Transformer.DASH.Section2_TrainingConvergence
import Transformer.DASH.Section2_TrainingScalarRoot
import Transformer.DASH.Section2_TrainingCycleStep
import Transformer.DASH.Section2_TrainingCounterexample
import Transformer.DASH.Section2_History
import Transformer.DASH.Section2_LoadBalancing
import Transformer.DASH.Section2_GraftingDomain
import Transformer.DASH.Section2_GraftingContinuity
import Transformer.DASH.Section3_CoupledNewton
import Transformer.DASH.Section3_CNConvergence
import Transformer.DASH.Section3_CNUniformConvergence
import Transformer.DASH.Section3_CNUnitScaling
import Transformer.DASH.Section3_Spectral
import Transformer.DASH.Section3_NewtonModels
import Transformer.DASH.Section3_OperationCounts
import Transformer.DASH.Section3_NewtonSpectra
import Transformer.DASH.Section3_MatrixConvergence
import Transformer.DASH.Section3_EVDExistence
import Transformer.DASH.Section3_InverseRootCorrectness
import Transformer.DASH.Section3_DomainCounterexamples
import Transformer.DASH.Section3_ChainedRoots
import Transformer.DASH.Section3_FiniteChaining
import Transformer.DASH.Section3_DyadicChaining
import Transformer.DASH.Section3_MatrixDyadicChaining
import Transformer.DASH.Section3_OnlyDyadicPowers
import Transformer.DASH.Section3_Scaling
import Transformer.DASH.Section3_RowBounds
import Transformer.DASH.Section3_GuardedScaling
import Transformer.DASH.Section3_GuardedNewton
import Transformer.DASH.Section3_GuardedRootCorrectness
import Transformer.DASH.Section3_GuardedExamples
import Transformer.DASH.Section3_OperatorNorm
import Transformer.DASH.Section3_OperatorDomains
import Transformer.DASH.Section3_Rayleigh
import Transformer.DASH.Section3_RayleighEquality
import Transformer.DASH.Section3_PowerIteration
import Transformer.DASH.Section3_PowerCoordinates
import Transformer.DASH.Section3_PowerConvergence
import Transformer.DASH.Section3_MultiPowerConvergence
import Transformer.DASH.Section3_PoolProbability
import Transformer.DASH.Section3_SampledPowerConvergence
import Transformer.DASH.Section3_ResidualBounds
import Transformer.DASH.Section3_StoppingAccuracy
import Transformer.DASH.Section3_MatrixStopping
import Transformer.DASH.SectionA_Clenshaw
import Transformer.DASH.SectionA_ScalarChebyshev
import Transformer.DASH.SectionA_WeightedOrthogonality
import Transformer.DASH.SectionA_Coefficients
import Transformer.DASH.SectionA_DiscreteOrthogonality
import Transformer.DASH.SectionA_Aliasing
import Transformer.DASH.SectionA_FitReproduction
import Transformer.DASH.SectionA_FitStability
import Transformer.DASH.SectionA_FitApproximation
import Transformer.DASH.SectionA_DegreeSpan
import Transformer.DASH.SectionA_BinomialReference
import Transformer.DASH.SectionA_ReferenceError
import Transformer.DASH.SectionA_FitConvergence
import Transformer.DASH.SectionA_MatrixFitConvergence
import Transformer.DASH.SectionA_MatrixChebyshev
import Transformer.DASH.SectionA_OptimizedClenshaw
import Transformer.DASH.SectionA_Approximation
import Transformer.DASH.SectionA_ApproximationExistence
import Transformer.DASH.SectionA_DenseCoefficients
import Transformer.DASH.SectionA_MatrixApproximationExistence
import Transformer.DASH.SectionB_Dampening
import Transformer.DASH.SectionB_ThresholdRank
import Transformer.DASH.Section4_Blocking
import Transformer.DASH.Section4_BatchedNewton
import Transformer.DASH.Section4_BatchedCounts
import Transformer.DASH.Section4_BatchedFourthRoot
import Transformer.DASH.Section4_GuardedNdb
import Transformer.DASH.Section4_GuardedCn
import Transformer.DASH.Section4_BatchedClenshaw
import Transformer.DASH.Section4_BatchedFitConvergence
import Transformer.DASH.Section4_GuardedChebyshev
import Transformer.DASH.Section4_GuardedPowerCorrectness
import Transformer.DASH.Section4_StackedNewton
import Transformer.DASH.Section4_BatchedHistory
import Transformer.DASH.Section4_BatchedUpdates
import Transformer.DASH.Section4_GuardedDirection
import Transformer.DASH.Section4_GuardedShampooStep
import Transformer.DASH.Section4_Dimensions
import Transformer.DASH.Section4_ResidualLayout
import Transformer.DASH.Section4_CachedRoots
import Transformer.DASH.Section5_Rank
import Transformer.DASH.Section5_HistoryRank
