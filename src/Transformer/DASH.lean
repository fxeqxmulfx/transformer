module

public import Transformer.DASH.Section2_Models
public import Transformer.DASH.Section2_TrainingConvergence
public import Transformer.DASH.Section2_TrainingScalarRoot
public import Transformer.DASH.Section2_TrainingCycleStep
public import Transformer.DASH.Section2_TrainingCounterexample
public import Transformer.DASH.Section2_History
public import Transformer.DASH.Section2_LoadBalancing
public import Transformer.DASH.Section2_GraftingDomain
public import Transformer.DASH.Section2_GraftingContinuity
public import Transformer.DASH.Section3_CoupledNewton
public import Transformer.DASH.Section3_CNConvergence
public import Transformer.DASH.Section3_CNUniformConvergence
public import Transformer.DASH.Section3_CNUnitScaling
public import Transformer.DASH.Section3_Spectral
public import Transformer.DASH.Section3_NewtonModels
public import Transformer.DASH.Section3_OperationCounts
public import Transformer.DASH.Section3_NewtonSpectra
public import Transformer.DASH.Section3_MatrixConvergence
public import Transformer.DASH.Section3_EVDExistence
public import Transformer.DASH.Section3_InverseRootCorrectness
public import Transformer.DASH.Section3_DomainCounterexamples
public import Transformer.DASH.Section3_ChainedRoots
public import Transformer.DASH.Section3_FiniteChaining
public import Transformer.DASH.Section3_DyadicChaining
public import Transformer.DASH.Section3_MatrixDyadicChaining
public import Transformer.DASH.Section3_OnlyDyadicPowers
public import Transformer.DASH.Section3_Scaling
public import Transformer.DASH.Section3_RowBounds
public import Transformer.DASH.Section3_GuardedScaling
public import Transformer.DASH.Section3_GuardedNewton
public import Transformer.DASH.Section3_GuardedRootCorrectness
public import Transformer.DASH.Section3_GuardedExamples
public import Transformer.DASH.Section3_OperatorNorm
public import Transformer.DASH.Section3_OperatorDomains
public import Transformer.DASH.Section3_Rayleigh
public import Transformer.DASH.Section3_RayleighEquality
public import Transformer.DASH.Section3_PowerIteration
public import Transformer.DASH.Section3_PowerCoordinates
public import Transformer.DASH.Section3_PowerConvergence
public import Transformer.DASH.Section3_MultiPowerConvergence
public import Transformer.DASH.Section3_PoolProbability
public import Transformer.DASH.Section3_SampledPowerConvergence
public import Transformer.DASH.Section3_ResidualBounds
public import Transformer.DASH.Section3_StoppingAccuracy
public import Transformer.DASH.Section3_MatrixStopping
public import Transformer.DASH.SectionA_Clenshaw
public import Transformer.DASH.SectionA_ScalarChebyshev
public import Transformer.DASH.SectionA_WeightedOrthogonality
public import Transformer.DASH.SectionA_Coefficients
public import Transformer.DASH.SectionA_DiscreteOrthogonality
public import Transformer.DASH.SectionA_Aliasing
public import Transformer.DASH.SectionA_FitReproduction
public import Transformer.DASH.SectionA_FitStability
public import Transformer.DASH.SectionA_FitApproximation
public import Transformer.DASH.SectionA_DegreeSpan
public import Transformer.DASH.SectionA_BinomialReference
public import Transformer.DASH.SectionA_ReferenceError
public import Transformer.DASH.SectionA_FitConvergence
public import Transformer.DASH.SectionA_MatrixFitConvergence
public import Transformer.DASH.SectionA_MatrixChebyshev
public import Transformer.DASH.SectionA_OptimizedClenshaw
public import Transformer.DASH.SectionA_Approximation
public import Transformer.DASH.SectionA_ApproximationExistence
public import Transformer.DASH.SectionA_DenseCoefficients
public import Transformer.DASH.SectionA_MatrixApproximationExistence
public import Transformer.DASH.SectionB_Dampening
public import Transformer.DASH.SectionB_ThresholdRank
public import Transformer.DASH.Section4_Blocking
public import Transformer.DASH.Section4_BatchedNewton
public import Transformer.DASH.Section4_BatchedCounts
public import Transformer.DASH.Section4_BatchedFourthRoot
public import Transformer.DASH.Section4_GuardedNdb
public import Transformer.DASH.Section4_GuardedCn
public import Transformer.DASH.Section4_BatchedClenshaw
public import Transformer.DASH.Section4_BatchedFitConvergence
public import Transformer.DASH.Section4_GuardedChebyshev
public import Transformer.DASH.Section4_GuardedPowerCorrectness
public import Transformer.DASH.Section4_StackedNewton
public import Transformer.DASH.Section4_BatchedHistory
public import Transformer.DASH.Section4_BatchedUpdates
public import Transformer.DASH.Section4_GuardedDirection
public import Transformer.DASH.Section4_GuardedShampooStep
public import Transformer.DASH.Section4_Dimensions
public import Transformer.DASH.Section4_ResidualLayout
public import Transformer.DASH.Section4_CachedRoots
public import Transformer.DASH.Section5_Rank
public import Transformer.DASH.Section5_HistoryRank

/-!
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

@[expose] public section
