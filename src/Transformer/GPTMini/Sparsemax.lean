import Transformer.GPTMini.Sparsemax.Basic
import Transformer.GPTMini.Sparsemax.ClosedForm
import Transformer.GPTMini.Sparsemax.SupportWindow
import Transformer.GPTMini.Sparsemax.NonSaturation
import Transformer.GPTMini.Sparsemax.ActiveDirection
import Transformer.GPTMini.Sparsemax.OuterSensitivity
import Transformer.GPTMini.Sparsemax.ValueSpan
import Transformer.GPTMini.Sparsemax.SeparatedValues
import Transformer.GPTMini.Sparsemax.ValuePlateau
import Transformer.GPTMini.Sparsemax.BoundedGain
import Transformer.GPTMini.Sparsemax.BoundedCoordinates
import Transformer.GPTMini.Sparsemax.AnchoredScores
import Transformer.GPTMini.Sparsemax.AnchoredValues
import Transformer.GPTMini.Sparsemax.AnchorTransfer
import Transformer.GPTMini.Sparsemax.TrainableAnchors
import Transformer.GPTMini.Sparsemax.AnchoredSquaredError
import Transformer.GPTMini.Sparsemax.AnchoredCorrection
import Transformer.GPTMini.Sparsemax.QKChart
import Transformer.GPTMini.Sparsemax.QKAnchors
import Transformer.GPTMini.Sparsemax.QKTaskDirections
import Transformer.GPTMini.Sparsemax.QKSquaredError
import Transformer.GPTMini.Sparsemax.QKProjection
import Transformer.GPTMini.Sparsemax.QKProjectedError
import Transformer.GPTMini.Sparsemax.InputDecoder
import Transformer.GPTMini.Sparsemax.ProjectionLift
import Transformer.GPTMini.Sparsemax.ProjectionUpdate
import Transformer.GPTMini.Sparsemax.MixedInputQK
import Transformer.GPTMini.Sparsemax.QKIndependentDirections
import Transformer.GPTMini.Sparsemax.QKIndependentError
import Transformer.GPTMini.Sparsemax.PrefixInputs
import Transformer.GPTMini.Sparsemax.LongContextQK
import Transformer.GPTMini.Sparsemax.QKPrefixMatrix
import Transformer.GPTMini.Sparsemax.QKPrefixDirections
import Transformer.GPTMini.Sparsemax.QKPrefixError
import Transformer.GPTMini.Sparsemax.SupportSegment
import Transformer.GPTMini.Sparsemax.ClippedKeySegment
import Transformer.GPTMini.Sparsemax.SharedRowsExample
import Transformer.GPTMini.Sparsemax.SharedRows
import Transformer.GPTMini.Sparsemax.SquaredSegment
import Transformer.GPTMini.Sparsemax.SharedRowLoss
import Transformer.GPTMini.Sparsemax.SharedRowMinimum
import Transformer.GPTMini.Sparsemax.SharedRowDerivative
import Transformer.GPTMini.Sparsemax.SharedRowCancellation
import Transformer.GPTMini.Sparsemax.EmbeddingGram
import Transformer.GPTMini.Sparsemax.GramRouting
import Transformer.GPTMini.Sparsemax.GramRoutingEnergy
import Transformer.GPTMini.Sparsemax.GramSupport
import Transformer.GPTMini.Sparsemax.GramBoundary
import Transformer.GPTMini.Sparsemax.NormalizedGram
import Transformer.GPTMini.Sparsemax.NormalizedGramExamples
import Transformer.GPTMini.Sparsemax.InvertibleGram
import Transformer.GPTMini.Sparsemax.GramValues
import Transformer.GPTMini.Sparsemax.JointGramValues
import Transformer.GPTMini.Sparsemax.GramValueExampleGrams
import Transformer.GPTMini.Sparsemax.JointGramValueExamples
import Transformer.GPTMini.Sparsemax.MemoryGram
import Transformer.GPTMini.Sparsemax.MemoryValues
import Transformer.GPTMini.Sparsemax.ContextMemory
import Transformer.GPTMini.Sparsemax.PrefixMemoryCodes
import Transformer.GPTMini.Sparsemax.SharedMemoryValues
import Transformer.GPTMini.Sparsemax.SharedMemoryGeometry
import Transformer.GPTMini.Sparsemax.MemoryExampleGrams
import Transformer.GPTMini.Sparsemax.SharedMemoryExamples
import Transformer.GPTMini.Sparsemax.CodeMemoryOutputs
import Transformer.GPTMini.Sparsemax.CausalPrefixKeys
import Transformer.GPTMini.Sparsemax.CausalMemoryUniversality
import Transformer.GPTMini.Sparsemax.PrefixFeatureCodes
import Transformer.GPTMini.Sparsemax.PositionalMemoryTargets
import Transformer.GPTMini.Sparsemax.BoundedGramWidth
import Transformer.GPTMini.Sparsemax.PrototypeKernelCodes
import Transformer.GPTMini.Sparsemax.PrototypeKernelFeatures
import Transformer.GPTMini.Sparsemax.PrototypeKernelTraining
import Transformer.GPTMini.Sparsemax.PrototypePrefixMemory
import Transformer.GPTMini.Sparsemax.PrototypeKernelGeometry
import Transformer.GPTMini.Sparsemax.NearestPrototypeCodes
import Transformer.GPTMini.Sparsemax.PrefixNearestCodes
import Transformer.GPTMini.Sparsemax.NearestMemoryGeneralization
import Transformer.GPTMini.Sparsemax.PermutationMemoryGram
import Transformer.GPTMini.Sparsemax.LocalMemoryWeights
import Transformer.GPTMini.Sparsemax.LocalMemoryCore
import Transformer.GPTMini.Sparsemax.LocalMemoryParameters
import Transformer.GPTMini.Sparsemax.LocalMemoryFeasibility
import Transformer.GPTMini.Sparsemax.LocalMemorySupport
import Transformer.GPTMini.Sparsemax.LocalJointMemory
import Transformer.GPTMini.Sparsemax.LocalNearestMemory
import Transformer.GPTMini.Sparsemax.LocalMemoryExamples
import Transformer.GPTMini.Sparsemax.LocalMemoryQuadratic
import Transformer.GPTMini.Sparsemax.LocalMemorySelection
import Transformer.GPTMini.Sparsemax.LocalRegularizedMemory
import Transformer.GPTMini.Sparsemax.BipartiteMemoryGram
import Transformer.GPTMini.Sparsemax.LocalIncidentWeights
import Transformer.GPTMini.Sparsemax.LocalMemoryScoreIdentities
import Transformer.GPTMini.Sparsemax.IncidentMemoryCore
import Transformer.GPTMini.Sparsemax.IncidentMemoryParameters
import Transformer.GPTMini.Sparsemax.IncidentMemoryFeasibility
import Transformer.GPTMini.Sparsemax.IncidentJointMemory
import Transformer.GPTMini.Sparsemax.IncidentNearestMemory
import Transformer.GPTMini.Sparsemax.Uniform
import Transformer.GPTMini.Sparsemax.SelfRoute
import Transformer.GPTMini.Sparsemax.Clipping
import Transformer.GPTMini.Sparsemax.Failure
import Transformer.GPTMini.Sparsemax.RoutingLoss
import Transformer.GPTMini.Sparsemax.Certificate.Results

/-!
# Sparsemax training boundaries and convex learned-memory constructions

Real-valued causal saturation is motivated by arXiv:2211.11052v1, §3.1.
Convex row inference does not make every outer training gradient useful.
The finite certificate is separate from floating-point PyTorch correctness.

arXiv:1602.02068v2, §2.2, Proposition 1 certifies clipped thresholds,
one-unit support windows and relative uniformity. The XSA combination,
arXiv:2603.09078v1, §2, is locally zero above epsilon on strict self routes;
a below-epsilon counterexample needs its explicit norm hypothesis.
The sparsemax paper's §3.2–§3.3 gives corrective supervised score gradients
with target positions supplied, without proving latent attention learning.

Derived bounded gaps and QKNorm gains ensure multiple active positions.
Nonzero active-pair directions reach an outer loss only if its derivative
distinguishes their values. Separated or spanning active value differences
prevent cancellation. Bounded anchors implement sparse transfers and correct
wrong squared outputs. Dedicated anchor channels remove the context-length
restriction for independent input directions. Fixed frames and value anchors
remain explicit assumptions; shared-row cancellation is a separate boundary.

Compatible shared-row endpoints with fixed values give an affine attention
path and loss `(1-t)^2 * initialLoss`. A fitting endpoint excludes positive
joint stationary points even at inactive ties. Repeated conflicting inputs
have a nonzero attainable minimum. Joint attainability is essential.

A bounded PSD Gram learns both Q/K families with exact feature recovery
and affine content scores. Score domains and squared projection energy are
convex; unrestricted sparsemax graphs and ordinary joint value mixtures are
nonconvex. Probability input scores make actual attention affine, including
changed supports. Genuine examples change both Q/K and support. Small fixed
rank may fail at their midpoint, so rank is not silently constrained.

For one causal context, self-weight floors yield a convex inverse domain.
Learn Gram and output coordinates jointly; one common value table is decoded
by the actual attention inverse. Outputs are affine and convex output
criteria stay convex. Sharing across contexts needs a data encoder.

A shared dictionary handles arbitrary fixed probability data codes. Genuine
mixture queries give attention M times memory; a floor above one half ensures
an inverse without triangular support restrictions. One global common value
table gives MZ outputs. Output-only fitting leaves the Gram unidentified.
Distinct complete causal signatures fit consistent arbitrary finite targets,
while identical prefixes must share outputs. Full signatures cost `(V+1)^T` slots.

A kernel on P observed distinct prefixes has a training-code right inverse,
from identity plus constant and squared-feature Grams. It fits arbitrary
prototype targets with genuine width 2P and convex joint output criteria.
This chart requires at least R slots for R independent target rows; its free
Gram has quadratic storage and dense routes.

Nearest observed-prefix codes are one-hot and identity on registered data.
Masked Hamming distance ignores future query/prototype tokens. Unseen error
is at most epsilon plus L times cover radius under explicit fit, regularity
and coverage; indistinguishable target functions refute unconditional bounds.

A compact path learns 3P-1 edge/norm coordinates. Its affine genuine Gram
permits independent Q/K norms; strict floors ensure the inverse. Actual
nearest queries have at most three routes, including unseen witnesses.
Shared values retain MZ outputs, arbitrary prototype fitting, conditional
unseen bounds and convex joint criteria. Possible connections and same-family
orthogonality are fixed. Width and prototype count grow; FFN is deferred.

An additional complete-coordinate squared criterion is strictly convex and
has one constrained minimum, even for an infeasible reference. Positive
weight preserves joint convexity for convex output criteria. Every attained
joint minimum selects the same Gram parameters, even at different outputs.
The reference supplies information beyond the unidentified output-only loss.

Separate incident-edge budgets enlarge the compact domain beyond its former
global budget. Explicit nonnegative score-weighted outer products prove PSD
without requiring the global identity coefficient to stay nonnegative.
The same affine Gram, variable Q/K norms, actual three-route support, inverse,
convex joint output criteria and conditional causal generalization survive.
The enlarged compact domain also has a unique squared-criterion minimum.
-/
