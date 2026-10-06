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
import Transformer.GPTMini.Sparsemax.Uniform
import Transformer.GPTMini.Sparsemax.SelfRoute
import Transformer.GPTMini.Sparsemax.Clipping
import Transformer.GPTMini.Sparsemax.Failure
import Transformer.GPTMini.Sparsemax.RoutingLoss
import Transformer.GPTMini.Sparsemax.Certificate.Results

/-!
# Sparsemax training boundaries and measured prediction certificates

Real-valued causal saturation is motivated by arXiv:2211.11052v1, §3.1.
Convex row inference does not make every outer training gradient useful.
The finite certificate is separate from floating-point PyTorch correctness.

arXiv:1602.02068v2, §2.2, Proposition 1 certifies the clipped threshold,
one-unit support window and full-support relative uniformity. The XSA
combination, arXiv:2603.09078v1, §2, is locally zero on a strict self route
above epsilon; a below-epsilon counterexample requires the norm hypothesis.

arXiv:1602.02068v2, §3.2–§3.3, supplies a corrective score-loss derivative
on a wrong saturated route with its target position given. This supervised
row result does not guarantee learning latent attention.

Derived restrictions from §2.2 and §2.5 need no routing targets: a top-two
gap below one, or a persistent QKNorm gain below one half, ensures two
active visible positions. A bounded sigmoid gain enforces this restriction.
The actual projection has a nonzero active-pair direction; an outer task
loss receives it only when its derivative distinguishes that pair. A
bounded-score counterexample retains a positive flat output loss with
two active positions, recording the limit of the score restriction.

Spanning active value differences prevents cancellation of a nonzero output
derivative; two distinct scalar values suffice for wrong squared-error outputs.
A separated assignment fits by bounded sparse transfer; collapsed values fail
the span premise. Translated, scaled basis anchors enforce the span in any
finite dimension using independent bounded active scores, retaining ordinary
values and exact inactive zeros. These are restricted row architectures.

Differentiable anchors realize exact active-pair transfers at boundaries.
Unit-key and shared-matrix decoders implement them through actual QKNorm.
Wrong squared-error outputs cannot be local minima and finite updates fit
targets. Full independent-input decoding requires context size at most input
width. Dedicated anchor channels remove that length restriction, allow
ordinary embeddings and preserve ordinary keys under smooth Q/K updates.
Repeated-token witnesses satisfy the partial-decoder premises. Fixed unit
frames and value anchors remain; shared-row cancellation is separate.

For shared rows, compatible endpoint supports and clipped QKNorm keys give
an affine score path. A target-fitting endpoint yields loss `(1-t)^2 *
initialLoss` and derivative `-2 * initialLoss`, excluding positive-error
joint stationary points even at inactive ties. A two-row example fits from
`9/128` to zero; conflicting repeated observations have minimum `1/2`.
Joint attainability, support compatibility and fixed values remain essential.

A bounded PSD Gram learns both Q/K families with exact feature recovery
and affine shared content scores. Causal score domains and squared projection
energy are convex without fixed supports; cap below one half excludes
singleton saturation. The unrestricted sparsemax graph and score energy
without its score-square term remain nonconvex. Requiring probability score
rows makes actual attention affine, including changed supports. A four-token
witness changes both Q/K families and loses one active position; midpoint
attention is the endpoint mean, but its Gram cannot keep one-feature width.
These finite-context restrictions replace QKNorm. Ordinary joint value
mixing still has a nonconvex graph, even with bounds and full support.

For one distinct-token causal context, self-weight floors give a convex
domain with invertible attention. Learn Gram and output coordinates jointly;
inverse decoding gives one common value table. Encoding and decoding are
mutually exact; actual outputs are affine and convex output criteria stay
convex. A two-token example changes Q/K, values and support. Its actual
midpoint output is the endpoint mean, unlike literal mean original values.
Sharing across contexts, small width and value penalties need separate results.

A shared dictionary handles arbitrary numbers of fixed causal data codes.
Genuine mixture-query scores give attention M times memory; a floor above
one half guarantees its inverse without triangular supports. One global
value table gives MZ outputs, a convex prediction class and convex output
criteria. Repeated-token witnesses are proved. This changes token attention
to parameter memory; output-only fitting leaves the Gram free.

Six position/token slots remove the frequency-code obstruction with a right
inverse fitting arbitrary targets, including `(0,0,1)`. Complete signatures
fit all consistent finite-window targets in the same convex chart; identical
prefixes must share outputs. Full signatures cost `(V+1)^T` slots.

A P-slot kernel on distinct observed prefixes has a proved training-code
right inverse, from identity plus constant and squared-feature Grams. One
common value table fits arbitrary prototype targets with genuine Q/K width
2P and convex joint output criteria. This affine chart needs at least R slots
for R independent target rows; Gram storage is quadratic and routes are dense.

A nearest variant gives one-hot codes and identity prototype codes in the
same chart. Masked Hamming distance ignores future query/prototype tokens.
Error is at most epsilon plus L times cover radius under explicit fit,
regularity and coverage; indistinguishable targets disprove unconditional bounds.

A compact path learns 3P-1 edge/norm coordinates in a convex linear domain.
Its affine genuine Gram permits independent Q/K norms; floor above one half
guarantees the inverse. Actual nearest queries have at most three routes,
attained by an unseen four-slot witness. Common values retain MZ outputs,
arbitrary prototype fit, conditional unseen bounds and convex joint criteria.
Possible connections and same-family orthogonality are fixed restrictions;
a noninjective forward retains output-only Gram freedom. Data count and width
still grow with observations; FFN and task-loss choice remain open.

An additional complete-coordinate squared criterion is strictly convex and
has a unique constrained minimum on the compact domain, even for an infeasible
reference. Positive weight preserves joint convexity with convex output criteria.
Every attained joint minimum uses the same selected Gram parameters, even
at different outputs. The reference supplies additional information for this choice.
-/
