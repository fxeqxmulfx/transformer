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
import Transformer.GPTMini.Sparsemax.Uniform
import Transformer.GPTMini.Sparsemax.SelfRoute
import Transformer.GPTMini.Sparsemax.Clipping
import Transformer.GPTMini.Sparsemax.Failure
import Transformer.GPTMini.Sparsemax.RoutingLoss
import Transformer.GPTMini.Sparsemax.Certificate.Results

/-!
# Sparsemax training boundaries and measured prediction certificates

Real-valued saturation of the repository's causal projection, motivated by
arXiv:2211.11052v1, §3.1. Convex row inference does not make every outer
training gradient useful. The finite experimental certificate is separate
from a proof of floating-point PyTorch execution.

arXiv:1602.02068v2, §2.2, Proposition 1 certifies the clipped-threshold
formula, the one-unit support window and full-support relative uniformity.
The XSA combination, arXiv:2603.09078v1, §2, is locally zero on a strict
self route above epsilon; a below-epsilon counterexample records why
that norm hypothesis is required by the implementation.

arXiv:1602.02068v2, §3.2–§3.3, supplies a score loss with a corrective
derivative on a wrong saturated route when its target position is given.
That supervised row result does not guarantee learning latent attention.

Derived restrictions from §2.2 and §2.5 need no routing targets: a top-two
gap below one, or a persistent QKNorm gain below one half, ensures two
active visible positions. A bounded sigmoid gain enforces this restriction.
The actual projection has a nonzero active-pair direction; an outer task
loss receives it only when its derivative distinguishes that pair. A
bounded-score counterexample retains a positive flat output loss with
two active positions, recording the limit of the score restriction.

A full span of active value differences prevents cancellation of a nonzero
output derivative. Two distinct active scalar values suffice for wrong
squared-error outputs. A separated assignment reaches zero error by bounded
sparse transfer, while collapsed values fail the span premise. Translated,
scaled basis anchors enforce the span in any finite output dimension through
independent bounded active scores, retaining arbitrary ordinary values and
exact inactive zeros. These are restricted row architectures.

The differentiable anchor chart realizes exact active-pair transfers at
support boundaries. A unit-key chart implements them through actual QKNorm.
An independent-input decoder transfers them to shared Q/K matrices while
retaining unseen directions. Wrong squared-error outputs cannot be local
minima, and a finite update fits the target. Full independence requires
context size at most input width; fixed frames and anchors remain explicit.

A partial decoder controls only anchors and kills ordinary inputs. Its
dedicated channels permit arbitrary ordinary embeddings and context lengths.
A smooth shared-matrix update preserves ordinary keys while realizing
anchored scores. Nonzero task derivatives reach joint Q/K matrices, and
wrong squared-error outputs remain excluded as local minima. Repeated-token
examples satisfy these premises while failing full input independence.
Unit frames and visible value anchors remain; shared-row cancellation is separate.

Summed squared loss uses shared matrices across rows and examples. Matching
endpoint supports make sparsemax affine on a score segment; clipped endpoint
keys keep actual QKNorm linear throughout it. A better compatible endpoint
excludes a joint local minimum. A shared target-fitting endpoint gives path
loss `(1-t)^2 * initialLoss` and derivative `-2 * initialLoss`, excluding
positive-error joint stationary points, including inactive threshold ties.
A two-row example fits distinct targets from loss `9/128` to zero. Repeated
observations with conflicting targets instead have global minimum `1/2`.
Joint attainability, support compatibility, clipped keys and fixed values
remain essential; unconstrained whole-model convergence is not asserted.

A new Gram architecture learns both query and key embedding families in
one bounded PSD matrix, with exact finite-coordinate recovery and affine
shared content scores. The joint causal-row domain and squared projection
energy are convex without fixing supports. An entry cap below one half
excludes singleton saturation. The unrestricted exact sparsemax graph is
still nonconvex, and dropping the score-square energy term also fails.
A stronger linear restriction makes each selected score row itself a
causal probability row. Actual sparsemax fixes it, so the exact learned
embedding/attention graph is convex even when supports change. A four-token
example changes both embedding families and loses one active position;
its midpoint attention is exactly the mean endpoint attention. Its midpoint
Gram cannot retain the endpoints' one-feature width. These are new finite-
context architectures replacing QKNorm, with no task loss or FFN yet.
Ordinary jointly learned value mixing has a separate nonconvex output graph,
even with bounded parameters and full support in its original coordinates.

For one distinct-token context with all causal rows, a positive self-weight
floor gives a convex structural domain with invertible actual attention.
Learn the Gram and output table together; decode one shared value table
by the attention inverse. Encoding and decoding are proved mutually exact,
and the actual attention/value output is affine in these joint coordinates.
Any future convex output objective remains convex; no task loss is selected.
A two-token example changes both Q/K families, values and sparse support.
The true midpoint output is the mean endpoint output; literal mean values
produce a different output. This finite-context chart leaves multi-context
sharing, fixed small width, value penalties and inverse conditioning open.

A shared learned dictionary now handles arbitrarily many causal data codes.
Their genuine Q/K mixture scores give actual attention M times the memory.
A convex diagonal floor above one half guarantees its inverse without
triangular supports. One global decoded value table gives M times Z for
all contexts; the exact prediction class and any convex output objective
remain convex. Repeated-token prefix codes and a three-context witness are
proved. This is memory attention with fixed data codes; ordinary token
self-attention is changed, and output-only training leaves the Gram free.

The frequency-code three-target obstruction is removed by richer data
features. Six shared position/token slots give a proved right inverse for
the same three prefixes, permitting arbitrary vector outputs and the
previously excluded triple `(0,0,1)`. Complete causal signatures realize
exactly all observation-consistent target tables for a finite window.
Identical visible prefixes must share outputs; different prefixes impose
no extra restriction. Both constructions retain the convex joint chart
for every feasible learned Gram. Full signatures cost `(V+1)^T` slots.

A compact kernel construction registers P distinct observed prefixes, with
P learned memory slots. Its data kernel is identity plus a constant Gram
and a squared-feature Gram, so normalized training codes have a proved
right inverse. One common decoded value table fits arbitrary vector targets
on these prototypes for every feasible learned Gram. Genuine learned Q/K
embeddings need width at most 2P, without a nonconvex rank restriction.
Actual forwards are causal normalized kernel sums, and any convex output
criterion remains convex jointly. This affine chart requires at least R
slots for R independent vector target rows; the construction attains that
bound. Gram parameters still grow quadratically, actual query routes are
dense, fixed data codes remain, and output-only training leaves the Gram
undetermined. Finite interpolation does not prove text generalization or
sample-independent compactness. FFN and task-loss selection stay open.
-/
