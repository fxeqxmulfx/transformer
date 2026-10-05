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

A full span of active value differences prevents cancellation of any
nonzero output derivative. For ordinary scalar squared error, two distinct
active values suffice whenever the output is wrong. A constructed separated
value assignment reaches zero error by a bounded sparse score transfer;
the previous collapsed value assignment is proved to fail the span premise.
The anchor construction enforces the span in any finite output dimension:
prepend a translated, positively scaled basis and keep its scores active
through independent bounded coordinates. The span persists for every
finite parameter assignment, with arbitrary ordinary values and exact
inactive zeros. This is a new row architecture, not a guarantee for the
existing query/key parameterization or for a zero output derivative.

The differentiable inverse of the anchor score chart realizes exact
active-pair transfers through learned parameters. Nonzero ordinary task
derivatives therefore survive in trainable anchor directions. For squared
output error, every wrong finite output fails to be a local minimum of
this row loss, even at support boundaries. A concrete finite parameter
step reaches zero error while retaining the third weight at zero.
These row results do not assert whole-model or shared-row convergence.

A differentiable unit-key chart now implements the anchored scores
through the actual QKNorm inner product. Its gain dominates every finite
ordinary score, while all keys have norm one. Nonzero ordinary task
derivatives reach actual K vectors, and wrong squared-error outputs
cannot be local minima in those vectors. For any linearly independent
input family, the same directions reach jointly trained shared Q/K
projection matrices. A proved continuous linear decoder and affine update
realize arbitrary key changes starting at the actual current matrix,
retaining its action on directions unseen by the decoder.
For ordinary squared error, wrong outputs are not local minima of the
joint projection loss either. A finite change of the actual key matrix
reaches zero loss in both standard and nonstandard sparse scalar examples.
The full-family independence condition requires context size at most input
width; a proved obstruction records the failure above that width.
The fixed unit frame, restricted key parameterization and independent
input family are explicit restrictions. They do not follow
from unconstrained QKNorm or arbitrary learned embeddings.

Full input independence is unnecessary when only anchors need control.
A partial decoder isolates the anchors and kills ordinary inputs. Its
dedicated-channel implementation permits arbitrary ordinary embeddings
and context lengths. A smooth anchor-only update through the current
shared matrix preserves all ordinary keys and realizes the anchored scores.
Nonzero ordinary task derivatives still reach joint Q/K matrices, and
wrong squared-error outputs remain excluded as local minima. A concrete
family with arbitrarily many repeated ordinary tokens satisfies the
weaker premises while long instances fail full input independence.
The unit-frame, key-family and visible value-anchor restrictions remain;
these single-row results do not exclude cancellation in a shared objective.

The summed ordinary squared loss now uses actual shared matrices for
arbitrarily many rows and examples. Matching endpoint sparse supports
make the actual projection affine on a score segment. A new explicit
restriction keeps projected endpoint keys in the epsilon ball, where
actual QKNorm is linear; the common matrix segment stays in that ball.
No independent queries, input decoder or row-specific parameters are used.
A better compatible endpoint excludes a joint local minimum of the sum.
If one compatible shared matrix fits all ordinary targets, the actual
path loss is `(1-t)^2 * initialLoss`, with right derivative `-2 * initialLoss`.
At positive error, a zero full joint Q/K derivative is impossible, including
inactive threshold ties. A two-row sparse example has opposite initial
errors and distinct targets fitted by one shared matrix: loss `9/128` to zero.
An actual counterexample refutes unconditional transfer: repeated observations
with conflicting targets zero and one have positive global minimum `1/2`,
two active anchors and exact ordinary zeros. Joint attainability, endpoint
support compatibility, clipped keys and fixed values remain essential;
these results do not assert convergence of an unconstrained full model.
-/
