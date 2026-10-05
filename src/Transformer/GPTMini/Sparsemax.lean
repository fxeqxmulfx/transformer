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
-/
