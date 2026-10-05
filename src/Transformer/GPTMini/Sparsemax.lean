import Transformer.GPTMini.Sparsemax.Basic
import Transformer.GPTMini.Sparsemax.ClosedForm
import Transformer.GPTMini.Sparsemax.SupportWindow
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
-/
