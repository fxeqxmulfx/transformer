/-
# Saddle avoidance

Gradient descent with a small enough step almost never converges to a strict
saddle.

| Module | Contents |
| --- | --- |
| `SaddleAvoidance.Defs` | the stay set, expanding splittings, unstable fixed points, gradient steps |
| `SaddleAvoidance.Basic` | orbits staying near a point; sets met once by every line of a direction |
| `SaddleAvoidance.Cone` | the cone estimate: the stay set meets such a line at most once |
| `SaddleAvoidance.Spectral` | the expanding splitting of a symmetric map |
| `SaddleAvoidance.Preimage` | preimages of null sets under a local diffeomorphism are null |
| `SaddleAvoidance.Global` | the starts converging to symmetric unstable fixed points are null |
| `SaddleAvoidance.GradientDescent` | the starts converging to strict saddles are null |

The sources are the gradient-descent corollary in §5.1 of arXiv:1710.07406
and `thm:main` of arXiv:1602.04915.  Both rest on the center-stable manifold
theorem; here a cone estimate replaces it, at the price of a symmetric
derivative at the unstable point, which gradient descent has.
-/

import Transformer.SaddleAvoidance.Defs
import Transformer.SaddleAvoidance.Basic
import Transformer.SaddleAvoidance.Cone
import Transformer.SaddleAvoidance.Spectral
import Transformer.SaddleAvoidance.Preimage
import Transformer.SaddleAvoidance.Global
import Transformer.SaddleAvoidance.GradientDescent
