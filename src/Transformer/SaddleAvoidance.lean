/-
# Saddle avoidance

Gradient descent with a small enough step almost never converges to a strict
saddle.  Under the Łojasiewicz inequality at the critical points, almost
every bounded orbit converges, to a local minimizer when every critical point
is a local minimizer or a strict saddle.  Almost every orbit with a limit
point up to translations where the Hessian is injective converges to a local
minimizer; for a periodic function whose critical points are all
nondegenerate, almost every orbit does.

| Module | Contents |
| --- | --- |
| `SaddleAvoidance.Defs` | the stay set, expanding splittings, unstable fixed points, gradient steps |
| `SaddleAvoidance.Basic` | orbits staying near a point; sets met once by every line of a direction |
| `SaddleAvoidance.Cone` | the cone estimate: the stay set meets such a line at most once |
| `SaddleAvoidance.Spectral` | the expanding splitting of a symmetric map |
| `SaddleAvoidance.Preimage` | preimages of null sets under a local diffeomorphism are null |
| `SaddleAvoidance.Global` | the starts converging to symmetric unstable fixed points are null |
| `SaddleAvoidance.GradientDescent` | the starts converging to strict saddles are null |
| `SaddleAvoidance.Descent` | the descent lemma, sufficient decrease, limits are critical points |
| `SaddleAvoidance.Lojasiewicz` | the Łojasiewicz inequality bounds a step by a drop |
| `SaddleAvoidance.Trap` | orbits starting near a Łojasiewicz point stay near it, with finite length |
| `SaddleAvoidance.Convergence` | orbits with a Łojasiewicz limit point converge |
| `SaddleAvoidance.Minimizers` | almost every limit is a local minimizer; almost every bounded orbit converges to one |
| `SaddleAvoidance.CriticalPoints` | limit points up to translations are critical; the Łojasiewicz inequality at nondegenerate critical points |
| `SaddleAvoidance.SecondOrder` | the second-order test; Hessians at nondegenerate critical points other than strict saddles are positive definite |
| `SaddleAvoidance.Nondegenerate` | almost every orbit with a nondegenerate limit point converges to a local minimizer |

The sources are the gradient-descent corollary in §5.1 of arXiv:1710.07406
and `thm:main` of arXiv:1602.04915.  Both rest on the center-stable manifold
theorem; here a cone estimate replaces it, at the price of a symmetric
derivative at the unstable point, which gradient descent has.  The
convergence follows Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim.
16, 2005), which arXiv:1602.04915 cites for `prop:rates`.
-/

import Transformer.SaddleAvoidance.Defs
import Transformer.SaddleAvoidance.Basic
import Transformer.SaddleAvoidance.Cone
import Transformer.SaddleAvoidance.Spectral
import Transformer.SaddleAvoidance.Preimage
import Transformer.SaddleAvoidance.Global
import Transformer.SaddleAvoidance.GradientDescent
import Transformer.SaddleAvoidance.Descent
import Transformer.SaddleAvoidance.Lojasiewicz
import Transformer.SaddleAvoidance.Trap
import Transformer.SaddleAvoidance.Convergence
import Transformer.SaddleAvoidance.Minimizers
import Transformer.SaddleAvoidance.CriticalPoints
import Transformer.SaddleAvoidance.SecondOrder
import Transformer.SaddleAvoidance.Nondegenerate
