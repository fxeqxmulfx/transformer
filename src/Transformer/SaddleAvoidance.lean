/-
# Saddle avoidance

Gradient descent with a small enough step almost never converges to a strict
saddle.  Under the Łojasiewicz inequality at the critical points, almost
every bounded orbit converges, to a local minimizer when every critical point
is a local minimizer or a strict saddle.  Almost every orbit with a limit
point up to translations where the Hessian is injective converges to a local
minimizer; for a periodic function whose critical points are all
nondegenerate, almost every orbit does.  A `C²` function invariant under
translations spanning the space, such as a function of angles, has a bounded
gradient and a globally Lipschitz gradient, as these theorems require.  Under
a translation symmetry `W`, an injective Hessian on `Wᗮ` still gives the
Łojasiewicz inequality.  A gradient flow with a limit point up to
translations where the Łojasiewicz inequality holds converges.  If every
critical point is a global minimizer or a strict saddle, almost every orbit
converges to a global minimizer.  The time-one map of the gradient flow of a
`C²` function with a bounded Lipschitz gradient is `C¹` with an invertible
derivative, and every strict saddle is a symmetric unstable fixed point of it.
A compact sublevel set replaces the global bound on the Hessian: for small
steps gradient descent maps it into itself, and the starts in it converging
to a strict saddle form a null set.  Gradient flows, descending or ascending,
almost never converge to a strict saddle: at integer times they are orbits of
the time-one map.  So if every critical point is a global minimizer or a
strict saddle, almost every trajectory converges to a global minimizer.

| Module | Contents |
| --- | --- |
| `SaddleAvoidance.Defs` | the stay set, expanding splittings, unstable fixed points, gradient steps |
| `SaddleAvoidance.Basic` | orbits staying near a point; sets met once by every line of a direction |
| `SaddleAvoidance.Cone` | the cone estimate: the stay set meets such a line at most once |
| `SaddleAvoidance.Spectral` | the expanding splitting of a symmetric map |
| `SaddleAvoidance.Preimage` | preimages of null sets are null where the derivative is invertible |
| `SaddleAvoidance.Global` | the starts converging to symmetric unstable fixed points are null, also within a forward-invariant set on which the derivative is invertible |
| `SaddleAvoidance.GradientDescent` | the starts converging to strict saddles are null |
| `SaddleAvoidance.Descent` | the descent lemma, sufficient decrease, limits are critical points |
| `SaddleAvoidance.Lojasiewicz` | the Łojasiewicz inequality bounds a step by a drop |
| `SaddleAvoidance.Trap` | orbits starting near a Łojasiewicz point stay near it, with finite length |
| `SaddleAvoidance.Convergence` | orbits with a Łojasiewicz limit point converge |
| `SaddleAvoidance.Minimizers` | almost every limit is a local minimizer; almost every bounded orbit converges to one |
| `SaddleAvoidance.CriticalPoints` | limit points up to translations are critical; the Łojasiewicz inequality at nondegenerate critical points |
| `SaddleAvoidance.SecondOrder` | the second-order test; Hessians at nondegenerate critical points other than strict saddles are positive definite |
| `SaddleAvoidance.Necessary` | the gradient vanishes and the Hessian is nonnegative at a local minimizer; a critical point with a direction of negative curvature is a strict saddle |
| `SaddleAvoidance.Nondegenerate` | almost every orbit with a nondegenerate limit point converges to a local minimizer |
| `SaddleAvoidance.LinearFlow` | the linear flow `exp (τ A)`: it solves `J' = A J`, is symmetric for symmetric `A`, and has the eigenvalue `e^μ` for each eigenvalue `μ` of `A` |
| `SaddleAvoidance.Periodic` | a `C²` function invariant under a spanning set of translations has a bounded gradient, a bounded Hessian and a Lipschitz gradient |
| `SaddleAvoidance.Invariant` | the Łojasiewicz inequality at critical points nondegenerate transversally to a translation symmetry |
| `SaddleAvoidance.LocalInequality` | the local gradient inequality of `Normalization` gives the Łojasiewicz inequality; analytic functions in dimension at most one satisfy it at every critical point |
| `SaddleAvoidance.FlowTrap` | gradient flows starting near a Łojasiewicz point stay near it, with finite length |
| `SaddleAvoidance.FlowConvergence` | gradient flows with a Łojasiewicz limit point up to translations converge |
| `SaddleAvoidance.Landscape` | if every critical point is a global minimizer or a strict saddle, almost every orbit converges to a global minimizer; for periodic functions without a hypothesis on the orbits |
| `SaddleAvoidance.TimeOne` | the gradient flow: its time-`t` maps are `C¹` with invertible derivatives, equal to `exp (-t ∇²f(z))` at a critical point `z`; strict saddles are symmetric unstable fixed points of the time-one map |
| `SaddleAvoidance.Sublevel` | avoidance on a forward-invariant set; small steps keep gradient descent in a compact sublevel set, from which almost no start converges to a strict saddle |
| `SaddleAvoidance.FlowAvoidance` | the gradient flow is a one-parameter group whose integer times iterate the time-one map; gradient descent and ascent flows almost never converge to a strict saddle |
| `SaddleAvoidance.FlowLandscape` | limits of gradient flows are critical; if every critical point is a global minimizer or a strict saddle, almost every trajectory converges to a global minimizer; for periodic functions without a hypothesis on the trajectories |

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
import Transformer.SaddleAvoidance.Necessary
import Transformer.SaddleAvoidance.Nondegenerate
import Transformer.SaddleAvoidance.LinearFlow
import Transformer.SaddleAvoidance.Periodic
import Transformer.SaddleAvoidance.Invariant
import Transformer.SaddleAvoidance.LocalInequality
import Transformer.SaddleAvoidance.FlowTrap
import Transformer.SaddleAvoidance.FlowConvergence
import Transformer.SaddleAvoidance.Landscape
import Transformer.SaddleAvoidance.TimeOne
import Transformer.SaddleAvoidance.Sublevel
import Transformer.SaddleAvoidance.FlowAvoidance
import Transformer.SaddleAvoidance.FlowLandscape
