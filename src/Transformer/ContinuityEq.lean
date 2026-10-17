/-
# The linear continuity equation

A field `w : ℝ → E → E` on a finite-dimensional space, continuous, bounded
and uniformly Lipschitz in space (`GlobalFlow.IsFlowField`), transports
measures by `∂_t μ + div(w μ) = 0`.

* `ContinuityEq/Weak.lean` extends the weak formulation from smooth test
  functions with compact support to `C¹` ones;
* `ContinuityEq/Duality.lean` differentiates `s ↦ ∫ g(Φ_{s,t} x) dμ_s` along a
  weak solution, `Φ` being the flow of `w`;
* `ContinuityEq/Lagrangian.lean` shows that measures pushed along
  trajectories are weak solutions and, when `w` is `C¹` in space, that every
  weak solution is one of them;
* `ContinuityEq/Stability.lean` bounds the `W_2` distance between measures
  pushed along two flows whose fields are close.

Source: the Lagrangian proof of `c:wellposedtransformers` and `e:w2estimate`
in arXiv:2305.05465v6, §6.
-/

import Transformer.ContinuityEq.Weak
import Transformer.ContinuityEq.Duality
import Transformer.ContinuityEq.Lagrangian
import Transformer.ContinuityEq.Stability
