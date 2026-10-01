/-
# AMSGradW: original decoupled-decay training convergence

User-requested deterministic extension of Tran and Le,
arXiv:1904.03590v4, Algorithm 1, §4 and §6, with decoupled weight decay
as discussed in Hägele et al., arXiv:2606.25971v2, §2 and §4.1.
Neither manuscript supplies the AMSGradW training theorem proved here.

## Exact update and scope

The genuine full gradient of the fixed loss updates the first moment,
second moment and coordinatewise maximum exactly as in AMSGrad.
The parameter update is
`x_next = (1 - eta*decay)*x - eta*m_next/(epsilon + sqrt(maximum_next))`.
Decay is absent from the moment gradients. All histories start at zero.
The model uses constant coefficients and learning rate, no bias correction
and no projection. No safeguard or direction replacement is added.

## Proved sufficient regime

The fixed objective is differentiable with a Lipschitz genuine gradient
in the coordinate maximum norm, with constant `L >= 0`. Parameters obey
`eta > 0`, `epsilon > 0`, `decay > 0`, `0 <= beta < 1`,
`0 <= beta2 <= 1`, `eta*decay <= 1` and `L < decay*epsilon`.
These are objective and parameter assumptions; bounded iterates,
convergent metrics, gradient alignment and a known equilibrium are
not supplied as hypotheses. The strict decay condition is sufficient
and restrictive; it is not asserted for typical training hyperparameters.

The actual weights and first moments converge. Every denominator has
a finite positive limit `D`, derived from monotone actual maximum
histories and proved state bounds. The original loss converges as well.
The limit satisfies `gradient f star + decay*D*star = 0` coordinatewise.
Existence and uniqueness of that equilibrium for the limiting `D` are
proved by contraction, including nonzero first-moment coefficients.

With a convex first-order lower model for the loss, the actual weights
converge to the unique global minimizer of
`f x + decay/2 * sum_i D_i*x_i^2`. The limiting diagonal is history
dependent, so this is not generally ordinary isotropic L2 regularization.
No geometric rate for the unfrozen adaptive run is asserted.

On `(w-1)^2/2`, with eta `1/4`, epsilon one, decay two, beta `9/10`
and beta2 `1/2`, actual AMSGradW converges to `1/(1+2D)` with `D >= 1`.
The original gradient converges to a nonzero value. This proves that
ordinary AMSGrad's unregularized stationarity conclusion does not
transfer to positive constant decoupled decay.

These are exact-real, fixed full-gradient results. Stochastic minibatches,
bias-corrected updates and floating-point implementations are not covered.
Every statement is proved without `sorry` or extra axioms.
-/

import Transformer.AMSGradW.Basic
import Transformer.AMSGradW.Bounds
import Transformer.AMSGradW.MetricLimit
import Transformer.AMSGradW.Contraction
import Transformer.AMSGradW.Error
import Transformer.AMSGradW.Perturbation
import Transformer.AMSGradW.Convergence
import Transformer.AMSGradW.Minimum
import Transformer.AMSGradW.ShiftedQuadratic
