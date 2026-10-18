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
converges to a global minimizer.  This hypothesis on the critical points
follows from the form in which papers state a landscape: every critical
point with a positive semidefinite Hessian is a minimizer, global or local,
everywhere or below a level.  The time-one map of the gradient flow of a
`C²` function with a bounded Lipschitz gradient is `C¹` with an invertible
derivative, and every strict saddle is a symmetric unstable fixed point of it.
A compact sublevel set replaces the global bound on the Hessian: for small
steps gradient descent maps it into itself, and the starts in it converging
to a strict saddle form a null set.  If every critical point in it is a local
minimizer or a strict saddle and satisfies the Łojasiewicz inequality, almost
every start in it converges to a local minimizer.  Gradient flows, descending or ascending,
almost never converge to a strict saddle: at integer times they are orbits of
the time-one map.  So if every critical point is a global minimizer or a
strict saddle, almost every trajectory converges to a global minimizer.  The
avoidance theorem extends to fixed points whose derivative is conjugate to a
symmetric map, and with it to the flows `ẋ = -G(x) ∇f(x)` with `G` symmetric
positive definite at the strict saddles.  A cutoff localizes it: for `f` of
class `C²` and `G` of class `C¹` on an open set, `G` symmetric positive definite
there, almost no solution of `ẋ = -G(x) ∇f(x)` that stays in the set converges
to a strict saddle in it.  The trap and the convergence of
gradient flows hold for every curve along which `f` decreases at least at a
rate `c₁ ‖∇f‖²` while it moves at most at the speed `c₂ ‖∇f‖`, among them the
flows `ẋ = -G(x) ∇f(x)` with `c₁ ≤ G ≤ c₂`.  An analytic function on a
finite-dimensional real inner product space satisfies the Łojasiewicz
inequality at every point, so for analytic energies that hypothesis is dropped:
an orbit of gradient descent or a trajectory of the gradient flow with a limit
point up to translations converges to a critical point, every one does when
the translations span the space, and so does every bounded one.
At a limit where the Łojasiewicz inequality holds with exponent `a`, orbits
of gradient descent and descent curves converge geometrically for `a ≤ 1/2`,
and as `k ^ (-(1 - a)/(2a - 1))` for `1/2 < a < 1`, in `f` and in norm.
A linear isometry preserving `f` commutes with gradient steps and gradient
flows.  The subspace fixed by a set of such isometries is invariant, and on it
gradient descent is that of the restriction of `f`.  Most landscape theorems
describe only the critical points below a level `c`.  Gradient descent and
gradient flows do not raise `f`, so for the starts below `c` the landscape
hypothesis and the Łojasiewicz inequality are needed only below `c`, and for
analytic `f` the latter holds anyway.  A run can check a bound on the Hessian
only along its own orbit, and that is enough: on the segment of a step it gives
the sufficient decrease, and the starts whose orbit has it at its points and
converges to a strict saddle form a null set.  Nor does the Łojasiewicz
convergence need a global Lipschitz constant: it needs only that every step of
the orbit lowers `f` by at least `η κ ‖∇f‖²` for some `κ > 0`.  So almost
every start below a level whose orbit has the bound on its segments converges to
a critical point that is not a strict saddle, or runs off to infinity; for analytic
`f` without the Łojasiewicz hypothesis and without a level.  The rates too need
only the decrease of every step, and a Hessian bound on the segments of the orbit
gives them with `κ = 1 - η L`.

| Module | Contents |
| --- | --- |
| `SaddleAvoidance.Defs` | the stay set, expanding splittings, unstable fixed points, gradient steps |
| `SaddleAvoidance.Basic` | orbits staying near a point; sets met once by every line of a direction |
| `SaddleAvoidance.Cone` | the cone estimate: the stay set meets such a line at most once |
| `SaddleAvoidance.Spectral` | the expanding splitting of a symmetric map |
| `SaddleAvoidance.Preimage` | preimages of null sets are null where the derivative is invertible |
| `SaddleAvoidance.Global` | the starts converging to symmetric unstable fixed points are null, also within a forward-invariant set on which the derivative is invertible |
| `SaddleAvoidance.GradientDescent` | the starts converging to strict saddles are null |
| `SaddleAvoidance.Descent` | the descent lemma, sufficient decrease at every step of an orbit (`iterate_succ_le_sub`), `f` decreases along an orbit whose steps lower it by `η κ ‖∇f‖²` (`antitone_iterate_of_le_sub`), limits are critical points |
| `SaddleAvoidance.Sequences` | sequences without gradients: a weighted AM-GM inequality, steps bounded by drops add up and trap a sequence near a point, the values of a decreasing sequence near a value it comes close to, limit points up to translations, and a bound along a sequence holds at them |
| `SaddleAvoidance.Lojasiewicz` | the Łojasiewicz inequality bounds a step lowering `f` by `η κ ‖∇f‖²` by a drop (`mul_norm_gradientStep_sub_le_of_le_sub`); it holds with every larger exponent (`IsLojasiewiczAt.mono`) |
| `SaddleAvoidance.Trap` | orbits starting near a Łojasiewicz point stay near it, with finite length, also when only a decrease along the orbit is known (`mul_norm_iterate_succ_sub_le_of_le_sub`, `iterate_mem_ball_of_le_sub`, `mul_norm_iterate_sub_le_of_le_sub`) |
| `SaddleAvoidance.Convergence` | orbits with a Łojasiewicz limit point converge, also when only a decrease along the orbit is known (`exists_tendsto_of_isLojasiewiczAt_of_le_sub`, `tendsto_of_isLojasiewiczAt_of_le_sub`) |
| `SaddleAvoidance.Decay` | sequences dropping by a power of themselves decay geometrically or polynomially |
| `SaddleAvoidance.RateEstimates` | the estimates behind the rates of gradient descent at a Łojasiewicz limit, along an orbit whose steps lower `f` by `η κ ‖∇f‖²` (`exists_rate_estimates_of_isLojasiewiczAt_of_le_sub`) |
| `SaddleAvoidance.Rates` | rates of gradient descent at a Łojasiewicz limit: geometric for `a ≤ 1/2` (`exists_le_mul_pow_of_isLojasiewiczAt`), `(k + 1) ^ (-(1 - a)/(2a - 1))` for `1/2 < a < 1` (`exists_le_mul_rpow_of_isLojasiewiczAt`), also when only a decrease along the orbit is known (`exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub`, `exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub`) |
| `SaddleAvoidance.Minimizers` | almost every limit is a local minimizer; almost every bounded orbit converges to one |
| `SaddleAvoidance.CriticalPoints` | limit points up to translations are critical, also when only a decrease along the orbit is known (`gradient_eq_zero_of_frequently_of_le_sub`); the Łojasiewicz inequality at nondegenerate critical points |
| `SaddleAvoidance.SecondOrder` | the second-order test; Hessians at nondegenerate critical points other than strict saddles are positive definite |
| `SaddleAvoidance.Necessary` | the gradient vanishes and the Hessian is nonnegative at a local minimizer; a critical point with a direction of negative curvature is a strict saddle |
| `SaddleAvoidance.SecondOrderCritical` | strict saddles are the critical points with a direction of negative curvature; the landscape hypothesis from its second-order form, for global and local minimizers, also below a level |
| `SaddleAvoidance.Nondegenerate` | almost every orbit with a nondegenerate limit point converges to a local minimizer |
| `SaddleAvoidance.LinearFlow` | the linear flow `exp (τ A)`: it solves `J' = A J`, is symmetric for symmetric `A`, and has the eigenvalue `e^μ` for each eigenvalue `μ` of `A` |
| `SaddleAvoidance.Periodic` | a `C²` function invariant under a spanning set of translations has a bounded gradient, a bounded Hessian and a Lipschitz gradient; an invariant `C¹` map is bounded and Lipschitz |
| `SaddleAvoidance.Invariant` | the Łojasiewicz inequality at critical points nondegenerate transversally to a translation symmetry |
| `SaddleAvoidance.LocalInequality` | the local gradient inequality of `Lojasiewicz` gives the Łojasiewicz inequality; it depends only on the germ of the function (`isLojasiewiczAt_congr`) and is invariant under linear isometries (`isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff`); analytic functions on a finite-dimensional real inner product space satisfy it at every point (`exists_isLojasiewiczAt_of_analyticAt`, `exists_isLojasiewiczAt_of_analyticOnNhd`) |
| `SaddleAvoidance.FlowTrap` | gradient flows starting near a Łojasiewicz point stay near it, with finite length |
| `SaddleAvoidance.FlowConvergence` | gradient flows with a Łojasiewicz limit point up to translations converge |
| `SaddleAvoidance.Landscape` | if every critical point is a global minimizer or a strict saddle, almost every orbit converges to a global minimizer; for periodic functions without a hypothesis on the orbits |
| `SaddleAvoidance.TimeOne` | the gradient flow: its time-`t` maps are `C¹` with invertible derivatives, equal to `exp (-t ∇²f(z))` at a critical point `z`; strict saddles are symmetric unstable fixed points of the time-one map |
| `SaddleAvoidance.Sublevel` | avoidance on a forward-invariant set; small steps keep gradient descent in a compact sublevel set, from which almost no start converges to a strict saddle |
| `SaddleAvoidance.FlowAvoidance` | the gradient flow is a one-parameter group whose integer times iterate the time-one map; gradient descent and ascent flows almost never converge to a strict saddle |
| `SaddleAvoidance.FlowLandscape` | limits of gradient flows are critical; if every critical point is a global minimizer or a strict saddle, almost every trajectory converges to a global minimizer; for periodic functions without a hypothesis on the trajectories |
| `SaddleAvoidance.Symmetrizable` | fixed points whose derivative is conjugate to a symmetric map: their stay sets are null, and so are the starts converging to them |
| `SaddleAvoidance.SquareRoot` | the square root of a positive definite map `P`; `P H` for symmetric `H` is conjugate to a symmetric map, with a negative eigenvalue if `H` has one |
| `SaddleAvoidance.MetricFlow` | the flow `ẋ = -G(x) ∇f(x)`: its time-`t` maps are `C¹` with invertible derivatives, equal to `exp (-t G(z) ∇²f(z))` at a critical point `z`; strict saddles are symmetrizable unstable fixed points of the time-one map |
| `SaddleAvoidance.MetricAvoidance` | the flow `ẋ = -G(x) ∇f(x)` is a one-parameter group and almost never converges to a strict saddle |
| `SaddleAvoidance.Cutoff` | smooth cutoffs equal to `1` near a compact subset of an open set; near a compact set a `C²` function agrees with one whose gradient is Lipschitz; countably many compact subsets catching every compact subset of an open set; strict saddles depend only on the germ of the function |
| `SaddleAvoidance.OpenAvoidance` | for `f` and `G` given on an open set, almost no solution of `ẋ = -G(x) ∇f(x)` that stays in a compact subset converges to a strict saddle, nor one that stays in the open set to a strict saddle in it |
| `SaddleAvoidance.SublevelConvergence` | orbits from a compact set that gradient descent maps into itself depend only on `f` near it, and converge for small steps under the Łojasiewicz inequality; from a compact sublevel set almost every orbit converges to a local minimizer for small steps, without a global Lipschitz constant, and for analytic `f` without the Łojasiewicz hypothesis |
| `SaddleAvoidance.DescentCurve` | curves descending `f` at two rates: gradient flows and flows in a metric; translates, monotonicity of `f`, and a curve along which `f` is constant stops |
| `SaddleAvoidance.DescentTrap` | descent curves starting near a Łojasiewicz point stay near it, with finite length |
| `SaddleAvoidance.DescentConvergence` | descent curves with a Łojasiewicz limit point up to translations converge, to a critical point |
| `SaddleAvoidance.FlowDecay` | functions decreasing at a power of themselves decay exponentially or polynomially |
| `SaddleAvoidance.DescentEstimates` | the estimates behind the rates of descent curves at a Łojasiewicz limit |
| `SaddleAvoidance.FlowRates` | rates of descent curves and gradient flows at a Łojasiewicz limit: exponential for `a ≤ 1/2` (`IsDescentCurve.exists_le_mul_exp_of_isLojasiewiczAt`), `(t + 1) ^ (-(1 - a)/(2a - 1))` for `1/2 < a < 1` (`IsDescentCurve.exists_le_mul_rpow_of_isLojasiewiczAt`) |
| `SaddleAvoidance.Symmetry` | isometries preserving `f` carry its gradient along and commute with gradient steps and flows; the subspace they fix is invariant, its critical points along it are critical (symmetric criticality), and gradient descent on it is that of the restriction of `f` |
| `SaddleAvoidance.Analytic` | for an analytic function, orbits of gradient descent and gradient flows with a limit point up to translations converge, to a critical point (`exists_tendsto_of_analyticOnNhd`, `exists_tendsto_gradientFlow_of_analyticOnNhd`); every one when the translations span the space (`exists_tendsto_of_span_eq_top_of_analyticOnNhd`, `exists_tendsto_gradientFlow_of_span_eq_top_of_analyticOnNhd`); every bounded one (`exists_tendsto_of_isBounded_of_analyticOnNhd`, `exists_tendsto_gradientFlow_of_isBounded_of_analyticOnNhd`) |
| `SaddleAvoidance.AnalyticLandscape` | the scheme theorems for analytic functions, without the Łojasiewicz hypothesis: `ae_exists_tendsto_isMinOn_of_analyticOnNhd`, `ae_exists_tendsto_isMinOn_of_span_eq_top_of_analyticOnNhd`, `ae_exists_tendsto_isLocalMin_of_analyticOnNhd`, `ae_exists_tendsto_gradientFlow_isMinOn_of_analyticOnNhd`, `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top_of_analyticOnNhd` |
| `SaddleAvoidance.LevelLandscape` | below a level `c`: almost every start of gradient descent below `c` converges to a critical point that is not a strict saddle, and to a global minimizer if every critical point below `c` is one or a strict saddle; the hypotheses are asked only below `c` |
| `SaddleAvoidance.FlowLevelLandscape` | the same for gradient flows |
| `SaddleAvoidance.AnalyticLevelLandscape` | both for analytic `f`, without the Łojasiewicz hypothesis |
| `SaddleAvoidance.Orbit` | a Hessian bound along the orbit: on the segment of a step it gives the sufficient decrease (`le_sub_of_gradientStep_of_segment`, `iterate_succ_le_sub_of_segment`, `antitone_iterate_of_segment`); the starts whose orbit has it at its points, or on its segments, and converges to a strict saddle are null (`measure_iterate_norm_le_tendsto_strictSaddle_eq_zero`, `measure_segment_tendsto_strictSaddle_eq_zero`, `ae_not_tendsto_strictSaddle_of_segment`) |
| `SaddleAvoidance.OrbitLandscape` | the scheme along the orbit: an orbit with `‖∇²f‖ ≤ L`, `η L < 1`, on the segment of every step converges to a critical point no higher than its start, or runs off to infinity (`exists_tendsto_or_tendsto_norm_of_segment`); for almost every start below a level, to one that is not a strict saddle (`ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment`), to a global minimizer if every critical point below the level is one or a strict saddle (`ae_exists_tendsto_isMinOn_or_tendsto_norm_of_lt_of_segment`) |
| `SaddleAvoidance.AnalyticOrbitLandscape` | the same for analytic `f`, with neither the Łojasiewicz hypothesis nor a level (`ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd`, `ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd`); the double well `x⁴/4 - x²/2`, whose Hessian is unbounded, has the bound along every orbit from `[-3/2, 3/2]` for `η = 1/10` |
| `SaddleAvoidance.OrbitRates` | the rates of gradient descent at a Łojasiewicz limit under a Hessian bound on the segments of the orbit (`exists_rate_estimates_of_isLojasiewiczAt_of_segment`, `exists_le_mul_pow_of_isLojasiewiczAt_of_segment`, `exists_le_mul_rpow_of_isLojasiewiczAt_of_segment`) |

The sources are the gradient-descent corollary in §5.1 of arXiv:1710.07406
and `thm:main` of arXiv:1602.04915.  Both rest on the center-stable manifold
theorem; here a cone estimate replaces it, at the price of a symmetric
derivative at the unstable point, which gradient descent has.  The
convergence follows Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim.
16, 2005), which arXiv:1602.04915 cites for `prop:rates`.
-/

import Transformer.Landscape.SaddleAvoidance.Defs
import Transformer.Landscape.SaddleAvoidance.Basic
import Transformer.Landscape.SaddleAvoidance.Cone
import Transformer.Landscape.SaddleAvoidance.Spectral
import Transformer.Landscape.SaddleAvoidance.Preimage
import Transformer.Landscape.SaddleAvoidance.Global
import Transformer.Landscape.SaddleAvoidance.GradientDescent
import Transformer.Landscape.SaddleAvoidance.Descent
import Transformer.Landscape.SaddleAvoidance.Sequences
import Transformer.Landscape.SaddleAvoidance.Lojasiewicz
import Transformer.Landscape.SaddleAvoidance.Trap
import Transformer.Landscape.SaddleAvoidance.Convergence
import Transformer.Landscape.SaddleAvoidance.Decay
import Transformer.Landscape.SaddleAvoidance.RateEstimates
import Transformer.Landscape.SaddleAvoidance.Rates
import Transformer.Landscape.SaddleAvoidance.Minimizers
import Transformer.Landscape.SaddleAvoidance.CriticalPoints
import Transformer.Landscape.SaddleAvoidance.SecondOrder
import Transformer.Landscape.SaddleAvoidance.Necessary
import Transformer.Landscape.SaddleAvoidance.SecondOrderCritical
import Transformer.Landscape.SaddleAvoidance.Nondegenerate
import Transformer.Landscape.SaddleAvoidance.LinearFlow
import Transformer.Landscape.SaddleAvoidance.Periodic
import Transformer.Landscape.SaddleAvoidance.Invariant
import Transformer.Landscape.SaddleAvoidance.LocalInequality
import Transformer.Landscape.SaddleAvoidance.FlowTrap
import Transformer.Landscape.SaddleAvoidance.FlowConvergence
import Transformer.Landscape.SaddleAvoidance.Landscape
import Transformer.Landscape.SaddleAvoidance.TimeOne
import Transformer.Landscape.SaddleAvoidance.Sublevel
import Transformer.Landscape.SaddleAvoidance.FlowAvoidance
import Transformer.Landscape.SaddleAvoidance.FlowLandscape
import Transformer.Landscape.SaddleAvoidance.Symmetrizable
import Transformer.Landscape.SaddleAvoidance.SquareRoot
import Transformer.Landscape.SaddleAvoidance.MetricFlow
import Transformer.Landscape.SaddleAvoidance.MetricAvoidance
import Transformer.Landscape.SaddleAvoidance.Cutoff
import Transformer.Landscape.SaddleAvoidance.OpenAvoidance
import Transformer.Landscape.SaddleAvoidance.SublevelConvergence
import Transformer.Landscape.SaddleAvoidance.DescentCurve
import Transformer.Landscape.SaddleAvoidance.DescentTrap
import Transformer.Landscape.SaddleAvoidance.DescentConvergence
import Transformer.Landscape.SaddleAvoidance.FlowDecay
import Transformer.Landscape.SaddleAvoidance.DescentEstimates
import Transformer.Landscape.SaddleAvoidance.FlowRates
import Transformer.Landscape.SaddleAvoidance.Symmetry
import Transformer.Landscape.SaddleAvoidance.Analytic
import Transformer.Landscape.SaddleAvoidance.AnalyticLandscape
import Transformer.Landscape.SaddleAvoidance.LevelLandscape
import Transformer.Landscape.SaddleAvoidance.FlowLevelLandscape
import Transformer.Landscape.SaddleAvoidance.AnalyticLevelLandscape
import Transformer.Landscape.SaddleAvoidance.Orbit
import Transformer.Landscape.SaddleAvoidance.OrbitLandscape
import Transformer.Landscape.SaddleAvoidance.AnalyticOrbitLandscape
import Transformer.Landscape.SaddleAvoidance.OrbitRates
