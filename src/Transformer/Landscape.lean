/-
# The landscape framework

What gradient descent and gradient flows do on a landscape, and everything this rests
on.  Each subdirectory has an aggregator beside it, `Landscape/<Subdirectory>.lean`,
with a map of its modules.

| Subdirectory | What it proves | Main theorems |
| --- | --- | --- |
| `SaddleAvoidance` | gradient descent and gradient flows, also in a metric, almost never converge to a strict saddle; under the Łojasiewicz inequality an orbit with a limit point converges, at known rates; if every critical point is a minimizer or a strict saddle, almost every orbit converges to a minimizer; the same below a level, on compact sublevel sets, for analytic functions and under symmetries | `measure_tendsto_strictSaddle_eq_zero`, `measure_inter_tendsto_strictSaddle_eq_zero`, `exists_tendsto_of_isLojasiewiczAt`, `exists_le_mul_rpow_of_isLojasiewiczAt`, `ae_exists_tendsto_isMinOn`, `ae_exists_tendsto_not_isStrictSaddle_of_lt`, `ae_exists_tendsto_isMinOn_of_analyticOnNhd` |
| `PhaseModel` | the phase energies `Σ_r (1 - cos ⟪m_r, ψ⟫)` of division in a cyclic group and in a finite abelian group: their zeros, their landscape below `n²/8`, and convergence to a character | `isStrictSaddle_of_divisionEnergy_lt`, `ae_exists_tendsto_characterPhase` |
| `Neural` | analytic SiLU and SwiGLU coordinates, positive-epsilon RMSNorm, finite losses, and complete tied Rotate90 residual classifiers in their actual Euclidean weights | `analyticAt_rmsNorm`, `analyticOnNhd_siluLoss`, `analyticOnNhd_gatedLoss`, `exists_isLojasiewiczAt_siluLoss`, `exists_isLojasiewiczAt_gatedLoss` |
| `GlobalFlow` | complete trajectories and flows of Lipschitz differential equations, `C¹` in the initial point | `exists_global`, `IsFlowField.contDiff_flow` |
| `Lojasiewicz` | Łojasiewicz's gradient inequality for analytic functions, and the convergence of modulated gradient flows of analytic energies (`lem: loj` of arXiv:2510.22026v2, App. D.1) | `analytic_gradient_inequality`, `lojasiewicz_modulated`, `lojasiewicz_modulated_of_finiteDimensional` |
| `Uniformization` | local uniformization of real analytic functions: compact monomial charts | `hasMonomialCharts_of_analyticAt` |
| `AnalyticPreparation` | real analytic Weierstrass preparation and division | `exists_analyticWeierstrassDivision`, `prepared_zero_fibers_finite` |
| `RealAnalyticGerms` | the rings of real analytic germs are Noetherian; Weierstrass division of germs | `analyticGerm_instIsNoetherianRing`, `exists_preparedGermDivision` |
| `Sturm` | Sturm's theorem and Sturm–Tarski sign queries for real polynomials | `IsSturmChain.sturm_Ioc`, `realRootQuery_eq` |

## `SaddleAvoidance` by layer

The layers share one directory: lemmas such as `IsStrictSaddle.isSymmetricUnstable`
and `IsDescentCurve.exists_le_mul_exp_of_isLojasiewiczAt` sit in other modules than
their type, and dot notation needs them in its namespace.

* Avoidance, layer A: `Defs`, `Basic`, `Cone`, `Spectral`, `Preimage`, `Global`,
  `GradientDescent`, `Sublevel`, `Symmetrizable`, `SquareRoot`, `LinearFlow`, `TimeOne`,
  `FlowAvoidance`, `MetricFlow`, `MetricAvoidance`, `Cutoff`, `OpenAvoidance`, `Orbit`.
* Convergence, layer B: `Descent`, `Sequences`, `Lojasiewicz`, `Trap`, `Convergence`,
  `CriticalPoints`, `Invariant`, `LocalInequality`, `Periodic`, `FlowTrap`,
  `FlowConvergence`, `DescentCurve`, `DescentTrap`, `DescentConvergence`,
  `SublevelConvergence`, `Analytic`, `Symmetry`.
* Rates: `Decay`, `RateEstimates`, `Rates`, `OrbitRates`, `FlowDecay`, `DescentEstimates`,
  `FlowRates`.
* The scheme, layer C: `Minimizers`, `SecondOrder`, `Necessary`, `SecondOrderCritical`,
  `Nondegenerate`, `Landscape`, `FlowLandscape`, `AnalyticLandscape`, `LevelLandscape`,
  `FlowLevelLandscape`, `AnalyticLevelLandscape`, `OrbitLandscape`, `AnalyticOrbitLandscape`.

## Dependencies

`PhaseModel` imports `SaddleAvoidance`, which imports `GlobalFlow` (in `TimeOne`) and
`Lojasiewicz` (in `LocalInequality`, `Symmetry` and `FlowConvergence`).  `Lojasiewicz`,
`Uniformization`, `AnalyticPreparation`, `RealAnalyticGerms` and `Sturm` import each
other's modules.  Outside this directory the framework imports only `Transformer.Basic`
and Mathlib.  `Normalization`, `Perspective`, `Clusters`, `Interpolation`, `Causal`,
`Kinetic`, `Imbalance` and `ContinuityEq` import it.
-/

import Transformer.Landscape.SaddleAvoidance
import Transformer.Landscape.PhaseModel
import Transformer.Landscape.Neural
import Transformer.Landscape.GlobalFlow
import Transformer.Landscape.Lojasiewicz
import Transformer.Landscape.Uniformization
import Transformer.Landscape.AnalyticPreparation
import Transformer.Landscape.RealAnalyticGerms
import Transformer.Landscape.Sturm
