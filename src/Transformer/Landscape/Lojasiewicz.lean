/-
# Łojasiewicz's theorem for analytic functions

A trajectory of a modulated gradient flow `ẋ = -M(t) ∇E(x)` of an analytic energy that
stays in a compact set converges to a critical point: `lem: loj` of Appendix D.1 of
arXiv:2510.22026v2, proved here for any analytic `E` on a finite-dimensional real inner
product space.  The proof rests on the Łojasiewicz gradient inequality at every point,
which `SaddleAvoidance.LocalInequality` turns into `IsLojasiewiczAt`.  The modules came
from `Transformer.Normalization`, the paper's directory, which keeps the statements that
use its dynamics `eq: NA`.

`Uniformization`, `RealAnalyticGerms` and `Sturm` use the three modules marked (shared)
below, and the inequality uses them in turn.

* The convergence theorem, `lem: loj` (`lojasiewicz_modulated`, in any
  finite-dimensional real inner product space
  `lojasiewicz_modulated_of_finiteDimensional`), and its conditional form, given the
  local gradient inequality at every critical point
  (`lojasiewicz_modulated_of_local_gradient_inequality`): `Modulated`,
  `LojasiewiczInnerProduct`, `LojasiewiczConditional`.
* Modulated gradient flows: operator bounds and energy dissipation, displacement bounds,
  the divergent clock and criticality of a limit, convergence from the gradient
  inequality; quadratic losses and dimensions at most one: `ModulatedEnergy`,
  `ModulatedLength`, `ModulatedCritical`, `ModulatedConvergence`, `ModulatedQuadratic`,
  `ModulatedScalar`, `QuadraticEnergy`.
* The gradient inequality `|E(y) - E(z)|^α ≤ k ‖∇E(y)‖` near a point
  (`analytic_gradient_inequality`): from compact monomial charts, which `Uniformization`
  provides (`hasCompactMonomialCharts_of_analyticAt`), and its special cases: dimensions
  zero and one, nondegenerate and blown-up critical points, normal crossings:
  `AnalyticMonomialCharts`, `CompactMonomialCharts`, `CoordinateMonomial`,
  `MonomialGradientBound`, `MonomialGradientInequality`, `NormalCrossingCharts`,
  `GradientPullback`, `GradientInequalityCoordinates`, `GradientInequalityCover`,
  `DirectionalCoordinates`, `OneDimensionalGradientInequality`,
  `ScalarGradientInequality`, `NondegenerateGradientInequality`,
  `BlowupGradientInequality`.
* The reduction of the inequality to analytic curve selection, an alternative route:
  gradient minima on energy levels, their lifting and selection along curves:
  `MinimumCurveSelection`, `MinimumCurveLifting`, `AnalyticComparisonArcs`,
  `AnalyticCurveEnergyLevels`, `AnalyticCurveGradient`, `AnalyticPlaneCurveSelection`,
  `PlaneCurveSelection`, `PlaneCurveOrientation`, `PlaneCurveExamples`,
  `ProjectedGradientMinimumLifting`, `EnergyGradientMinimumLifting`,
  `GradientFiberMinimumLifting`, `GradientEnergyImage`, `GradientMinimumCluster`,
  `GradientMinimumPolar`, `GlobalMinimumLiftingExamples`, `CompactFiberGradient`,
  `CompactEnergyAtlasCover`, `FiniteAnalyticMinima`, `AnalyticSignStability` (shared),
  `AnalyticDirection`, `AnalyticImplicitBranch`.
* Preparation of the energy graph and the roots of analytic monic families: Weierstrass
  coordinates, Newton scaling, ramified roots and their lifting:
  `WeierstrassCoordinates` (shared), `EnergyLevelPreparation`, `EnergyChartCoordinates`,
  `AnalyticPreparedSlice`, `ZeroAxisProjection`, `PolynomialFamily`,
  `PolynomialFamilyCoefficients`, `PolynomialCentering`, `PolynomialTranslation` (shared),
  `PolynomialDegreeReduction`, `PolynomialRootBounds`, `PolynomialRootCluster`,
  `PolynomialRootDivision`, `PolynomialRootOrder`, `NewtonCoefficientScaling`,
  `NewtonPolynomialScaling`, `RamifiedPolynomialRoot`, `RamifiedPolynomialBranches`,
  `RamifiedBranchIdentity`, `RamifiedSquareRoot`, `PositiveRamification`,
  `PreparedRootAccumulation`, `ConstrainedRootLifting`, `QuadraticCurveLifting`,
  `AnalyticEquationCurveLifting`, `ApproachingFiberWitnesses`.
* Exact queries: minima over analytic fibers, boxes, balls and atlases as Sturm–Tarski
  sign tests: `AnalyticFiberConstraints`, `AnalyticFiberMinimaReduction`,
  `AnalyticFiberMinimumLifting`, `AnalyticFiberMinimumQuery`,
  `AnalyticFiberPolynomialValues`, `AnalyticFiberValueRelation`,
  `PolynomialFiberArithmetic`, `PolynomialFiberMinimum`, `PolynomialFiberThreshold`,
  `AnalyticBoxMinimumQuery`, `AnalyticAtlasSource`, `AnalyticAtlasGradientQuery`,
  `AnalyticAtlasMinimumQuery`, `AnalyticAtlasQueryExamples`, `BallEnergyPatchSource`,
  `BallEnergyQueryPreparation`, `BallEnergyQueryValues`, `CompactBallGradientQueries`,
  `CompactBallQueryAtlas`.
-/

import Transformer.Landscape.Lojasiewicz.Modulated
import Transformer.Landscape.Lojasiewicz.LojasiewiczInnerProduct
