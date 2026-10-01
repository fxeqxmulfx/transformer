/-
# Real analytic preparation and ramified-root dependency audit

Run after building the imported modules:

    lake env lean scripts/AnalyticPreparationAxioms.lean

The copied preparation, real-germ, and Sturm declarations and the external dependency
roots of preparation, division, Newton--Puiseux, Rückert, signed queries, and fiber-value
proofs are checked transitively. This audit
supplements statement and definition inspection and the full-tree audit.
-/

import Transformer.Normalization.EnergyLevelPreparation
import Transformer.Normalization.QuadraticCurveLifting
import Transformer.Normalization.AnalyticPlaneCurveSelection
import Transformer.Normalization.GlobalMinimumLiftingExamples
import Transformer.Normalization.AnalyticFiberConstraints
import Transformer.Normalization.AnalyticFiberMinimaReduction
import Transformer.Normalization.AnalyticFiberValueRelation
import Transformer.Normalization.AnalyticFiberMinimumQuery
import Transformer.Normalization.AnalyticBoxMinimumQuery
import Transformer.Normalization.AnalyticAtlasQueryExamples
import Transformer.Normalization.CompactBallGradientQueries
import Transformer.RealAnalyticGerms
import Transformer.Sturm
import Lean.Util.FoldConsts

open Lean Elab Command

#print axioms Transformer.Normalization.analytic_implicit_branch
#print axioms Transformer.Normalization.directionCoordinates_axis
#print axioms Transformer.Normalization.exists_regular_energy_coordinates
#print axioms Transformer.Normalization.exists_weierstrass_energy_coordinates
#print axioms Transformer.Normalization.analytic_energy_level_fibers_finite
#print axioms Transformer.Normalization.exists_finite_fiber_energy_coordinates
#print axioms Transformer.Normalization.analytic_square_root_after_ramification
#print axioms Transformer.Normalization.analytic_quadratic_root_branches
#print axioms Transformer.Normalization.analytic_newton_coefficient_scaling
#print axioms Transformer.Normalization.centered_polynomial_root_multiplicity_lt
#print axioms Transformer.Normalization.analytic_polynomial_preparation_at_root
#print axioms Transformer.Normalization.analytic_ramified_polynomial_root
#print axioms Transformer.Normalization.analytic_ramified_polynomial_branches
#print axioms Transformer.Normalization.analytic_root_with_sign_constraints
#print axioms Transformer.Normalization.analytic_equation_curve_lifting
#print axioms Transformer.Normalization.analytic_zero_axis_isolated
#print axioms Transformer.Normalization.prepared_plane_curve_selection
#print axioms Transformer.Normalization.analytic_plane_level_curve_selection
#print axioms Transformer.Normalization.analytic_finite_minimum_branch
#print axioms Transformer.Normalization.analytic_polynomial_fiber_minimum
#print axioms Transformer.Normalization.exists_prepared_analytic_slice
#print axioms Transformer.Normalization.analytic_fiber_minimum_lifting
#print axioms Transformer.Normalization.exists_energy_gradient_minimum_lifting_coordinates
#print axioms Transformer.Normalization.analytic_global_gradient_minimum_lifting
#print axioms Transformer.AnalyticPreparation.exists_isWeierstrassPreparation
#print axioms Transformer.AnalyticPreparation.prepared_zero_fibers_finite
#print axioms Transformer.AnalyticPreparation.exists_analyticWeierstrassDivision
#print axioms Transformer.AnalyticPreparation.analyticWeierstrassDivision_unique
#print axioms Transformer.RealAnalyticGerms.analyticGerm_isNoetherian_core
#print axioms Transformer.RealAnalyticGerms.analyticGerm_finite_equations
#print axioms Transformer.RealAnalyticGerms.analyticGerm_finite_parameterized_equations
#print axioms Transformer.RealAnalyticGerms.preparedQuotientRemainderEquiv
#print axioms Transformer.RealAnalyticGerms.preparedQuotient_moduleFinite
#print axioms Transformer.RealAnalyticGerms.prepared_analytic_germ_integral_relation
#print axioms Transformer.RealAnalyticGerms.prepared_analytic_germ_relation_degree
#print axioms Transformer.RealAnalyticGerms.monic_analytic_germ_relation_representative
#print axioms Transformer.Normalization.analytic_fiber_sign_conditions_polynomial
#print axioms Transformer.Normalization.analytic_fiber_minima_polynomial
#print axioms Transformer.Normalization.analytic_prepared_fiber_value_relation
#print axioms Transformer.Sturm.IsSturmChain.sturm_Ioc
#print axioms Transformer.Sturm.signedRemainderChain_isSturm
#print axioms Transformer.Sturm.squarefreePart_real_roots
#print axioms Transformer.Sturm.signedRemainderChain_query_Ioc
#print axioms Transformer.Sturm.realRootQuery_eq
#print axioms Transformer.Sturm.exists_root_with_signs_iff
#print axioms Transformer.Sturm.polynomial_root_minimum_iff
#print axioms Transformer.Normalization.analytic_fiber_minima_arithmetic
#print axioms Transformer.Normalization.analytic_box_minima_arithmetic
#print axioms Transformer.Normalization.analytic_atlas_lower_bounds_arithmetic
#print axioms Transformer.Normalization.analytic_atlas_gradient_minima_arithmetic
#print axioms Transformer.Normalization.exists_ball_energy_query_preparation
#print axioms Transformer.Normalization.finite_cover_near_energy_level
#print axioms Transformer.Normalization.analytic_compact_ball_lower_bounds_arithmetic
#print axioms Transformer.Normalization.analytic_compact_ball_gradient_minima_arithmetic
#print axioms Transformer.Sturm.polynomial_root_lower_bound_iff
#print axioms Algebra.aeval_self_charpoly_lmul
#print axioms OpenPartialHomeomorph.analyticAt_symm'
#print axioms analyticAt_log
#print axioms analyticAt_rexp
#print axioms FormalMultilinearSeries.changeOrigin_eval
#print axioms FormalMultilinearSeries.changeOriginSeries_summable_aux₁
#print axioms analyticAt_inverse
#print axioms Units.oneSub
#print axioms Polynomial.finite_setOfPred_isRoot
#print axioms Polynomial.pow_mul_divByMonic_rootMultiplicity_eq
#print axioms Polynomial.eval_divByMonic_pow_rootMultiplicity_ne_zero
#print axioms Real.rpow_inv_natCast_pow
#print axioms IsCompact.tendsto_subseq
#print axioms Finset.exists_min_image
#print axioms AnalyticAt.fderiv

run_cmd do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut audited : Nat := 0
  for (name, ci) in env.constants.toList do
    if !((`Transformer.AnalyticPreparation).isPrefixOf name ||
        (`Transformer.RealAnalyticGerms).isPrefixOf name ||
        (`Transformer.Sturm).isPrefixOf name) || name.isInternal then continue
    let axioms ← liftCoreM (collectAxioms name)
    for ax in axioms do
      if !standard.contains ax then throwError "Unaccepted axiom in {name}: {ax}"
    if ci.type.isConstOf ``True then throwError "Vacuous result: {name}"
    audited := audited + 1
  logInfo m!"Audited {audited} real preparation, germ, and Sturm declarations: all axioms accepted."

run_cmd do
  let mut pending : Array Name := #[
    ``Transformer.AnalyticPreparation.exists_isWeierstrassPreparation,
    ``Transformer.AnalyticPreparation.exists_analyticWeierstrassDivision,
    ``Transformer.AnalyticPreparation.analyticWeierstrassDivision_unique,
    ``Transformer.RealAnalyticGerms.analyticGerm_isNoetherian_core,
    ``Transformer.RealAnalyticGerms.analyticGerm_finite_parameterized_equations,
    ``Transformer.Normalization.analytic_fiber_sign_conditions_polynomial,
    ``Transformer.Normalization.analytic_fiber_minima_polynomial,
    ``Transformer.Normalization.analytic_fiber_minima_arithmetic,
    ``Transformer.Normalization.analytic_box_minima_arithmetic,
    ``Transformer.Normalization.analytic_atlas_gradient_minima_arithmetic,
    ``Transformer.Normalization.analytic_compact_ball_gradient_minima_arithmetic,
    ``Transformer.Sturm.IsSturmChain.sturm_Ioc,
    ``Transformer.Sturm.realRootQuery_eq,
    ``Transformer.Sturm.exists_root_with_signs_iff,
    ``Transformer.Sturm.polynomial_root_minimum_iff,
    ``Transformer.Normalization.analytic_prepared_fiber_value_relation,
    ``Transformer.Normalization.analytic_plane_level_curve_selection,
    ``Transformer.Normalization.analytic_global_gradient_minimum_lifting,
    ``Transformer.Normalization.exists_energy_gradient_minimum_lifting_coordinates]
  let mut seen : NameSet := {}
  let mut external : NameSet := {}
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    if !(`Transformer).isPrefixOf name then
      external := external.insert name
      continue
    let ci ← liftCoreM (getConstInfo name)
    pending := pending ++ ci.type.getUsedConstants
    if let some value := ci.value? (allowOpaque := true) then
      pending := pending ++ value.getUsedConstants
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  for name in external.toList do
    let axioms ← liftCoreM (collectAxioms name)
    for ax in axioms do
      if !standard.contains ax then throwError "Unaccepted external axiom in {name}: {ax}"
  logInfo m!"Audited {external.size} external dependency roots of preparation, real germs, Sturm queries, and lifting."
