module

public import Transformer.Landscape.Lojasiewicz.EnergyLevelPreparation
public import Transformer.Landscape.Lojasiewicz.QuadraticCurveLifting
public import Transformer.Landscape.Lojasiewicz.AnalyticPlaneCurveSelection
public import Transformer.Landscape.Lojasiewicz.GlobalMinimumLiftingExamples
public import Transformer.Landscape.Lojasiewicz.AnalyticFiberConstraints
public import Transformer.Landscape.Lojasiewicz.AnalyticFiberMinimaReduction
public import Transformer.Landscape.Lojasiewicz.AnalyticFiberValueRelation
public import Transformer.Landscape.Lojasiewicz.AnalyticFiberMinimumQuery
public import Transformer.Landscape.Lojasiewicz.AnalyticBoxMinimumQuery
public import Transformer.Landscape.Lojasiewicz.AnalyticAtlasQueryExamples
public import Transformer.Landscape.Lojasiewicz.CompactBallGradientQueries
public import Transformer.Landscape.RealAnalyticGerms
public import Transformer.Landscape.Sturm
public import Lean.Util.FoldConsts

/-!
# Real analytic preparation and ramified-root dependency audit

Run after building the imported modules:

    lake env lean scripts/AnalyticPreparationAxioms.lean

The copied preparation, real-germ, and Sturm declarations and the external dependency
roots of preparation, division, Newton--Puiseux, Rückert, signed queries, and fiber-value
proofs are checked transitively. This audit
supplements statement and definition inspection and the full-tree audit.
-/

@[expose] public section

open Lean Elab Command

#print axioms Transformer.Landscape.Lojasiewicz.analytic_implicit_branch
#print axioms Transformer.Landscape.Lojasiewicz.directionCoordinates_axis
#print axioms Transformer.Landscape.Lojasiewicz.exists_regular_energy_coordinates
#print axioms Transformer.Landscape.Lojasiewicz.exists_weierstrass_energy_coordinates
#print axioms Transformer.Landscape.Lojasiewicz.analytic_energy_level_fibers_finite
#print axioms Transformer.Landscape.Lojasiewicz.exists_finite_fiber_energy_coordinates
#print axioms Transformer.Landscape.Lojasiewicz.analytic_square_root_after_ramification
#print axioms Transformer.Landscape.Lojasiewicz.analytic_quadratic_root_branches
#print axioms Transformer.Landscape.Lojasiewicz.analytic_newton_coefficient_scaling
#print axioms Transformer.Landscape.Lojasiewicz.centered_polynomial_root_multiplicity_lt
#print axioms Transformer.Landscape.Lojasiewicz.analytic_polynomial_preparation_at_root
#print axioms Transformer.Landscape.Lojasiewicz.analytic_ramified_polynomial_root
#print axioms Transformer.Landscape.Lojasiewicz.analytic_ramified_polynomial_branches
#print axioms Transformer.Landscape.Lojasiewicz.analytic_root_with_sign_constraints
#print axioms Transformer.Landscape.Lojasiewicz.analytic_equation_curve_lifting
#print axioms Transformer.Landscape.Lojasiewicz.analytic_zero_axis_isolated
#print axioms Transformer.Landscape.Lojasiewicz.prepared_plane_curve_selection
#print axioms Transformer.Landscape.Lojasiewicz.analytic_plane_level_curve_selection
#print axioms Transformer.Landscape.Lojasiewicz.analytic_finite_minimum_branch
#print axioms Transformer.Landscape.Lojasiewicz.analytic_polynomial_fiber_minimum
#print axioms Transformer.Landscape.Lojasiewicz.exists_prepared_analytic_slice
#print axioms Transformer.Landscape.Lojasiewicz.analytic_fiber_minimum_lifting
#print axioms Transformer.Landscape.Lojasiewicz.exists_energy_gradient_minimum_lifting_coordinates
#print axioms Transformer.Landscape.Lojasiewicz.analytic_global_gradient_minimum_lifting
#print axioms Transformer.Landscape.AnalyticPreparation.exists_isWeierstrassPreparation
#print axioms Transformer.Landscape.AnalyticPreparation.prepared_zero_fibers_finite
#print axioms Transformer.Landscape.AnalyticPreparation.exists_analyticWeierstrassDivision
#print axioms Transformer.Landscape.AnalyticPreparation.analyticWeierstrassDivision_unique
#print axioms Transformer.Landscape.RealAnalyticGerms.analyticGerm_isNoetherian_core
#print axioms Transformer.Landscape.RealAnalyticGerms.analyticGerm_finite_equations
#print axioms Transformer.Landscape.RealAnalyticGerms.analyticGerm_finite_parameterized_equations
#print axioms Transformer.Landscape.RealAnalyticGerms.preparedQuotientRemainderEquiv
#print axioms Transformer.Landscape.RealAnalyticGerms.preparedQuotient_moduleFinite
#print axioms Transformer.Landscape.RealAnalyticGerms.prepared_analytic_germ_integral_relation
#print axioms Transformer.Landscape.RealAnalyticGerms.prepared_analytic_germ_relation_degree
#print axioms Transformer.Landscape.RealAnalyticGerms.monic_analytic_germ_relation_representative
#print axioms Transformer.Landscape.Lojasiewicz.analytic_fiber_sign_conditions_polynomial
#print axioms Transformer.Landscape.Lojasiewicz.analytic_fiber_minima_polynomial
#print axioms Transformer.Landscape.Lojasiewicz.analytic_prepared_fiber_value_relation
#print axioms Transformer.Landscape.Sturm.IsSturmChain.sturm_Ioc
#print axioms Transformer.Landscape.Sturm.signedRemainderChain_isSturm
#print axioms Transformer.Landscape.Sturm.squarefreePart_real_roots
#print axioms Transformer.Landscape.Sturm.signedRemainderChain_query_Ioc
#print axioms Transformer.Landscape.Sturm.realRootQuery_eq
#print axioms Transformer.Landscape.Sturm.exists_root_with_signs_iff
#print axioms Transformer.Landscape.Sturm.polynomial_root_minimum_iff
#print axioms Transformer.Landscape.Lojasiewicz.analytic_fiber_minima_arithmetic
#print axioms Transformer.Landscape.Lojasiewicz.analytic_box_minima_arithmetic
#print axioms Transformer.Landscape.Lojasiewicz.analytic_atlas_lower_bounds_arithmetic
#print axioms Transformer.Landscape.Lojasiewicz.analytic_atlas_gradient_minima_arithmetic
#print axioms Transformer.Landscape.Lojasiewicz.exists_ball_energy_query_preparation
#print axioms Transformer.Landscape.Lojasiewicz.finite_cover_near_energy_level
#print axioms Transformer.Landscape.Lojasiewicz.analytic_compact_ball_lower_bounds_arithmetic
#print axioms Transformer.Landscape.Lojasiewicz.analytic_compact_ball_gradient_minima_arithmetic
#print axioms Transformer.Landscape.Sturm.polynomial_root_lower_bound_iff
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
    if !((`Transformer.Landscape.AnalyticPreparation).isPrefixOf name ||
        (`Transformer.Landscape.RealAnalyticGerms).isPrefixOf name ||
        (`Transformer.Landscape.Sturm).isPrefixOf name) || name.isInternal then continue
    let axioms ← liftCoreM (collectAxioms name)
    for ax in axioms do
      if !standard.contains ax then throwError "Unaccepted axiom in {name}: {ax}"
    if ci.type.isConstOf ``True then throwError "Vacuous result: {name}"
    audited := audited + 1
  logInfo m!"Audited {audited} real preparation, germ, and Sturm declarations: all axioms accepted."

run_cmd do
  let mut pending : Array Name := #[
    ``Transformer.Landscape.AnalyticPreparation.exists_isWeierstrassPreparation,
    ``Transformer.Landscape.AnalyticPreparation.exists_analyticWeierstrassDivision,
    ``Transformer.Landscape.AnalyticPreparation.analyticWeierstrassDivision_unique,
    ``Transformer.Landscape.RealAnalyticGerms.analyticGerm_isNoetherian_core,
    ``Transformer.Landscape.RealAnalyticGerms.analyticGerm_finite_parameterized_equations,
    ``Transformer.Landscape.Lojasiewicz.analytic_fiber_sign_conditions_polynomial,
    ``Transformer.Landscape.Lojasiewicz.analytic_fiber_minima_polynomial,
    ``Transformer.Landscape.Lojasiewicz.analytic_fiber_minima_arithmetic,
    ``Transformer.Landscape.Lojasiewicz.analytic_box_minima_arithmetic,
    ``Transformer.Landscape.Lojasiewicz.analytic_atlas_gradient_minima_arithmetic,
    ``Transformer.Landscape.Lojasiewicz.analytic_compact_ball_gradient_minima_arithmetic,
    ``Transformer.Landscape.Sturm.IsSturmChain.sturm_Ioc,
    ``Transformer.Landscape.Sturm.realRootQuery_eq,
    ``Transformer.Landscape.Sturm.exists_root_with_signs_iff,
    ``Transformer.Landscape.Sturm.polynomial_root_minimum_iff,
    ``Transformer.Landscape.Lojasiewicz.analytic_prepared_fiber_value_relation,
    ``Transformer.Landscape.Lojasiewicz.analytic_plane_level_curve_selection,
    ``Transformer.Landscape.Lojasiewicz.analytic_global_gradient_minimum_lifting,
    ``Transformer.Landscape.Lojasiewicz.exists_energy_gradient_minimum_lifting_coordinates]
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
