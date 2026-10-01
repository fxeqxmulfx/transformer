/-
# Real analytic germs: PreparedAssociate

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/PreparedAssociate.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.Preparation
import Transformer.RealAnalyticGerms.Division

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Transport an analytic WPT unit from product coordinates to standard coordinates.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def standardPreparationUnitGerm {n : ℕ} (u : Ambient n → ℝ)
    (hu : AnalyticAt ℝ u 0) : AnalyticGerm (n + 1) :=
  AnalyticGerm.ofFunction (fun x ↦ u (wptAmbientEquiv n x))
    (by
      have hu' : AnalyticAt ℝ u (wptAmbientEquiv n 0) := by rw [map_zero]; exact hu
      simpa [Function.comp_def] using hu'.compContinuousLinearMap
        (u := (wptAmbientEquiv n : Base (n + 1) →L[ℝ] Ambient n))
        (x := 0))

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem standardPreparationUnitGerm_isUnit {n : ℕ} (u : Ambient n → ℝ)
    (hu : AnalyticAt ℝ u 0) (hu0 : u 0 ≠ 0) :
    IsUnit (standardPreparationUnitGerm u hu) := by
  apply (analyticGerm_isUnit_iff _).2
  change u (wptAmbientEquiv n 0) ≠ 0
  simpa only [map_zero] using hu0

/--
The raw representative identity supplied by regularized preparation descends to
an equality saying that the coordinate pullback is a unit times the prepared
polynomial germ.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coordinatePullback_eq_unit_mul_preparedPolynomialGerm
    {n d : ℕ} {f : AnalyticGerm (n + 1)}
    (L : Base (n + 1) ≃L[ℝ] Base (n + 1))
    (H : Ambient n → ℝ) (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ)
    (hcoord :
      ((fun x : Base (n + 1) ↦ H (wptAmbientEquiv n x)) :
          FunctionGerm (n + 1)) =
        (coordinatePullback L f : FunctionGerm (n + 1)))
    (hprep : IsWeierstrassPreparation H d a u) :
    coordinatePullback L f =
      standardPreparationUnitGerm u hprep.2.2.1 *
        preparedPolynomialGerm a hprep.1 := by
  apply Subtype.ext
  rw [← hcoord]
  change ((fun x : Base (n + 1) ↦ H (wptAmbientEquiv n x)) :
      FunctionGerm (n + 1)) =
    ((fun x ↦ u (wptAmbientEquiv n x)) : FunctionGerm (n + 1)) *
      (preparedPolynomialFunction a : FunctionGerm (n + 1))
  rw [← Filter.Germ.coe_mul]
  apply Filter.Germ.coe_eq.mpr
  have htendsto : Tendsto (wptAmbientEquiv n) (𝓝 0) (𝓝 0) := by
    have h : Tendsto (wptAmbientEquiv n) (𝓝 0)
        (𝓝 (wptAmbientEquiv n 0)) :=
      (wptAmbientEquiv n).continuous.continuousAt
    rwa [map_zero] at h
  have hfactor := hprep.2.2.2.2.comp_tendsto htendsto
  filter_upwards [hfactor] with x hx
  exact hx

/-- The prepared divisor is an associate of the coordinate pullback.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coordinatePullback_associated_preparedPolynomialGerm
    {n d : ℕ} {f : AnalyticGerm (n + 1)}
    (L : Base (n + 1) ≃L[ℝ] Base (n + 1))
    (H : Ambient n → ℝ) (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ)
    (hcoord :
      ((fun x : Base (n + 1) ↦ H (wptAmbientEquiv n x)) :
          FunctionGerm (n + 1)) =
        (coordinatePullback L f : FunctionGerm (n + 1)))
    (hprep : IsWeierstrassPreparation H d a u) :
    Associated (coordinatePullback L f) (preparedPolynomialGerm a hprep.1) := by
  rw [coordinatePullback_eq_unit_mul_preparedPolynomialGerm L H a u hcoord hprep]
  exact associated_unit_mul_left _ _
    (standardPreparationUnitGerm_isUnit u hprep.2.2.1 hprep.2.2.2.1)

end Transformer.RealAnalyticGerms
