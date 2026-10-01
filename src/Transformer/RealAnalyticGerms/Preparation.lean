/-
# Real analytic germs: Preparation

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/Preparation.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.RegularCoordinates

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- A representative of a nonzero analytic germ is not locally zero.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem representative_not_eventually_zero {n : ℕ}
    {f : AnalyticGerm n} (hf : f ≠ 0)
    {F : Base n → ℝ}
    (hrep : (F : FunctionGerm n) = (f : FunctionGerm n)) :
    ¬ F =ᶠ[𝓝 0] (fun _ ↦ 0) := by
  intro hzero
  apply hf
  apply Subtype.ext
  change (f : FunctionGerm n) = 0
  rw [← hrep, ← Filter.Germ.coe_zero]
  exact Filter.Germ.coe_eq.mpr hzero

/-- Evaluation of an analytic representative agrees with evaluation of its germ.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem representative_value_eq_evalAtOrigin {n : ℕ}
    {f : AnalyticGerm n} {F : Base n → ℝ}
    (hrep : (F : FunctionGerm n) = (f : FunctionGerm n)) :
    F 0 = evalAtOrigin f := by
  have h := congrArg Filter.Germ.value hrep
  change F 0 = Filter.Germ.value (f : FunctionGerm n)
  simpa using h

/--
Regularized preparation of a nonzero analytic germ.

`H` is written in WPT's product coordinates.  Composing it with the standard
successor-coordinate splitting represents exactly the coordinate pullback of
the original germ.  The equality `H 0 = evalAtOrigin f` makes the positive-order
corollary independent of representatives.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_regularized_weierstrassPreparation
    {n : ℕ} {f : AnalyticGerm (n + 1)} (hf_ne : f ≠ 0) :
    ∃ (L : Base (n + 1) ≃L[ℝ] Base (n + 1))
      (d : ℕ) (H : Ambient n → ℝ)
      (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ),
      AnalyticAt ℝ H 0 ∧
      ((fun x : Base (n + 1) ↦ H (wptAmbientEquiv n x)) :
          FunctionGerm (n + 1)) =
        (coordinatePullback L f : FunctionGerm (n + 1)) ∧
      H 0 = evalAtOrigin f ∧
      ExactOrderInLastVariable H d ∧
      IsWeierstrassPreparation H d a u := by
  obtain ⟨F, hF, hrep⟩ := AnalyticGerm.exists_rep f
  have hF_ne : ¬ F =ᶠ[𝓝 0] (fun _ ↦ 0) :=
    representative_not_eventually_zero hf_ne hrep
  obtain ⟨L, d, horder⟩ := exists_regularizing_coordinateEquiv hF hF_ne
  let H : Ambient n → ℝ := fun x ↦ F (L ((wptAmbientEquiv n).symm x))
  have hH : AnalyticAt ℝ H 0 := by
    let A := L.toContinuousLinearMap.comp
      (wptAmbientEquiv n).symm.toContinuousLinearMap
    have hFA : AnalyticAt ℝ F (A 0) := by rw [map_zero]; exact hF
    have hcomp := hFA.compContinuousLinearMap (u := A) (x := 0)
    change AnalyticAt ℝ (F ∘ A) 0
    exact hcomp
  obtain ⟨a, u, hprep⟩ :=
    exists_isWeierstrassPreparation hH horder
  refine ⟨L, d, H, a, u, hH, ?_, ?_, horder, hprep⟩
  · change ((fun x : Base (n + 1) ↦ H (wptAmbientEquiv n x)) :
        FunctionGerm (n + 1)) =
      functionGermPullbackHom L (f : FunctionGerm (n + 1))
    rw [← hrep]
    apply Filter.Germ.coe_eq.mpr
    apply Filter.Eventually.of_forall
    intro x
    exact congrArg (fun y ↦ F (L y)) ((wptAmbientEquiv n).symm_apply_apply x)
  · calc
      H 0 = F (L ((wptAmbientEquiv n).symm 0)) := rfl
      _ = F (L 0) := by rw [map_zero]
      _ = F 0 := by rw [map_zero]
      _ = evalAtOrigin f := representative_value_eq_evalAtOrigin hrep


end Transformer.RealAnalyticGerms
