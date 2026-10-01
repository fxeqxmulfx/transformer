/-
# Joint witnesses for analytic germ representatives and pullbacks

Nonconstant real functions supply the actual analytic representatives,
coordinate changes, and nonvanishing units. Auxiliary examples for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.RealAnalyticGerms.AnalyticPullback

open Filter
open scoped Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- A nonconstant exponential gives an actual analytic unit, a nonzero
germ, and a valid regularized preparation and associated prepared divisor.
This jointly witnesses the representative, unit, and association hypotheses.
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (f : AnalyticGerm 2) (L : Base 2 ≃L[ℝ] Base 2)
    (d : ℕ) (H : Ambient 1 → ℝ) (a : Fin d → Base 1 → ℝ) (u : Ambient 1 → ℝ),
    f ≠ 0 ∧ IsUnit f ∧ AnalyticAt ℝ H 0 ∧
      ((fun x : Base 2 => H (wptAmbientEquiv 1 x)) : FunctionGerm 2) =
        (coordinatePullback L f : FunctionGerm 2) ∧
      H 0 = evalAtOrigin f ∧ ExactOrderInLastVariable H d ∧
      ∃ hprep : IsWeierstrassPreparation H d a u,
      IsUnit (standardPreparationUnitGerm u hprep.2.2.1) ∧
      Associated (coordinatePullback L f)
        (preparedPolynomialGerm a hprep.1) := by
  let F : Base 2 → ℝ := fun x => Real.exp (x 0)
  have hF : AnalyticAt ℝ F 0 := analyticAt_rexp.comp
    ((ContinuousLinearMap.proj (0 : Fin 2) : Base 2 →L[ℝ] ℝ).analyticAt 0)
  let f := AnalyticGerm.ofFunction F hF
  have hf0 : evalAtOrigin f ≠ 0 := by simp [f, F]
  have hfne : f ≠ 0 := by
    intro hz
    exact hf0 (by rw [hz]; exact map_zero (evalAtOriginHom 2))
  obtain ⟨L, d, H, a, u, hH, hcoord, hH0, horder, hprep⟩ :=
    exists_regularized_weierstrassPreparation hfne
  exact ⟨f, L, d, H, a, u, hfne, (analyticGerm_isUnit_iff f).mpr hf0,
    hH, hcoord, hH0, horder, hprep,
    standardPreparationUnitGerm_isUnit u hprep.2.2.1 hprep.2.2.2.1,
    coordinatePullback_associated_preparedPolynomialGerm L H a u hcoord hprep⟩

/-- A nonlinear map `t ↦ (t²,t³)` and the nonconstant germ `x₀+x₁`
jointly witness analytic pullback and its representative formula. The
resulting function is exactly `t²+t³`. Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : ∃ (T : Base 1 → Base 2) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0)
    (F : Base 2 → ℝ) (hF : AnalyticAt ℝ F 0),
    (analyticPullbackHom T hT hT0 (AnalyticGerm.ofFunction F hF) : FunctionGerm 1) =
      ((fun t : Base 1 => (t 0) ^ 2 + (t 0) ^ 3) : FunctionGerm 1) := by
  let T : Base 1 → Base 2 := fun t i => if i = 0 then (t 0) ^ 2 else (t 0) ^ 3
  have ht : AnalyticAt ℝ (fun t : Base 1 => t 0) 0 :=
    (ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0
  have hT : AnalyticAt ℝ T 0 := by
    apply analyticAt_pi_iff.mpr
    intro i
    split_ifs
    · exact ht.fun_pow 2
    · exact ht.fun_pow 3
  have hT0 : T 0 = 0 := by funext i; simp [T]
  let F : Base 2 → ℝ := fun x => x 0 + x 1
  have hF : AnalyticAt ℝ F 0 :=
    ((ContinuousLinearMap.proj (0 : Fin 2) : Base 2 →L[ℝ] ℝ).analyticAt 0).fun_add
      ((ContinuousLinearMap.proj (1 : Fin 2) : Base 2 →L[ℝ] ℝ).analyticAt 0)
  refine ⟨T, hT, hT0, F, hF, ?_⟩
  rw [analyticPullbackHom_ofFunction]
  apply congrArg Filter.Germ.ofFun
  funext t
  simp [T, F]

end Transformer.RealAnalyticGerms
