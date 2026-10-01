/-
# Isolating the central distinguished-variable zero

Preparation with a nonvanishing unit makes the zero at the central
distinguished-variable axis isolated. Accumulating nonzero points of a
prepared zero set must therefore have nonzero base parameters.
-/

import Transformer.Normalization.AnalyticEquationCurveLifting

open Filter

namespace Transformer.Normalization

open AnalyticPreparation

/-- The central distinguished-variable axis has no nearby nonzero zero
of a real analytic germ of finite distinguished order. The proof cancels
the actual preparation unit and uses its zero-origin coefficients.
Auxiliary for curve selection in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_zero_axis_isolated {n d : ℕ} (F : Ambient n → ℝ)
    (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d) :
    ∀ᶠ x in nhds (0 : Ambient n), x.1 = 0 → F x = 0 → x = 0 := by
  obtain ⟨A, U, hprep⟩ := exists_isWeierstrassPreparation hF horder
  have hunit : ∀ᶠ x in nhds (0 : Ambient n), U x ≠ 0 :=
    hprep.2.2.1.continuousAt.eventually_ne hprep.2.2.2.1
  filter_upwards [hprep.2.2.2.2, hunit] with x hx hU
  intro hbase hzero
  have hpoly : preparedPolynomial d A x = 0 :=
    (mul_eq_zero.mp (hx.symm.trans hzero)).resolve_left hU
  have hpow : x.2 ^ d = 0 := by
    simpa [preparedPolynomial, hbase, hprep.2.1] using hpoly
  have hx2 : x.2 = 0 := by
    by_contra h
    exact pow_ne_zero _ h hpow
  exact Prod.ext hbase hx2

/-- The one-dimensional parameter space with a distinguished variable
is continuously linearly equivalent to the real plane. The inverse
puts the scalar parameter into the unique base coordinate. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def oneParameterCoordinates : Ambient 1 ≃L[ℝ] ℝ × ℝ :=
  (ContinuousLinearEquiv.funUnique (Fin 1) ℝ ℝ).prodCongr (ContinuousLinearEquiv.refl ℝ ℝ)

/-- The inverse plane coordinates are the actual constant one-coordinate
base function and the distinguished variable. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem oneParameterCoordinates_symm_apply (x : ℝ × ℝ) :
    oneParameterCoordinates.symm x = ((fun _ => x.1), x.2) := rfl

/-- The cusp `y³ = z₀²` satisfies the analytic and exact-order hypotheses
jointly. Its only sufficiently small zero on the central base axis is the
origin, despite the multiple distinguished-variable zero. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∀ᶠ x in nhds (0 : Ambient 1), x.1 = 0 →
    x.2 ^ 3 - (x.1 0) ^ 2 = 0 → x = 0 := by
  let F : Ambient 1 → ℝ := fun x => x.2 ^ 3 - (x.1 0) ^ 2
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 := (analyticAt_snd.fun_pow 3).fun_sub ((L.analyticAt 0).fun_pow 2)
  have horder : ExactOrderInLastVariable F 3 := by
    have hslice : lastSlice F = fun t : ℝ => t ^ 3 := by funext t; simp [lastSlice, F]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 3) 0 = 3 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 3
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 3)).mp hord
  exact analytic_zero_axis_isolated F hF horder

end Transformer.Normalization
