/-
# Saddle avoidance — the landscape hypothesis in second-order form

Papers state a benign landscape as the absence of spurious second-order
critical points: every critical point at which the Hessian is positive
semidefinite is a minimizer.  The landscape theorems ask instead that every
critical point be a minimizer or a strict saddle (`hcrit`).  For `f` of class
`C²` on a finite-dimensional space the two agree.  A critical point is a
strict saddle exactly when it has a direction `h` of negative curvature,
`⟪∇²f(z) h, h⟫ < 0` (`isStrictSaddle_iff_exists_inner_neg`), so a property
holds at it or it is a strict saddle exactly when the property holds there if
the Hessian is positive semidefinite (`or_isStrictSaddle_iff`).  This gives
`hcrit` for global minimizers (`isMinOn_or_isStrictSaddle_of_inner_nonneg`)
and local minimizers (`isLocalMin_or_isStrictSaddle_of_inner_nonneg`), and
for both below a level `c` (the forms ending in `_of_lt`).
-/

import Transformer.Landscape.SaddleAvoidance.Necessary

open Set

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **A strict saddle has a direction of negative curvature**: an eigenvector
`v` of the Hessian `A` with eigenvalue `μ < 0` has `⟪A v, v⟫ = μ ‖v‖² < 0`. -/
theorem IsStrictSaddle.exists_inner_neg {z : E} (hz : IsStrictSaddle f z) :
    ∃ h, ⟪fderiv ℝ (gradient f) z h, h⟫ < 0 := by
  obtain ⟨-, μ, hμ, hev⟩ := hz
  obtain ⟨v, hv, hv0⟩ := hev.exists_hasEigenvector
  have hAv : fderiv ℝ (gradient f) z v = μ • v := Module.End.mem_eigenspace_iff.1 hv
  refine ⟨v, ?_⟩
  rw [hAv, real_inner_smul_left]
  exact mul_neg_of_neg_of_pos hμ (real_inner_self_pos.2 hv0)

variable [FiniteDimensional ℝ E]

/-- **The strict saddles of a `C²` function are its critical points with a
direction of negative curvature**: `IsStrictSaddle.exists_inner_neg` one way,
`isStrictSaddle_of_inner_neg` the other. -/
theorem isStrictSaddle_iff_exists_inner_neg (hf : ContDiff ℝ 2 f) {z : E}
    (hz : gradient f z = 0) :
    IsStrictSaddle f z ↔ ∃ h, ⟪fderiv ℝ (gradient f) z h, h⟫ < 0 :=
  ⟨IsStrictSaddle.exists_inner_neg, fun ⟨_, hh⟩ => isStrictSaddle_of_inner_neg hf hz hh⟩

/-- **The case split.**  At a critical point `z` of a `C²` function, a
proposition `p` holds or `z` is a strict saddle exactly when `p` follows from
the Hessian at `z` being positive semidefinite: where it is not, some `h` has
`⟪∇²f(z) h, h⟫ < 0`, and `z` is a strict saddle. -/
theorem or_isStrictSaddle_iff (hf : ContDiff ℝ 2 f) {z : E} (hz : gradient f z = 0)
    {p : Prop} :
    p ∨ IsStrictSaddle f z ↔ ((∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) → p) := by
  rw [isStrictSaddle_iff_exists_inner_neg hf hz]
  refine ⟨fun hp hpsd => hp.resolve_right fun ⟨h, hh⟩ => not_lt.2 (hpsd h) hh, fun hp => ?_⟩
  by_cases hpsd : ∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫
  · exact Or.inl (hp hpsd)
  · push Not at hpsd
    exact Or.inr hpsd

/-- **The landscape hypothesis for global minimizers, in second-order form.**
If every critical point of a `C²` function at which the Hessian is positive
semidefinite is a global minimizer, every critical point is a global
minimizer or a strict saddle: the hypothesis `hcrit` of
`ae_exists_tendsto_isMinOn` and `ae_exists_tendsto_gradientFlow_isMinOn`.
Conversely, `hcrit` gives back this hypothesis (`or_isStrictSaddle_iff`). -/
theorem isMinOn_or_isStrictSaddle_of_inner_nonneg (hf : ContDiff ℝ 2 f)
    (h2 : ∀ z, gradient f z = 0 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
      IsMinOn f univ z) :
    ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z :=
  fun z hz => (or_isStrictSaddle_iff hf hz).2 (h2 z hz)

/-- **The landscape hypothesis for local minimizers, in second-order form.**
If every critical point of a `C²` function at which the Hessian is positive
semidefinite is a local minimizer, every critical point is a local minimizer
or a strict saddle: the hypothesis `hcrit` of `ae_exists_tendsto_isLocalMin`.
Conversely, `hcrit` gives back this hypothesis (`or_isStrictSaddle_iff`). -/
theorem isLocalMin_or_isStrictSaddle_of_inner_nonneg (hf : ContDiff ℝ 2 f)
    (h2 : ∀ z, gradient f z = 0 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
      IsLocalMin f z) :
    ∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z :=
  fun z hz => (or_isStrictSaddle_iff hf hz).2 (h2 z hz)

/-- **The landscape hypothesis below a level `c`, for global minimizers.**  If
every critical point `z` of a `C²` function with `f z < c` at which the
Hessian is positive semidefinite is a global minimizer, every critical point
below `c` is a global minimizer or a strict saddle.  Any other restriction of
the critical points is the same application of `or_isStrictSaddle_iff`. -/
theorem isMinOn_or_isStrictSaddle_of_inner_nonneg_of_lt (hf : ContDiff ℝ 2 f) {c : ℝ}
    (h2 : ∀ z, gradient f z = 0 → f z < c → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
      IsMinOn f univ z) :
    ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z :=
  fun z hz hc => (or_isStrictSaddle_iff hf hz).2 (h2 z hz hc)

/-- **The landscape hypothesis below a level `c`, for local minimizers.**  If
every critical point `z` of a `C²` function with `f z < c` at which the
Hessian is positive semidefinite is a local minimizer, every critical point
below `c` is a local minimizer or a strict saddle. -/
theorem isLocalMin_or_isStrictSaddle_of_inner_nonneg_of_lt (hf : ContDiff ℝ 2 f) {c : ℝ}
    (h2 : ∀ z, gradient f z = 0 → f z < c → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
      IsLocalMin f z) :
    ∀ z, gradient f z = 0 → f z < c → IsLocalMin f z ∨ IsStrictSaddle f z :=
  fun z hz hc => (or_isStrictSaddle_iff hf hz).2 (h2 z hz hc)

/-- The hypotheses of the theorems above hold for the double well
`f(x) = (x² - 1)² / 4` on the line, at its critical point `0` and with
`c = 1`.  Its gradient is `x³ - x` and its Hessian `3 x² - 1`: the critical
point `0`, where the Hessian is `-1`, is a strict saddle, and the critical
points `±1`, where it is `2`, are global minimizers. -/
example : let f := fun x : ℝ => (x ^ 2 - 1) ^ 2 / 4
    ContDiff ℝ 2 f ∧ gradient f 0 = 0 ∧ IsStrictSaddle f 0 ∧
      (∀ z, gradient f z = 0 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
        IsMinOn f univ z) ∧
      (∀ z, gradient f z = 0 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) → IsLocalMin f z) ∧
      (∀ z, gradient f z = 0 → f z < 1 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
        IsMinOn f univ z) ∧
      (∀ z, gradient f z = 0 → f z < 1 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
        IsLocalMin f z) := by
  intro f
  have hf : ContDiff ℝ 2 f := by fun_prop
  have hg : ∀ x, gradient f x = x ^ 3 - x := fun x => by
    rw [gradient_eq_deriv']
    exact ((((hasDerivAt_pow 2 x).sub_const 1).pow 2).div_const 4).deriv.trans (by norm_num; ring)
  have hd : ∀ z, ⟪fderiv ℝ (gradient f) z 1, 1⟫ = 3 * z ^ 2 - 1 := fun z => by
    rw [fderiv_apply_one_eq_deriv, funext hg, Real.inner_apply, mul_one,
      ((hasDerivAt_pow 3 z).fun_sub (hasDerivAt_id' z)).deriv]
    norm_num
  have hmin : ∀ z, gradient f z = 0 → (∀ h, 0 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫) →
      IsMinOn f univ z := by
    intro z hz hpsd
    have h1 := hpsd 1
    rw [hd] at h1
    rw [hg] at hz
    have hz2 : z ^ 2 = 1 := by
      rcases mul_eq_zero.1 (show z * (z ^ 2 - 1) = 0 by linear_combination hz) with h0 | h0
      · rw [h0] at h1
        norm_num at h1
      · linarith
    refine isMinOn_univ_iff.2 fun x => ?_
    show (z ^ 2 - 1) ^ 2 / 4 ≤ (x ^ 2 - 1) ^ 2 / 4
    rw [hz2, sub_self, zero_pow two_ne_zero, zero_div]
    positivity
  refine ⟨hf, by rw [hg]; norm_num,
    isStrictSaddle_of_inner_neg hf (by rw [hg]; norm_num) (h := 1) (by rw [hd]; norm_num), hmin,
    fun z hz hp => (hmin z hz hp).isLocalMin Filter.univ_mem, fun z hz _ hp => hmin z hz hp,
    fun z hz _ hp => (hmin z hz hp).isLocalMin Filter.univ_mem⟩

end Transformer.Landscape.SaddleAvoidance
