import Transformer.Modes.Section5_UnitPhase
import Transformer.Modes.Section2_Gt

/-!
# Derivatives and curvature of the Gaussian phase curve

The phase in arXiv:2412.09080v3, §5.5, is a linear combination of
`g(u) = u exp (-βu²/2)` and `g'(u)`. This module proves the printed
formulas for `g''`, `g'''`, and the determinant of the first two curve
derivatives, using `u = t - x` so the functions agree with `Section2_Gt`.
The formulas are linked to the actual derivatives by `HasDerivAt`.

For `β > 0` the determinant is strictly negative:
`g' g''' - (g'')² = -β (β²u⁴ + 3) exp (-βu²)`.
Consequently the first two derivatives of a phase with nonzero direction
cannot both vanish. This gives the source's pointwise nondegeneracy;
quantitative lower bounds on bounded intervals require a further estimate.

The same paragraph incorrectly states that any stationary point of
`φ_θ` satisfies `g'(u) = 0`. For direction `(0, 1)`, the phase is `g'`:
it is stationary at `u = 0`, while `g'(0) = 1`. A theorem below proves
this counterexample with a unit direction. The determinant argument for
nondegeneracy remains valid without that sentence.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The source's second Gaussian derivative, evaluated at `t - x`.
Source: arXiv:2412.09080v3, §5.5, the display for `g''`. -/
noncomputable def bigG'' (β t x : ℝ) : ℝ :=
  Real.exp (-(β / 2) * (t - x) ^ 2) * (β * (t - x) * (β * (t - x) ^ 2 - 3))

/-- The source's third Gaussian derivative, evaluated at `t - x`.
Source: arXiv:2412.09080v3, §5.5, the display for `g'''`. -/
noncomputable def bigG''' (β t x : ℝ) : ℝ :=
  Real.exp (-(β / 2) * (t - x) ^ 2) *
    (β * (-(β ^ 2) * (t - x) ^ 4 + 6 * β * (t - x) ^ 2 - 3))

/-- The printed formula for `g''` is the actual derivative of `g'`.
Source: arXiv:2412.09080v3, §5.5, the Gaussian derivative display. -/
theorem hasDerivAt_bigG' (β x t : ℝ) :
    HasDerivAt (fun s => bigG' β s x) (bigG'' β t x) t := by
  have hu := (hasDerivAt_id t).sub_const x
  have hpoly := (hasDerivAt_const t (1 : ℝ)).fun_sub ((hu.fun_pow 2).const_mul β)
  have h := (hasDerivAt_bump β x t).fun_mul hpoly
  unfold bigG'
  exact h.congr_deriv (by unfold bigG''; simp only [id_eq]; ring)

/-- The printed formula for `g'''` is the actual derivative of `g''`.
Source: arXiv:2412.09080v3, §5.5, the Gaussian derivative display. -/
theorem hasDerivAt_bigG'' (β x t : ℝ) :
    HasDerivAt (fun s => bigG'' β s x) (bigG''' β t x) t := by
  have hu := (hasDerivAt_id t).sub_const x
  have hpoly := (hu.const_mul β).fun_mul (((hu.fun_pow 2).const_mul β).sub_const 3)
  have h := (hasDerivAt_bump β x t).fun_mul hpoly
  unfold bigG''
  exact h.congr_deriv (by unfold bigG'''; simp only [id_eq]; ring)

/-- The determinant formula for the Gaussian curve `(g, g')`.
Source: arXiv:2412.09080v3, §5.5, the determinant display. -/
theorem gaussian_curve_determinant (β t x : ℝ) :
    bigG' β t x * bigG''' β t x - (bigG'' β t x) ^ 2 =
      -β * (β ^ 2 * (t - x) ^ 4 + 3) * Real.exp (-β * (t - x) ^ 2) := by
  have he : Real.exp (-(β / 2) * (t - x) ^ 2) *
      Real.exp (-(β / 2) * (t - x) ^ 2) = Real.exp (-β * (t - x) ^ 2) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc _ = -β * (β ^ 2 * (t - x) ^ 4 + 3) *
          (Real.exp (-(β / 2) * (t - x) ^ 2) * Real.exp (-(β / 2) * (t - x) ^ 2)) := by
        unfold bigG' bigG'' bigG'''
        ring
    _ = _ := by rw [he]

/-- The curve determinant is strictly negative for positive `β`.
Source: arXiv:2412.09080v3, §5.5, the nonvanishing determinant. -/
theorem gaussian_curve_determinant_neg {β : ℝ} (hβ : 0 < β) (t x : ℝ) :
    bigG' β t x * bigG''' β t x - (bigG'' β t x) ^ 2 < 0 := by
  rw [gaussian_curve_determinant]
  have hpoly : 0 < β ^ 2 * (t - x) ^ 4 + 3 := by positivity
  have he := Real.exp_pos (-β * (t - x) ^ 2)
  nlinarith [mul_pos hβ hpoly, mul_pos (mul_pos hβ hpoly) he]

example : (0 : ℝ) < 1 := one_pos

/-- The first derivative of the actual Gaussian phase.
Source: arXiv:2412.09080v3, §5.5, the formula for `φ_θ'`. -/
theorem hasDerivAt_gaussianPhase (β x t : ℝ) (θ : ℝ × ℝ) :
    HasDerivAt (fun s => θ.1 * bigG β s x + θ.2 * bigG' β s x)
      (θ.1 * bigG' β t x + θ.2 * bigG'' β t x) t := by
  exact ((hasDerivAt_bigG β x t).const_mul θ.1).fun_add
    ((hasDerivAt_bigG' β x t).const_mul θ.2)

/-- The second derivative of the actual Gaussian phase.
Source: arXiv:2412.09080v3, §5.5, the formula for `φ_θ''`. -/
theorem hasDerivAt_gaussianPhase' (β x t : ℝ) (θ : ℝ × ℝ) :
    HasDerivAt (fun s => θ.1 * bigG' β s x + θ.2 * bigG'' β s x)
      (θ.1 * bigG'' β t x + θ.2 * bigG''' β t x) t := by
  exact ((hasDerivAt_bigG' β x t).const_mul θ.1).fun_add
    ((hasDerivAt_bigG'' β x t).const_mul θ.2)

/-- A nonzero phase direction has no simultaneous zero of its first
and second derivatives. Source: arXiv:2412.09080v3, §5.5, nondegeneracy. -/
theorem gaussian_phase_derivatives_ne {β : ℝ} (hβ : 0 < β) {θ : ℝ × ℝ}
    (hθ : θ ≠ 0) (t x : ℝ) :
    (θ.1 * bigG' β t x + θ.2 * bigG'' β t x) ≠ 0 ∨
      (θ.1 * bigG'' β t x + θ.2 * bigG''' β t x) ≠ 0 := by
  by_contra h
  push Not at h
  obtain ⟨hp, hq⟩ := h
  have hd := gaussian_curve_determinant_neg hβ t x
  have h1 : θ.1 * (bigG' β t x * bigG''' β t x - (bigG'' β t x) ^ 2) = 0 := by
    linear_combination bigG''' β t x * hp - bigG'' β t x * hq
  have h2 : θ.2 * (bigG' β t x * bigG''' β t x - (bigG'' β t x) ^ 2) = 0 := by
    linear_combination bigG' β t x * hq - bigG'' β t x * hp
  have hθ1 : θ.1 = 0 := (mul_eq_zero.mp h1).resolve_right hd.ne
  have hθ2 : θ.2 = 0 := (mul_eq_zero.mp h2).resolve_right hd.ne
  exact hθ (Prod.ext hθ1 hθ2)

example : (0 : ℝ) < 1 ∧ ((0, 1) : ℝ × ℝ) ≠ 0 := by
  refine ⟨one_pos, ?_⟩
  intro h
  have h1 := congrArg Prod.snd h
  norm_num at h1

/-- Every stationary point of a Gaussian phase with nonzero direction
is nondegenerate. The hypothesis is the actual stationary equation,
without the source's incorrect assertion that `g'` must vanish there.
Source: arXiv:2412.09080v3, §5.5, the nondegeneracy paragraph. -/
theorem gaussian_stationary_phase_nondegenerate {β : ℝ} (hβ : 0 < β) {θ : ℝ × ℝ}
    (hθ : θ ≠ 0) (t x : ℝ)
    (hstationary : θ.1 * bigG' β t x + θ.2 * bigG'' β t x = 0) :
    θ.1 * bigG'' β t x + θ.2 * bigG''' β t x ≠ 0 := by
  rcases gaussian_phase_derivatives_ne hβ hθ t x with hp | hq
  · exact False.elim (hp hstationary)
  · exact hq

example : (0 : ℝ) < 1 ∧ ((0, 1) : ℝ × ℝ) ≠ 0 ∧
    (0 : ℝ) * bigG' 1 0 0 + 1 * bigG'' 1 0 0 = 0 := by
  refine ⟨one_pos, ?_, ?_⟩
  · intro h
    have h1 := congrArg Prod.snd h
    norm_num at h1
  · norm_num [bigG'']

/-- The source's claim that every stationary point of `φ_θ` satisfies
`g'(u) = 0` is false: direction `(0, 1)` gives phase `g'`, stationary
at zero although `g'(0) = 1`.
Source: arXiv:2412.09080v3, §5.5, the paragraph after the determinant. -/
theorem stationary_phase_need_not_zero_bigG' :
    ∃ θ : ℝ × ℝ, θ.1 ^ 2 + θ.2 ^ 2 = 1 ∧
      HasDerivAt (fun s => θ.1 * bigG 1 s 0 + θ.2 * bigG' 1 s 0) 0 0 ∧
      bigG' 1 0 0 = 1 := by
  refine ⟨(0, 1), by norm_num, ?_, ?_⟩
  · have h := hasDerivAt_gaussianPhase 1 0 0 (0, 1)
    simpa [bigG''] using h
  · norm_num [bigG']

end Transformer.Modes
