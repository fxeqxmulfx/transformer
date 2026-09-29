/-
# The number of modes of a Gaussian KDE — pointwise bounds behind the width of the integral over `T'`

The elementary half of the last step of `lem:main-int-phi` on `T'`
(`Section2_WidthIntegral.lean`).  With `R = ρ e^{-u/2} ≥ 1` and `u = t²`, the
integrand `e^{-κ R u}(1 + K √R (4 + 6u))` of the proxy Kac–Rice integral is
bounded by a sum of three Gaussians in `t = √u`:

* for `u ≥ 1` the factor `√R ≤ R u` is absorbed by `e^{-κ R u/2}`, since
  `x e^{-b x} ≤ 1/b`;
* for `u ≤ 1`, `ρ/2 ≤ R ≤ ρ`: the factor `√R ≤ √ρ` stays, and it is paid for by
  the width `ρ^{-1/2}` of the Gaussian `e^{-κ ρ u/2}` in `t`, in the integral.

Source: arXiv:2412.09080v3, `eq:int-phi-b` and the proof of `lem:main-int-phi`.
-/

import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral

open Real

namespace Transformer
namespace Modes

/-- `x e^{-b x} ≤ 1/b` for `b > 0`: from `b x ≤ e^{b x}`. -/
theorem mul_exp_neg_le {b x : ℝ} (hb : 0 < b) :
    x * Real.exp (-b * x) ≤ b⁻¹ := by
  have h1 : b * x * Real.exp (-b * x) ≤ 1 := by
    calc b * x * Real.exp (-b * x) ≤ Real.exp (b * x) * Real.exp (-b * x) :=
          mul_le_mul_of_nonneg_right (by linarith [Real.add_one_le_exp (b * x)])
            (Real.exp_pos _).le
      _ = 1 := by rw [← Real.exp_add]; simp
  calc x * Real.exp (-b * x) = (b * x * Real.exp (-b * x)) / b := by field_simp
    _ ≤ 1 / b := by gcongr
    _ = b⁻¹ := one_div b

/-- `(4 + 6u) e^{-κu/2} ≤ (4 + 24/κ) e^{-κu/4}`. -/
theorem poly_mul_exp_le {κ u : ℝ} (hκ : 0 < κ) (hu : 0 ≤ u) :
    (4 + 6 * u) * Real.exp (-(κ / 2) * u) ≤ (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by
  have hE := Real.exp_pos (-(κ / 4) * u)
  have hE1 : Real.exp (-(κ / 4) * u) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by nlinarith [mul_nonneg hκ.le hu])
  have hsplit : Real.exp (-(κ / 2) * u) =
      Real.exp (-(κ / 4) * u) * Real.exp (-(κ / 4) * u) := by
    rw [← Real.exp_add]; congr 1; ring
  have hu' : u * Real.exp (-(κ / 4) * u) ≤ (κ / 4)⁻¹ := mul_exp_neg_le (by positivity)
  have hinv : (κ / 4)⁻¹ = 4 / κ := by field_simp
  have hbound : (4 + 6 * u) * Real.exp (-(κ / 4) * u) ≤ 4 + 24 / κ := by
    calc (4 + 6 * u) * Real.exp (-(κ / 4) * u)
        = 4 * Real.exp (-(κ / 4) * u) + 6 * (u * Real.exp (-(κ / 4) * u)) := by ring
      _ ≤ 4 * 1 + 6 * (4 / κ) := by
          rw [hinv] at hu'
          gcongr
      _ = 4 + 24 / κ := by ring
  calc (4 + 6 * u) * Real.exp (-(κ / 2) * u)
      = ((4 + 6 * u) * Real.exp (-(κ / 4) * u)) * Real.exp (-(κ / 4) * u) := by
        rw [hsplit]; ring
    _ ≤ (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := mul_le_mul_of_nonneg_right hbound hE.le

/-- For `u ≥ 1`, `R ≥ 1`: `√R` is absorbed by `e^{-κ R u/2}`, since `√R ≤ R u`. -/
theorem width_term_large {κ K₁ R u : ℝ} (hκ : 0 < κ) (hK₁ : 0 ≤ K₁) (hR : 1 ≤ R)
    (hu1 : 1 ≤ u) :
    Real.exp (-κ * (R * u)) * (K₁ * Real.sqrt R * (4 + 6 * u)) ≤
      K₁ * (2 / κ) * (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by
  have hR0 : 0 ≤ R := by linarith
  have hu0 : 0 ≤ u := by linarith
  have hRu : 0 ≤ R * u := mul_nonneg hR0 hu0
  have hsqrt : Real.sqrt R ≤ R * u := by
    calc Real.sqrt R ≤ R := by rw [Real.sqrt_le_iff]; exact ⟨hR0, by nlinarith⟩
      _ ≤ R * u := by nlinarith
  have hsplit : Real.exp (-κ * (R * u)) =
      Real.exp (-(κ / 2) * (R * u)) * Real.exp (-(κ / 2) * (R * u)) := by
    rw [← Real.exp_add]; congr 1; ring
  have h1 : R * u * Real.exp (-(κ / 2) * (R * u)) ≤ (κ / 2)⁻¹ :=
    mul_exp_neg_le (by positivity)
  have h2 : Real.exp (-(κ / 2) * (R * u)) ≤ Real.exp (-(κ / 2) * u) :=
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ.le hu0) (sub_nonneg.2 hR)])
  have h3 := poly_mul_exp_le hκ hu0
  have h4 : (4 + 6 * u) * Real.exp (-(κ / 2) * (R * u)) ≤
      (4 + 6 * u) * Real.exp (-(κ / 2) * u) := by gcongr
  have hinv : (κ / 2)⁻¹ = 2 / κ := by field_simp
  calc Real.exp (-κ * (R * u)) * (K₁ * Real.sqrt R * (4 + 6 * u))
      ≤ Real.exp (-κ * (R * u)) * (K₁ * (R * u) * (4 + 6 * u)) := by gcongr
    _ = K₁ * (R * u * Real.exp (-(κ / 2) * (R * u))) *
          ((4 + 6 * u) * Real.exp (-(κ / 2) * (R * u))) := by rw [hsplit]; ring
    _ ≤ K₁ * (κ / 2)⁻¹ * ((4 + 24 / κ) * Real.exp (-(κ / 4) * u)) := by
        refine mul_le_mul (mul_le_mul_of_nonneg_left h1 hK₁) (h4.trans h3) (by positivity)
          (by positivity)
    _ = K₁ * (2 / κ) * (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by rw [hinv]; ring

/-- For `u ≤ 1` and `ρ/2 ≤ R ≤ ρ`: the factor `√R ≤ √ρ` is paid for by the width
`e^{-κ ρ u/2}` of the integral, whose integral is `√(2π/(κρ))`. -/
theorem width_term_small {κ K₁ ρ R u : ℝ} (hκ : 0 < κ) (hK₁ : 0 ≤ K₁) (hu : 0 ≤ u)
    (hRρ : R ≤ ρ) (hρR : ρ / 2 ≤ R) (hu1 : u ≤ 1) :
    Real.exp (-κ * (R * u)) * (K₁ * Real.sqrt R * (4 + 6 * u)) ≤
      10 * K₁ * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * u) := by
  have hE : Real.exp (-κ * (R * u)) ≤ Real.exp (-(κ / 2 * ρ) * u) :=
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ.le hu) (sub_nonneg.2 hρR)])
  have hS : Real.sqrt R ≤ Real.sqrt ρ := Real.sqrt_le_sqrt hRρ
  have h10 : 4 + 6 * u ≤ 10 := by linarith
  calc Real.exp (-κ * (R * u)) * (K₁ * Real.sqrt R * (4 + 6 * u))
      ≤ Real.exp (-(κ / 2 * ρ) * u) * (K₁ * Real.sqrt ρ * 10) := by
        refine mul_le_mul hE (mul_le_mul (mul_le_mul_of_nonneg_left hS hK₁) h10
          (by positivity) (by positivity)) (by positivity) (Real.exp_pos _).le
    _ = 10 * K₁ * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * u) := by ring

/-- **The pointwise majorant.**  If `R = ρ e^{-u/2} ≥ 1` and `u ≥ 0`, then
`e^{-κ R u}(1 + K √R (4 + 6u)) ≤ e^{-κ u} + 10 K √ρ e^{-κ ρ u/2}
  + K (2/κ)(4 + 24/κ) e^{-κ u/4}`; the three terms are Gaussians in `t = √u`. -/
theorem width_integrand_le {κ K₁ ρ u : ℝ} (hκ : 0 < κ) (hK₁ : 0 ≤ K₁) (hu : 0 ≤ u)
    (hR : 1 ≤ ρ * Real.exp (-u / 2)) :
    Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) *
        (1 + K₁ * Real.sqrt (ρ * Real.exp (-u / 2)) * (4 + 6 * u)) ≤
      Real.exp (-κ * u) + 10 * K₁ * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * u) +
        K₁ * (2 / κ) * (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by
  have he := Real.exp_pos (-u / 2)
  have he1 : Real.exp (-u / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hρ : 0 < ρ := by
    by_contra h
    push Not at h
    nlinarith
  have hRρ : ρ * Real.exp (-u / 2) ≤ ρ := by nlinarith
  have hA : Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) ≤ Real.exp (-κ * u) :=
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ.le hu) (sub_nonneg.2 hR)])
  have hB : Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) *
        (K₁ * Real.sqrt (ρ * Real.exp (-u / 2)) * (4 + 6 * u)) ≤
      10 * K₁ * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * u) +
        K₁ * (2 / κ) * (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by
    have hp1 : 0 ≤ 10 * K₁ * Real.sqrt ρ * Real.exp (-(κ / 2 * ρ) * u) := by positivity
    have hp2 : 0 ≤ K₁ * (2 / κ) * (4 + 24 / κ) * Real.exp (-(κ / 4) * u) := by positivity
    by_cases hu1 : u ≤ 1
    · have h2 : 1 / 2 ≤ Real.exp (-u / 2) := by linarith [Real.add_one_le_exp (-u / 2)]
      have := width_term_small hκ hK₁ hu hRρ (by nlinarith) hu1
      linarith
    · have := width_term_large hκ hK₁ hR (le_of_lt (not_le.mp hu1))
      linarith
  calc Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) *
        (1 + K₁ * Real.sqrt (ρ * Real.exp (-u / 2)) * (4 + 6 * u))
      = Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) +
        Real.exp (-κ * (ρ * Real.exp (-u / 2) * u)) *
          (K₁ * Real.sqrt (ρ * Real.exp (-u / 2)) * (4 + 6 * u)) := by ring
    _ ≤ _ := by linarith

/-- The hypotheses of `width_integrand_le` are satisfiable: `ρ = 1`, `u = 0`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ 1 ≤ (1 : ℝ) * Real.exp (-0 / 2) := by
  refine ⟨one_pos, le_rfl, le_rfl, by simp⟩

end Modes
end Transformer
