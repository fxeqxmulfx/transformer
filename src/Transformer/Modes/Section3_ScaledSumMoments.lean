import Transformer.Modes.Section3_ScaledSumCube

/-
# The number of modes of a Gaussian KDE — the mixed third moments of a
normalized sum

`Section3_ScaledSumCube.lean` proves the scaling law for every linear form at
once, `𝔼 (s (S_n)₁ + t (S_n)₂)³ = n^{-1/2} 𝔼 (s Y₁ + t Y₂)³`.  Here the four
mixed third moments are read off it.

Both sides are cubic forms in `(s, t)`:
`𝔼 (s x₁ + t x₂)³ = s³ m₃₀ + 3s²t m₂₁ + 3st² m₁₂ + t³ m₀₃`.  Evaluating at
`(1,0)`, `(0,1)`, `(1,1)` and `(1,-1)` gives four linear equations whose only
solution identifies the coefficients one by one — the cubic form determines its
coefficients, which is why the linear form loses nothing.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Real MeasureTheory

namespace Transformer
namespace Modes

variable {μ ν : Measure (ℝ × ℝ)} {ε : ℝ} {n : ℕ}

/-- **The cube of a linear form, integrated, is the cubic form in `(s, t)` built
from the four mixed third moments.**

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem integral_dot_cube_expand (hE : HasExpMomentsOn ν ε) (hε : 0 < ε) (s t : ℝ) :
    ∫ x, (s * x.1 + t * x.2) ^ 3 ∂ν
      = s ^ 3 * (∫ x, x.1 ^ 3 * x.2 ^ 0 ∂ν) + 3 * s ^ 2 * t * (∫ x, x.1 ^ 2 * x.2 ^ 1 ∂ν)
        + 3 * s * t ^ 2 * (∫ x, x.1 ^ 1 * x.2 ^ 2 ∂ν) + t ^ 3 * (∫ x, x.1 ^ 0 * x.2 ^ 3 ∂ν) := by
  have h1 := (integrable_pow_mul_pow hE hε 3 0).const_mul (s ^ 3)
  have h2 := (integrable_pow_mul_pow hE hε 2 1).const_mul (3 * s ^ 2 * t)
  have h3 := (integrable_pow_mul_pow hE hε 1 2).const_mul (3 * s * t ^ 2)
  have h4 := (integrable_pow_mul_pow hE hε 0 3).const_mul (t ^ 3)
  have hsplit : ∀ x : ℝ × ℝ, (s * x.1 + t * x.2) ^ 3
      = s ^ 3 * (x.1 ^ 3 * x.2 ^ 0) + 3 * s ^ 2 * t * (x.1 ^ 2 * x.2 ^ 1)
        + 3 * s * t ^ 2 * (x.1 ^ 1 * x.2 ^ 2) + t ^ 3 * (x.1 ^ 0 * x.2 ^ 3) := fun x => by ring
  have h12 : Integrable (fun x : ℝ × ℝ =>
      s ^ 3 * (x.1 ^ 3 * x.2 ^ 0) + 3 * s ^ 2 * t * (x.1 ^ 2 * x.2 ^ 1)) ν := h1.add h2
  have h123 : Integrable (fun x : ℝ × ℝ =>
      s ^ 3 * (x.1 ^ 3 * x.2 ^ 0) + 3 * s ^ 2 * t * (x.1 ^ 2 * x.2 ^ 1)
        + 3 * s * t ^ 2 * (x.1 ^ 1 * x.2 ^ 2)) ν := h12.add h3
  simp only [hsplit]
  rw [integral_add h123 h4, integral_add h12 h3, integral_add h1 h2,
    integral_const_mul, integral_const_mul, integral_const_mul, integral_const_mul]

/-- The hypotheses of `integral_dot_cube_expand` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 := ⟨hasExpMomentsOn_stdGauss2, one_pos⟩

/-- **Each mixed third moment of the normalized sum carries one factor
`n^{-1/2}`**: `𝔼 (S_n)₁^k (S_n)₂^{3-k} = n^{-1/2} 𝔼 Y₁^k Y₂^{3-k}` for
`k ≤ 3`, for a centred `μ` with exponential moments.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`, where the factor
`n^{-1/2}` is omitted (see `Section3_Cumulants.lean`). -/
theorem integral_map_scaledSum_monomial [IsProbabilityMeasure μ] (hE : HasExpMomentsOn μ ε)
    (hε : 0 < ε) (hfst : ∫ x, x.1 ∂μ = 0) (hsnd : ∫ x, x.2 ∂μ = 0) (hn : 1 ≤ n)
    (k : ℕ) (hk : k ≤ 3) :
    ∫ x, x.1 ^ k * x.2 ^ (3 - k) ∂((Measure.pi fun _ : Fin n => μ).map (scaledSum n))
      = (Real.sqrt n)⁻¹ * ∫ x, x.1 ^ k * x.2 ^ (3 - k) ∂μ := by
  set ρ := (Measure.pi fun _ : Fin n => μ).map (scaledSum n) with hρ
  have hEρ : HasExpMomentsOn ρ ε := hasExpMomentsOn_map_scaledSum hE hn
  have hcube : ∀ s t : ℝ,
      s ^ 3 * (∫ x, x.1 ^ 3 * x.2 ^ 0 ∂ρ) + 3 * s ^ 2 * t * (∫ x, x.1 ^ 2 * x.2 ^ 1 ∂ρ)
          + 3 * s * t ^ 2 * (∫ x, x.1 ^ 1 * x.2 ^ 2 ∂ρ) + t ^ 3 * (∫ x, x.1 ^ 0 * x.2 ^ 3 ∂ρ)
        = (Real.sqrt n)⁻¹ *
          (s ^ 3 * (∫ x, x.1 ^ 3 * x.2 ^ 0 ∂μ) + 3 * s ^ 2 * t * (∫ x, x.1 ^ 2 * x.2 ^ 1 ∂μ)
            + 3 * s * t ^ 2 * (∫ x, x.1 ^ 1 * x.2 ^ 2 ∂μ)
            + t ^ 3 * (∫ x, x.1 ^ 0 * x.2 ^ 3 ∂μ)) := by
    intro s t
    rw [← integral_dot_cube_expand hEρ hε s t, ← integral_dot_cube_expand hE hε s t, hρ]
    exact integral_map_scaledSum_dot_cube hE hε hfst hsnd hn s t
  have e10 := hcube 1 0
  have e01 := hcube 0 1
  have e11 := hcube 1 1
  have e1m := hcube 1 (-1)
  norm_num at e10 e01 e11 e1m ⊢
  interval_cases k <;> norm_num <;> linarith

/-- The hypotheses of `integral_map_scaledSum_monomial` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 ∧ ∫ x, x.1 ∂stdGauss2 = 0 ∧
    ∫ x, x.2 ∂stdGauss2 = 0 ∧ 1 ≤ 1 ∧ (0 : ℕ) ≤ 3 :=
  ⟨hasExpMomentsOn_stdGauss2, one_pos, isStandardized_stdGauss2.mean_fst,
    isStandardized_stdGauss2.mean_snd, le_rfl, Nat.zero_le 3⟩

end Modes
end Transformer
