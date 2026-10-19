module

public import Transformer.Causal.Section4_AnisotropicBoundary

/-!
# Actual coordinate margins for the anisotropic causal field

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5` and
`eqn:csa`, at the actual matrices of `Section4_AnisotropicCircle`.
The region keeps the leader near its leading eigenvector and the
follower in the negative-cosine semicircle with sine between one
quarter and one half. These are actual coordinate inequalities.

The six margins below are just distances to these coordinate edges.
Their displayed velocities are proved to be their actual derivatives
along the original field. Every feasible boundary has strictly inward
derivative. The two cosine edges cannot be reached simultaneously
with the other coordinate bounds, by the original unit-circle identity.
No invariant-region or convergence conclusion is included in a definition.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The six actual coordinate margins, before any invariance assertion.
Source: arXiv:2411.04990v2, §4, the circle coordinates in `eqn:csa`. -/
noncomputable def anisotropicMargins (φ : Fin 2 → ℝ) : Fin 6 → ℝ :=
  ![1 / 100 - Real.sin (φ 0), 1 / 100 + Real.sin (φ 0), Real.cos (φ 0) - 99 / 100,
    Real.sin (φ 1) - 1 / 4, 1 / 2 - Real.sin (φ 1), -Real.cos (φ 1)]

/-- The displayed derivatives of the original six coordinate margins.
Their identification with the actual derivatives is proved below.
Source: arXiv:2411.04990v2, §4, differentiating the actual `eqn:csa`. -/
noncomputable def anisotropicMarginVelocities (φ : Fin 2 → ℝ) : Fin 6 → ℝ :=
  ![Real.cos (φ 0) ^ 2 * Real.sin (φ 0), -Real.cos (φ 0) ^ 2 * Real.sin (φ 0),
    Real.cos (φ 0) * Real.sin (φ 0) ^ 2, Real.cos (φ 1) * anisotropicAngleField φ 1,
    -Real.cos (φ 1) * anisotropicAngleField φ 1, Real.sin (φ 1) * anisotropicAngleField φ 1]

/-- The actual margin derivatives along every genuine original angular
solution. Source: arXiv:2411.04990v2, §4, the coordinate chain rule for `eqn:csa`. -/
theorem hasDerivAt_anisotropicMargins (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) (t : ℝ) (i : Fin 6) :
    HasDerivAt (fun s => anisotropicMargins (φ s) i) (anisotropicMarginVelocities (φ t) i) t := by
  have hs₀ := hasDerivAt_anisotropic_leader_sin φ hφ t
  have hc₀ := hasDerivAt_anisotropic_leader_cos φ hφ t
  have h₁ := hasDerivAt_anisotropicAngle_coordinate φ hφ t 1
  fin_cases i
  · change HasDerivAt (fun s => 1 / 100 - Real.sin (φ s 0))
      (Real.cos (φ t 0) ^ 2 * Real.sin (φ t 0)) t
    simpa only [neg_mul, neg_neg] using hs₀.const_sub (1 / 100)
  · change HasDerivAt (fun s => 1 / 100 + Real.sin (φ s 0))
      (-Real.cos (φ t 0) ^ 2 * Real.sin (φ t 0)) t
    exact hs₀.const_add (1 / 100)
  · change HasDerivAt (fun s => Real.cos (φ s 0) - 99 / 100)
      (Real.cos (φ t 0) * Real.sin (φ t 0) ^ 2) t
    exact hc₀.sub_const (99 / 100)
  · change HasDerivAt (fun s => Real.sin (φ s 1) - 1 / 4)
      (Real.cos (φ t 1) * anisotropicAngleField (φ t) 1) t
    exact h₁.sin.sub_const (1 / 4)
  · change HasDerivAt (fun s => 1 / 2 - Real.sin (φ s 1))
      (-Real.cos (φ t 1) * anisotropicAngleField (φ t) 1) t
    simpa only [neg_mul] using h₁.sin.const_sub (1 / 2)
  · change HasDerivAt (fun s => -Real.cos (φ s 1))
      (Real.sin (φ t 1) * anisotropicAngleField (φ t) 1) t
    simpa only [neg_mul, neg_neg] using h₁.cos.fun_neg

/-- Every feasible boundary of the original coordinate region has a
strictly positive margin derivative. Source: arXiv:2411.04990v2,
§4, `eqn:csa`, the boundary test for its anisotropic two-token field.
The strict follower signs are proved from the original exponential weights. -/
theorem anisotropicMarginVelocities_pos_on_boundary (φ : Fin 2 → ℝ)
    (hall : ∀ j : Fin 6, 0 ≤ anisotropicMargins φ j) (i : Fin 6)
    (hi : anisotropicMargins φ i = 0) : 0 < anisotropicMarginVelocities φ i := by
  have h₀ := hall 0
  have h₁ := hall 1
  have h₂ := hall 2
  have h₃ := hall 3
  have h₄ := hall 4
  have h₅ := hall 5
  change 0 ≤ 1 / 100 - Real.sin (φ 0) at h₀
  change 0 ≤ 1 / 100 + Real.sin (φ 0) at h₁
  change 0 ≤ Real.cos (φ 0) - 99 / 100 at h₂
  change 0 ≤ Real.sin (φ 1) - 1 / 4 at h₃
  change 0 ≤ 1 / 2 - Real.sin (φ 1) at h₄
  change 0 ≤ -Real.cos (φ 1) at h₅
  have hs₀ : |Real.sin (φ 0)| ≤ 1 / 100 := abs_le.mpr ⟨by linarith, by linarith⟩
  have hc₀ : 99 / 100 ≤ Real.cos (φ 0) := by linarith
  have hc₀pos : 0 < Real.cos (φ 0) := by linarith
  have hs₁ : Real.sin (φ 1) ∈ Icc (1 / 4) (1 / 2) := ⟨by linarith, by linarith⟩
  have hc₁ : Real.cos (φ 1) ≤ 0 := by linarith
  fin_cases i
  · change 1 / 100 - Real.sin (φ 0) = 0 at hi
    have hs : Real.sin (φ 0) = 1 / 100 := by linarith
    change 0 < Real.cos (φ 0) ^ 2 * Real.sin (φ 0)
    rw [hs]
    positivity
  · change 1 / 100 + Real.sin (φ 0) = 0 at hi
    have hs : Real.sin (φ 0) = -(1 / 100) := by linarith
    change 0 < -Real.cos (φ 0) ^ 2 * Real.sin (φ 0)
    rw [hs, mul_neg, neg_mul, neg_neg]
    positivity
  · change Real.cos (φ 0) - 99 / 100 = 0 at hi
    have hsq := (sq_le_sq₀ (abs_nonneg (Real.sin (φ 0)))
      (by norm_num : (0 : ℝ) ≤ 1 / 100)).mpr hs₀
    rw [sq_abs] at hsq
    nlinarith [Real.sin_sq_add_cos_sq (φ 0)]
  · change Real.sin (φ 1) - 1 / 4 = 0 at hi
    have hs : Real.sin (φ 1) = 1 / 4 := by linarith
    have hcn : Real.cos (φ 1) < 0 := by nlinarith [Real.sin_sq_add_cos_sq (φ 1)]
    exact mul_pos_of_neg_of_neg hcn (anisotropicAngleField_neg_at_quarter φ hc₀ hs₀ hs hc₁)
  · change 1 / 2 - Real.sin (φ 1) = 0 at hi
    have hs : Real.sin (φ 1) = 1 / 2 := by linarith
    have hcn : Real.cos (φ 1) < 0 := by nlinarith [Real.sin_sq_add_cos_sq (φ 1)]
    exact mul_pos (neg_pos.mpr hcn) (anisotropicAngleField_pos_at_half φ hs₀ hs hc₁)
  · change -Real.cos (φ 1) = 0 at hi
    have hsq := (sq_le_sq₀ (show 0 ≤ Real.sin (φ 1) by linarith [hs₁.1])
      (by norm_num : (0 : ℝ) ≤ 1 / 2)).mpr hs₁.2
    nlinarith [Real.sin_sq_add_cos_sq (φ 1)]

/-- Nonnegative margins give the actual weak leader and follower bounds.
Source: arXiv:2411.04990v2, §4, the coordinate region for the original `eqn:csa`. -/
theorem coordinate_bounds_of_anisotropicMargins_nonneg (φ : Fin 2 → ℝ)
    (hall : ∀ j : Fin 6, 0 ≤ anisotropicMargins φ j) :
    |Real.sin (φ 0)| ≤ 1 / 100 ∧ 99 / 100 ≤ Real.cos (φ 0) ∧
      Real.sin (φ 1) ∈ Icc (1 / 4) (1 / 2) ∧ Real.cos (φ 1) ≤ 0 := by
  have h₀ := hall 0; have h₁ := hall 1; have h₂ := hall 2
  have h₃ := hall 3; have h₄ := hall 4; have h₅ := hall 5
  change 0 ≤ 1 / 100 - Real.sin (φ 0) at h₀
  change 0 ≤ 1 / 100 + Real.sin (φ 0) at h₁
  change 0 ≤ Real.cos (φ 0) - 99 / 100 at h₂
  change 0 ≤ Real.sin (φ 1) - 1 / 4 at h₃
  change 0 ≤ 1 / 2 - Real.sin (φ 1) at h₄
  change 0 ≤ -Real.cos (φ 1) at h₅
  exact ⟨abs_le.mpr ⟨by linarith, by linarith⟩, by linarith, ⟨by linarith, by linarith⟩, by linarith⟩

/-- All derivative and nonnegative-boundary hypotheses hold together for
an actual global solution starting at the quarter-coordinate edge.
Source: arXiv:2411.04990v2, §4, the original `eqn:csa` on the nonempty circle. -/
example : ∃ φ : ℝ → Fin 2 → ℝ,
    (∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) ∧
    (∀ j : Fin 6, 0 ≤ anisotropicMargins (φ 0) j) ∧ anisotropicMargins (φ 0) 3 = 0 := by
  obtain ⟨φ, h0, hφ⟩ := exists_anisotropicAngleFlow ![0, π - Real.arcsin (1 / 4)]
  have hs : Real.sin (π - Real.arcsin (1 / 4 : ℝ)) = 1 / 4 := by
    rw [Real.sin_pi_sub, Real.sin_arcsin (by norm_num) (by norm_num)]
  refine ⟨φ, hφ, ?_, ?_⟩
  · rw [h0]
    intro j
    fin_cases j
    · norm_num [anisotropicMargins]
    · norm_num [anisotropicMargins]
    · norm_num [anisotropicMargins]
    · change 0 ≤ Real.sin (π - Real.arcsin (1 / 4 : ℝ)) - 1 / 4
      rw [hs]
      norm_num
    · change 0 ≤ 1 / 2 - Real.sin (π - Real.arcsin (1 / 4 : ℝ))
      rw [hs]
      norm_num
    · change 0 ≤ -Real.cos (π - Real.arcsin (1 / 4 : ℝ))
      rw [Real.cos_pi_sub, Real.cos_arcsin, neg_neg]
      exact Real.sqrt_nonneg _
  · rw [h0]
    change Real.sin (π - Real.arcsin (1 / 4 : ℝ)) - 1 / 4 = 0
    rw [hs, sub_self]

end Transformer.Causal
