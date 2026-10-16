/-
# Lower bounds for the radial rotation approximation error

Source: arXiv:2604.01978v1, §6.1, `lem:stability_generator`, and
§2.3.2, the improved estimate following `thm:weak_error_clean`.

The exact head, law, chain, and modified equation are verified in
`Section2_RadialRotation` and `Section2_RadialChain`.
This module derives explicit lower bounds for their discrepancies.
The observable is the second coordinate of the sole planar token;
it is linear and therefore satisfies the source's C⁴ requirement.

For one step the error is at least `η²/3` when `0<η≤1`.
At time one, with `η=1/m` and `L=m`, it is at least
`cos(1)/(m+1)`. Both estimates use the actual normalized chain.
They defeat every proposed constant in the source's improved
rates, including when the comparison is restricted to grid times.
-/

import Transformer.Homogenized.Section2_RadialChain

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The actual observable `φ(X)=(x₀)₁` is linear and C⁴.
Source: arXiv:2604.01978v1, §6.1, the observable in `lem:stability_generator`. -/
theorem contDiff_radialRotObservable : ContDiff ℝ 4 (fun x : Idx 1 → EucSpace 2 => x 0 1) :=
  ((EuclideanSpace.proj (1 : Fin 2) : EucSpace 2 →L[ℝ] ℝ).comp
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Idx 1 => EucSpace 2) 0)).contDiff

/-- The radial value component slows the actual first step by at least `η²/2`.
Source: arXiv:2604.01978v1, §2.1, the normalization in `eq:update_tokens`. -/
theorem radialRotChain_one_le {η : ℝ} (hη : 0 < η) (hη1 : η ≤ 1) (β : ℝ) :
    radialRotChain η β 1 0 1 ≤ η - η ^ 2 / 2 := by
  rw [radialRotChain_one]
  have hs : 1 ≤ Real.sqrt (1 + (η / (1 + η)) ^ 2) := by
    simpa only [Real.sqrt_one] using
      Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + (η / (1 + η)) ^ 2 from
        le_add_of_nonneg_right (sq_nonneg (η / (1 + η))))
  refine (div_le_self (by positivity) hs).trans ?_
  apply (div_le_iff₀ (by linarith : 0 < 1 + η)).mpr
  nlinarith [mul_nonneg (sq_nonneg η) (sub_nonneg.mpr hη1)]

/-- The actual one-step weak discrepancy is at least `η²/3`.
Source: arXiv:2604.01978v1, §6.1, the two processes in `lem:stability_generator`. -/
theorem radialRotChain_one_gap {η : ℝ} (hη : 0 < η) (hη1 : η ≤ 1) (β : ℝ) :
    η ^ 2 / 3 ≤ Real.sin η - radialRotChain η β 1 0 1 := by
  have hd := radialRotChain_one_le hη hη1 β
  have hs := Real.sin_ge_sub_cube hη.le
  have hc := mul_le_mul_of_nonneg_left hη1 (sq_nonneg η)
  nlinarith

/-- Every candidate constant fails the source's one-step `Cη³` estimate.
Both expectations here use the actual probability measure on `Unit`.
Source: arXiv:2604.01978v1, §6.1, `lem:stability_generator`. -/
theorem radialRot_one_step_constant (C : ℝ) (hC : 0 < C) :
    ∃ η : ℝ, 0 < η ∧ η ≤ 1 ∧
      C * η ^ 2 * max η (alphaOf η 0 1) <
        |(∫ _ : Unit, (rotFlow η 0) 1 ∂Measure.dirac ()) -
          ∫ _ : Unit, radialRotChain η 1 1 0 1 ∂Measure.dirac ()| := by
  let η : ℝ := 1 / (3 * C + 1)
  have hden : 0 < 3 * C + 1 := by linarith
  have hη : 0 < η := one_div_pos.mpr hden
  have hη1 : η ≤ 1 := (div_le_iff₀ hden).mpr (by linarith)
  have hCη : C * η < 1 / 3 := by
    dsimp only [η]
    rw [mul_one_div, div_lt_iff₀ hden]
    linarith
  have hg := radialRotChain_one_gap hη hη1 1
  have hm := mul_lt_mul_of_pos_right hCη (sq_pos_of_pos hη)
  refine ⟨η, hη, hη1, ?_⟩
  simp only [integral_dirac, rotFlow_apply, Matrix.cons_val_one, Matrix.cons_val_fin_one,
    alphaOf, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div, max_eq_left hη.le]
  have hab := le_abs_self (Real.sin η - radialRotChain η 1 1 0 1)
  nlinarith


/-- On the angular interval `[0,1]`, sine increases at slope at least `cos(1)`.
This converts the actual grid-time phase lag into an observable discrepancy.
Source: arXiv:2604.01978v1, §2.3.2, the improved weak approximation. -/
theorem sin_one_sub_lower {θ : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    Real.cos 1 * (1 - θ) ≤ Real.sin 1 - Real.sin θ := by
  apply (convex_Icc (0 : ℝ) 1).mul_sub_le_image_sub_of_le_deriv
    Real.continuous_sin.continuousOn Real.differentiable_sin.differentiableOn
    (fun x hx => ?_) θ ⟨hθ0, hθ1⟩ 1 ⟨by norm_num, le_rfl⟩ hθ1
  rw [(Real.hasDerivAt_sin x).deriv]
  have hx' := interior_subset hx
  exact Real.cos_le_cos_of_nonneg_of_le_pi hx'.1 (by linarith [Real.two_le_pi]) hx'.2

/-- At physical time one the chain has an observable error of order `1/m`.
The exact phase is `m arctan(1/(m+1))`, whereas the modified SDE has phase one.
Source: arXiv:2604.01978v1, §2.3.2, the improved grid-time approximation. -/
theorem radialRotChain_grid_gap (m : ℕ) (hm : 0 < m) (β : ℝ) :
    Real.cos 1 / ((m : ℝ) + 1) ≤ Real.sin 1 - radialRotChain (1 / (m : ℝ)) β m 0 1 := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have he : (1 / (m : ℝ)) / (1 + 1 / m) = 1 / ((m : ℝ) + 1) := by
    field_simp
  rw [radialRotChain, he, rotChain_eq_rotFlow, rotFlow_apply]
  simp only [Matrix.cons_val_one, Matrix.cons_val_fin_one]
  let θ : ℝ := (m : ℝ) * Real.arctan (1 / ((m : ℝ) + 1))
  have hθ0 : 0 ≤ θ := mul_nonneg hm0.le (Real.arctan_nonneg.mpr (by positivity))
  have hθb : θ ≤ (m : ℝ) / ((m : ℝ) + 1) := by
    simpa only [mul_one_div] using
      mul_le_mul_of_nonneg_left (Real.arctan_le_self (by positivity : 0 ≤ 1 / ((m : ℝ) + 1))) hm0.le
  have hdiv : (m : ℝ) / ((m : ℝ) + 1) + 1 / ((m : ℝ) + 1) = 1 := by
    field_simp
  have hc : 0 < Real.cos 1 := Real.cos_pos_of_le_one (by norm_num)
  have hinv : 0 < 1 / ((m : ℝ) + 1) := by positivity
  have hs := sin_one_sub_lower hθ0 (by linarith : θ ≤ 1)
  have hmul := mul_le_mul_of_nonneg_left (show 1 / ((m : ℝ) + 1) ≤ 1 - θ by linarith) hc.le
  simpa only [mul_one_div] using hmul.trans hs

/-- Every candidate constant fails the improved estimate already at a grid time.
Here `η=1/m`, `L=m`, and the time horizon is exactly one.
Source: arXiv:2604.01978v1, §2.3.2, the paragraph after `thm:weak_error_clean`. -/
theorem radialRot_grid_constant (C : ℝ) (hC : 1 ≤ C) :
    ∃ m : ℕ, 0 < m ∧
      C * Real.exp C * (1 / (m : ℝ)) * (1 + 1) * (1 / (m : ℝ)) <
        |Real.sin 1 - radialRotChain (1 / (m : ℝ)) 1 m 0 1| := by
  have hc : 0 < Real.cos 1 := Real.cos_pos_of_le_one (by norm_num)
  obtain ⟨m, hm⟩ := exists_nat_gt (max 1 (4 * (C * Real.exp C) / Real.cos 1))
  have hm1 : (1 : ℝ) < m := (le_max_left _ _).trans_lt hm
  have hm0 : (0 : ℝ) < m := by linarith
  have hmN : 0 < m := by exact_mod_cast hm0
  have hlarge : 4 * (C * Real.exp C) < (m : ℝ) * Real.cos 1 :=
    (div_lt_iff₀ hc).mp ((le_max_right _ _).trans_lt hm)
  have hA : 0 < C * Real.exp C := mul_pos (by linarith) (Real.exp_pos _)
  have hsmall : 2 * (C * Real.exp C) * ((m : ℝ) + 1) < Real.cos 1 * (m : ℝ) ^ 2 := by
    have hmul := mul_lt_mul_of_pos_right hlarge hm0
    have hmul2 := mul_pos hA (sub_pos.mpr hm1)
    nlinarith
  have hdiv : 2 * (C * Real.exp C) / (m : ℝ) ^ 2 < Real.cos 1 / ((m : ℝ) + 1) :=
    (div_lt_div_iff₀ (sq_pos_of_pos hm0) (by linarith)).mpr hsmall
  have hgap := radialRotChain_grid_gap m hmN 1
  have hab := le_abs_self (Real.sin 1 - radialRotChain (1 / (m : ℝ)) 1 m 0 1)
  refine ⟨m, hmN, ?_⟩
  have he : C * Real.exp C * (1 / (m : ℝ)) * (1 + 1) * (1 / (m : ℝ)) =
      2 * (C * Real.exp C) / (m : ℝ) ^ 2 := by ring
  rw [he]
  exact hdiv.trans_le (hgap.trans hab)

/-- The small-step hypotheses hold jointly at a positive physical step. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := ⟨zero_lt_one, le_rfl⟩

/-- A genuine angular value satisfies both hypotheses of the sine estimate. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- Positive layer counts and candidate constants exist. -/
example : 0 < (1 : ℕ) ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 :=
  ⟨Nat.one_pos, zero_lt_one, le_rfl⟩

end Transformer.Homogenized
