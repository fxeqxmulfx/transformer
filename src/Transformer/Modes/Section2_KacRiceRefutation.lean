/-
# Counterexample to the printed general Kac–Rice claim

Source: arXiv:2412.09080v3, §2.1, `thm:kac-rice` and
`eq:krf`. The source uses the derivative-modulus probability
condition `P(ω(η)>ε)=O(η)`. With that precise condition its
claimed general formula is false.

Let the signed scale `R` be uniform on `(-1,1)` and let the
independent complex parameter `Z` be uniform on the unit
disk. The actual smooth random field is
`F(t)=R*(2+re(Z*exp(i*t/R)))`. The null zero-scale path is
handled explicitly by the definitions and probability law.

Every nonzero-scale path has the sign of `R` and avoids
level zero, so its expected zero-level upcrossing count on
the compact interval `[0,1]` is zero. The actual field and
derivative are bounded under the probability law. Their
actual joint density and the scalar marginal were derived
by exact changes of variables and Tonelli.

Both densities have the original required continuity near
level zero, with the joint density continuous for every real
derivative coordinate. The actual global derivative modulus
exceeds `ε` only when `|R|<η/ε` on the almost-sure disk
support, giving the precise printed `O(η)` hypothesis.

Nevertheless the actual joint density at level zero is
strictly positive for every derivative value in `(0,1)`.
The printed right side is therefore positive while its left
side is zero. The full original hypothesis structure is
jointly satisfied by this same field, law, and actual
densities. Both sides of the failed equality are finite.
No stronger modulus assumption or replacement
general Kac–Rice theorem is asserted here.
-/

import Transformer.Modes.Section2_ProbabilityRegularity

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The original inner Kac–Rice integral at level zero is strictly positive for the actual density of the actual oscillating field.
Source: arXiv:2412.09080v3, §2.1, the right side of `eq:krf` for the counterexample to `thm:kac-rice`. -/
theorem riceCounter_rice_integral_pos :
    0 < ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * riceCounterJointDensity (0, y)) := by
  have hm : Measurable (fun y : ℝ => ENNReal.ofReal (y * riceCounterJointDensity (0, y))) :=
    (measurable_id.mul (measurable_riceCounterJointDensity.comp (measurable_const.prodMk measurable_id))).ennreal_ofReal
  apply (setLIntegral_pos_iff hm).mpr
  have hsub : Ioo (0 : ℝ) 1 ⊆
      Function.support (fun y : ℝ => ENNReal.ofReal (y * riceCounterJointDensity (0, y))) ∩ Ioi 0 := by
    intro y hy
    refine ⟨?_, hy.1⟩
    change ENNReal.ofReal (y * riceCounterJointDensity (0, y)) ≠ 0
    have habs : |y| < 1 := by rw [abs_of_pos hy.1]; exact hy.2
    exact (ENNReal.ofReal_pos.mpr (mul_pos hy.1 (riceCounterJointDensity_pos 0 y (by norm_num) habs))).ne'
  have hvol : (0 : ℝ≥0∞) < volume (Ioo (0 : ℝ) 1) := by norm_num
  exact hvol.trans_le (measure_mono hsub)

/-- The original positive-derivative density integral is finite, with its actual global density bound.
Source: arXiv:2412.09080v3, §2.1, the finite right side of `eq:krf` in the counterexample to `thm:kac-rice`. -/
theorem riceCounter_rice_integral_le :
    (∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * riceCounterJointDensity (0, y))) ≤ ENNReal.ofReal Real.pi⁻¹ := by
  have hpoint (y : ℝ) : (Ioi (0 : ℝ)).indicator
      (fun y => ENNReal.ofReal (y * riceCounterJointDensity (0, y))) y ≤
        (Ioo (0 : ℝ) 1).indicator (fun _ => ENNReal.ofReal Real.pi⁻¹) y := by
    by_cases hy0 : 0 < y
    · rw [indicator_of_mem (show y ∈ Ioi (0 : ℝ) from hy0)]
      by_cases hy1 : y < 1
      · rw [indicator_of_mem (show y ∈ Ioo (0 : ℝ) 1 from ⟨hy0, hy1⟩)]
        apply ENNReal.ofReal_le_ofReal
        rw [riceCounterJointDensity_zero_level]
        calc
          y * riceCounterSliceDensity y ≤ 1 * riceCounterSliceDensity y :=
            mul_le_mul_of_nonneg_right hy1.le (riceCounterSliceDensity_nonneg y)
          _ ≤ Real.pi⁻¹ := by simpa only [one_mul] using riceCounterSliceDensity_le y
      · have hyabs : 1 ≤ |y| := by rw [abs_of_pos hy0]; exact le_of_not_gt hy1
        rw [riceCounterJointDensity_zero_level, riceCounterSliceDensity_eq_zero y hyabs, mul_zero, ENNReal.ofReal_zero]
        exact zero_le
    · rw [indicator_of_notMem (show y ∉ Ioi (0 : ℝ) from hy0)]
      exact zero_le
  calc
    _ = ∫⁻ y, (Ioi (0 : ℝ)).indicator (fun y => ENNReal.ofReal (y * riceCounterJointDensity (0, y))) y :=
      (lintegral_indicator measurableSet_Ioi _).symm
    _ ≤ ∫⁻ y, (Ioo (0 : ℝ) 1).indicator (fun _ => ENNReal.ofReal Real.pi⁻¹) y := lintegral_mono hpoint
    _ = _ := by
      rw [lintegral_indicator measurableSet_Ioo, setLIntegral_const, Real.volume_Ioo]
      norm_num

/-- The original full Kac–Rice right side is positive on the actual compact time interval of positive length.
Source: arXiv:2412.09080v3, §2.1, the exact right side of `eq:krf` for the counterexample to `thm:kac-rice`. -/
theorem riceCounter_rice_rhs_pos :
    0 < ∫⁻ _ in Icc (0 : ℝ) 1, ∫⁻ y in Ioi (0 : ℝ),
      ENNReal.ofReal (y * riceCounterJointDensity (0, y)) := by
  simpa only [setLIntegral_const, Real.volume_Icc, sub_zero, ENNReal.ofReal_one, mul_one] using
    riceCounter_rice_integral_pos

/-- The original full right side is finite as well as positive on the actual compact time interval.
Source: arXiv:2412.09080v3, §2.1, the finite strict mismatch in `eq:krf` for the counterexample to `thm:kac-rice`. -/
theorem riceCounter_rice_rhs_ne_top :
    (∫⁻ _ in Icc (0 : ℝ) 1, ∫⁻ y in Ioi (0 : ℝ),
      ENNReal.ofReal (y * riceCounterJointDensity (0, y))) ≠ ⊤ := by
  have h := ne_top_of_le_ne_top ENNReal.ofReal_ne_top riceCounter_rice_integral_le
  simpa only [setLIntegral_const, Real.volume_Icc, sub_zero, ENNReal.ofReal_one, mul_one] using h

/-- The actual expected upcrossing count differs from the original Kac–Rice right side, for the same actual field and its derived density.
Source: arXiv:2412.09080v3, §2.1, the failed equality `eq:krf` in the printed `thm:kac-rice`. -/
theorem riceCounter_kacRice_ne :
    expectedUpcrossings riceCounterLaw riceCounterField 0 (Icc 0 1) ≠
      ∫⁻ _ in Icc (0 : ℝ) 1, ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * riceCounterJointDensity (0, y)) := by
  rw [expectedUpcrossings_riceCounter_eq_zero]
  exact ne_of_lt riceCounter_rice_rhs_pos

/-- **Counterexample to the printed `thm:kac-rice`.** An actual probability law, compact positive-length time set, smooth field, and its actual densities satisfy every original hypothesis while the exact printed equality fails.
Source: arXiv:2412.09080v3, §2.1, `thm:kac-rice`, `eq:krf`; the source's `O(η)` modulus condition is retained unchanged. -/
theorem kacRice_counterexample :
    ∃ (P : Measure (ℝ × ℂ)) (Ψ : (ℝ × ℂ) → ℝ → ℝ) (u : ℝ) (T : Set ℝ)
      (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ),
      IsProbabilityMeasure P ∧ IsCompact T ∧ 0 < volume T ∧ IsKacRiceField P Ψ u T p1 p ∧
        expectedUpcrossings P Ψ u T ≠ ∫⁻ t in T, ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * p t u y) := by
  exact ⟨riceCounterLaw, riceCounterField, 0, Icc 0 1, (fun _ x => riceCounterValueDensity x),
    (fun _ x y => riceCounterJointDensity (x, y)), inferInstance, isCompact_Icc,
    by norm_num, isKacRiceField_riceCounter _, riceCounter_kacRice_ne⟩

/-- The printed universal Kac–Rice claim cannot hold even on the explicit original product parameter space.
Source: arXiv:2412.09080v3, §2.1, the universal assertion of `thm:kac-rice` with its exact original hypotheses and conclusion. -/
theorem not_kacRice : ¬(∀ (P : Measure (ℝ × ℂ)), IsProbabilityMeasure P →
    ∀ (Ψ : (ℝ × ℂ) → ℝ → ℝ) (u : ℝ) (T : Set ℝ), IsCompact T →
    ∀ (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ), IsKacRiceField P Ψ u T p1 p →
      expectedUpcrossings P Ψ u T = ∫⁻ t in T, ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * p t u y)) := by
  intro h
  exact riceCounter_kacRice_ne (h riceCounterLaw inferInstance riceCounterField 0 (Icc 0 1) isCompact_Icc
    (fun _ x => riceCounterValueDensity x) (fun _ x y => riceCounterJointDensity (x, y))
      (isKacRiceField_riceCounter _))

/-- The same actual law, compact interval, field, and actual densities jointly realize all original premises and both incompatible sides of the printed formula.
Source: arXiv:2412.09080v3, §2.1, the simultaneous nonvacuous witness refuting `thm:kac-rice` and `eq:krf`. -/
example : IsProbabilityMeasure riceCounterLaw ∧ IsCompact (Icc (0 : ℝ) 1) ∧
    volume (Icc (0 : ℝ) 1) = 1 ∧
    IsKacRiceField riceCounterLaw riceCounterField 0 (Icc 0 1)
      (fun _ x => riceCounterValueDensity x) (fun _ x y => riceCounterJointDensity (x, y)) ∧
    expectedUpcrossings riceCounterLaw riceCounterField 0 (Icc 0 1) = 0 ∧
    (0 < ∫⁻ _ in Icc (0 : ℝ) 1, ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * riceCounterJointDensity (0, y))) ∧
    (∫⁻ _ in Icc (0 : ℝ) 1, ∫⁻ y in Ioi (0 : ℝ), ENNReal.ofReal (y * riceCounterJointDensity (0, y))) ≠ ⊤ :=
  ⟨inferInstance, isCompact_Icc, by norm_num, isKacRiceField_riceCounter _,
    expectedUpcrossings_riceCounter_eq_zero _, riceCounter_rice_rhs_pos, riceCounter_rice_rhs_ne_top⟩

end Transformer.Modes
