/-
# Causal attention — The meridian section of the cone over a cap on `𝕊^2`

In cylindrical coordinates `(a, ρ, θ)` around the axis `e₀` of `ℝ³`, the cone
over the cap of radius `δ` is invariant under rotation in `θ`, and its volume is
`2π ∫∫ ρ da dρ` over its section by the half-plane `ρ > 0`.  That section is
the circular sector `s < 1, 0 < φ ≤ δ` in polar coordinates `(a, ρ) =
(s cos φ, s sin φ)`, and the integral is

  `∫_0^1 s² ds · ∫_0^δ sin φ dφ = (1 - cos δ) / 3`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers" (the
cap mass on `𝕊^2`).
-/

import Transformer.Causal.CapPolar

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

/-- A point at height `a` and distance `ρ` from the axis lies in the cone over
the cap of radius `δ` around the axis. -/
def ConeCond (δ a ρ : ℝ) : Prop :=
  0 < √(a ^ 2 + ρ ^ 2) ∧ √(a ^ 2 + ρ ^ 2) < 1 ∧ √(a ^ 2 + ρ ^ 2) * cos δ ≤ a

theorem measurableSet_coneCond (δ : ℝ) :
    MeasurableSet { z : ℝ × ℝ | ConeCond δ z.1 z.2 } := by
  have hc : Continuous fun p : ℝ × ℝ => √(p.1 ^ 2 + p.2 ^ 2) := by fun_prop
  exact (measurableSet_lt measurable_const hc.measurable).inter
    ((measurableSet_lt hc.measurable measurable_const).inter
      (measurableSet_le (hc.mul continuous_const).measurable measurable_fst))

/-- The section of the cone by the half-plane `ρ > 0`. -/
def halfCone (δ : ℝ) : Set (ℝ × ℝ) := { z | 0 < z.2 ∧ ConeCond δ z.1 z.2 }

theorem measurableSet_halfCone (δ : ℝ) : MeasurableSet (halfCone δ) :=
  (measurableSet_lt measurable_const measurable_snd).inter (measurableSet_coneCond δ)

/-- `ConeCond` in polar coordinates `(a, ρ) = (s cos φ, s sin φ)`, `s > 0`,
`φ ∈ (-π, π)`, on the side `ρ > 0`. -/
theorem mem_halfCone_polar {δ s φ : ℝ} (hδ : δ ∈ Set.Icc 0 π) (hs : 0 < s)
    (hφ : φ ∈ Set.Ioo (-π) π) :
    (s * cos φ, s * sin φ) ∈ halfCone δ ↔ s < 1 ∧ φ ∈ Set.Ioc 0 δ := by
  have hsq : √((s * cos φ) ^ 2 + (s * sin φ) ^ 2) = s := by
    rw [mul_pow, mul_pow, ← mul_add, cos_sq_add_sin_sq, mul_one, Real.sqrt_sq hs.le]
  simp only [halfCone, ConeCond, Set.mem_ofPred_eq, hsq, hs, true_and, Set.mem_Ioc]
  rw [mul_comm s (cos δ), mul_comm s (cos φ), mul_le_mul_iff_left₀ hs,
    cos_le_cos_iff_abs_le hδ hφ, mul_pos_iff_of_pos_left hs]
  have hsin : 0 < sin φ ↔ 0 < φ := by
    refine ⟨fun h => ?_, fun h => Real.sin_pos_of_pos_of_lt_pi h hφ.2⟩
    by_contra hle
    exact absurd h (not_lt.2 (Real.sin_nonpos_of_nonpos_of_neg_pi_le (not_lt.1 hle) hφ.1.le))
  rw [hsin]
  constructor
  · rintro ⟨h0, h1, h2⟩
    exact ⟨h1, h0, (le_abs_self φ).trans h2⟩
  · rintro ⟨h1, h0, h2⟩
    exact ⟨h0, h1, by rw [abs_of_pos h0]; exact h2⟩

/-- **The meridian integral.**  `∫∫_{halfCone δ} ρ da dρ = (1 - cos δ) / 3`. -/
theorem lintegral_halfCone {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) :
    ∫⁻ z in halfCone δ, ENNReal.ofReal z.2 = ENNReal.ofReal ((1 - cos δ) / 3) := by
  set B : Set ℝ := Set.Ioc 0 δ ∩ Set.Iio π
  have hB : MeasurableSet B := measurableSet_Ioc.inter measurableSet_Iio
  have hint : ∀ p ∈ polarCoord.target,
      ENNReal.ofReal p.1 • (halfCone δ).indicator (fun z => ENNReal.ofReal z.2)
        (polarCoord.symm p)
      = (Set.Ioo 0 1 ×ˢ B).indicator
          (fun p => ENNReal.ofReal (p.1 ^ 2) * ENNReal.ofReal (sin p.2)) p := by
    rintro ⟨s, φ⟩ hp
    rw [polarCoord_target, Set.mem_prod] at hp
    obtain ⟨hs, hφ⟩ := hp
    have hs : 0 < s := hs
    have hmem : polarCoord.symm (s, φ) ∈ halfCone δ ↔ (s, φ) ∈ Set.Ioo 0 1 ×ˢ B := by
      rw [polarCoord_symm_apply, mem_halfCone_polar hδ hs hφ]
      simp only [Set.mem_prod, Set.mem_Ioo, B, Set.mem_inter_iff, Set.mem_Iio, hs, true_and,
        hφ.2, and_true]
    by_cases h : (s, φ) ∈ Set.Ioo 0 1 ×ˢ B
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem (hmem.2 h), polarCoord_symm_apply,
        smul_eq_mul, ← ENNReal.ofReal_mul hs.le, ← ENNReal.ofReal_mul (sq_nonneg s)]
      congr 1
      ring
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem (mt hmem.1 h), smul_zero]
  have hT : MeasurableSet polarCoord.target := by
    rw [polarCoord_target]; exact measurableSet_Ioi.prod measurableSet_Ioo
  have hAB : MeasurableSet (Set.Ioo (0 : ℝ) 1 ×ˢ B) := measurableSet_Ioo.prod hB
  have hsub : Set.Ioo (0 : ℝ) 1 ×ˢ B ⊆ polarCoord.target := by
    rw [polarCoord_target]
    exact Set.prod_mono (fun r hr => hr.1) fun φ hφ =>
      ⟨by linarith [hφ.1.1, pi_pos], hφ.2⟩
  have h₁ : Set.Ioo 0 δ ⊆ B := fun φ hφ =>
    ⟨⟨hφ.1, hφ.2.le⟩, show φ < π from hφ.2.trans_le hδ.2⟩
  have h₂ : B ⊆ Set.Icc 0 δ := fun φ hφ => ⟨hφ.1.1.le, hφ.1.2⟩
  rw [← lintegral_indicator (measurableSet_halfCone δ), ← lintegral_comp_polarCoord_symm,
    setLIntegral_congr_fun hT hint, lintegral_indicator hAB, Measure.restrict_restrict hAB,
    Set.inter_eq_left.2 hsub, Measure.volume_eq_prod, ← Measure.prod_restrict,
    lintegral_prod_mul (f := fun s => ENNReal.ofReal (s ^ 2))
      (g := fun φ => ENNReal.ofReal (sin φ)) (by fun_prop) (by fun_prop),
    lintegral_Ioo_ofReal (f := fun s => s ^ 2) zero_le_one (continuous_pow 2)
      fun x _ => sq_nonneg x, integral_pow,
    setLIntegral_eq_of_Ioo_subset_of_subset_Icc h₁ h₂,
    lintegral_Ioo_ofReal hδ.1 continuous_sin
      fun φ hφ => Real.sin_nonneg_of_nonneg_of_le_pi hφ.1.le (hφ.2.le.trans hδ.2),
    integral_sin, cos_zero, ← ENNReal.ofReal_mul (by norm_num)]
  congr 1
  push_cast
  ring

/-- The hypothesis of `lintegral_halfCone` is satisfiable: `δ = π`. -/
example : π ∈ Set.Icc 0 π := ⟨pi_pos.le, le_rfl⟩

end Causal
end Transformer
