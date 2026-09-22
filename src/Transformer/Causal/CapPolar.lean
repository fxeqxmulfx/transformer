/-
# Causal attention — The cone over a cap on the circle, in polar coordinates

In polar coordinates `(r, θ) ∈ (0, ∞) × (-π, π)` of the plane, the cone over
the cap of radius `δ` around `e₀` is `r < 1, |θ| ≤ δ`, and its area is
`∫_0^1 r dr · 2δ = δ`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers" (the
cap mass `δ/π` on `𝕊^1`).
-/

import Transformer.Causal.CapCone
import Mathlib.Analysis.SpecialFunctions.PolarCoord

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

/-- On `(-π, π)`, `cos δ ≤ cos θ ↔ |θ| ≤ δ` for `δ ∈ [0, π]`. -/
theorem cos_le_cos_iff_abs_le {δ θ : ℝ} (hδ : δ ∈ Set.Icc 0 π) (hθ : θ ∈ Set.Ioo (-π) π) :
    cos δ ≤ cos θ ↔ |θ| ≤ δ := by
  have hθπ : |θ| ≤ π := abs_le.2 ⟨hθ.1.le, hθ.2.le⟩
  rw [← Real.cos_abs θ]
  constructor
  · intro h
    by_contra hlt
    exact absurd h (not_le.2 (Real.cos_lt_cos_of_nonneg_of_le_pi hδ.1 hθπ (not_le.1 hlt)))
  · exact Real.cos_le_cos_of_nonneg_of_le_pi (abs_nonneg θ) hδ.2

/-- `∫⁻_{(a,b)} f = ∫_a^b f` for a continuous `f ≥ 0` on `(a, b)`. -/
theorem lintegral_Ioo_ofReal {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (hf : Continuous f)
    (h0 : ∀ x ∈ Set.Ioo a b, 0 ≤ f x) :
    ∫⁻ x in Set.Ioo a b, ENNReal.ofReal (f x) = ENNReal.ofReal (∫ x in a..b, f x) := by
  rw [intervalIntegral.integral_of_le hab, integral_Ioc_eq_integral_Ioo,
    ofReal_integral_eq_lintegral_ofReal]
  · exact hf.integrableOn_Icc.mono_set Set.Ioo_subset_Icc_self
  · exact (ae_restrict_iff' measurableSet_Ioo).2 (ae_of_all _ h0)

/-- A set squeezed between `(a, b)` and `[a, b]` carries the same integral. -/
theorem setLIntegral_eq_of_Ioo_subset_of_subset_Icc {f : ℝ → ℝ≥0∞} {a b : ℝ} {B : Set ℝ}
    (h₁ : Set.Ioo a b ⊆ B) (h₂ : B ⊆ Set.Icc a b) :
    ∫⁻ x in B, f x = ∫⁻ x in Set.Ioo a b, f x :=
  le_antisymm ((lintegral_mono_set h₂).trans_eq (setLIntegral_congr Ioo_ae_eq_Icc).symm)
    (lintegral_mono_set h₁)

/-- The plane `EucSpace 2` in coordinates `(y₀, y₁)`. -/
def planeMap (y : EucSpace 2) : ℝ × ℝ := (y 0, y 1)

theorem planeMap_measurePreserving : MeasurePreserving planeMap :=
  (volume_preserving_finTwoArrow ℝ).comp (PiLp.volume_preserving_ofLp (Fin 2))

theorem norm_eucSpace_two (y : EucSpace 2) : ‖y‖ = √(y 0 ^ 2 + y 1 ^ 2) := by
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_two, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs,
    sq_abs]

/-- The first basis vector of the plane. -/
noncomputable def e₂ : EucSpace 2 := EuclideanSpace.single 0 1

/-- **Area of the cone over a cap on the circle.**  `λ(coneCap e₀ δ) = δ`. -/
theorem volume_coneCap_two {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) :
    volume (coneCap e₂ δ) = ENNReal.ofReal δ := by
  set S : Set (ℝ × ℝ) := { p | 0 < √(p.1 ^ 2 + p.2 ^ 2) ∧ √(p.1 ^ 2 + p.2 ^ 2) < 1 ∧
    √(p.1 ^ 2 + p.2 ^ 2) * cos δ ≤ p.1 }
  have hS : MeasurableSet S := by
    have hc : Continuous fun p : ℝ × ℝ => √(p.1 ^ 2 + p.2 ^ 2) := by fun_prop
    exact (measurableSet_lt measurable_const hc.measurable).inter
      ((measurableSet_lt hc.measurable measurable_const).inter
        (measurableSet_le (hc.mul continuous_const).measurable measurable_fst))
  have hpre : planeMap ⁻¹' S = coneCap e₂ δ := by
    ext y
    simp [S, coneCap, planeMap, norm_eucSpace_two, e₂, EuclideanSpace.inner_single_right]
  rw [← hpre, planeMap_measurePreserving.measure_preimage hS.nullMeasurableSet,
    ← lintegral_indicator_one hS, ← lintegral_comp_polarCoord_symm]
  set B : Set ℝ := Set.Icc (-δ) δ ∩ Set.Ioo (-π) π
  have hB : MeasurableSet B := measurableSet_Icc.inter measurableSet_Ioo
  have hint : ∀ p ∈ polarCoord.target, ENNReal.ofReal p.1 • S.indicator 1 (polarCoord.symm p)
      = (Set.Ioo 0 1 ×ˢ B).indicator (fun p => ENNReal.ofReal p.1) p := by
    rintro ⟨r, θ⟩ hp
    rw [polarCoord_target, Set.mem_prod] at hp
    obtain ⟨hr, hθ⟩ := hp
    have hr : 0 < r := hr
    have hsq : √((r * cos θ) ^ 2 + (r * sin θ) ^ 2) = r := by
      rw [mul_pow, mul_pow, ← mul_add, cos_sq_add_sin_sq, mul_one, Real.sqrt_sq hr.le]
    have hmem : polarCoord.symm (r, θ) ∈ S ↔ (r, θ) ∈ Set.Ioo 0 1 ×ˢ B := by
      simp only [polarCoord_symm_apply, S, Set.mem_ofPred_eq, hsq, Set.mem_prod, Set.mem_Ioo, B,
        Set.mem_inter_iff, Set.mem_Icc, hθ, and_true, hr, true_and]
      rw [mul_comm r, mul_comm r, mul_le_mul_iff_left₀ hr, cos_le_cos_iff_abs_le hδ hθ, abs_le]
    by_cases h : (r, θ) ∈ Set.Ioo 0 1 ×ˢ B
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem (hmem.2 h)]
      simp
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem (mt hmem.1 h), smul_zero]
  have hT : MeasurableSet polarCoord.target := by
    rw [polarCoord_target]; exact measurableSet_Ioi.prod measurableSet_Ioo
  have hAB : MeasurableSet (Set.Ioo (0 : ℝ) 1 ×ˢ B) := measurableSet_Ioo.prod hB
  have hsub : Set.Ioo (0 : ℝ) 1 ×ˢ B ⊆ polarCoord.target := by
    rw [polarCoord_target]
    exact Set.prod_mono (fun r hr => hr.1) Set.inter_subset_right
  rw [setLIntegral_congr_fun hT hint, lintegral_indicator hAB, Measure.restrict_restrict hAB,
    Set.inter_eq_left.2 hsub, Measure.volume_eq_prod, ← Measure.prod_restrict,
    lintegral_prod _ (measurable_fst.ennreal_ofReal).aemeasurable]
  simp only [lintegral_const, Measure.restrict_apply_univ]
  rw [lintegral_mul_const _ ENNReal.measurable_ofReal,
    lintegral_Ioo_ofReal (f := fun x => x) zero_le_one continuous_id fun x hx => hx.1.le,
    integral_id]
  have hvB : volume B = ENNReal.ofReal (δ - -δ) := by
    refine le_antisymm ((measure_mono Set.inter_subset_left).trans_eq Real.volume_Icc) ?_
    rw [← Real.volume_Ioo]
    refine measure_mono fun θ hθ => ⟨Set.Ioo_subset_Icc_self hθ, ?_, ?_⟩
    · linarith [hθ.1, hδ.2]
    · linarith [hθ.2, hδ.2]
  rw [hvB, ← ENNReal.ofReal_mul (by norm_num)]
  congr 1
  ring

/-- The hypotheses of `cos_le_cos_iff_abs_le` and `volume_coneCap_two` are
satisfiable: `δ = π`, `θ = 0`. -/
example : π ∈ Set.Icc 0 π ∧ (0 : ℝ) ∈ Set.Ioo (-π) π :=
  ⟨⟨pi_pos.le, le_rfl⟩, ⟨by linarith [pi_pos], pi_pos⟩⟩

/-- The hypotheses of `lintegral_Ioo_ofReal` and
`setLIntegral_eq_of_Ioo_subset_of_subset_Icc` are satisfiable: `f = 1` on
`(0, 1)`, and `B = [0, 1]`. -/
example : (0 : ℝ) ≤ 1 ∧ Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ x ∈ Set.Ioo (0 : ℝ) 1, (0 : ℝ) ≤ 1) ∧
    Set.Ioo (0 : ℝ) 1 ⊆ Set.Icc 0 1 ∧ Set.Icc (0 : ℝ) 1 ⊆ Set.Icc 0 1 :=
  ⟨zero_le_one, continuous_const, fun _ _ => zero_le_one, Set.Ioo_subset_Icc_self, le_rfl⟩

end Causal
end Transformer
