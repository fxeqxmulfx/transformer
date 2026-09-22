/-
# Causal attention — The volume of the cone over a cap on `𝕊^2`

Around the axis `e₀` of `ℝ³`, a point `(a, q)` with `q ∈ ℝ²` lies in the cone
over the cap of radius `δ` exactly when `(a, |q|)` does, so by Tonelli and
polar coordinates in `q`,

  `λ(coneCap e₀ δ) = 2π ∫_0^∞ ρ ∫ 1[(a, ρ) ∈ cone] da dρ = 2π (1 - cos δ) / 3`,

the inner integral being the meridian integral of `Causal.CapHalfPlane`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers" (the
cap mass on `𝕊^2`).
-/

import Transformer.Causal.CapHalfPlane

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

/-- `EucSpace 3` in coordinates `(y₀, (y₁, y₂))`. -/
def spaceMap (y : EucSpace 3) : ℝ × (ℝ × ℝ) := (y 0, (y 1, y 2))

theorem spaceMap_measurePreserving : MeasurePreserving spaceMap := by
  have h := ((MeasurePreserving.id (volume : Measure ℝ)).prod
    (volume_preserving_finTwoArrow ℝ)).comp
      ((volume_preserving_piFinSuccAbove (fun _ : Fin 3 => ℝ) 0).comp
        (PiLp.volume_preserving_ofLp (Fin 3)))
  exact h

theorem norm_eucSpace_three (y : EucSpace 3) :
    ‖y‖ = √(y 0 ^ 2 + √(y 1 ^ 2 + y 2 ^ 2) ^ 2) := by
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_three, Real.sq_sqrt (by positivity),
    Real.norm_eq_abs, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs, sq_abs, add_assoc]

/-- The first basis vector of `ℝ³`. -/
noncomputable def e₃ : EucSpace 3 := EuclideanSpace.single 0 1

/-- The mass of the cone along the axis at distance `ρ` from it. -/
noncomputable def axialMass (δ ρ : ℝ) : ℝ≥0∞ :=
  ∫⁻ a, { z : ℝ × ℝ | ConeCond δ z.1 z.2 }.indicator 1 (a, ρ)

theorem measurable_axialMass (δ : ℝ) : Measurable (axialMass δ) :=
  Measurable.lintegral_prod_left' (measurable_one.indicator (measurableSet_coneCond δ))

/-- `∫_0^∞ ρ · axialMass(ρ) dρ` is the meridian integral. -/
theorem lintegral_Ioi_axialMass (δ : ℝ) :
    ∫⁻ ρ in Set.Ioi 0, ENNReal.ofReal ρ * axialMass δ ρ
      = ∫⁻ z in halfCone δ, ENNReal.ofReal z.2 := by
  have hpt : ∀ ρ, (Set.Ioi 0).indicator (fun ρ => ENNReal.ofReal ρ * axialMass δ ρ) ρ
      = ∫⁻ a, (halfCone δ).indicator (fun z => ENNReal.ofReal z.2) (a, ρ) := by
    intro ρ
    by_cases hρ : 0 < ρ
    · have hm : Measurable fun a : ℝ =>
          { z : ℝ × ℝ | ConeCond δ z.1 z.2 }.indicator (1 : ℝ × ℝ → ℝ≥0∞) (a, ρ) :=
        (measurable_one.indicator (measurableSet_coneCond δ)).comp
          (measurable_id.prodMk measurable_const)
      rw [Set.indicator_of_mem (show ρ ∈ Set.Ioi 0 from hρ), axialMass,
        ← lintegral_const_mul _ hm]
      refine lintegral_congr fun a => ?_
      by_cases h : ConeCond δ a ρ
      · rw [Set.indicator_of_mem (show (a, ρ) ∈ halfCone δ from ⟨hρ, h⟩)]
        simp [Set.indicator_of_mem (show (a, ρ) ∈ { z : ℝ × ℝ | ConeCond δ z.1 z.2 } from h)]
      · rw [Set.indicator_of_notMem (show (a, ρ) ∉ halfCone δ from fun h' => h h'.2)]
        simp [Set.indicator_of_notMem (show (a, ρ) ∉ { z : ℝ × ℝ | ConeCond δ z.1 z.2 } from h)]
    · rw [Set.indicator_of_notMem (show ρ ∉ Set.Ioi 0 from hρ)]
      refine ((lintegral_congr fun a => Set.indicator_of_notMem
        (show (a, ρ) ∉ halfCone δ from fun h' => hρ h'.1) _).trans lintegral_zero).symm
  rw [← lintegral_indicator measurableSet_Ioi, ← lintegral_indicator (measurableSet_halfCone δ),
    Measure.volume_eq_prod, lintegral_prod_symm _
      (measurable_snd.ennreal_ofReal.indicator (measurableSet_halfCone δ)).aemeasurable]
  exact lintegral_congr hpt

/-- **Volume of the cone over a cap on `𝕊^2`.**
`λ(coneCap e₀ δ) = 2π (1 - cos δ) / 3`. -/
theorem volume_coneCap_three {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) :
    volume (coneCap e₃ δ) = ENNReal.ofReal (2 * π * (1 - cos δ) / 3) := by
  set S : Set (ℝ × (ℝ × ℝ)) := { z | ConeCond δ z.1 √(z.2.1 ^ 2 + z.2.2 ^ 2) }
  have hq : Measurable fun q : ℝ × ℝ => √(q.1 ^ 2 + q.2 ^ 2) := by fun_prop
  have hS : MeasurableSet S :=
    measurableSet_coneCond δ |>.preimage (measurable_fst.prodMk (hq.comp measurable_snd))
  have hpre : spaceMap ⁻¹' S = coneCap e₃ δ := by
    ext y
    simp [S, ConeCond, coneCap, spaceMap, norm_eucSpace_three, e₃,
      EuclideanSpace.inner_single_right]
  have hT : MeasurableSet polarCoord.target := by
    rw [polarCoord_target]; exact measurableSet_Ioi.prod measurableSet_Ioo
  have hpolar : ∀ p ∈ polarCoord.target,
      ENNReal.ofReal p.1 • axialMass δ √((polarCoord.symm p).1 ^ 2 + (polarCoord.symm p).2 ^ 2)
        = ENNReal.ofReal p.1 * axialMass δ p.1 := by
    rintro ⟨r, θ⟩ hp
    rw [polarCoord_target, Set.mem_prod] at hp
    have hr : 0 < r := hp.1
    rw [polarCoord_symm_apply, smul_eq_mul]
    congr 2
    rw [mul_pow, mul_pow, ← mul_add, cos_sq_add_sin_sq, mul_one, Real.sqrt_sq hr.le]
  rw [← hpre, spaceMap_measurePreserving.measure_preimage hS.nullMeasurableSet,
    ← lintegral_indicator_one hS, Measure.volume_eq_prod, lintegral_prod_symm _
      ((measurable_one.indicator hS).aemeasurable)]
  change ∫⁻ q : ℝ × ℝ, axialMass δ √(q.1 ^ 2 + q.2 ^ 2) = _
  rw [← lintegral_comp_polarCoord_symm, setLIntegral_congr_fun hT hpolar, polarCoord_target,
    Measure.volume_eq_prod, ← Measure.prod_restrict,
    lintegral_prod (fun x : ℝ × ℝ => ENNReal.ofReal x.1 * axialMass δ x.1)
      ((measurable_fst.ennreal_ofReal.mul ((measurable_axialMass δ).comp measurable_fst)).aemeasurable)]
  simp only [lintegral_const, Measure.restrict_apply_univ, Real.volume_Ioo]
  rw [lintegral_mul_const (f := fun x => ENNReal.ofReal x * axialMass δ x) _
      (ENNReal.measurable_ofReal.mul (measurable_axialMass δ)),
    lintegral_Ioi_axialMass, lintegral_halfCone hδ, ← ENNReal.ofReal_mul (by
      have := Real.cos_le_one δ; linarith)]
  congr 1
  ring

/-- The hypothesis of `volume_coneCap_three` is satisfiable: `δ = π`. -/
example : π ∈ Set.Icc 0 π := ⟨pi_pos.le, le_rfl⟩

end Causal
end Transformer
