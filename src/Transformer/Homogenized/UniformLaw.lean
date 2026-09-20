/-
# Homogenized Transformers — the uniform law on the sphere

The measure `σ_d` that `ass:low-temperature` of arXiv:2604.01978v1,
*Homogenized Transformers*, §3 writes its density against.

It is pinned down by the property that characterizes it — a probability
measure on `ℝ^d` carried by the unit sphere and invariant under every linear
isometry of the ambient space is `σ_d`, and there is only one — so a statement
quantified over every measure satisfying `IsUniformAmbient` says exactly what
the source says about `σ_d`.  This is `Metastability.IsUniformOn`'s device,
read in ambient coordinates because §4 and §5 of this paper work with measures
on `ℝ^d` carried by the sphere rather than with measures on a subtype.

That the device is not vacuous is `isUniformAmbient_uniformAmbient`, and it is
proved in every dimension `d ≥ 1`: the radial projection of Lebesgue measure
on the punctured unit ball, normalized, is carried by the sphere because the
projection lands there, and is rotation invariant because a linear isometry
commutes with the projection and preserves both Lebesgue measure and the ball.
-/

import Transformer.Homogenized.MvGenerator
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The uniform law on the sphere, in ambient coordinates -/

/-- `σ` is **the uniform law `σ_d` on `𝕊^{d-1}`**, read as a measure on the
ambient `ℝ^d`: a probability measure carried by the unit sphere and invariant
under every linear isometry of `ℝ^d`.

Such a measure is unique, so quantifying over all of them is not a
strengthening; see the module docstring.

Source: arXiv:2604.01978v1, `ass:low-temperature`. -/
def IsUniformAmbient (d : ℕ) (σ : Measure (EucSpace d)) : Prop :=
  IsProbabilityMeasure σ ∧
    σ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0 ∧
      ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map U = σ

/-! ### The construction -/

/-- The radial projection `x ↦ ‖x‖⁻¹ x`, which carries `ℝ^d ∖ {0}` onto the
unit sphere. -/
noncomputable def radialProj (d : ℕ) (x : EucSpace d) : EucSpace d := ‖x‖⁻¹ • x

theorem measurable_radialProj (d : ℕ) : Measurable (radialProj d) :=
  (measurable_norm.inv).smul measurable_id

theorem norm_radialProj {d : ℕ} {x : EucSpace d} (hx : x ≠ 0) :
    ‖radialProj d x‖ = 1 := by
  have hn : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  rw [radialProj, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity), inv_mul_cancel₀ hn]

/-- A linear isometry commutes with the radial projection. -/
theorem radialProj_comp_isometry {d : ℕ} (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    ⇑U ∘ radialProj d = radialProj d ∘ ⇑U := by
  funext x
  simp only [Function.comp_apply, radialProj, map_smul, U.norm_map]

/-- The punctured unit ball of `ℝ^d`: the set the radial projection is applied
to, chosen open so that its Lebesgue measure is positive without computing it. -/
def puncturedBall (d : ℕ) : Set (EucSpace d) := Metric.ball 0 1 \ {0}

theorem isOpen_puncturedBall (d : ℕ) : IsOpen (puncturedBall d) :=
  Metric.isOpen_ball.sdiff isClosed_singleton

theorem measurableSet_puncturedBall (d : ℕ) : MeasurableSet (puncturedBall d) :=
  (isOpen_puncturedBall d).measurableSet

theorem ne_zero_of_mem_puncturedBall {d : ℕ} {x : EucSpace d} (hx : x ∈ puncturedBall d) :
    x ≠ 0 := by
  simpa [puncturedBall] using hx.2

theorem measure_puncturedBall_pos {d : ℕ} (hd : 0 < d) :
    0 < volume (puncturedBall d) := by
  refine (isOpen_puncturedBall d).measure_pos volume ⟨EuclideanSpace.single ⟨0, hd⟩ (1 / 2 : ℝ), ?_⟩
  have hnorm : ‖EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 / 2 : ℝ)‖ = 1 / 2 := by
    simp [PiLp.norm_single]
  refine ⟨?_, ?_⟩
  · simpa [Metric.mem_ball, hnorm] using (by norm_num : (1 : ℝ) / 2 < 1)
  · simp only [Set.mem_singleton_iff]
    exact norm_ne_zero_iff.mp (by rw [hnorm]; norm_num)

theorem measure_puncturedBall_lt_top (d : ℕ) : volume (puncturedBall d) < ⊤ :=
  lt_of_le_of_lt (measure_mono Set.sdiff_subset) measure_ball_lt_top

/-- **The uniform law on `𝕊^{d-1}`**, in ambient coordinates: the radial
projection of Lebesgue measure on the punctured unit ball, normalized. -/
noncomputable def uniformAmbient (d : ℕ) : Measure (EucSpace d) :=
  (volume (puncturedBall d))⁻¹ •
    Measure.map (radialProj d) (volume.restrict (puncturedBall d))

theorem uniformAmbient_apply {d : ℕ} {s : Set (EucSpace d)} (hs : MeasurableSet s) :
    uniformAmbient d s =
      (volume (puncturedBall d))⁻¹ *
        volume (radialProj d ⁻¹' s ∩ puncturedBall d) := by
  rw [uniformAmbient, Measure.smul_apply, smul_eq_mul,
    Measure.map_apply (measurable_radialProj d) hs,
    Measure.restrict_apply ((measurable_radialProj d) hs)]

/-- **`IsUniformAmbient` is satisfiable, in every dimension `d ≥ 1`.** -/
theorem isUniformAmbient_uniformAmbient {d : ℕ} (hd : 0 < d) :
    IsUniformAmbient d (uniformAmbient d) := by
  have hpos := measure_puncturedBall_pos hd
  have hlt := measure_puncturedBall_lt_top d
  refine ⟨⟨?_⟩, ?_, ?_⟩
  · rw [uniformAmbient_apply MeasurableSet.univ, Set.preimage_univ, Set.univ_inter,
      ENNReal.inv_mul_cancel hpos.ne' hlt.ne]
  · have hmeas : MeasurableSet {y : EucSpace d | ‖y‖ = 1}ᶜ :=
      (measurableSet_eq_fun measurable_norm measurable_const).compl
    rw [uniformAmbient_apply hmeas]
    have hempty : radialProj d ⁻¹' {y : EucSpace d | ‖y‖ = 1}ᶜ ∩ puncturedBall d = ∅ := by
      ext x
      refine ⟨fun hx => ?_, fun hx => hx.elim⟩
      exact absurd (norm_radialProj (ne_zero_of_mem_puncturedBall hx.2)) hx.1
    rw [hempty, measure_empty, mul_zero]
  · intro U
    have hUmeas : Measurable U := U.continuous.measurable
    have hpre : U ⁻¹' puncturedBall d = puncturedBall d := by
      ext x
      simp [puncturedBall, Metric.mem_ball, dist_eq_norm, U.map_eq_zero_iff]
    have hrestrict : Measure.map U (volume.restrict (puncturedBall d))
        = volume.restrict (puncturedBall d) := by
      rw [← hpre, ← Measure.restrict_map hUmeas (measurableSet_puncturedBall d),
        (U.measurePreserving).map_eq, hpre]
    rw [uniformAmbient, Measure.map_smul _ hUmeas.aemeasurable,
      Measure.map_map hUmeas (measurable_radialProj d), radialProj_comp_isometry U,
      ← Measure.map_map (measurable_radialProj d) hUmeas, hrestrict]

end Homogenized
end Transformer
