/-
# Causal attention — The cone over a geodesic cap

`Measure.toSphere` gives a set `C` of the sphere the mass `d · λ((0,1) · C)`,
the Lebesgue measure of the cone over `C` inside the unit ball.  For the
closed geodesic cap of radius `δ` around `x` that cone is

  `{y : 0 < ‖y‖ < 1, ‖y‖ cos δ ≤ ⟨y, x⟩}`   (`coneCap`),

and its volume does not depend on `x`, a reflection carrying one centre to
another.  This reduces the cap masses of `Causal.CapMass` to one volume per
dimension, computed in polar coordinates in `Causal.CapPolar`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers".
-/

import Transformer.Causal.ParkingCount

open scoped ENNReal Pointwise
open Real MeasureTheory

namespace Transformer
namespace Causal

variable {d : ℕ}

/-- The cone `(0,1) · C` over the closed geodesic cap `C` of radius `δ`
around `x`. -/
def coneCap (x : EucSpace d) (δ : ℝ) : Set (EucSpace d) :=
  { y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ ‖y‖ * cos δ ≤ inner (𝕜 := ℝ) y x }

theorem measurableSet_coneCap (x : EucSpace d) (δ : ℝ) : MeasurableSet (coneCap x δ) :=
  (measurableSet_lt measurable_const continuous_norm.measurable).inter
    ((measurableSet_lt continuous_norm.measurable measurable_const).inter
      (measurableSet_le (continuous_norm.mul continuous_const).measurable
        (continuous_id.inner continuous_const).measurable))

/-- On `[0, π]`, `arccos t ≤ δ ↔ cos δ ≤ t` for `t ∈ [-1, 1]`. -/
theorem geoDist_le_iff {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) (x y : SSphere d) :
    geoDist d x y ≤ δ ↔ cos δ ≤ inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) := by
  have h := abs_real_inner_le_norm (x : EucSpace d) (y : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2, mul_one] at h
  obtain ⟨h₁, h₂⟩ := abs_le.1 h
  unfold geoDist
  constructor
  · intro hle
    have := Real.cos_le_cos_of_nonneg_of_le_pi (Real.arccos_nonneg _) hδ.2 hle
    rwa [Real.cos_arccos h₁ h₂] at this
  · intro hle
    exact (Real.arccos_le_arccos hle).trans_eq (Real.arccos_cos hδ.1 hδ.2)

/-- The closed geodesic cap as a measurable set. -/
theorem measurableSet_geoDist_le (x : SSphere d) (δ : ℝ) :
    MeasurableSet { y : SSphere d | geoDist d x y ≤ δ } :=
  measurableSet_le ((continuous_geoDist d).comp
    (continuous_const.prodMk continuous_id)).measurable measurable_const

/-- **The cone over a cap.**  `(0,1) · {y : dist(x, y) ≤ δ} = coneCap x δ`. -/
theorem Ioo_smul_cap {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) (x : SSphere d) :
    Set.Ioo (0 : ℝ) 1 • (((↑) : SSphere d → EucSpace d) '' { y | geoDist d x y ≤ δ })
      = coneCap (x : EucSpace d) δ := by
  ext z
  simp only [Set.mem_smul, Set.mem_image, Set.mem_ofPred_eq, geoDist_le_iff hδ]
  constructor
  · rintro ⟨t, ⟨ht₀, ht₁⟩, _, ⟨y, hy, rfl⟩, rfl⟩
    have hy1 : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
    have hn : ‖t • (y : EucSpace d)‖ = t := by
      rw [norm_smul, hy1, mul_one, Real.norm_eq_abs, abs_of_pos ht₀]
    refine ⟨hn.symm ▸ ht₀, hn.symm ▸ ht₁, ?_⟩
    rw [hn, real_inner_smul_left, real_inner_comm]
    exact mul_le_mul_of_nonneg_left hy ht₀.le
  · rintro ⟨h₀, h₁, h⟩
    have hs : ‖(‖z‖⁻¹ • z)‖ = 1 := by
      rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_norm, inv_mul_cancel₀ h₀.ne']
    refine ⟨‖z‖, ⟨h₀, h₁⟩, _, ⟨⟨‖z‖⁻¹ • z, mem_sphere_zero_iff_norm.mpr hs⟩, ?_, rfl⟩, ?_⟩
    · show cos δ ≤ inner (𝕜 := ℝ) (x : EucSpace d) (‖z‖⁻¹ • z)
      rw [real_inner_smul_right, real_inner_comm, le_inv_mul_iff₀ h₀]
      linarith [h]
    · show ‖z‖ • ‖z‖⁻¹ • z = z
      rw [smul_smul, mul_inv_cancel₀ h₀.ne', one_smul]

/-- **The cone volume does not depend on the centre**: the reflection in the
hyperplane orthogonal to `x - y` carries `x` to `y` and preserves volume. -/
theorem volume_coneCap_eq (x y : SSphere d) (δ : ℝ) :
    volume (coneCap (x : EucSpace d) δ) = volume (coneCap (y : EucSpace d) δ) := by
  set U := Submodule.reflection (ℝ ∙ ((x : EucSpace d) - (y : EucSpace d)))ᗮ
  have hU : U (x : EucSpace d) = y := Submodule.reflection_sub (by
    rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2])
  have hpre : U ⁻¹' coneCap (y : EucSpace d) δ = coneCap (x : EucSpace d) δ := by
    ext z
    simp only [Set.mem_preimage, coneCap, Set.mem_ofPred_eq, LinearIsometryEquiv.norm_map, ← hU,
      LinearIsometryEquiv.inner_map_map]
  rw [← hpre, U.measurePreserving.measure_preimage
    (measurableSet_coneCap _ δ).nullMeasurableSet]

/-- **Cap mass as a cone volume.**  `σ(cap) = d · λ(coneCap x δ)`, where `σ` is
the spherical part of Lebesgue measure. -/
theorem toSphere_cap {δ : ℝ} (hδ : δ ∈ Set.Icc 0 π) (x : SSphere d) :
    (volume : Measure (EucSpace d)).toSphere { y | geoDist d x y ≤ δ }
      = d * volume (coneCap (x : EucSpace d) δ) := by
  rw [Measure.toSphere_apply' _ (measurableSet_geoDist_le x δ), Ioo_smul_cap hδ,
    finrank_euclideanSpace_fin]

/-- The cap of radius `π` is the whole sphere. -/
theorem cap_pi (x : SSphere d) : { y : SSphere d | geoDist d x y ≤ π } = Set.univ :=
  Set.eq_univ_of_forall fun _ => Real.arccos_le_pi _

/-- The hypothesis of `geoDist_le_iff`, `Ioo_smul_cap` and `toSphere_cap` is
satisfiable: `δ = π`. -/
example : π ∈ Set.Icc 0 π := ⟨pi_pos.le, le_rfl⟩

end Causal
end Transformer
