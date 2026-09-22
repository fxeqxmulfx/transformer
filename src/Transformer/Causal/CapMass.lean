/-
# Causal attention — The uniform cap masses in dimensions two and three

The count of strong R'enyi centers, `Transformer.Causal.ParkingCount`, is the
reciprocal of the common mass of the geodesic caps of radius `δ`.  For the
uniform measure on the sphere that mass is an explicit function of `δ`, and
the survey records it in the two dimensions it draws pictures in.  The mass of
a cap is the ratio `λ(coneCap x δ) / λ(coneCap x π)` of the volumes of the
cones over it and over the whole sphere (`Causal.CapCone`), computed in polar
coordinates on the plane (`Causal.CapPolar`) and in cylindrical coordinates in
space (`Causal.CapSpace`).

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers".
-/

import Transformer.Causal.CapSpace

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (d : ℕ)

/-- The **uniform probability measure on `𝕊^{d-1}`**: the spherical part of
Lebesgue measure, normalized to unit mass.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def uniformSphere : Measure (SSphere d) :=
  ((volume : Measure (EucSpace d)).toSphere Set.univ)⁻¹ •
    (volume : Measure (EucSpace d)).toSphere

/-- `uniformSphere d` is a probability measure once the sphere is nonempty:
the spherical part of Lebesgue measure is finite, and nonzero in positive
dimension, so dividing by its total mass normalizes it.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers" (the
uniform law on `𝕊^{d-1}`). -/
theorem isProbabilityMeasure_uniformSphere (hd : 1 ≤ d) :
    IsProbabilityMeasure (uniformSphere d) := by
  have : Nontrivial (EucSpace d) :=
    nontrivial_of_ne (EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 : ℝ)) 0
      (by simp)
  have h0 : (volume : Measure (EucSpace d)).toSphere Set.univ ≠ 0 :=
    Measure.measure_univ_eq_zero.not.mpr (Measure.toSphere_ne_zero _)
  have htop : (volume : Measure (EucSpace d)).toSphere Set.univ ≠ ⊤ :=
    measure_ne_top _ _
  exact ⟨by
    simp only [uniformSphere, Measure.smul_apply, smul_eq_mul]
    exact ENNReal.inv_mul_cancel h0 htop⟩

/-- The hypothesis of `isProbabilityMeasure_uniformSphere` is satisfiable:
the circle, `d = 2`. -/
example : 1 ≤ 2 := by norm_num

/-- **Cap mass as a ratio of cone volumes**, measured around any centre `e`:
`σ(cap(x, δ)) = λ(coneCap e δ) / λ(coneCap e π)`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers" (the
reduction behind both cap masses). -/
theorem uniformSphere_cap {δ : ℝ} (hδ : δ ∈ Set.Icc 0 Real.pi) (hd : d ≠ 0)
    (x e : SSphere d) :
    uniformSphere d { y : SSphere d | geoDist d x y ≤ δ }
      = volume (coneCap (e : EucSpace d) δ) / volume (coneCap (e : EucSpace d) Real.pi) := by
  simp only [uniformSphere, Measure.smul_apply, smul_eq_mul]
  rw [← cap_pi x, toSphere_cap hδ x, toSphere_cap ⟨pi_pos.le, le_rfl⟩ x,
    volume_coneCap_eq x e δ, volume_coneCap_eq x e Real.pi, ← ENNReal.div_eq_inv_mul,
    ENNReal.mul_div_mul_left _ _ (by exact_mod_cast hd) (ENNReal.natCast_ne_top d)]

/-- The hypotheses of `uniformSphere_cap` are satisfiable: `δ = π`, `d = 2`. -/
example : Real.pi ∈ Set.Icc 0 Real.pi ∧ (2 : ℕ) ≠ 0 := ⟨⟨pi_pos.le, le_rfl⟩, by norm_num⟩

/-- **Cap mass in dimension two.**

On the circle `𝕊^1`, the uniform geodesic cap of radius `δ` has mass `δ / π`,
so the expected number of strong R'enyi centers is `π δ^{-1}`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_two (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 2,
      uniformSphere 2 { y : SSphere 2 | geoDist 2 x y ≤ δ }
        = ENNReal.ofReal (δ / Real.pi) := by
  intro x
  let e : SSphere 2 := ⟨e₂, mem_sphere_zero_iff_norm.mpr (by simp [e₂, PiLp.norm_single])⟩
  rw [uniformSphere_cap 2 hδ (by norm_num) x e, volume_coneCap_two hδ,
    volume_coneCap_two ⟨pi_pos.le, le_rfl⟩, ENNReal.ofReal_div_of_pos pi_pos]

/-- The hypothesis of `uniform_cap_mass_two` is satisfiable: `δ = 0`. -/
example : (0 : ℝ) ∈ Set.Icc 0 Real.pi :=
  ⟨le_rfl, Real.pi_pos.le⟩

/-- **Cap mass in dimension three.**

On `𝕊^2` the uniform geodesic cap of radius `δ` has mass `sin²(δ/2)`, so the
expected number of strong R'enyi centers is `(sin²(δ/2))^{-1}`.

The survey writes this cap mass as `3 sin²(δ/2)`; that cannot be a normalized
cap mass, since at `δ = π` the cap is the whole sphere and the value would be
`3`.  What is stated here is the cap mass itself, `(1 - cos δ) / 2 =
sin²(δ/2)`, which agrees with the two-dimensional case in being a probability.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_three (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 3,
      uniformSphere 3 { y : SSphere 3 | geoDist 3 x y ≤ δ }
        = ENNReal.ofReal (Real.sin (δ / 2) ^ 2) := by
  intro x
  let e : SSphere 3 := ⟨e₃, mem_sphere_zero_iff_norm.mpr (by simp [e₃, PiLp.norm_single])⟩
  have hπ : 0 < 2 * Real.pi * (1 - Real.cos Real.pi) / 3 := by
    rw [Real.cos_pi]; positivity
  rw [uniformSphere_cap 3 hδ (by norm_num) x e, volume_coneCap_three hδ,
    volume_coneCap_three ⟨pi_pos.le, le_rfl⟩, ← ENNReal.ofReal_div_of_pos hπ,
    Real.sin_sq_eq_half_sub, show 2 * (δ / 2) = δ by ring, Real.cos_pi]
  congr 1
  field_simp
  ring

/-- The hypothesis of `uniform_cap_mass_three` is satisfiable: `δ = π`, where
the cap is the whole sphere and `sin²(π/2) = 1`. -/
example : Real.pi ∈ Set.Icc 0 Real.pi ∧ Real.sin (Real.pi / 2) ^ 2 = 1 := by
  refine ⟨⟨Real.pi_pos.le, le_rfl⟩, ?_⟩
  simp

end Causal
end Transformer
