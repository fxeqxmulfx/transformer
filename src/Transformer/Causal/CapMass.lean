/-
# Causal attention — The uniform cap masses in dimensions two and three

The count of strong R'enyi centers, `Transformer.Causal.ParkingCount`, is the
reciprocal of the common mass of the geodesic caps of radius `δ`.  For the
uniform measure on the sphere that mass is an explicit function of `δ`, and
the survey records it in the two dimensions it draws pictures in.  Both are
stated and not proved: the mass of a cap is the spherical part of Lebesgue
measure of a region cut out in polar coordinates, which this development does
not compute.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers".
-/

import Transformer.Causal.ParkingCount

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

/-- **Cap mass in dimension two.**

On the circle `𝕊^1`, the uniform geodesic cap of radius `δ` has mass `δ / π`,
so the expected number of strong R'enyi centers is `π δ^{-1}`.

Not proved here.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_two (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 2,
      uniformSphere 2 { y : SSphere 2 | geoDist 2 x y ≤ δ }
        = ENNReal.ofReal (δ / Real.pi) := by
  sorry

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

Not proved here.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_three (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 3,
      uniformSphere 3 { y : SSphere 3 | geoDist 3 x y ≤ δ }
        = ENNReal.ofReal (Real.sin (δ / 2) ^ 2) := by
  sorry

/-- The hypothesis of `uniform_cap_mass_three` is satisfiable: `δ = π`, where
the cap is the whole sphere and `sin²(π/2) = 1`. -/
example : Real.pi ∈ Set.Icc 0 Real.pi ∧ Real.sin (Real.pi / 2) ^ 2 = 1 := by
  refine ⟨⟨Real.pi_pos.le, le_rfl⟩, ?_⟩
  simp

end Causal
end Transformer
