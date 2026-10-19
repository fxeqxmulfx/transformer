module

public import Transformer.Interpolation.Section5_ZeroTemperatureMoments
public import Transformer.Perspective.Section2_EnergyKernel

/-!
# Concentration controls the actual attention flux

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`, and §5,
`prop:generic-discrete` at `β = 0`.

For a unit vector `a`, the mean concentration defect is `1 - ⟪a, m⟫`.
It is the integral of the nonnegative function `1 - ⟪a, x⟫` on the sphere.
The zero-temperature flux of the linear test `⟪a, x⟫` has absolute value
at most twice this defect. A vanishing defect forces the entire probability
law to be the Dirac law at `a`.

These are estimates for arbitrary spherical probability laws, including
non-atomic ones. They will give uniqueness of every actual weak solution
from a one-token empirical initial law, as required by the quantified
cluster hypothesis in `prop:generic-discrete`.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The spherical concentration defect is its actual nonnegative test
integral. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`,
using the linear test centered at a unit vector. -/
theorem integral_spherical_defect (μ : Perspective.ProbSphere d) (a : SSphere d) :
    (∫ x : SSphere d, (1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d))
      ∂(μ : Measure (SSphere d))) =
      1 - inner (𝕜 := ℝ) (a : EucSpace d)
        (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))) := by
  have hi := integrable_spherical_continuous μ
    (fun x : SSphere d => inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d)) (by fun_prop)
  rw [integral_sub (integrable_const _) hi, integral_const, probReal_univ, one_smul,
    integral_inner (integrable_spherical_coe μ)]

/-- The first moment of a spherical probability law has inner product at
most one with every unit vector. Source: arXiv:2411.04551v3, §1,
the unweighted attention mean in `eq: vf` at `β = 0`. -/
theorem inner_spherical_mean_le_one (μ : Perspective.ProbSphere d) (a : SSphere d) :
    inner (𝕜 := ℝ) (a : EucSpace d)
      (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))) ≤ 1 := by
  calc
    _ ≤ ‖(a : EucSpace d)‖ *
        ‖∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))‖ :=
      real_inner_le_norm _ _
    _ = ‖∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))‖ := by
      rw [mem_sphere_zero_iff_norm.mp a.2, one_mul]
    _ ≤ 1 := norm_spherical_mean_le_one μ

/-- The scalar mean term in the zero-temperature flux is bounded by the
concentration defect. Source: arXiv:2411.04551v3, §1, `eq: vf` at `β = 0`.
The two bounds follow from `‖m‖ ≤ 1` and `‖m - a‖² ≥ 0`. -/
theorem abs_inner_mean_sub_norm_sq_le_defect (μ : Perspective.ProbSphere d) (a : SSphere d) :
    |inner (𝕜 := ℝ) (a : EucSpace d)
        (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))) -
      ‖∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))‖ ^ 2| ≤
      1 - inner (𝕜 := ℝ) (a : EucSpace d)
        (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))) := by
  let m := ∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))
  have hm : ‖m‖ ≤ 1 := norm_spherical_mean_le_one μ
  have hs : 0 ≤ ‖m - (a : EucSpace d)‖ ^ 2 := sq_nonneg _
  rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp a.2,
    ← real_inner_comm m (a : EucSpace d)] at hs
  exact abs_le.mpr ⟨by nlinarith [norm_nonneg m], by nlinarith⟩

/-- **The actual zero-temperature linear flux is controlled by twice the
concentration defect.** Source: arXiv:2411.04551v3, §1, `eq: vf`.
The constant is explicit and independent of the dimension and the law. -/
theorem zero_temperature_flux_abs_le_defect (μ : Perspective.ProbSphere d) (a : SSphere d) :
    |∫ x : SSphere d, inner (𝕜 := ℝ) (a : EucSpace d)
      (proj d (x : EucSpace d) (∫ y : SSphere d, (y : EucSpace d)
        ∂(μ : Measure (SSphere d)))) ∂(μ : Measure (SSphere d))| ≤
      2 * (1 - inner (𝕜 := ℝ) (a : EucSpace d)
        (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d)))) := by
  let m := ∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))
  have hD (x : SSphere d) : 0 ≤ 1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d) :=
    sub_nonneg.mpr (Perspective.inner_sphere_le_one d a x)
  have hiD := integrable_spherical_continuous μ
    (fun x : SSphere d => 1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d))
    (by fun_prop)
  have hb (x : SSphere d) : |inner (𝕜 := ℝ) (x : EucSpace d) m| ≤ 1 := by
    calc
      _ ≤ ‖(x : EucSpace d)‖ * ‖m‖ := abs_real_inner_le_norm _ _
      _ = ‖m‖ := by rw [mem_sphere_zero_iff_norm.mp x.2, one_mul]
      _ ≤ 1 := norm_spherical_mean_le_one μ
  have hC : |∫ x : SSphere d, (1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d)) *
      inner (𝕜 := ℝ) (x : EucSpace d) m ∂(μ : Measure (SSphere d))| ≤
      1 - inner (𝕜 := ℝ) (a : EucSpace d) m := by
    have hnorm := norm_integral_le_of_norm_le hiD (ae_of_all _ fun x => show
      ‖(1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d)) *
        inner (𝕜 := ℝ) (x : EucSpace d) m‖ ≤
        1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d) from by
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (hD x)]
      simpa only [mul_one] using mul_le_mul_of_nonneg_left (hb x) (hD x))
    simpa only [Real.norm_eq_abs, integral_spherical_defect] using hnorm
  rw [zero_temperature_flux_defect]
  exact (abs_add_le _ _).trans (by linarith [abs_inner_mean_sub_norm_sq_le_defect μ a])

/-- **A spherical probability law with zero concentration defect is the
Dirac law at that point.** Source: arXiv:2411.04551v3, §5,
`prop:generic-discrete`, the one-token cluster case.
Nonnegativity makes the defect vanish almost everywhere; its unique zero
on the sphere is the specified point. -/
theorem probSphere_eq_dirac_of_inner_mean_eq_one (μ : Perspective.ProbSphere d) (a : SSphere d)
    (h : inner (𝕜 := ℝ) (a : EucSpace d)
      (∫ x : SSphere d, (x : EucSpace d) ∂(μ : Measure (SSphere d))) = 1) :
    μ = Perspective.diracProb d a := by
  have hiD := integrable_spherical_continuous μ
    (fun x : SSphere d => 1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d))
    (by fun_prop)
  have hD : 0 ≤ (fun x : SSphere d =>
      1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d)) :=
    fun x => sub_nonneg.mpr (Perspective.inner_sphere_le_one d a x)
  have hz : (∫ x : SSphere d, (1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d))
      ∂(μ : Measure (SSphere d))) = 0 := by
    rw [integral_spherical_defect, h, sub_self]
  have hae : (id : SSphere d → SSphere d) =ᵐ[(μ : Measure (SSphere d))] (fun _ => a) := by
    filter_upwards [(integral_eq_zero_iff_of_nonneg hD hiD).mp hz] with x hx
    have hp : inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d) = 1 := by
      change 1 - inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d) = 0 at hx
      linarith
    exact (Perspective.eq_of_inner_sphere_eq_one d hp).symm
  apply Subtype.ext
  change (μ : Measure (SSphere d)) = Measure.dirac a
  calc
    _ = (μ : Measure (SSphere d)).map id := Measure.map_id.symm
    _ = (μ : Measure (SSphere d)).map (fun _ => a) := Measure.map_congr hae
    _ = Measure.dirac a := by rw [Measure.map_const, measure_univ, one_smul]

/-- The complete concentration hypothesis is attained by a nonzero unit
Dirac law, including in the dimension of the source proposition. -/
example (a : SSphere 3) : inner (𝕜 := ℝ) (a : EucSpace 3)
    (∫ x : SSphere 3, (x : EucSpace 3)
      ∂(Perspective.diracProb 3 a : Measure (SSphere 3))) = 1 := by
  change inner (𝕜 := ℝ) (a : EucSpace 3)
    (∫ x : SSphere 3, (x : EucSpace 3) ∂Measure.dirac a) = 1
  rw [integral_dirac, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp a.2]
  norm_num

end Transformer.Interpolation
