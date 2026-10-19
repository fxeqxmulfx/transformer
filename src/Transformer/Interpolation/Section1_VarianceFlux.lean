module

public import Transformer.Interpolation.Section1_FullFieldLipschitz
public import Transformer.Interpolation.Section1_SquaredDistance
public import Transformer.Interpolation.Section3_PerturbationGeometry
public import Mathlib.MeasureTheory.Integral.Prod

/-!
# Original spherical variance and tangent-field flux

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`.
For any continuous tangent field on the original sphere, the mean flux
is exactly minus half its two-point relative-velocity pairing. The
corresponding two-point squared distance is twice the original variance
1-‖barycenter‖². A sphere Lipschitz constant therefore bounds variance
growth without evaluating the field at an off-sphere mean.

These are identities and estimates for the actual probability law.
No atomicity, characteristic representation, or weak uniqueness is
assumed. They apply to every original network field at a fixed time.
-/

@[expose] public section

open Real MeasureTheory
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original two-point squared distance is twice the original
spherical variance. Source auxiliary: arXiv:2411.04551v3, §1,
the atomic-data geometry of `eq: cauchy.pb`. -/
theorem integral_pair_squared_distance (ν : Perspective.ProbSphere d) :
    (∫ x : SSphere d, ∫ y : SSphere d, ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2
      ∂(ν : Measure (SSphere d)) ∂(ν : Measure (SSphere d))) =
      2 * (1 - ‖barycenter d ν‖ ^ 2) := by
  have he (x : SSphere d) :
      (∫ y : SSphere d, ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d))) =
        2 - 2 * inner ℝ (barycenter d ν) (x : EucSpace d) := by
    simp_rw [norm_sub_rev (x : EucSpace d)]
    rw [integral_spherical_squared_distance, integral_inner (integrable_spherical_coe ν)]
    rw [real_inner_comm]
    rfl
  simp_rw [he]
  have hi := integrable_spherical_continuous ν
    (fun x : SSphere d => inner ℝ (barycenter d ν) (x : EucSpace d))
    (continuous_const.inner continuous_subtype_val)
  rw [integral_sub (integrable_const 2) (hi.const_mul 2), integral_const_mul, integral_const,
    probReal_univ, one_smul, integral_inner (integrable_spherical_coe ν)]
  change 2 - 2 * inner ℝ (barycenter d ν) (barycenter d ν) = 2 * (1 - ‖barycenter d ν‖ ^ 2)
  rw [real_inner_self_eq_norm_sq]
  ring

/-- Tangency identifies the two-point relative-velocity pairing with
the original mean flux. Source auxiliary: arXiv:2411.04551v3, §1,
`eq: vf` and its original spherical variance identity. -/
theorem integral_pair_relative_velocity (ν : Perspective.ProbSphere d)
    (F : SSphere d → EucSpace d) (hF : Continuous F)
    (htan : ∀ x : SSphere d, inner ℝ (x : EucSpace d) (F x) = 0) :
    (∫ x : SSphere d, ∫ y : SSphere d,
      inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y)
        ∂(ν : Measure (SSphere d)) ∂(ν : Measure (SSphere d))) =
      -2 * inner ℝ (barycenter d ν) (∫ x, F x ∂(ν : Measure (SSphere d))) := by
  have hiF : Integrable F (ν : Measure (SSphere d)) :=
    hF.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hp (x y : SSphere d) :
      inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y) =
        -inner ℝ (F x) (y : EucSpace d) - inner ℝ (x : EucSpace d) (F y) := by
    simp only [inner_sub_left, inner_sub_right, htan x, htan y,
      real_inner_comm (y : EucSpace d) (F x)]
    ring
  have he (x : SSphere d) :
      (∫ y : SSphere d, inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y)
        ∂(ν : Measure (SSphere d))) =
        -inner ℝ (barycenter d ν) (F x) -
          inner ℝ (∫ y, F y ∂(ν : Measure (SSphere d))) (x : EucSpace d) := by
    simp_rw [hp]
    have h₁ := integrable_spherical_continuous ν
      (fun y : SSphere d => inner ℝ (F x) (y : EucSpace d))
      (continuous_const.inner continuous_subtype_val)
    have h₂ := integrable_spherical_continuous ν
      (fun y : SSphere d => inner ℝ (x : EucSpace d) (F y)) (continuous_const.inner hF)
    have hneg : Integrable (fun y : SSphere d => -inner ℝ (F x) (y : EucSpace d))
        (ν : Measure (SSphere d)) := h₁.neg
    rw [integral_sub hneg h₂, integral_neg,
      integral_inner (integrable_spherical_coe ν), integral_inner hiF]
    rw [real_inner_comm (F x), real_inner_comm (x : EucSpace d)]
    rfl
  simp_rw [he]
  have h₁ := integrable_spherical_continuous ν
    (fun x : SSphere d => inner ℝ (barycenter d ν) (F x)) (continuous_const.inner hF)
  have h₂ := integrable_spherical_continuous ν
    (fun x : SSphere d => inner ℝ (∫ y, F y ∂(ν : Measure (SSphere d))) (x : EucSpace d))
    (continuous_const.inner continuous_subtype_val)
  have hneg : Integrable (fun x : SSphere d => -inner ℝ (barycenter d ν) (F x))
      (ν : Measure (SSphere d)) := h₁.neg
  rw [integral_sub hneg h₂, integral_neg, integral_inner hiF,
    integral_inner (integrable_spherical_coe ν)]
  change -inner ℝ (barycenter d ν) (∫ x, F x ∂(ν : Measure (SSphere d))) -
    inner ℝ (∫ x, F x ∂(ν : Measure (SSphere d))) (barycenter d ν) = _
  rw [real_inner_comm (∫ x, F x ∂(ν : Measure (SSphere d)))]
  ring

/-- A tangent sphere Lipschitz field bounds the original variance flux.
Source auxiliary: arXiv:2411.04551v3, §1, weak well-posedness of
`eq: cauchy.pb`, and §1.1, original parameter-dependent stability. -/
theorem neg_inner_mean_flux_le_variance (ν : Perspective.ProbSphere d)
    (F : SSphere d → EucSpace d) (K : ℝ≥0) (hF : LipschitzWith K F)
    (htan : ∀ x : SSphere d, inner ℝ (x : EucSpace d) (F x) = 0) :
    -inner ℝ (barycenter d ν) (∫ x, F x ∂(ν : Measure (SSphere d))) ≤
      (K : ℝ) * (1 - ‖barycenter d ν‖ ^ 2) := by
  have hp (x y : SSphere d) :
      inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y) ≤
        (K : ℝ) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 := by
    have hh := hF.dist_le_mul x y
    change ‖F x - F y‖ ≤ (K : ℝ) * ‖(x : EucSpace d) - (y : EucSpace d)‖ at hh
    exact (real_inner_le_norm _ _).trans
      ((mul_le_mul_of_nonneg_left hh (norm_nonneg _)).trans_eq (by ring))
  have hi (x : SSphere d) := integral_mono
    (integrable_spherical_continuous ν _
      ((continuous_const.sub continuous_subtype_val).inner (continuous_const.sub hF.continuous)))
    (integrable_spherical_continuous ν _ (by fun_prop)) (hp x)
  have he : (fun x : SSphere d => ∫ y : SSphere d,
      (K : ℝ) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d))) =
      (fun x : SSphere d => (K : ℝ) * (2 - 2 * inner ℝ (barycenter d ν) (x : EucSpace d))) := by
    funext x
    rw [integral_const_mul]
    simp_rw [norm_sub_rev (x : EucSpace d)]
    rw [integral_spherical_squared_distance, integral_inner (integrable_spherical_coe ν), real_inner_comm]
    rfl
  have hright : Integrable (fun x : SSphere d => ∫ y : SSphere d,
      (K : ℝ) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ∂(ν : Measure (SSphere d)))
      (ν : Measure (SSphere d)) := by
    rw [he]
    exact integrable_spherical_continuous ν _ (by fun_prop)
  have hleft : Integrable (fun x : SSphere d => ∫ y : SSphere d,
      inner ℝ ((x : EucSpace d) - (y : EucSpace d)) (F x - F y) ∂(ν : Measure (SSphere d)))
      (ν : Measure (SSphere d)) := by
    have hc : Continuous (fun q : SSphere d × SSphere d =>
        inner ℝ ((q.1 : EucSpace d) - (q.2 : EucSpace d)) (F q.1 - F q.2)) :=
      ((continuous_subtype_val.comp continuous_fst).sub
        (continuous_subtype_val.comp continuous_snd)).inner
          ((hF.continuous.comp continuous_fst).sub (hF.continuous.comp continuous_snd))
    have hprod : Integrable (fun q : SSphere d × SSphere d =>
        inner ℝ ((q.1 : EucSpace d) - (q.2 : EucSpace d)) (F q.1 - F q.2))
        ((ν : Measure (SSphere d)).prod ν) :=
      hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
    exact hprod.integral_prod_left
  have hh := integral_mono hleft hright hi
  simp_rw [integral_const_mul] at hh
  rw [integral_pair_relative_velocity ν F hF.continuous htan, integral_pair_squared_distance] at hh
  linarith

/-- An actual projection field is tangent, continuous and sphere
Lipschitz. A symmetric two-atom law has strictly positive original
variance, so the flux estimate is tested away from zero variance. -/
example : ∃ (ν : Perspective.ProbSphere 2) (F : SSphere 2 → EucSpace 2) (K : ℝ≥0),
    Continuous F ∧ LipschitzWith K F ∧
    (∀ x : SSphere 2, inner ℝ (x : EucSpace 2) (F x) = 0) ∧ 0 < 1 - ‖barycenter 2 ν‖ ^ 2 := by
  let p : SSphere 2 := basePoint 1
  let q : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let ν := halfDirac 2 p (antipode 2 p)
  let F (x : SSphere 2) := proj 2 (x : EucSpace 2) (q : EucSpace 2)
  have hF : LipschitzWith 2 F := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    change ‖F x - F y‖ ≤ 2 * ‖(x : EucSpace 2) - (y : EucSpace 2)‖
    simpa only [mem_sphere_zero_iff_norm.mp q.2, mul_one] using
      Perspective.norm_proj_sub_proj_le_mul (mem_sphere_zero_iff_norm.mp x.2)
        (mem_sphere_zero_iff_norm.mp y.2) (q : EucSpace 2)
  have hm : barycenter 2 ν = 0 := by
    rw [barycenter_halfDirac]
    change (2 : ℝ)⁻¹ • (p : EucSpace 2) + (2 : ℝ)⁻¹ • -(p : EucSpace 2) = 0
    simp
  refine ⟨ν, F, 2, hF.continuous, hF, fun x => inner_proj_eq_zero
    (mem_sphere_zero_iff_norm.mp x.2) _, ?_⟩
  rw [hm, norm_zero]
  norm_num

end Transformer.Interpolation
