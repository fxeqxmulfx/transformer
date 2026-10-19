module

public import Transformer.Interpolation.Section3_CapCoefficients
public import Transformer.Interpolation.Section3_CapCenter

/-!
# Tangential error in the actual two-neuron cap construction

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
and the original projected perceptron in §1, `eq: vf`.

On the region where both inward activations are nonnegative, the
calibrated field splits exactly into a target-directed projection and a
scalar error times the center-directed projection. The error vanishes at
the target and is bounded by the distance to it, with an explicit constant.
The remaining projected inner product is bounded by that same distance.
Their product is therefore quadratic in the target distance.

The final identity retains the complete original two-neuron velocity.
It isolates its contracting term and this quadratic calibration error,
which a sufficiently large finite radial coefficient can dominate.
-/

@[expose] public section

open Real Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A unit tangent projection controls the residual target test by distance.
Auxiliary for arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`. -/
theorem abs_inner_proj_target_le (x ω z : SSphere d) :
    |inner ℝ (proj d x ω) (z : EucSpace d)| ≤ ‖(x : EucSpace d) - (ω : EucSpace d)‖ := by
  have he : proj d x ω = proj d x ((ω : EucSpace d) - (x : EucSpace d)) := by
    simp only [proj, inner_sub_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2]
    module
  have hc := abs_real_inner_le_norm (proj d x ω) (z : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp z.2, mul_one, he] at hc
  rw [he]
  exact hc.trans
    ((norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) _).trans_eq (norm_sub_rev _ _))

/-- The center projection tested against the displacement to the target
equals the negative residual target test. Auxiliary for arXiv:2411.04551v3,
Appendix, the two-neuron construction toward `lem: two.balls`. -/
theorem inner_proj_sub_target (x ω z : SSphere d) :
    inner ℝ (proj d x z) ((x : EucSpace d) - (ω : EucSpace d)) =
      -inner ℝ (proj d x ω) (z : EucSpace d) := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2,
    real_inner_comm (z : EucSpace d) (x : EucSpace d),
    real_inner_comm (z : EucSpace d) (ω : EucSpace d)]
  ring

/-- The actual scalar calibration error vanishes at the target and grows
at most linearly with the target distance. Auxiliary for arXiv:2411.04551v3,
Appendix, `lem: two.balls`. -/
theorem cap_calibration_error_le (z w ω x : SSphere d) (c h : ℝ)
    (hb : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) :
    |(inner ℝ (x : EucSpace d) (z : EucSpace d) - c) -
      ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
        (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) *
          (inner ℝ (x : EucSpace d) (w : EucSpace d) - h)| ≤
      ‖(z : EucSpace d) - ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
        (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ *
          ‖(x : EucSpace d) - (ω : EucSpace d)‖ := by
  let a := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let b := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  have hab : a / b * b = a := div_mul_cancel₀ a hb.ne'
  have he : (inner ℝ (x : EucSpace d) (z : EucSpace d) - c) -
      (a / b) * (inner ℝ (x : EucSpace d) (w : EucSpace d) - h) =
        inner ℝ ((x : EucSpace d) - (ω : EucSpace d)) ((z : EucSpace d) - (a / b) • (w : EucSpace d)) := by
    simp only [inner_sub_left, inner_sub_right,
      real_inner_smul_right]
    dsimp only [a, b] at hab
    dsimp only [a, b]
    nlinarith
  rw [he]
  exact (abs_real_inner_le_norm _ _).trans_eq (mul_comm _ _)

/-- The exact original two-neuron velocity splits into its target term and
its calibration error when both inward activations are nonnegative.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidgeField_cap_decomposition (z w ω x : SSphere d) (c h ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) - h) :
    twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) (capTargetOutput z w ω c h ℓ)
      (-(w : EucSpace d)) (-h) x =
      (ℓ * (inner ℝ (x : EucSpace d) (w : EucSpace d) - h) /
        (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • proj d x ω +
      ((inner ℝ (x : EucSpace d) (z : EucSpace d) - c) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) *
            (inner ℝ (x : EucSpace d) (w : EucSpace d) - h)) • proj d x z := by
  have haeq : -c - inner ℝ (x : EucSpace d) (-(z : EucSpace d)) =
      inner ℝ (x : EucSpace d) (z : EucSpace d) - c := by rw [inner_neg_right]; ring
  have hbeq : -h - inner ℝ (x : EucSpace d) (-(w : EucSpace d)) =
      inner ℝ (x : EucSpace d) (w : EucSpace d) - h := by rw [inner_neg_right]; ring
  rw [twoRidgeField, haeq, hbeq, max_eq_left hα, max_eq_left hβ]
  simp only [capTargetOutput, proj, inner_sub_right,
    real_inner_smul_right, div_eq_mul_inv]
  module

/-- Exact squared-distance dissipation for the calibrated original field.
Auxiliary for arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere` and
the constant-control construction toward `lem: two.balls`. -/
theorem inner_twoRidge_cap_sub_target (z w ω x : SSphere d) (c h ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) - h) :
    inner ℝ (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c)
      (capTargetOutput z w ω c h ℓ) (-(w : EucSpace d)) (-h) x)
      ((x : EucSpace d) - (ω : EucSpace d)) =
      -((ℓ * (inner ℝ (x : EucSpace d) (w : EucSpace d) - h) /
        (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) *
        (1 + inner ℝ (x : EucSpace d) (ω : EucSpace d)) / 2) *
          ‖(x : EucSpace d) - (ω : EucSpace d)‖ ^ 2 -
      ((inner ℝ (x : EucSpace d) (z : EucSpace d) - c) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) *
            (inner ℝ (x : EucSpace d) (w : EucSpace d) - h)) *
              inner ℝ (proj d x ω) (z : EucSpace d) := by
  have hr := inner_ridge_velocity_sub_output ω x 0 1
  simp only [inner_zero_right, sub_zero, max_eq_left zero_le_one,
    one_smul, one_mul] at hr
  rw [twoRidgeField_cap_decomposition z w ω x c h ℓ hα hβ,
    inner_add_left, real_inner_smul_left, real_inner_smul_left, hr,
    inner_proj_sub_target]
  ring

/-- A transverse unit particle has positive outer activation, zero inner
activation and a target with strictly positive inner activation.
Thus all decomposition and error hypotheses hold on a nontrivial geometry. -/
example : ∃ z w ω x : SSphere 2,
    0 < inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - 0 ∧
    0 ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) - (-1 / 2) ∧
    0 ≤ inner ℝ (x : EucSpace 2) (w : EucSpace 2) - 0 ∧
    x ≠ ω := by
  let a : SSphere 2 := basePoint 1
  let x : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  refine ⟨a, a, a, x,
    ?_, ?_, ?_, ?_⟩
  · norm_num [a, basePoint]
  · norm_num [x, a, basePoint, EuclideanSpace.inner_single_left]
  · norm_num [x, a, basePoint, EuclideanSpace.inner_single_left]
  · intro he
    have hi : inner ℝ (x : EucSpace 2) (a : EucSpace 2) = 0 := by
      norm_num [x, a, basePoint, EuclideanSpace.inner_single_left]
    rw [he, real_inner_self_eq_norm_sq,
      mem_sphere_zero_iff_norm.mp a.2] at hi
    norm_num at hi

end Transformer.Interpolation
