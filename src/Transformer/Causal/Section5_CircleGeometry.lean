/-
# Circle geometry behind the strong-center barrier

ArXiv:2411.04990v2, §5, `lemma:meta`, and `sec:proof_lemma_meta`.

The closest preceding token can only be pulled toward a strong center by
tokens on the opposite side of the empty arc. Their distance from that
closest token is at least the center's current gap. This is the geometric
step preceding `eq:bddotphi` in the proof of the metastability estimate.

We prove that step directly for the actual unit vectors in `ℝ²`, using their
Gram identity. No choice of angle representatives or common angular chart
is assumed. The argument therefore also covers configurations straddling
the cut of a polar-coordinate chart.
-/

import Transformer.Causal.Basic

open Real

namespace Transformer.Causal

/-- The actual inner product in the two-dimensional circle model is the
sum of the two coordinate products. Source: arXiv:2411.04990v2, §5,
the polar-coordinate calculation in the proof of `lemma:meta`. -/
theorem circle_inner_coordinates (x y : EucSpace 2) :
    inner (𝕜 := ℝ) x y = x 0 * y 0 + x 1 * y 1 := by
  rw [PiLp.inner_apply, Fin.sum_univ_two, Real.inner_apply, Real.inner_apply]

/-- The coordinates of an actual circle token satisfy the unit equation.
Source: arXiv:2411.04990v2, §5, `lemma:meta`, whose tokens lie on `𝕊¹`. -/
theorem circle_unit_coordinates (x : SSphere 2) :
    (x : EucSpace 2) 0 ^ 2 + (x : EucSpace 2) 1 ^ 2 = 1 := by
  have hx := EuclideanSpace.real_norm_sq_eq (x : EucSpace 2)
  rw [mem_sphere_zero_iff_norm.mp x.2, Fin.sum_univ_two] at hx
  simpa only [one_pow] using hx.symm

/-- Circle inner products have absolute value at most one.
Source: arXiv:2411.04990v2, §5, the distances in `lemma:meta`. -/
theorem abs_circle_inner_le_one (x y : SSphere 2) :
    |inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)| ≤ 1 := by
  have hb := abs_real_inner_le_norm (x : EucSpace 2) (y : EucSpace 2)
  simpa only [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2,
    one_mul] using hb

/-- **The three-token Gram identity on the circle.** In two dimensions
the conditional inner product has exactly the product of the two tangent
norms. Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`.
The equality is what makes the closest-token argument specific to `d = 2`. -/
theorem circle_gram_identity (x y z : SSphere 2) :
    (inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) -
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) *
        inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2)) ^ 2 =
      (1 - inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) ^ 2) *
        (1 - inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) ^ 2) := by
  let D (u v : EucSpace 2) := u 0 * v 1 - u 1 * v 0
  have hD (u v : SSphere 2) : D u v ^ 2 =
      1 - inner (𝕜 := ℝ) (u : EucSpace 2) (v : EucSpace 2) ^ 2 := by
    have heq : inner (𝕜 := ℝ) (u : EucSpace 2) (v : EucSpace 2) ^ 2 + D u v ^ 2 =
        ((u : EucSpace 2) 0 ^ 2 + (u : EucSpace 2) 1 ^ 2) *
          ((v : EucSpace 2) 0 ^ 2 + (v : EucSpace 2) 1 ^ 2) := by
      rw [circle_inner_coordinates]
      dsimp [D]
      ring
    rw [circle_unit_coordinates, circle_unit_coordinates, mul_one] at heq
    linarith
  have hc : inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) -
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) *
        inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) = -D x y * D y z := by
    rw [circle_inner_coordinates, circle_inner_coordinates, circle_inner_coordinates]
    dsimp [D]
    linear_combination -((x : EucSpace 2) 0 * (z : EucSpace 2) 0 +
      (x : EucSpace 2) 1 * (z : EucSpace 2) 1) * circle_unit_coordinates y
  rw [hc, mul_pow, neg_sq, hD x y, hD y z]

/-- The actual projected direction has the sine-squared norm of the angle
between its unit vectors. Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. -/
theorem circle_norm_proj_sq (x y : SSphere 2) :
    ‖proj 2 (x : EucSpace 2) (y : EucSpace 2)‖ ^ 2 =
      1 - inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) ^ 2 := by
  rw [proj, norm_sub_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs,
    mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2, mul_one, sq_abs]
  rw [real_inner_comm (y : EucSpace 2) (x : EucSpace 2)]
  ring

/-- The radial derivative of a center-to-predecessor chord has the
conditional inner product appearing in the Gram identity.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`. -/
theorem circle_chord_inner_proj (x y z : SSphere 2) :
    inner (𝕜 := ℝ) ((y : EucSpace 2) - (x : EucSpace 2))
      (proj 2 (y : EucSpace 2) (z : EucSpace 2)) =
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) *
        inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) -
      inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) := by
  rw [inner_sub_left, inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp y.2)]
  simp only [proj, inner_sub_right, real_inner_smul_right]
  ring

/-- **A token pulling the closest predecessor inward is farther from
that predecessor than the center is.** The hypothesis `hgap` says that `y`
is at least as close to the center `x` as `z` is. The strict inward sign
forces `⟪y,z⟫ ≤ ⟪x,y⟫`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the bound on
`-φ̇_{r*}` following `eq:bddotphi`. -/
theorem inner_le_of_circle_inward (x y z : SSphere 2)
    (hgap : inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) ≤
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2))
    (hinward : inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) *
      inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) <
        inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2)) :
    inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) ≤
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) := by
  let a := inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)
  let b := inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2)
  let c := inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2)
  have ha1 : a ≤ 1 := (le_abs_self a).trans (abs_circle_inner_le_one x y)
  have hb1 : b ≤ 1 := (le_abs_self b).trans (abs_circle_inner_le_one y z)
  have hcov : 0 < c - a * b := sub_pos.mpr hinward
  have hgram : (c - a * b) ^ 2 = (1 - a ^ 2) * (1 - b ^ 2) := circle_gram_identity x y z
  have ha : 0 < a := by
    by_contra h
    have hn : 0 ≤ -a * (1 - b) := mul_nonneg (by linarith) (by linarith)
    nlinarith
  have hb : b < 1 := by
    by_contra h
    have heq : b = 1 := by linarith
    rw [heq] at hgram
    nlinarith
  have hupper : c - a * b ≤ a * (1 - b) := by linarith
  have hsq := mul_self_le_mul_self hcov.le hupper
  have hfac : (1 - b) * (b + 1 - 2 * a ^ 2) ≤ 0 := by nlinarith [hgram]
  have hbound : b ≤ 2 * a ^ 2 - 1 := by
    by_contra h
    have hp : 0 < (1 - b) * (b + 1 - 2 * a ^ 2) :=
      mul_pos (by linarith) (by linarith)
    linarith
  have hp : 0 ≤ (2 * a + 1) * (1 - a) := mul_nonneg (by linarith) (by linarith)
  change b ≤ a
  nlinarith

/-- All hypotheses of the inward-pull lemma hold for the explicit unit
tokens `(1,0)`, `(3/5,4/5)` and `(0,-1)`. -/
example : ∃ x y z : SSphere 2,
    inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) ≤
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) ∧
    inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2) *
      inner (𝕜 := ℝ) (y : EucSpace 2) (z : EucSpace 2) <
        inner (𝕜 := ℝ) (x : EucSpace 2) (z : EucSpace 2) := by
  let v : EucSpace 2 := EuclideanSpace.single 0 (3 / 5 : ℝ) +
    EuclideanSpace.single 1 (4 / 5 : ℝ)
  have hv : ‖v‖ = 1 := by
    have hs : ‖v‖ ^ 2 = 1 := by
      rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
      norm_num [v, PiLp.single_apply]
    nlinarith [norm_nonneg v]
  let y : SSphere 2 := ⟨v, mem_sphere_zero_iff_norm.mpr hv⟩
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 (-1 : ℝ), by simp [PiLp.norm_single]⟩
  refine ⟨basePoint 1, y, z, ?_, ?_⟩ <;>
    norm_num [circle_inner_coordinates, basePoint, y, z, v, PiLp.single_apply]

end Transformer.Causal
