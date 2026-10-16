/-
# Inward drift and invariant cap strips for the actual two-neuron flow

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
and §1, the original particle equation `eq: ips`.

On a thin strip inside the inner cap, the first neuron points strictly
inward and its activation has a uniform positive lower bound. An explicit
cone inequality makes the second neuron's inward contribution nonnegative.
The total cap-height derivative is therefore uniformly positive there.

A scalar fencing argument then prevents an actual particle from crossing
any positive level in this strip in the outward direction. The proof uses
the true ReLU field and does not require differentiating the activation.
-/

import Transformer.Interpolation.Section3_TwoRidgeLaw
import Transformer.Interpolation.SeparationField

open Real Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The inner cap height has the derivative given by the actual velocity.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem hasDerivAt_twoRidge_cap_activation (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) (w x₀ : SSphere d) (h t : ℝ) :
    HasDerivAt (fun s => inner ℝ (twoRidgeFlow p u b q v c x₀ s : EucSpace d) (w : EucSpace d) - h)
      (inner ℝ (twoRidgeField p u b q v c (twoRidgeFlow p u b q v c x₀ t)) (w : EucSpace d)) t := by
  simpa only [inner_zero_right, zero_add] using
    ((hasDerivAt_twoRidgeFlow p u b q v c x₀ t).inner ℝ (hasDerivAt_const t (w : EucSpace d))).sub_const h

/-- The actual two-neuron velocity points uniformly inward on the strip.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidge_cap_boundary_drift (z w x : SSphere d) (q : EucSpace d) (c h m σ : ℝ)
    (hh : 0 ≤ h) (hm : 0 < m) (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hα : m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) - h)
    (hβσ : inner ℝ (x : EucSpace d) (w : EucSpace d) - h ≤ σ)
    (hq : (h + σ) * ‖q‖ ≤ inner ℝ q (w : EucSpace d)) :
    m * (inner ℝ (z : EucSpace d) (w : EucSpace d) - h - σ) ≤
      inner ℝ (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x)
        (w : EucSpace d) := by
  have ha : 0 ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c := by linarith
  have hxw : 0 ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) := by linarith
  have hxz := abs_real_inner_le_norm (x : EucSpace d) (z : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp z.2, mul_one] at hxz
  have hprod := mul_le_mul_of_nonneg_right (abs_le.mp hxz).2 hxw
  have hg : 0 ≤ inner ℝ (z : EucSpace d) (w : EucSpace d) - h - σ := by linarith
  have hfirst := mul_le_mul hα
    (show inner ℝ (z : EucSpace d) (w : EucSpace d) - h - σ ≤
      inner ℝ (z : EucSpace d) (w : EucSpace d) -
        inner ℝ (x : EucSpace d) (z : EucSpace d) * inner ℝ (x : EucSpace d) (w : EucSpace d) by nlinarith)
    hg ha
  have hxq := abs_real_inner_le_norm (x : EucSpace d) q
  rw [mem_sphere_zero_iff_norm.mp x.2, one_mul] at hxq
  have hqprod := mul_le_mul ((le_abs_self _).trans hxq)
    (show inner ℝ (x : EucSpace d) (w : EucSpace d) ≤ h + σ by linarith) hxw (norm_nonneg q)
  have hsecond : 0 ≤ (inner ℝ (x : EucSpace d) (w : EucSpace d) - h) *
      (inner ℝ q (w : EucSpace d) - inner ℝ (x : EucSpace d) q * inner ℝ (x : EucSpace d) (w : EucSpace d)) :=
    mul_nonneg hβ (by nlinarith)
  have haeq : -c - inner ℝ (x : EucSpace d) (-(z : EucSpace d)) =
      inner ℝ (x : EucSpace d) (z : EucSpace d) - c := by rw [inner_neg_right]; ring
  have hbeq : -h - inner ℝ (x : EucSpace d) (-(w : EucSpace d)) =
      inner ℝ (x : EucSpace d) (w : EucSpace d) - h := by rw [inner_neg_right]; ring
  rw [twoRidgeField, haeq, hbeq, max_eq_left ha, max_eq_left hβ]
  simp only [inner_add_left, real_inner_smul_left, proj, inner_sub_left]
  linarith

/-- Strictly positive derivative at a level prevents a downward crossing.
Auxiliary for arXiv:2411.04551v3, Appendix, the cap-invariance step in
`lem: two.balls`; this is the scalar fencing argument used there. -/
theorem scalar_lower_barrier (f f' : ℝ → ℝ) (a : ℝ)
    (hd : ∀ s : ℝ, 0 ≤ s → HasDerivAt f (f' s) s) (h0 : a ≤ f 0)
    (hpos : ∀ s : ℝ, 0 ≤ s → f s = a → 0 < f' s) (t : ℝ) (ht : 0 ≤ t) : a ≤ f t := by
  have hf (s : ℝ) (hs : 0 ≤ s) : HasDerivAt (fun r => a - f r) (-f' s) s :=
    (hd s hs).const_sub a
  have hb := image_le_of_deriv_right_lt_deriv_boundary
    (f := fun s => a - f s) (f' := fun s => -f' s) (B := fun _ => 0) (B' := fun _ => 0)
    (a := 0) (b := t)
    (fun s hs => (hf s hs.1).continuousAt.continuousWithinAt)
    (fun s hs => (hf s hs.1).hasDerivWithinAt)
    (by linarith : a - f 0 ≤ 0) (fun s => hasDerivAt_const s 0)
    (fun s hs he => by have hp := hpos s hs.1 (by linarith); linarith)
    (x := t) ⟨ht, le_rfl⟩
  linarith

/-- Every positive level in the strip is forward invariant under the
actual original two-neuron flow. Auxiliary for arXiv:2411.04551v3,
Appendix, `lem: two.balls`. -/
theorem twoRidge_cap_strip_invariant (z w x₀ : SSphere d) (q : EucSpace d) (c h m σ δ : ℝ)
    (hh : 0 ≤ h) (hm : 0 < m) (hσz : σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h)
    (hmargin : ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
      m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c)
    (hq : (h + σ) * ‖q‖ ≤ inner ℝ q (w : EucSpace d)) (hδ : 0 < δ) (hδσ : δ ≤ σ)
    (h0 : δ ≤ inner ℝ (x₀ : EucSpace d) (w : EucSpace d) - h) (t : ℝ) (ht : 0 ≤ t) :
    δ ≤ inner ℝ
      (twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀ t : EucSpace d)
      (w : EucSpace d) - h := by
  let X := twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x₀
  apply scalar_lower_barrier _
    (fun s => inner ℝ (twoRidgeField (z : EucSpace d) (-(z : EucSpace d)) (-c) q
      (-(w : EucSpace d)) (-h) (X s)) (w : EucSpace d)) δ
    (fun s _ => hasDerivAt_twoRidge_cap_activation _ _ _ _ _ _ _ _ _ _) ?_ ?_ t ht
  · simpa only [X, twoRidgeFlow_zero] using h0
  · intro s hs he
    have hx : h ≤ inner ℝ (X s : EucSpace d) (w : EucSpace d) := by linarith
    have hd := twoRidge_cap_boundary_drift z w (X s) q c h m σ hh hm hσz
      (hmargin (X s) hx) (by linarith) (by linarith) hq
    have hp : 0 < m * (inner ℝ (z : EucSpace d) (w : EucSpace d) - h - σ) :=
      mul_pos hm (by linarith)
    exact hp.trans_le hd

/-- The scalar barrier assumptions hold jointly for a strictly increasing
trajectory starting exactly on its barrier. -/
example : (∀ s : ℝ, 0 ≤ s → HasDerivAt (fun r : ℝ => r) 1 s) ∧
    (0 : ℝ) ≤ (fun r : ℝ => r) 0 ∧
    (∀ s : ℝ, 0 ≤ s → (fun r : ℝ => r) s = 0 → (0 : ℝ) < 1) ∧ (0 : ℝ) ≤ 1 :=
  ⟨fun s _ => hasDerivAt_id s, le_rfl, fun _ _ _ => zero_lt_one, zero_le_one⟩

/-- A nonzero two-neuron field has a genuine positive activation margin,
inward cone, boundary strip, invariant level and admissible initial point. -/
example : ∃ (z w x₀ : SSphere 2) (q : EucSpace 2),
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧
    (1 / 4 : ℝ) < inner ℝ (z : EucSpace 2) (w : EucSpace 2) - 1 / 2 ∧
    (∀ x : SSphere 2, (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (w : EucSpace 2) →
      (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (z : EucSpace 2) - 0) ∧
    (1 / 2 + 1 / 4 : ℝ) * ‖q‖ ≤ inner ℝ q (w : EucSpace 2) ∧
    (0 : ℝ) < 1 / 16 ∧ (1 / 16 : ℝ) ≤ 1 / 4 ∧
    (1 / 16 : ℝ) ≤ inner ℝ (x₀ : EucSpace 2) (w : EucSpace 2) - 1 / 2 ∧
    (1 / 2 : ℝ) ≤ inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) - 0 ∧
    0 ≤ inner ℝ (x₀ : EucSpace 2) (w : EucSpace 2) - 1 / 2 ∧
    inner ℝ (x₀ : EucSpace 2) (w : EucSpace 2) - 1 / 2 ≤ (1 / 4 : ℝ) ∧ (0 : ℝ) ≤ 1 := by
  let z : SSphere 2 := basePoint 1
  let v : EucSpace 2 := (3 / 5 : ℝ) • (z : EucSpace 2) + (4 / 5 : ℝ) • EuclideanSpace.single 1 1
  have hv : ‖v‖ = 1 := by
    have hs : ‖v‖ ^ 2 = 1 := by
      norm_num [v, z, basePoint, norm_add_sq_real, norm_smul,
        real_inner_smul_left, real_inner_smul_right, EuclideanSpace.inner_single_left]
    nlinarith [norm_nonneg v]
  let x₀ : SSphere 2 := ⟨v, mem_sphere_zero_iff_norm.mpr hv⟩
  have hi : inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) = 3 / 5 := by
    norm_num [x₀, v, z, basePoint, inner_add_left,
      real_inner_smul_left, EuclideanSpace.inner_single_left]
  refine ⟨z, z, x₀, (2 : ℝ) • (z : EucSpace 2), by norm_num, by norm_num, ?_, ?_, ?_,
    by norm_num, by norm_num, ?_, ?_, ?_, ?_, zero_le_one⟩
  · norm_num [z, basePoint]
  · intro x hx
    simpa only [sub_zero] using hx
  · norm_num [norm_smul, real_inner_smul_left, z, basePoint, EuclideanSpace.inner_single_left]
  all_goals rw [hi]; norm_num

end Transformer.Interpolation
