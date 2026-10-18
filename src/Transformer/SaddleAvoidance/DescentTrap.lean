/-
# Saddle avoidance — descent curves are trapped near a Łojasiewicz point

The trap of `SaddleAvoidance.FlowTrap`, for the curves that descend `f` at the
rates `c₁` and `c₂` (`IsDescentCurve`) instead of the gradient flow, with the
constants carried along.  Where `m (f - f z) ^ a ≤ ‖∇f‖` with `0 ≤ a < 1` and
`f > f z`, the speed `‖ẋ‖ ≤ c₂ ‖∇f‖` is at most `c₂ / (c₁ m (1 - a))` times the
rate at which `(f - f z) ^ (1 - a)` decreases, because `f` decreases at least
at the rate `c₁ ‖∇f‖²`.  So the length of the path is bounded by the drop of
`(f - f z) ^ (1 - a)` (`IsDescentCurve.mul_norm_sub_le`, through Mathlib's
fencing theorem `image_norm_le_of_norm_deriv_right_le_deriv_boundary'`), and a
descent curve that starts close to `z`, with `f` close to `f z` and above it,
never leaves the ball where the Łojasiewicz inequality holds: at its first
exit time it would still be inside (`IsDescentCurve.mem_ball`).

Source: arXiv:2312.10794v5, §4, after `thm: beta.interval`, and Appendix A:
the argument of Łojasiewicz's theorem (lojasiewicz1963propriete), for the
gradient flows of `𝖤_β` in the metric of `SA` (§3.4) as in that of `USA`.
-/

import Transformer.SaddleAvoidance.DescentCurve

open Filter Metric Set
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a m c₁ c₂ t₀ : ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **The length of a descent curve is bounded by the drop of
`(f - f z) ^ (1 - a)`** while `f` stays above `f z` and
`m (f - f z) ^ a ≤ ‖∇f‖`, with `m, c₁ > 0`, `c₂ ≥ 0` and `a < 1`. -/
theorem IsDescentCurve.mul_norm_sub_le (hf : Differentiable ℝ f) {s t : ℝ} (hst : s ≤ t)
    (hx : IsDescentCurve f c₁ c₂ s x) (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) (hm : 0 < m)
    (ha1 : a < 1) {z : E} (hgap : ∀ u ∈ Icc s t, f z < f (x u))
    (hloj : ∀ u ∈ Ico s t, m * (f (x u) - f z) ^ a ≤ ‖gradient f (x u)‖) :
    c₁ * m * (1 - a) * ‖x t - x s‖ ≤
      c₂ * ((f (x s) - f z) ^ (1 - a) - (f (x t) - f z) ^ (1 - a)) := by
  have hκ : 0 < c₁ * m * (1 - a) := mul_pos (mul_pos hc₁ hm) (sub_pos.2 ha1)
  set φ : ℝ → ℝ := fun u => (f (x u) - f z) ^ (1 - a) with hφ
  have hφd : ∀ u ∈ Icc s t, HasDerivAt φ
      (⟪gradient f (x u), deriv x u⟫ * (1 - a) * (f (x u) - f z) ^ (1 - a - 1)) u := fun u hu =>
    ((hx.hasDerivAt_comp hu.1 (hf _)).sub_const (f z)).rpow_const
      (Or.inl (sub_pos.2 (hgap u hu)).ne')
  have hfence := image_norm_le_of_norm_deriv_right_le_deriv_boundary'
    (f := fun u => x u - x s) (f' := deriv x) (a := s) (b := t)
    (B := fun u => c₂ * (φ s - φ u) / (c₁ * m * (1 - a)))
    (B' := fun u => c₂ * -⟪gradient f (x u), deriv x u⟫ * (f (x u) - f z) ^ (1 - a - 1) /
      (c₁ * m))
    (fun u hu => ((hx.hasDerivAt hu.1).sub_const (x s)).continuousAt.continuousWithinAt)
    (fun u hu => ((hx.hasDerivAt hu.1).sub_const (x s)).hasDerivWithinAt)
    (by simp) (fun u hu =>
      (((continuousAt_const.sub (hφd u hu).continuousAt).const_mul c₂).div_const
        _).continuousWithinAt)
    (fun u hu => ?_) (fun u hu => ?_) (right_mem_Icc.2 hst)
  · rw [le_div_iff₀ hκ] at hfence
    simpa only [hφ, mul_comm] using hfence
  · convert ((((hφd u (Ico_subset_Icc_self hu)).const_sub (φ s)).const_mul c₂).div_const
      (c₁ * m * (1 - a))).hasDerivWithinAt using 1
    have h1a : 1 - a ≠ 0 := (sub_pos.2 ha1).ne'
    field_simp
  · have hD := sub_pos.2 (hgap u (Ico_subset_Icc_self hu))
    have hDa : (f (x u) - f z) ^ (1 - a - 1) = ((f (x u) - f z) ^ a)⁻¹ := by
      rw [show 1 - a - 1 = -a by ring, Real.rpow_neg hD.le]
    have hpos : 0 < (f (x u) - f z) ^ a * (c₁ * m) :=
      mul_pos (Real.rpow_pos_of_pos hD a) (mul_pos hc₁ hm)
    obtain ⟨-, h₁, h₂⟩ := hx u hu.1
    have hl := hloj u hu
    rw [hDa, mul_assoc, ← div_eq_mul_inv, mul_div_assoc, div_div, ← mul_div_assoc,
      le_div_iff₀ hpos]
    have hv := norm_nonneg (deriv x u)
    have hg := norm_nonneg (gradient f (x u))
    nlinarith [mul_le_mul_of_nonneg_left hl (mul_nonneg hv hc₁.le),
      mul_le_mul_of_nonneg_left h₂ (mul_nonneg hc₁.le hg), mul_le_mul_of_nonneg_left h₁ hc₂]

/-- **A descent curve is trapped near a point where the Łojasiewicz inequality
holds.**  If the inequality holds on `B(z, r)` below `f z + ε`, the curve stays
above `f z` from time `t₀` on, starts below `f z + ε`, and
`c₁ m (1 - a) ‖x t₀ - z‖ + c₂ (f (x t₀) - f z) ^ (1 - a) < c₁ m (1 - a) r`, then
it never leaves `B(z, r)`. -/
theorem IsDescentCurve.mem_ball (hf : Differentiable ℝ f) (hx : IsDescentCurve f c₁ c₂ t₀ x)
    (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) (hm : 0 < m) (ha1 : a < 1) {z : E} {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hgap : ∀ t, t₀ ≤ t → f z < f (x t)) (hε : f (x t₀) < f z + ε)
    (hstart : c₁ * m * (1 - a) * ‖x t₀ - z‖ + c₂ * (f (x t₀) - f z) ^ (1 - a) <
      c₁ * m * (1 - a) * r)
    {t : ℝ} (ht : t₀ ≤ t) : x t ∈ ball z r := by
  by_contra hout
  set S := Ici t₀ ∩ x ⁻¹' (ball z r)ᶜ with hS
  have hSc : IsClosed S := ContinuousOn.preimage_isClosed_of_isClosed
    (fun u hu => (hx.hasDerivAt hu).continuousAt.continuousWithinAt) isClosed_Ici
    isOpen_ball.isClosed_compl
  have hSb : BddBelow S := ⟨t₀, fun u hu => hu.1⟩
  have hT := hSc.csInf_mem ⟨t, ht, hout⟩ hSb
  have hin : ∀ u ∈ Ico t₀ (sInf S), x u ∈ ball z r := fun u hu => by
    by_contra h
    exact (csInf_le hSb ⟨hu.1, h⟩).not_gt hu.2
  have hanti := hx.antitoneOn hf hc₁.le
  have hlen := hx.mul_norm_sub_le hf (t := sInf S) hT.1 hc₁ hc₂ hm ha1
    (fun u hu => hgap u hu.1) fun u hu => by
      have hl := hloj _ (hin u hu) (hgap u hu.1)
        ((hanti self_mem_Ici hu.1 hu.1).trans_lt hε)
      rwa [abs_of_pos (sub_pos.2 (hgap u hu.1))] at hl
  have hκ : 0 < c₁ * m * (1 - a) := mul_pos (mul_pos hc₁ hm) (sub_pos.2 ha1)
  have hφ := mul_nonneg hc₂ (Real.rpow_nonneg (sub_pos.2 (hgap _ hT.1)).le (1 - a))
  have htri := mul_le_mul_of_nonneg_left
    (norm_sub_le_norm_sub_add_norm_sub (x (sInf S)) (x t₀) z) hκ.le
  refine hT.2 (mem_ball_iff_norm.2 (lt_of_mul_lt_mul_left ?_ hκ.le))
  linarith

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line and its gradient flow `x t = e^{-t}`, which descends `f` at the rates `1`
and `1`, with `a = 1/2`, `m = 1`, `z = 0`, `r = 4`, `ε = 1` and the times
`0 ≤ 1`: then `c₁ m (1 - a) ‖x 0‖ = 1/2` and `c₂ (f (x 0)) ^ (1/2) ≤ 1` add up
to less than `c₁ m (1 - a) r = 2`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ IsDescentCurve f 1 1 0 x ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
      (1 / 2 : ℝ) < 1 ∧ (∀ t, f 0 < f (x t)) ∧
      (∀ t, 1 * (f (x t) - f 0) ^ (1 / 2 : ℝ) ≤ ‖gradient f (x t)‖) ∧
      (∀ y ∈ ball (0 : ℝ) 4, f 0 < f y → f y < f 0 + 1 →
        1 * |f y - f 0| ^ (1 / 2 : ℝ) ≤ ‖gradient f y‖) ∧ f (x 0) < f 0 + 1 ∧
      1 * 1 * (1 - 1 / 2) * ‖x 0 - 0‖ + 1 * (f (x 0) - f 0) ^ (1 - 1 / 2 : ℝ) <
        1 * 1 * (1 - 1 / 2) * 4 := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hf0 : ∀ y, f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := fun y => by
    simp only [f, norm_zero]
    ring
  have hx0 : x 0 = 1 := by simp [x]
  have hroot : (f (x 0) - f 0) ^ (1 / 2 : ℝ) ≤ ‖x 0‖ := by
    rw [hf0]
    exact half_mul_norm_sq_rpow_le _
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero,
    isDescentCurve_of_gradientFlow fun t _ => ?_, zero_le_one, one_pos, by norm_num,
    fun t => ?_, fun t => ?_, fun y _ _ _ => ?_, ?_, ?_⟩
  · rw [hg]
    simpa [x] using (hasDerivAt_neg t).exp
  · rw [← sub_pos, hf0]
    have := Real.exp_pos (-t)
    simp only [x, Real.norm_eq_abs, abs_of_pos this]
    positivity
  · rw [hf0, one_mul, hg, id]
    exact half_mul_norm_sq_rpow_le _
  · rw [hf0, abs_of_nonneg (by positivity), one_mul, hg, id]
    exact half_mul_norm_sq_rpow_le y
  · rw [← sub_lt_iff_lt_add', hf0, hx0]
    norm_num
  · rw [hx0] at hroot ⊢
    norm_num at hroot ⊢
    linarith

end Transformer.SaddleAvoidance
