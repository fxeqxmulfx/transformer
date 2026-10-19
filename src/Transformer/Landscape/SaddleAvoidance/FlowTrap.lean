module

public import Mathlib.Analysis.Calculus.Deriv.MeanValue

public import Transformer.Landscape.SaddleAvoidance.Convergence
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# Saddle avoidance — gradient flows are trapped near a Łojasiewicz point

Along a trajectory of the gradient flow `ẋ = -∇f(x)`, `f` decreases at the
rate `‖∇f‖²` (`hasDerivAt_comp_gradientFlow`, `antitoneOn_comp_gradientFlow`).
Where `m (f - f z) ^ a ≤ ‖∇f‖` with `0 ≤ a < 1` and `f > f z`, the speed
`‖∇f‖` is at most the rate at which `(f - f z) ^ (1 - a) / (m (1 - a))`
decreases, so the length of the path is bounded by the drop of
`(f - f z) ^ (1 - a)` (`mul_norm_sub_le_of_gradientFlow`, through Mathlib's
fencing theorem `image_norm_le_of_norm_deriv_right_le_deriv_boundary'`). A
trajectory that starts close to `z`, with `f` close to `f z` and above it,
therefore never leaves the ball where the Łojasiewicz inequality holds: at
its first exit time it would still be inside (`mem_ball_of_gradientFlow`).

This is the continuous-time trap of `Landscape.SaddleAvoidance.Trap`: the argument of
Łojasiewicz's theorem (lojasiewicz1963propriete), which arXiv:2312.10794v5
invokes in Appendix A for the convergence of every trajectory, here with
the inequality assumed only near one point.
`Landscape.Lojasiewicz.displacement_le_potential_drop` carries out the same length
estimate on `ℝᴺ`, with the bound on the speed asked on the whole closed
interval; the trap needs the estimate up to an exit time, where the
Łojasiewicz inequality may fail.
-/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a m : ℝ}

/-- **`f` decreases along the gradient flow at the rate `‖∇f‖²`.** -/
theorem hasDerivAt_comp_gradientFlow {t : ℝ} (hf : DifferentiableAt ℝ f (x t))
    (hx : HasDerivAt x (-gradient f (x t)) t) :
    HasDerivAt (fun s => f (x s)) (-‖gradient f (x t)‖ ^ 2) t := by
  have h := hf.hasFDerivAt.comp_hasDerivAt t hx
  rwa [← inner_gradient_left, inner_neg_right, real_inner_self_eq_norm_sq] at h

theorem antitoneOn_comp_gradientFlow (hf : Differentiable ℝ f) {t₀ : ℝ}
    (hx : ∀ t, t₀ ≤ t → HasDerivAt x (-gradient f (x t)) t) :
    AntitoneOn (fun t => f (x t)) (Ici t₀) := by
  refine antitoneOn_of_hasDerivWithinAt_nonpos (f' := fun t => -‖gradient f (x t)‖ ^ 2)
    (convex_Ici t₀)
    (fun t ht => (hasDerivAt_comp_gradientFlow (hf _) (hx t ht)).continuousAt.continuousWithinAt)
    (fun t ht => ?_) fun _ _ => neg_nonpos.2 (sq_nonneg _)
  exact (hasDerivAt_comp_gradientFlow (hf _) (hx t (interior_subset ht))).hasDerivWithinAt

/-- **The length of a gradient-flow path is bounded by the drop of
`(f - f z) ^ (1 - a)`** while `f` stays above `f z` and
`m (f - f z) ^ a ≤ ‖∇f‖`, with `m > 0` and `0 ≤ a < 1`. -/
theorem mul_norm_sub_le_of_gradientFlow (hf : Differentiable ℝ f) {s t : ℝ} (hst : s ≤ t)
    (hx : ∀ u ∈ Icc s t, HasDerivAt x (-gradient f (x u)) u) (hm : 0 < m) (ha1 : a < 1)
    {z : E} (hgap : ∀ u ∈ Icc s t, f z < f (x u))
    (hloj : ∀ u ∈ Ico s t, m * (f (x u) - f z) ^ a ≤ ‖gradient f (x u)‖) :
    m * (1 - a) * ‖x t - x s‖ ≤ (f (x s) - f z) ^ (1 - a) - (f (x t) - f z) ^ (1 - a) := by
  have hκ : 0 < m * (1 - a) := mul_pos hm (by linarith)
  set φ : ℝ → ℝ := fun u => (f (x u) - f z) ^ (1 - a) with hφ
  have hφd : ∀ u ∈ Icc s t, HasDerivAt φ
      (-‖gradient f (x u)‖ ^ 2 * (1 - a) * (f (x u) - f z) ^ (1 - a - 1)) u := fun u hu =>
    ((hasDerivAt_comp_gradientFlow (hf _) (hx u hu)).sub_const (f z)).rpow_const
      (Or.inl (sub_pos.2 (hgap u hu)).ne')
  have hfence := image_norm_le_of_norm_deriv_right_le_deriv_boundary'
    (f := fun u => x u - x s) (f' := fun u => -gradient f (x u)) (a := s) (b := t)
    (B := fun u => (φ s - φ u) / (m * (1 - a)))
    (B' := fun u => ‖gradient f (x u)‖ ^ 2 * (f (x u) - f z) ^ (1 - a - 1) / m)
    (fun u hu => ((hx u hu).sub_const (x s)).continuousAt.continuousWithinAt)
    (fun u hu => ((hx u (Ico_subset_Icc_self hu)).sub_const (x s)).hasDerivWithinAt)
    (by simp) (fun u hu =>
      ((continuousAt_const.sub (hφd u hu).continuousAt).div_const _).continuousWithinAt)
    (fun u hu => ?_) (fun u hu => ?_) (right_mem_Icc.2 hst)
  · rw [le_div_iff₀ hκ, mul_comm] at hfence
    simpa [hφ] using hfence
  · convert (((hφd u (Ico_subset_Icc_self hu)).const_sub (φ s)).div_const
      (m * (1 - a))).hasDerivWithinAt using 1
    have h1a : 1 - a ≠ 0 := (sub_pos.2 ha1).ne'
    field_simp
  · have hD := sub_pos.2 (hgap u (Ico_subset_Icc_self hu))
    have hDa : (f (x u) - f z) ^ (1 - a - 1) = ((f (x u) - f z) ^ a)⁻¹ := by
      rw [show 1 - a - 1 = -a by ring, Real.rpow_neg hD.le]
    have hpos : 0 < (f (x u) - f z) ^ a * m := mul_pos (Real.rpow_pos_of_pos hD a) hm
    have hG := hloj u hu
    rw [norm_neg, hDa, ← div_eq_mul_inv, div_div, le_div_iff₀ hpos]
    nlinarith [norm_nonneg (gradient f (x u))]

/-- **A gradient flow is trapped near a point where the Łojasiewicz
inequality holds.**  If the inequality holds on `B(z, r)` below `f z + ε`,
the trajectory stays above `f z` from time `t₀` on, starts below `f z + ε`,
and `m (1 - a) ‖x t₀ - z‖ + (f (x t₀) - f z) ^ (1 - a) < m (1 - a) r`, then it
never leaves `B(z, r)`. -/
theorem mem_ball_of_gradientFlow (hf : Differentiable ℝ f) {t₀ : ℝ}
    (hx : ∀ t, t₀ ≤ t → HasDerivAt x (-gradient f (x t)) t) (hm : 0 < m) (ha1 : a < 1)
    {z : E} {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hgap : ∀ t, t₀ ≤ t → f z < f (x t)) (hε : f (x t₀) < f z + ε)
    (hstart : m * (1 - a) * ‖x t₀ - z‖ + (f (x t₀) - f z) ^ (1 - a) < m * (1 - a) * r)
    {t : ℝ} (ht : t₀ ≤ t) : x t ∈ ball z r := by
  by_contra hout
  set S := Ici t₀ ∩ x ⁻¹' (ball z r)ᶜ with hS
  have hSc : IsClosed S := ContinuousOn.preimage_isClosed_of_isClosed
    (fun u hu => (hx u hu).continuousAt.continuousWithinAt) isClosed_Ici
    isOpen_ball.isClosed_compl
  have hSb : BddBelow S := ⟨t₀, fun u hu => hu.1⟩
  have hT := hSc.csInf_mem ⟨t, ht, hout⟩ hSb
  have hin : ∀ u ∈ Ico t₀ (sInf S), x u ∈ ball z r := fun u hu => by
    by_contra h
    exact (csInf_le hSb ⟨hu.1, h⟩).not_gt hu.2
  have hanti := antitoneOn_comp_gradientFlow hf hx
  have hlen := mul_norm_sub_le_of_gradientFlow hf (t := sInf S) hT.1
    (fun u hu => hx u hu.1) hm ha1 (fun u hu => hgap u hu.1) fun u hu => by
      have hl := hloj _ (hin u hu) (hgap u hu.1)
        ((hanti self_mem_Ici hu.1 hu.1).trans_lt hε)
      rwa [abs_of_pos (sub_pos.2 (hgap u hu.1))] at hl
  have hφ := Real.rpow_nonneg (sub_pos.2 (hgap _ hT.1)).le (1 - a)
  have htri := mul_le_mul_of_nonneg_left (norm_sub_le_norm_sub_add_norm_sub (x (sInf S)) (x t₀) z)
    (mul_pos hm (sub_pos.2 ha1)).le
  refine hT.2 (mem_ball_iff_norm.2 (lt_of_mul_lt_mul_left ?_ (mul_pos hm (sub_pos.2 ha1)).le))
  linarith

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line and its gradient flow `x t = e^{-t}`, with `a = 1/2`, `m = 1`, `z = 0`,
`r = 4`, `ε = 1` and the times `0 ≤ 1`: then `m (1 - a) ‖x 0‖ = 1/2` and
`(f (x 0)) ^ (1/2) ≤ 1` add up to less than `m (1 - a) r = 2`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ (∀ t, HasDerivAt x (-gradient f (x t)) t) ∧ (0 : ℝ) ≤ 1 ∧
      (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) < 1 ∧ (∀ t, f 0 < f (x t)) ∧
      (∀ t, 1 * (f (x t) - f 0) ^ (1 / 2 : ℝ) ≤ ‖gradient f (x t)‖) ∧
      (∀ y ∈ ball (0 : ℝ) 4, f 0 < f y → f y < f 0 + 1 →
        1 * |f y - f 0| ^ (1 / 2 : ℝ) ≤ ‖gradient f y‖) ∧ f (x 0) < f 0 + 1 ∧
      1 * (1 - 1 / 2) * ‖x 0 - 0‖ + (f (x 0) - f 0) ^ (1 - 1 / 2 : ℝ) <
        1 * (1 - 1 / 2) * 4 := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hf0 : ∀ y, f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := fun y => by
    simp only [f, norm_zero]
    ring
  have hx0 : x 0 = 1 := by simp [x]
  have hroot : (f (x 0) - f 0) ^ (1 / 2 : ℝ) ≤ ‖x 0‖ := by
    rw [hf0]
    exact half_mul_norm_sq_rpow_le _
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, fun t => ?_,
    zero_le_one, one_pos, by norm_num, fun t => ?_, fun t => ?_, fun y _ _ _ => ?_, ?_, ?_⟩
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

end Transformer.Landscape.SaddleAvoidance
