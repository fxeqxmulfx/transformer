/-
# Saddle avoidance — gradient descent is trapped near a Łojasiewicz point

Steps bounded by the drops of a sequence `φ` add up
(`mul_norm_sub_le_of_step`); a sequence that starts close to `z`, and whose
steps from inside `B(z, r)` are so bounded, never leaves `B(z, r)`
(`mem_ball_of_step`).  By `mul_norm_gradientStep_sub_le` the steps of
gradient descent are so bounded near a point `z` where the Łojasiewicz
inequality holds (`mul_norm_iterate_succ_sub_le`).  So an orbit that starts
close to `z`, with `f` close to `f z` and never below it, stays near `z`
(`iterate_mem_ball`) and has finite length (`mul_norm_iterate_sub_le`), as
in the proof of Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim. 16,
2005).
-/

import Transformer.Landscape.SaddleAvoidance.Lojasiewicz

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

section Sequence

variable {E : Type*} [NormedAddCommGroup E]

/-- **Steps bounded by drops add up.**  If `κ ‖u (k + 1) - u k‖ ≤
φ k - φ (k + 1)` for every `k`, then `κ ‖u k - u n‖ ≤ φ n - φ k` for
`n ≤ k`. -/
theorem mul_norm_sub_le_of_step {u : ℕ → E} {φ : ℕ → ℝ} {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ k, κ * ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1)) {n k : ℕ} (hnk : n ≤ k) :
    κ * ‖u k - u n‖ ≤ φ n - φ k := by
  induction k, hnk using Nat.le_induction with
  | base => simp
  | succ k _ ih =>
    have htri := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u k) (u n)) hκ
    linarith [h k]

/-- **A trap.**  If every step of `u` from inside `B(z, r)` is bounded by the
drop of a nonnegative sequence `φ`, `κ ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1)`
with `κ > 0`, and `κ ‖u 0 - z‖ + φ 0 < κ r`, then `u` never leaves
`B(z, r)`. -/
theorem mem_ball_of_step {u : ℕ → E} {φ : ℕ → ℝ} {κ r : ℝ} {z : E} (hκ : 0 < κ)
    (hφ : ∀ k, 0 ≤ φ k) (h : ∀ k, u k ∈ ball z r → κ * ‖u (k + 1) - u k‖ ≤ φ k - φ (k + 1))
    (h0 : κ * ‖u 0 - z‖ + φ 0 < κ * r) (k : ℕ) : u k ∈ ball z r := by
  suffices H : ∀ k, u k ∈ ball z r ∧ κ * ‖u k - u 0‖ ≤ φ 0 - φ k from (H k).1
  intro k
  induction k with
  | zero =>
    refine ⟨?_, by simp⟩
    rw [mem_ball, dist_eq_norm]
    exact lt_of_mul_lt_mul_left (by linarith [hφ 0]) hκ.le
  | succ k ih =>
    have htri := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u k) (u 0)) hκ.le
    have hlen : κ * ‖u (k + 1) - u 0‖ ≤ φ 0 - φ (k + 1) := by linarith [h k ih.1, ih.2]
    have hz := mul_le_mul_of_nonneg_left
      (norm_sub_le_norm_sub_add_norm_sub (u (k + 1)) (u 0) z) hκ.le
    refine ⟨?_, hlen⟩
    rw [mem_ball, dist_eq_norm]
    exact lt_of_mul_lt_mul_left (by linarith [hφ (k + 1)]) hκ.le

end Sequence

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η a m : ℝ}

/-- The step estimate along an orbit: from a point of `B(z, r)`, where the
gradient inequality holds below `f z + ε`, the step is bounded by the drop of
`(f - f z) ^ (1 - a)`, if the orbit starts below `f z + ε` and never goes
below `f z`. -/
theorem mul_norm_iterate_succ_sub_le (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z x : E} {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε) {k : ℕ}
    (hk : (gradientStep f η)^[k] x ∈ ball z r) :
    m * (1 - a) * (1 - η * L) * ‖(gradientStep f η)^[k + 1] x - (gradientStep f η)^[k] x‖ ≤
      (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a) -
        (f ((gradientStep f η)^[k + 1] x) - f z) ^ (1 - a) := by
  have hz := hfz (k + 1)
  rw [Function.iterate_succ_apply'] at hz ⊢
  exact mul_norm_gradientStep_sub_le hf hL hη hηL ha0 ha1
    (fun h => hloj _ hk h ((antitone_iterate hf hL hη.le hηL.le x (Nat.zero_le k)).trans_lt hε))
    hz

/-- **Gradient descent is trapped near a point where the Łojasiewicz
inequality holds.**  If the inequality holds on `B(z, r)` below `f z + ε`,
the orbit of `x` never goes below `f z`, `f x < f z + ε`, and
`κ ‖x - z‖ + (f x - f z) ^ (1 - a) < κ r` for `κ = m (1 - a)(1 - η L)`, then
the orbit never leaves `B(z, r)`. -/
theorem iterate_mem_ball (hf : Differentiable ℝ f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a) (ha1 : a < 1) (hm : 0 < m) {z x : E}
    {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε)
    (hstart : m * (1 - a) * (1 - η * L) * ‖x - z‖ + (f x - f z) ^ (1 - a) <
      m * (1 - a) * (1 - η * L) * r) (k : ℕ) :
    (gradientStep f η)^[k] x ∈ ball z r :=
  mem_ball_of_step (u := fun k => (gradientStep f η)^[k] x)
    (φ := fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a))
    (mul_pos (mul_pos hm (by linarith)) (by linarith))
    (fun k => Real.rpow_nonneg (sub_nonneg.2 (hfz k)) _)
    (fun _ hk => mul_norm_iterate_succ_sub_le hf hL hη hηL ha0 ha1 hloj hfz hε hk) hstart k

/-- **A trapped orbit has finite length**:
`κ ‖x_k - x_n‖ ≤ (f x_n - f z) ^ (1 - a) - (f x_k - f z) ^ (1 - a)` for
`n ≤ k`, under the hypotheses of `iterate_mem_ball`. -/
theorem mul_norm_iterate_sub_le (hf : Differentiable ℝ f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a) (ha1 : a < 1) (hm : 0 < m) {z x : E}
    {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε)
    (hstart : m * (1 - a) * (1 - η * L) * ‖x - z‖ + (f x - f z) ^ (1 - a) <
      m * (1 - a) * (1 - η * L) * r) {n k : ℕ} (hnk : n ≤ k) :
    m * (1 - a) * (1 - η * L) * ‖(gradientStep f η)^[k] x - (gradientStep f η)^[n] x‖ ≤
      (f ((gradientStep f η)^[n] x) - f z) ^ (1 - a) -
        (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a) :=
  mul_norm_sub_le_of_step (u := fun k => (gradientStep f η)^[k] x)
    (φ := fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a))
    (mul_nonneg (mul_nonneg hm.le (by linarith)) (by linarith))
    (fun k => mul_norm_iterate_succ_sub_le hf hL hη hηL ha0 ha1 hloj hfz hε
      (iterate_mem_ball hf hL hη hηL ha0 ha1 hm hloj hfz hε hstart k)) hnk

/-- The hypotheses of the theorems above hold for the constant sequence `0`
with `φ = 0`, `κ = 1` and `r = 1`, and for `f(x) = ‖x‖² / 2` on the line with
`η = 1/2`, `L = 1`, `a = 1/2`, `m = 1`, `z = 0`, `r = 2`, `ε = 1` and the
start `x = 1/4`: then `κ = 1/4`, and `κ ‖x‖ + (f x) ^ (1/2) ≤ 1/16 + 1/4`
is less than `κ r = 1/2`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    ((0 : ℝ) < 1 ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 0) ∧
      (∀ _ : ℕ, (0 : ℝ) ∈ ball (0 : ℝ) 1 → 1 * ‖(0 : ℝ) - 0‖ ≤ 0 - 0) ∧
      1 * ‖(0 : ℝ) - 0‖ + 0 < 1 * 1) ∧
    Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ y ∈ ball (0 : ℝ) 2, f 0 < f y → f y < f 0 + 1 →
        1 * |f y - f 0| ^ (1 / 2 : ℝ) ≤ ‖gradient f y‖) ∧
      (∀ k, f 0 ≤ f ((gradientStep f (1 / 2))^[k] (1 / 4))) ∧ f (1 / 4) < f 0 + 1 ∧
      1 * (1 - 1 / 2) * (1 - 1 / 2 * ((1 : ℝ≥0) : ℝ)) * ‖(1 / 4 : ℝ) - 0‖ +
          (f (1 / 4) - f 0) ^ (1 - 1 / 2 : ℝ) <
        1 * (1 - 1 / 2) * (1 - 1 / 2 * ((1 : ℝ≥0) : ℝ)) * 2 := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hf0 : ∀ y, f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := fun y => by
    simp only [f, norm_zero]
    ring
  have hroot : (f (1 / 4) - f 0) ^ (1 / 2 : ℝ) ≤ ‖(1 / 4 : ℝ)‖ := by
    rw [hf0]
    exact half_mul_norm_sq_rpow_le _
  refine ⟨⟨one_pos, fun _ => le_rfl, fun _ _ => by simp, by norm_num⟩,
    (contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, ?_, by norm_num,
    by norm_num, by norm_num, by norm_num, one_pos, fun y _ _ _ => ?_, fun k => ?_, ?_, ?_⟩
  · rw [hg]
    exact LipschitzWith.id
  · rw [hf0, abs_of_nonneg (by positivity), one_mul, hg, id]
    exact half_mul_norm_sq_rpow_le y
  · nlinarith [hf0 ((gradientStep f (1 / 2))^[k] (1 / 4)),
      sq_nonneg ‖(gradientStep f (1 / 2))^[k] (1 / 4)‖]
  · rw [← sub_lt_iff_lt_add', hf0]
    norm_num
  · norm_num at hroot ⊢
    linarith
end Transformer.Landscape.SaddleAvoidance
