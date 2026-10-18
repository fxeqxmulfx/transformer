/-
# Saddle avoidance — gradient descent is trapped near a Łojasiewicz point

Along an orbit whose steps lower `f` by at least `η κ ‖∇f‖²`, `κ > 0`, a step from
near a point `z` where the Łojasiewicz inequality holds is bounded by the drop of
`(f - f z) ^ (1 - a)` (`mul_norm_iterate_succ_sub_le_of_le_sub`, by
`mul_norm_gradientStep_sub_le_of_le_sub`).  Steps so bounded add up
(`mul_norm_sub_le_of_step`) and keep a sequence that starts close to `z` near `z`
(`mem_ball_of_step`).  So such an orbit that starts close to `z`, with `f` close to
`f z` and never below it, stays near `z` (`iterate_mem_ball_of_le_sub`) and has finite
length (`mul_norm_iterate_sub_le_of_le_sub`), as in the proof of Theorem 3.2 of Absil,
Mahony and Andrews (SIAM J. Optim. 16, 2005).  An `L`-Lipschitz gradient with
`η L < 1` gives the decrease with `κ = 1 - η L` (`mul_norm_iterate_succ_sub_le`,
`iterate_mem_ball`, `mul_norm_iterate_sub_le`).
-/

import Transformer.Landscape.SaddleAvoidance.Lojasiewicz

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η κ a m : ℝ}

/-- The step estimate along an orbit whose steps lower `f` by at least
`η κ ‖∇f‖²`, `κ > 0`: from a point of `B(z, r)`, where the gradient inequality holds
below `f z + ε`, the step is bounded by the drop of `(f - f z) ^ (1 - a)`, if the
orbit starts below `f z + ε` and never goes below `f z`. -/
theorem mul_norm_iterate_succ_sub_le_of_le_sub (hη : 0 < η) (hκ : 0 < κ) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z x : E} {r ε : ℝ}
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε) {k : ℕ}
    (hk : (gradientStep f η)^[k] x ∈ ball z r) :
    m * (1 - a) * κ * ‖(gradientStep f η)^[k + 1] x - (gradientStep f η)^[k] x‖ ≤
      (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a) -
        (f ((gradientStep f η)^[k + 1] x) - f z) ^ (1 - a) := by
  have hz := hfz (k + 1)
  have hd := hdec k
  rw [Function.iterate_succ_apply'] at hz hd ⊢
  exact mul_norm_gradientStep_sub_le_of_le_sub hη hκ ha0 ha1 hd (fun h => hloj _ hk h
    ((antitone_iterate_of_le_sub hη.le hκ.le hdec (Nat.zero_le k)).trans_lt hε)) hz

/-- The step estimate along an orbit for an `L`-Lipschitz gradient and `η L < 1`:
`mul_norm_iterate_succ_sub_le_of_le_sub` with `κ = 1 - η L`. -/
theorem mul_norm_iterate_succ_sub_le (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z x : E} {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε) {k : ℕ}
    (hk : (gradientStep f η)^[k] x ∈ ball z r) :
    m * (1 - a) * (1 - η * L) * ‖(gradientStep f η)^[k + 1] x - (gradientStep f η)^[k] x‖ ≤
      (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a) -
        (f ((gradientStep f η)^[k + 1] x) - f z) ^ (1 - a) :=
  mul_norm_iterate_succ_sub_le_of_le_sub hη (by linarith) ha0 ha1
    (iterate_succ_le_sub hf hL η x) hloj hfz hε hk

/-- **Gradient descent is trapped near a point where the Łojasiewicz inequality
holds**, along an orbit whose steps lower `f` by at least `η κ ‖∇f‖²`, `κ > 0`: if
the inequality holds on `B(z, r)` below `f z + ε`, the orbit of `x` never goes below
`f z`, `f x < f z + ε`, and `c ‖x - z‖ + (f x - f z) ^ (1 - a) < c r` for
`c = m (1 - a) κ`, then the orbit never leaves `B(z, r)`. -/
theorem iterate_mem_ball_of_le_sub (hη : 0 < η) (hκ : 0 < κ) (ha0 : 0 ≤ a) (ha1 : a < 1)
    (hm : 0 < m) {z x : E} {r ε : ℝ}
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε)
    (hstart : m * (1 - a) * κ * ‖x - z‖ + (f x - f z) ^ (1 - a) < m * (1 - a) * κ * r)
    (k : ℕ) : (gradientStep f η)^[k] x ∈ ball z r :=
  mem_ball_of_step (u := fun k => (gradientStep f η)^[k] x)
    (φ := fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a))
    (mul_pos (mul_pos hm (by linarith)) hκ)
    (fun k => Real.rpow_nonneg (sub_nonneg.2 (hfz k)) _)
    (fun _ hk => mul_norm_iterate_succ_sub_le_of_le_sub hη hκ ha0 ha1 hdec hloj hfz hε hk)
    hstart k

/-- **Gradient descent is trapped near a point where the Łojasiewicz inequality
holds**, for an `L`-Lipschitz gradient and `η L < 1`: `iterate_mem_ball_of_le_sub`
with `κ = 1 - η L`. -/
theorem iterate_mem_ball (hf : Differentiable ℝ f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a) (ha1 : a < 1) (hm : 0 < m) {z x : E}
    {r ε : ℝ}
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε)
    (hstart : m * (1 - a) * (1 - η * L) * ‖x - z‖ + (f x - f z) ^ (1 - a) <
      m * (1 - a) * (1 - η * L) * r) (k : ℕ) :
    (gradientStep f η)^[k] x ∈ ball z r :=
  iterate_mem_ball_of_le_sub hη (by linarith) ha0 ha1 hm (iterate_succ_le_sub hf hL η x) hloj
    hfz hε hstart k

/-- **A trapped orbit has finite length**:
`c ‖x_k - x_n‖ ≤ (f x_n - f z) ^ (1 - a) - (f x_k - f z) ^ (1 - a)` for `n ≤ k` and
`c = m (1 - a) κ`, under the hypotheses of `iterate_mem_ball_of_le_sub`. -/
theorem mul_norm_iterate_sub_le_of_le_sub (hη : 0 < η) (hκ : 0 < κ) (ha0 : 0 ≤ a)
    (ha1 : a < 1) (hm : 0 < m) {z x : E} {r ε : ℝ}
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hloj : ∀ y ∈ ball z r, f z < f y → f y < f z + ε → m * |f y - f z| ^ a ≤ ‖gradient f y‖)
    (hfz : ∀ k, f z ≤ f ((gradientStep f η)^[k] x)) (hε : f x < f z + ε)
    (hstart : m * (1 - a) * κ * ‖x - z‖ + (f x - f z) ^ (1 - a) < m * (1 - a) * κ * r)
    {n k : ℕ} (hnk : n ≤ k) :
    m * (1 - a) * κ * ‖(gradientStep f η)^[k] x - (gradientStep f η)^[n] x‖ ≤
      (f ((gradientStep f η)^[n] x) - f z) ^ (1 - a) -
        (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a) :=
  mul_norm_sub_le_of_step (u := fun k => (gradientStep f η)^[k] x)
    (φ := fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a))
    (mul_nonneg (mul_nonneg hm.le (by linarith)) hκ.le)
    (fun k => mul_norm_iterate_succ_sub_le_of_le_sub hη hκ ha0 ha1 hdec hloj hfz hε
      (iterate_mem_ball_of_le_sub hη hκ ha0 ha1 hm hdec hloj hfz hε hstart k)) hnk

/-- **A trapped orbit has finite length**, for an `L`-Lipschitz gradient and
`η L < 1`: `mul_norm_iterate_sub_le_of_le_sub` with `κ = 1 - η L`. -/
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
  mul_norm_iterate_sub_le_of_le_sub hη (by linarith) ha0 ha1 hm (iterate_succ_le_sub hf hL η x)
    hloj hfz hε hstart hnk

/-- The hypotheses of the theorems for an `L`-Lipschitz gradient hold for
`f(x) = ‖x‖² / 2` on the line with `η = 1/2`, `L = 1`, `a = 1/2`, `m = 1`, `z = 0`,
`r = 2`, `ε = 1` and the start `x = 1/4`: then `c = m (1 - a)(1 - η L) = 1/4`, and
`c ‖x‖ + (f x) ^ (1/2) ≤ 1/16 + 1/4` is less than `c r = 1/2`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
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
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, ?_, by norm_num,
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

/-- With `η = 3/2`, beyond `η L < 1` for every Lipschitz constant `L ≥ 1` of the
gradient, the orbit of `x = 1/4` for `‖x‖² / 2` lowers `f` by `η κ ‖∇f‖²` at every
step, `κ = 1 - η / 2 = 1/4`; with `a = 1/2`, `m = 1`, `z = 0`, `r = 3` and `ε = 1`,
`c = m (1 - a) κ = 1/8` and `c ‖x‖ + (f x) ^ (1/2) ≤ 1/32 + 1/4 < c r = 3/8`. -/
example := mul_norm_iterate_succ_sub_le_of_le_sub (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (a := 1 / 2) (m := 1) (z := 0) (x := 1 / 4) (r := 3)
  (ε := 1) (k := 0) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (fun y _ _ _ => abs_sub_rpow_le_norm_gradient_half_mul_norm_sq y)
  (fun _ => by simp; positivity) (by norm_num) (by norm_num)

example := iterate_mem_ball_of_le_sub (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (η := 3 / 2)
  (κ := 1 - 3 / 2 / 2) (a := 1 / 2) (m := 1) (z := 0) (x := 1 / 4) (r := 3) (ε := 1)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (fun y _ _ _ => abs_sub_rpow_le_norm_gradient_half_mul_norm_sq y)
  (fun _ => by simp; positivity) (by norm_num)
  (by have := half_mul_norm_sq_rpow_le (1 / 4 : ℝ); norm_num at this ⊢; linarith) 5

example := mul_norm_iterate_sub_le_of_le_sub (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (a := 1 / 2) (m := 1) (z := 0) (x := 1 / 4) (r := 3)
  (ε := 1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (fun y _ _ _ => abs_sub_rpow_le_norm_gradient_half_mul_norm_sq y)
  (fun _ => by simp; positivity) (by norm_num)
  (by have := half_mul_norm_sq_rpow_le (1 / 4 : ℝ); norm_num at this ⊢; linarith)
  (Nat.zero_le 5)

end Transformer.Landscape.SaddleAvoidance
