/-
# Saddle avoidance — the Łojasiewicz inequality bounds gradient steps

`f` satisfies the Łojasiewicz gradient inequality at `z` with exponent `a`
when `m |f x - f z| ^ a ≤ ‖∇f(x)‖` for all `x` near `z` with
`f z < f x < f z + ε` (`IsLojasiewiczAt`); `‖x‖² / 2` satisfies it at `0`
with `a = 1/2` (`isLojasiewiczAt_half_mul_norm_sq`), every `f` satisfies
it where its gradient is continuous and does not vanish
(`isLojasiewiczAt_of_gradient_ne_zero`), and it holds with every exponent
larger than one it holds with (`IsLojasiewiczAt.mono`).  With `0 ≤ a < 1` it
bounds a gradient step by the drop of `(f - f z) ^ (1 - a)`
(`mul_norm_gradientStep_sub_le`): sufficient decrease and the concavity of
`s ↦ s ^ (1 - a)`, a weighted AM-GM inequality (`mul_sub_le_rpow_mul_sub`),
give `m (1 - a)(1 - η L) η ‖∇f‖² ≤ ‖∇f‖ · drop`.

This is the step estimate of the proof of Theorem 3.2 of Absil, Mahony and
Andrews, *Convergence of the iterates of descent methods for analytic cost
functions* (SIAM J. Optim. 16, 2005), which arXiv:1602.04915 cites for
`prop:rates`.
-/

import Transformer.Landscape.SaddleAvoidance.Descent
import Mathlib.Analysis.MeanInequalities

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance


variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- **The Łojasiewicz gradient inequality** at `z` with exponent `a`: for
some `m, ε > 0`, `m |f x - f z| ^ a ≤ ‖∇f(x)‖` for all `x` near `z` with
`f z < f x < f z + ε`.  This is `def:gradineq` of arXiv:1602.04915, which
also asks that `z` be a critical point and `0 ≤ a < 1`; here these are
hypotheses of the theorems that use it. -/
def IsLojasiewiczAt (f : E → ℝ) (z : E) (a : ℝ) : Prop :=
  ∃ m > 0, ∃ ε > 0, ∀ᶠ x in 𝓝 z,
    f z < f x → f x < f z + ε → m * |f x - f z| ^ a ≤ ‖gradient f x‖

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
/-- `(‖x‖² / 2) ^ (1/2) ≤ ‖x‖`. -/
theorem half_mul_norm_sq_rpow_le (x : E) : ((1 / 2 : ℝ) * ‖x‖ ^ 2) ^ (1 / 2 : ℝ) ≤ ‖x‖ := by
  calc ((1 / 2 : ℝ) * ‖x‖ ^ 2) ^ (1 / 2 : ℝ) ≤ (‖x‖ ^ 2) ^ (1 / 2 : ℝ) :=
        Real.rpow_le_rpow (by positivity) (by nlinarith [sq_nonneg ‖x‖]) (by norm_num)
    _ = ‖x‖ := by rw [← Real.sqrt_eq_rpow, Real.sqrt_sq (norm_nonneg x)]

/-- `‖x‖² / 2` satisfies the Łojasiewicz inequality at its minimum `0` with
exponent `1/2` and `m = 1`. -/
theorem isLojasiewiczAt_half_mul_norm_sq :
    IsLojasiewiczAt (fun x : E => (1 / 2 : ℝ) * ‖x‖ ^ 2) 0 (1 / 2) := by
  refine ⟨1, one_pos, 1, one_pos, Eventually.of_forall fun y _ _ => ?_⟩
  have h0 : (1 / 2 : ℝ) * ‖y‖ ^ 2 - 1 / 2 * ‖(0 : E)‖ ^ 2 = 1 / 2 * ‖y‖ ^ 2 := by
    rw [norm_zero]
    ring
  rw [h0, abs_of_nonneg (by positivity), one_mul, gradient_half_mul_norm_sq, id]
  exact half_mul_norm_sq_rpow_le y

/-- Concavity of `s ↦ s ^ (1 - a)` as the Łojasiewicz argument uses it:
`(1 - a)(s - t) ≤ s ^ a (s ^ (1 - a) - t ^ (1 - a))`, the weighted AM-GM
inequality `s ^ a t ^ (1 - a) ≤ a s + (1 - a) t`. -/
theorem mul_sub_le_rpow_mul_sub {a s t : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hs : 0 ≤ s)
    (ht : 0 ≤ t) : (1 - a) * (s - t) ≤ s ^ a * (s ^ (1 - a) - t ^ (1 - a)) := by
  have hamgm := Real.geom_mean_le_arith_mean2_weighted ha0 (sub_nonneg.2 ha1) hs ht
    (by ring)
  have hss : s ^ a * s ^ (1 - a) = s := by
    rw [← Real.rpow_add' hs (by norm_num), add_sub_cancel, Real.rpow_one]
  have hlin : (1 - a) * (s - t) = s - (a * s + (1 - a) * t) := by ring
  rw [hlin, mul_sub, hss]
  linarith

variable {f : E → ℝ} {L : ℝ≥0} {η a m : ℝ}

/-- **Away from critical points the inequality is free**: where the gradient
is continuous and does not vanish, it holds with every exponent `0 ≤ a`, with
`m = ‖∇f(z)‖ / 2` and `ε = 1`. -/
theorem isLojasiewiczAt_of_gradient_ne_zero {z : E} (hc : ContinuousAt (gradient f) z)
    (hz : gradient f z ≠ 0) (ha : 0 ≤ a) : IsLojasiewiczAt f z a := by
  have hpos : 0 < ‖gradient f z‖ := norm_pos_iff.2 hz
  refine ⟨‖gradient f z‖ / 2, half_pos hpos, 1, one_pos, ?_⟩
  filter_upwards [hc.norm.tendsto.eventually (lt_mem_nhds (half_lt_self hpos))] with y hy h1 h2
  have h01 : |f y - f z| ^ a ≤ 1 :=
    Real.rpow_le_one (abs_nonneg _) (abs_le.2 ⟨by linarith, by linarith⟩) ha
  have := mul_le_mul_of_nonneg_left h01 (half_pos hpos).le
  linarith

/-- **A larger exponent.**  The Łojasiewicz inequality with exponent `a`
gives it with every exponent `b ≥ a`: below `f z + 1`,
`|f x - f z| ^ b ≤ |f x - f z| ^ a`. -/
theorem IsLojasiewiczAt.mono {z : E} {b : ℝ} (h : IsLojasiewiczAt f z a) (hab : a ≤ b) :
    IsLojasiewiczAt f z b := by
  obtain ⟨m, hm, ε, hε, hev⟩ := h
  refine ⟨m, hm, min ε 1, lt_min hε one_pos, hev.mono fun y hy h1 h2 => ?_⟩
  have hpos : 0 < f y - f z := sub_pos.2 h1
  have hle : |f y - f z| ≤ 1 := (abs_of_pos hpos).trans_le (by linarith [min_le_right ε 1])
  calc m * |f y - f z| ^ b ≤ m * |f y - f z| ^ a :=
        mul_le_mul_of_nonneg_left
          (Real.rpow_le_rpow_of_exponent_ge (abs_pos.2 hpos.ne') hle hab) hm.le
    _ ≤ ‖gradient f y‖ := hy h1 (h2.trans_le (by linarith [min_le_left ε 1]))

/-- **A gradient step is bounded by the drop of `(f - f z) ^ (1 - a)`.**  If
`f z` stays below `f` after the step from `x`, and the gradient inequality
holds at `x` when `f z < f x`, then
`m (1 - a)(1 - η L) ‖x' - x‖ ≤ (f x - f z) ^ (1 - a) - (f x' - f z) ^ (1 - a)`
for the step `x'`. -/
theorem mul_norm_gradientStep_sub_le (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z x : E} (hloj : f z < f x → m * |f x - f z| ^ a ≤ ‖gradient f x‖)
    (hz : f z ≤ f (gradientStep f η x)) :
    m * (1 - a) * (1 - η * L) * ‖gradientStep f η x - x‖ ≤
      (f x - f z) ^ (1 - a) - (f (gradientStep f η x) - f z) ^ (1 - a) := by
  have hdec := le_sub_of_gradientStep hf hL η x
  have hstep : ‖gradientStep f η x - x‖ = η * ‖gradient f x‖ := by
    simp [gradientStep, norm_smul, abs_of_pos hη]
  set G := ‖gradient f x‖ with hG
  set s := f x - f z with hs
  set t := f (gradientStep f η x) - f z with htdef
  have hc : 0 < 1 - η * L := by linarith
  have ht : 0 ≤ t := sub_nonneg.2 hz
  have hst : η * (1 - η * L) * G ^ 2 ≤ s - t := by linarith
  have hts : t ≤ s := by nlinarith [mul_nonneg (mul_nonneg hη.le hc.le) (sq_nonneg G)]
  have hdrop : 0 ≤ s ^ (1 - a) - t ^ (1 - a) :=
    sub_nonneg.2 (Real.rpow_le_rpow ht hts (by linarith))
  rw [hstep]
  rcases le_or_gt m 0 with hm | hm
  · have hP : 0 ≤ (1 - a) * (1 - η * L) * (η * G) :=
      mul_nonneg (mul_nonneg (by linarith) hc.le) (mul_nonneg hη.le (norm_nonneg _))
    have := mul_nonpos_of_nonpos_of_nonneg hm hP
    linarith
  rcases (show (0 : ℝ) ≤ G from norm_nonneg _).eq_or_lt with hG0 | hGpos
  · rw [← hG0]
    simpa using hdrop
  have hspos : 0 < s := by nlinarith [mul_pos (mul_pos hη hc) (pow_pos hGpos 2)]
  have hl := hloj (by linarith)
  rw [abs_of_pos hspos] at hl
  have hamgm := mul_sub_le_rpow_mul_sub ha0 ha1.le hspos.le ht
  have key : G * (m * (1 - a) * (1 - η * L) * (η * G)) ≤ G * (s ^ (1 - a) - t ^ (1 - a)) := by
    calc G * (m * (1 - a) * (1 - η * L) * (η * G))
        = m * ((1 - a) * (η * (1 - η * L) * G ^ 2)) := by ring
      _ ≤ m * ((1 - a) * (s - t)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hst (by linarith)) hm.le
      _ ≤ m * (s ^ a * (s ^ (1 - a) - t ^ (1 - a))) := mul_le_mul_of_nonneg_left hamgm hm.le
      _ = m * s ^ a * (s ^ (1 - a) - t ^ (1 - a)) := by ring
      _ ≤ G * (s ^ (1 - a) - t ^ (1 - a)) := mul_le_mul_of_nonneg_right hl hdrop
  exact le_of_mul_le_mul_left key hGpos

/-- The hypotheses of the theorems above hold for `a = 1/2`, `s = 1` and
`t = 0`, and for `f(x) = ‖x‖² / 2` on the line with `η = 1/2`, `L = 1`,
`a = 1/2`, `m = 1`, `z = 0` and `x = 1/4`, where the inequality holds with
`a = 1/2 ≤ 1`, and at `z = 1`, where the gradient is `1`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    ((0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0) ∧
    Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
      (f 0 < f (1 / 4) → 1 * |f (1 / 4) - f 0| ^ (1 / 2 : ℝ) ≤ ‖gradient f (1 / 4)‖) ∧
      f 0 ≤ f (gradientStep f (1 / 2) (1 / 4)) ∧ IsLojasiewiczAt f 0 (1 / 2) ∧
      ContinuousAt (gradient f) 1 ∧ gradient f 1 ≠ 0 := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hf0 : ∀ y, f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := fun y => by
    simp only [f, norm_zero]
    ring
  refine ⟨by norm_num, (contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, ?_,
    by norm_num, by norm_num, by norm_num, by norm_num, fun _ => ?_, ?_,
    isLojasiewiczAt_half_mul_norm_sq, ?_, ?_⟩
  · rw [hg]
    exact LipschitzWith.id
  · rw [hf0, abs_of_nonneg (by positivity), one_mul, hg, id]
    exact half_mul_norm_sq_rpow_le _
  · nlinarith [hf0 (gradientStep f (1 / 2) (1 / 4)), sq_nonneg ‖gradientStep f (1 / 2) (1 / 4)‖]
  · rw [hg]
    exact continuousAt_id
  · rw [hg]
    exact one_ne_zero

end Transformer.Landscape.SaddleAvoidance
