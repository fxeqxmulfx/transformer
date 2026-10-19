module

public import Transformer.ScheduleFree.SectionA_Conversion
public import Transformer.ScheduleFree.SectionD_Smooth

/-!
# The negative gradient-variation term

arXiv:2405.15682v4, Appendix D, Corollary 1. We prove the local estimate
that makes the Bregman remainders telescope. Multiplying the interpolation
condition by its positive denominator avoids every division by `α`.
-/

/-!
# Smooth online-to-batch conversion

arXiv:2405.15682v4, Appendix D, Corollary 1. This is the pathwise
population-gradient bound. Taking expectations and replacing population
pairings by unbiased oracle pairings gives the stated stochastic version.
The initial query is arbitrary because its coefficient is zero.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The interpolation residual pays for the mismatch of the two gradients. -/
theorem smooth_mix_energy {F : E → ℝ} {grad : E → E} {L W w α : ℝ}
    (hL : 0 < L) (hgrad : ∀ x, IsSubgradient F x (grad x))
    (hsmooth : SmoothUpper F grad L) (hW : 0 ≤ W)
    (hα : 0 ≤ α) (hα1 : α ≤ 1) (hsmall : 10 * W * α ≤ w) (x z : E) :
    10 * W * ‖grad x - grad (mix α x z)‖ ^ 2 ≤
      2 * L * w * mixGap F grad α x z := by
  let y := mix α x z
  have hb1 := (div_le_iff₀ (by positivity : 0 < 2 * L)).mp
    (smooth_bregman hL hgrad hsmooth y x)
  have hb2 := (div_le_iff₀ (by positivity : 0 < 2 * L)).mp
    (smooth_bregman hL hgrad hsmooth x y)
  rw [norm_sub_rev] at hb1
  have hid : α * mixGap F grad α x z =
      bregman F grad y x + (1 - α) * bregman F grad x y := by
    by_cases ha : α = 0
    · simp [ha, y, mix, bregman]
    · rw [mixGap_eq_div F grad α x z ha]
      dsimp [y]
      field_simp
  have hg := mixGap_nonneg hα hα1 (hgrad x) (hgrad y)
  have hm := mul_le_mul_of_nonneg_left hb2 (sub_nonneg.mpr hα1)
  have hidL := congrArg (fun s : ℝ => 2 * L * s) hid
  have hc : (2 - α) * ‖grad x - grad y‖ ^ 2 ≤ 2 * L * α * mixGap F grad α x z := by
    nlinarith
  have hweighted := mul_le_mul_of_nonneg_left hc (by positivity : 0 ≤ 10 * W)
  have hcoeff := mul_le_mul_of_nonneg_right hsmall
    (mul_nonneg (by positivity : 0 ≤ 2 * L) hg)
  have he : ‖grad x - grad y‖ ^ 2 ≤ (2 - α) * ‖grad x - grad y‖ ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hα1) (sq_nonneg ‖grad x - grad y‖)]
  exact (mul_le_mul_of_nonneg_left he (by positivity : 0 ≤ 10 * W)).trans
    (hweighted.trans (by nlinarith [hcoeff]))

example (x z : ℝ) : 10 * 1 * ‖(0 : ℝ) - 0‖ ^ 2 ≤
    2 * 1 * 1 * mixGap (fun _ => 0) (fun _ => 0) 0 x z :=
  smooth_mix_energy (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro a b; simp) (by intro a b; simp only [inner_zero_left, zero_add]; positivity)
    zero_le_one le_rfl zero_le_one (by norm_num) x z

/-- The local estimate behind Corollary 1, with a telescoping endpoint mismatch. -/
theorem smooth_defect_step {F : E → ℝ} {grad : E → E} {L U w α : ℝ}
    (hL : 0 < L) (hgrad : ∀ x, IsSubgradient F x (grad x))
    (hsmooth : SmoothUpper F grad L) (hU : 0 ≤ U) (hw : 0 ≤ w)
    (hα : 0 ≤ α) (hα1 : α ≤ 1) (hsmall : 10 * (U + w) * α ≤ w)
    (p x z yp u : E) :
    U / (6 * L) * ‖grad (mix α x z) - grad yp‖ ^ 2 +
      (U + w) / (2 * L) * ‖grad x - grad (mix α x z)‖ ^ 2 -
      U / (2 * L) * ‖grad p - grad yp‖ ^ 2 ≤
      w * mixGap F grad α x z + U * bregman F grad p x +
        w * bregman F grad u (mix α x z) := by
  let y := mix α x z
  have hm := smooth_mix_energy hL hgrad hsmooth (add_nonneg hU hw) hα hα1 hsmall x z
  have hb := (div_le_iff₀ (by positivity : 0 < 2 * L)).mp
    (smooth_bregman hL hgrad hsmooth p x)
  rw [norm_sub_rev] at hb
  have ht := norm_three_sq (grad y - grad x) (grad x - grad p) (grad p - grad yp)
  rw [show (grad y - grad x) + (grad x - grad p) + (grad p - grad yp) =
    grad y - grad yp by module, norm_sub_rev (grad y) (grad x)] at ht
  have htU := mul_le_mul_of_nonneg_left ht hU
  have hbU := mul_le_mul_of_nonneg_left hb hU
  have hstar := mul_nonneg (mul_nonneg (by positivity : 0 ≤ 6 * L) hw)
    (bregman_nonneg (hgrad (mix α x z)) (a := u))
  have hrest := mul_nonneg (show 0 ≤ 24 * U + 27 * w by positivity)
    (sq_nonneg ‖grad x - grad y‖)
  apply (mul_le_mul_iff_left₀ (by positivity : 0 < 6 * L)).mp
  have he1 : U / (6 * L) * (6 * L) = U := by field_simp
  have he2 : (U + w) / (2 * L) * (6 * L) = 3 * (U + w) := by field_simp; ring
  have he3 : U / (2 * L) * (6 * L) = 3 * U := by field_simp; ring
  simp only [sub_mul, add_mul, mul_assoc]
  have hcalc :
      (U / (6 * L) * ‖grad y - grad yp‖ ^ 2 +
        (U + w) / (2 * L) * ‖grad x - grad y‖ ^ 2 -
        U / (2 * L) * ‖grad p - grad yp‖ ^ 2) * (6 * L) =
      U * ‖grad y - grad yp‖ ^ 2 + 3 * (U + w) * ‖grad x - grad y‖ ^ 2 -
        3 * U * ‖grad p - grad yp‖ ^ 2 := by
    linear_combination he1 * ‖grad y - grad yp‖ ^ 2 +
      he2 * ‖grad x - grad y‖ ^ 2 - he3 * ‖grad p - grad yp‖ ^ 2
  dsimp [y] at *
  nlinarith [hcalc]

example (p x z yp u : ℝ) :
    1 / (6 * 1) * ‖(0 : ℝ) - 0‖ ^ 2 + (1 + 1) / (2 * 1) * ‖(0 : ℝ) - 0‖ ^ 2 -
      1 / (2 * 1) * ‖(0 : ℝ) - 0‖ ^ 2 ≤
      1 * mixGap (fun _ => 0) (fun _ => 0) 0 x z +
        1 * bregman (fun _ => 0) (fun _ => 0) p x +
          1 * bregman (fun _ => 0) (fun _ => 0) u (mix 0 x z) :=
  smooth_defect_step (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro a b; simp) (by intro a b; simp only [inner_zero_left, zero_add]; positivity)
    zero_le_one zero_le_one le_rfl zero_le_one (by norm_num) p x z yp u

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Query history indexed like the averages: entry `t+1` is the query at step `t`. -/
def pastQuery (w α : ℕ → ℝ) (z : ℕ → E) (initial : E) : ℕ → E
  | 0 => initial
  | t + 1 => query w α z t

/-- Corollary 1: smoothness pays for changes of gradient at successive queries.
The small-interpolation condition is the paper's `α_t≤w_t/(10w_{1:t})`,
written after multiplication by its positive denominator. -/
theorem negative_bregman {F : E → ℝ} {grad : E → E} {L : ℝ}
    (hL : 0 < L) (hgrad : ∀ x, IsSubgradient F x (grad x))
    (hsmooth : SmoothUpper F grad L) {w α : ℕ → ℝ}
    (hw : ∀ t, 0 ≤ w t) (hfirst : 0 < w 0) (hα : ∀ t, 0 ≤ α t ∧ α t ≤ 1)
    (hsmall : ∀ t, 10 * weightSum w (t + 1) * α t ≤ w t)
    (z : ℕ → E) (u : E) (T : ℕ) :
    weightSum w T * (F (average w z T) - F u) ≤
      (∑ t ∈ Finset.range T, w t * inner ℝ (grad (query w α z t)) (z t - u)) -
        ∑ t ∈ Finset.range T, weightSum w t / (6 * L) *
          ‖grad (query w α z t) - grad (pastQuery w α z u t)‖ ^ 2 := by
  let P := fun t => weightSum w t / (2 * L) *
    ‖grad (average w z t) - grad (pastQuery w α z u t)‖ ^ 2
  have hstep (t : ℕ) : weightSum w t / (6 * L) *
      ‖grad (query w α z t) - grad (pastQuery w α z u t)‖ ^ 2 + (P (t + 1) - P t) ≤
      defect F grad w α z u t := by
    have h := smooth_defect_step hL hgrad hsmooth (weightSum_nonneg hw t) (hw t)
      (hα t).1 (hα t).2 (by simpa only [← weightSum_succ] using hsmall t)
      (average w z t) (average w z (t + 1)) (z t) (pastQuery w α z u t) u
    change weightSum w t / (6 * L) *
      ‖grad (query w α z t) - grad (pastQuery w α z u t)‖ ^ 2 +
      (weightSum w (t + 1) / (2 * L) *
        ‖grad (average w z (t + 1)) - grad (query w α z t)‖ ^ 2 - P t) ≤ _
    dsimp only [P, defect, query] at h ⊢
    rw [weightSum_succ]
    convert h using 1
    ring
  have hs := Finset.sum_le_sum (s := Finset.range T) (fun t _ => hstep t)
  rw [Finset.sum_add_distrib, Finset.sum_range_sub P T] at hs
  have hP0 : P 0 = 0 := by simp [P, weightSum_zero]
  have hPT : 0 ≤ P T := by
    exact mul_nonneg (div_nonneg (weightSum_nonneg hw T) (by positivity)) (sq_nonneg _)
  rw [hP0, sub_zero] at hs
  have hid := conversion_identity F grad w α z u T
    (fun t _ => (weightSum_pos_succ hw hfirst t).ne')
  rw [Finset.sum_sub_distrib] at hid
  linarith

example (z : ℕ → ℝ) (u : ℝ) (T : ℕ) :
    weightSum (fun _ => 1) T * ((0 : ℝ) - 0) ≤
      (∑ t ∈ Finset.range T, (1 : ℝ) * inner ℝ (0 : ℝ) (z t - u)) -
        ∑ t ∈ Finset.range T, weightSum (fun _ => 1) t / (6 * 1) * ‖(0 : ℝ) - 0‖ ^ 2 :=
  negative_bregman (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro a b; simp) (by intro a b; simp only [inner_zero_left, zero_add]; positivity)
    (fun _ => zero_le_one) zero_lt_one (fun _ => ⟨le_rfl, zero_le_one⟩)
    (by intro t; simp) z u T

end Transformer.ScheduleFree

end
