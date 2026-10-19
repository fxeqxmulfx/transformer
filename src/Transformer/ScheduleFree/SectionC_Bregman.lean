module

public import Transformer.ScheduleFree.Section2_Averaging

/-!
# The exact Bregman identity

arXiv:2405.15682v4, Appendix C, Theorem 5 (`thm:genbregman`).
The interpolation remainder is written without division. For `α ≠ 0`
it is exactly the paper's two Bregman terms; at `α = 0` it is zero.
This supplies the continuous endpoint convention missing from the printed
formula, without excluding primal averaging.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix C: Bregman divergence for a specified subgradient selection. -/
def bregman (F : E → ℝ) (grad : E → E) (a b : E) : ℝ :=
  F a - F b - inner ℝ (grad b) (a - b)

/-- Appendix C: division-free interpolation remainder, including `α=0`. -/
def mixGap (F : E → ℝ) (grad : E → E) (α : ℝ) (x z : E) : ℝ :=
  bregman F grad (mix α x z) x +
    inner ℝ (grad (mix α x z) - grad x) (z - mix α x z)

/-- Appendix C: supporting subgradients make every Bregman divergence nonnegative. -/
theorem bregman_nonneg {F : E → ℝ} {grad : E → E} {a b : E}
    (h : IsSubgradient F b (grad b)) : 0 ≤ bregman F grad a b := by
  have hab := h a
  unfold bregman
  linarith

example (a b : ℝ) : 0 ≤ bregman (fun _ => 0) (fun _ => 0) a b :=
  bregman_nonneg (by intro u; simp)

/-- Appendix C, eq. (13): the symmetric Bregman identity. -/
theorem bregman_symm (F : E → ℝ) (grad : E → E) (a b : E) :
    bregman F grad a b + bregman F grad b a =
      inner ℝ (grad a - grad b) (a - b) := by
  simp only [bregman, inner_sub_left, inner_sub_right]
  ring

/-- Appendix C's printed interpolation terms, away from the zero denominator. -/
theorem mixGap_eq_div (F : E → ℝ) (grad : E → E) (α : ℝ) (x z : E)
    (hα : α ≠ 0) :
    mixGap F grad α x z = bregman F grad (mix α x z) x / α +
      (1 - α) / α * bregman F grad x (mix α x z) := by
  simp only [mixGap, bregman, mix, inner_sub_left, inner_sub_right,
    inner_add_right, real_inner_smul_right]
  field_simp
  ring

example (F : E → ℝ) (grad : E → E) (x z : E) :
    mixGap F grad 1 x z = bregman F grad (mix 1 x z) x / 1 +
      (1 - 1) / 1 * bregman F grad x (mix 1 x z) :=
  mixGap_eq_div F grad 1 x z one_ne_zero

/-- The primal-averaging endpoint omitted by Appendix C's division notation. -/
theorem mixGap_zero (F : E → ℝ) (grad : E → E) (x z : E) :
    mixGap F grad 0 x z = 0 := by
  simp [mixGap, bregman, mix]

/-- Appendix C: both interpolation terms have nonnegative coefficients. -/
theorem mixGap_nonneg {F : E → ℝ} {grad : E → E} {α : ℝ} {x z : E}
    (hα : 0 ≤ α) (hα1 : α ≤ 1)
    (hx : IsSubgradient F x (grad x))
    (hy : IsSubgradient F (mix α x z) (grad (mix α x z))) :
    0 ≤ mixGap F grad α x z := by
  by_cases ha : α = 0
  · rw [ha, mixGap_zero]
  · rw [mixGap_eq_div F grad α x z ha]
    exact add_nonneg (div_nonneg (bregman_nonneg hx) hα)
      (mul_nonneg (div_nonneg (sub_nonneg.mpr hα1) hα) (bregman_nonneg hy))

example (α : ℝ) (hα : 0 ≤ α) (hα1 : α ≤ 1) (x z : ℝ) :
    0 ≤ mixGap (fun _ => 0) (fun _ => 0) α x z :=
  mixGap_nonneg hα hα1 (by intro u; simp) (by intro u; simp)

/-- Appendix C, eq. (14): exact one-step conversion, including `α=0`.
Convexity is unnecessary for the identity; it gives the signs of its remainders. -/
theorem bregman_step (F : E → ℝ) (grad : E → E) (α U w : ℝ) (p x z u : E)
    (hbalance : (U + w) • x = U • p + w • z) :
    (U + w) * (F x - F u) - U * (F p - F u) =
      w * inner ℝ (grad (mix α x z)) (z - u) -
      (w * mixGap F grad α x z + U * bregman F grad p x +
        w * bregman F grad u (mix α x z)) := by
  have h := congrArg (fun v => inner ℝ (grad x) v) (balance_displacement hbalance)
  simp only [real_inner_smul_right, inner_sub_right] at h
  simp only [mixGap, bregman, inner_sub_left, inner_sub_right]
  nlinarith [h]

example (F : E → ℝ) (grad : E → E) (α : ℝ) (x u : E) :
    (0 + 1) * (F x - F u) - 0 * (F x - F u) =
      1 * inner ℝ (grad (mix α x x)) (x - u) -
      (1 * mixGap F grad α x x + 0 * bregman F grad x x +
        1 * bregman F grad u (mix α x x)) :=
  bregman_step F grad α 0 1 x x x u (by simp)

end Transformer.ScheduleFree
