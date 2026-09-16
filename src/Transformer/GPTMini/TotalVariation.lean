/-
# ℓ¹ distance between two finite probability vectors

Three facts about probability vectors on a finite index set, none of them
about transformers, all of them needed to say how far the softmax of
`reference/model.py` (`CausalMHA.forward`) can move:

  - `l1_le_two`: any two of them are within `2`, the trivial bound;
  - `l1_le_of_le_mul`: if `a ≤ c · b` pointwise they are within `2 (c - 1)`,
    by the identity `|t| = 2 t⁺ - t` whose linear part cancels;
  - `l1_le_of_le_exp`: combining the two at `c = e^{2r}` gives the *linear*
    bound `8 r`, uniformly in `r ≥ 0` — the multiplicative estimate for small
    `r`, the trivial one for large `r`.

The last is what makes a softmax attention head globally, and not merely
locally, Lipschitz: `2 (e^{2r} - 1)` grows without bound in `r`, but the ℓ¹
distance it estimates never exceeds `2`.
-/

import Transformer.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- **Any two probability vectors are within `2` in ℓ¹.**

`|a - b| ≤ a + b` termwise, and both sides sum to `1`. -/
theorem l1_le_two {T : ℕ} (a b : Fin T → ℝ)
    (ha : ∀ j, 0 ≤ a j) (hb : ∀ j, 0 ≤ b j)
    (ha1 : ∑ j, a j = 1) (hb1 : ∑ j, b j = 1) :
    ∑ j, |a j - b j| ≤ 2 := by
  calc ∑ j, |a j - b j| ≤ ∑ j, (a j + b j) :=
        Finset.sum_le_sum fun j _ =>
          abs_sub_le_iff.mpr ⟨by linarith [hb j], by linarith [ha j]⟩
    _ = 2 := by rw [Finset.sum_add_distrib, ha1, hb1]; norm_num

/-- The hypotheses are satisfiable: the uniform vector on two points. -/
example : ∑ _j : Fin 2, |(1 / 2 : ℝ) - 1 / 2| ≤ 2 :=
  l1_le_two (fun _ => 1 / 2) (fun _ => 1 / 2) (fun _ => by norm_num) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])

/-- **ℓ¹ distance of two distributions from a pointwise ratio bound.**

If `a` and `b` are probability vectors with `a j ≤ c · b j` for every `j`,
then `Σ_j |a j - b j| ≤ 2 (c - 1)`.

The proof is the standard total-variation identity `|t| = 2 t⁺ - t`: summing
it, the linear part cancels because both vectors sum to `1`, and the positive
part is bounded by `(c - 1) b j` termwise.  Note `1 ≤ c` is not assumed — it
follows by summing the hypothesis.

Source: the classical bound on the total variation distance. -/
theorem l1_le_of_le_mul {T : ℕ} (a b : Fin T → ℝ) (c : ℝ)
    (hb : ∀ j, 0 ≤ b j) (hab : ∀ j, a j ≤ c * b j)
    (ha1 : ∑ j, a j = 1) (hb1 : ∑ j, b j = 1) :
    ∑ j, |a j - b j| ≤ 2 * (c - 1) := by
  have habs : ∀ t : ℝ, |t| = 2 * max 0 t - t := by
    intro t
    rcases le_total 0 t with h | h
    · rw [abs_of_nonneg h, max_eq_right h]; ring
    · rw [abs_of_nonpos h, max_eq_left h]; ring
  have hrw : (∑ j, |a j - b j|)
      = 2 * (∑ j, max 0 (a j - b j)) - ((∑ j, a j) - ∑ j, b j) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => habs _
  have hc : (1 : ℝ) ≤ c := by
    have h := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hab j
    rwa [ha1, ← Finset.mul_sum, hb1, mul_one] at h
  have hmax : ∀ j, max 0 (a j - b j) ≤ (c - 1) * b j := by
    intro j
    refine max_le ?_ ?_
    · nlinarith [hb j]
    · nlinarith [hab j]
  have hsum : (∑ j, max 0 (a j - b j)) ≤ (c - 1) * ∑ j, b j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => hmax j
  rw [hb1, mul_one] at hsum
  rw [hrw, ha1, hb1, sub_self, sub_zero]
  linarith

/-- The hypotheses are satisfiable: any probability vector is related to
itself with `c = 1`. -/
example : ∑ _j : Fin 2, |(1 / 2 : ℝ) - 1 / 2| ≤ 2 * ((1 : ℝ) - 1) :=
  l1_le_of_le_mul (fun _ => 1 / 2) (fun _ => 1 / 2) 1
    (fun _ => by norm_num) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])

/-- **`e^{2r} - 1 ≤ 2 r e^{2r}`.**

The convexity bound `x + 1 ≤ e^x` at `x = -2r`, multiplied by `e^{2r} > 0`.
It converts a multiplicative estimate into an additive one. -/
theorem exp_two_mul_sub_one_le (r : ℝ) :
    Real.exp (2 * r) - 1 ≤ 2 * r * Real.exp (2 * r) := by
  have h := Real.add_one_le_exp (-(2 * r))
  have hpos : (0 : ℝ) < Real.exp (2 * r) := Real.exp_pos _
  have hmul : (-(2 * r) + 1) * Real.exp (2 * r)
      ≤ Real.exp (-(2 * r)) * Real.exp (2 * r) :=
    mul_le_mul_of_nonneg_right h hpos.le
  rw [← Real.exp_add, neg_add_cancel, Real.exp_zero] at hmul
  nlinarith

/-- **A pointwise factor `e^{2r}` costs at most `8 r` in ℓ¹.**

For `r ≤ 1/4` the multiplicative bound `2 (e^{2r} - 1) ≤ 4 r e^{2r}` applies
with `e^{2r} ≤ e^{1/2} ≤ 2`; for `r ≥ 1/4` the trivial bound `2 ≤ 8 r` does.
The constant is not sharp — the true bound is `2 tanh r` — but it is linear,
which is what a global Lipschitz constant needs. -/
theorem l1_le_of_le_exp {T : ℕ} (a b : Fin T → ℝ) (r : ℝ) (hr : 0 ≤ r)
    (ha : ∀ j, 0 ≤ a j) (hb : ∀ j, 0 ≤ b j)
    (hab : ∀ j, a j ≤ Real.exp (2 * r) * b j)
    (ha1 : ∑ j, a j = 1) (hb1 : ∑ j, b j = 1) :
    ∑ j, |a j - b j| ≤ 8 * r := by
  have htriv := l1_le_two a b ha hb ha1 hb1
  rcases le_or_gt r (1 / 4) with h | h
  · have h1 := l1_le_of_le_mul a b (Real.exp (2 * r)) hb hab ha1 hb1
    have h2 := exp_two_mul_sub_one_le r
    have h3 : Real.exp (2 * r) ≤ Real.exp (1 / 2 : ℝ) :=
      Real.exp_le_exp.mpr (by linarith)
    have h4 : Real.exp (1 / 2 : ℝ) ≤ 2 := by
      have hsq : Real.exp (1 / 2 : ℝ) * Real.exp (1 / 2 : ℝ) = Real.exp 1 := by
        rw [← Real.exp_add]; norm_num
      nlinarith [Real.exp_one_lt_three, Real.exp_pos (1 / 2 : ℝ)]
    nlinarith [mul_nonneg hr (by linarith : (0 : ℝ) ≤ 2 - Real.exp (2 * r))]
  · linarith

/-- The hypotheses are satisfiable: the uniform vector at `r = 0`. -/
example : ∑ _j : Fin 2, |(1 / 2 : ℝ) - 1 / 2| ≤ 8 * (0 : ℝ) :=
  l1_le_of_le_exp (fun _ => 1 / 2) (fun _ => 1 / 2) 0 le_rfl
    (fun _ => by norm_num) (fun _ => by norm_num) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])

end GPTMini
end Transformer
