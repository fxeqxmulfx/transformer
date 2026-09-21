/-
# Softmax weights under bounded scores, and the fixed-point quantizer

Two elementary facts that the context-length threshold of
`Transformer.Precision.ContextLength` is made of.

*Dispersion.*  If the attention scores of `n` tokens lie within a window of
width `D` (`s i - s j ≤ D` for all `i, j`), every softmax weight lies in
`[e^{-D}/n, e^{D}/n]`.  This is the dispersion of softmax of Veličković,
Perivolaropoulos, Barbero, Pascanu, arXiv:2410.01104, "softmax is not enough
(for sharp out-of-distribution)", §2: with bounded logits, attention
coefficients spread out as the number of items grows.

*The dead zone of a rounding.*  The `b`-bit fixed-point format stores a number
in `[0, 1]` as a multiple of `δ = 2^{-b}`, rounded to nearest.  Everything
below `δ/2` rounds to `0`, and nothing at or above `δ/2` does.
-/

import Transformer.Basic
import Mathlib.Algebra.Order.Round

open scoped BigOperators

namespace Transformer
namespace Precision

variable {n : ℕ}

/-- The softmax weight of token `i` among the scores `s`:
`σ(s)_i = e^{s_i} / Σ_j e^{s_j}` (arXiv:2410.01104, §2, the softmax function). -/
noncomputable def softmax (s : Idx n → ℝ) (i : Idx n) : ℝ :=
  Real.exp (s i) / ∑ j, Real.exp (s j)

/-- The softmax partition function is positive as soon as there is a token. -/
theorem softmax_denom_pos (s : Idx n → ℝ) (i : Idx n) : 0 < ∑ j, Real.exp (s j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (s j)) ⟨i, Finset.mem_univ i⟩

theorem softmax_nonneg (s : Idx n → ℝ) (i : Idx n) : 0 ≤ softmax s i :=
  div_nonneg (Real.exp_pos _).le (softmax_denom_pos s i).le

/-- **Dispersion, from above.**  Scores within a window of width `D` give every
token a weight of at most `e^D / n` (arXiv:2410.01104, §2). -/
theorem softmax_le {D : ℝ} {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D) (i : Idx n) :
    softmax s i ≤ Real.exp D / n := by
  have hZ := softmax_denom_pos s i
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hlow : (n : ℝ) * Real.exp (s i - D) ≤ ∑ j, Real.exp (s j) := by
    have := Finset.card_nsmul_le_sum Finset.univ (fun j => Real.exp (s j))
      (Real.exp (s i - D)) (fun j _ => Real.exp_le_exp.2 (by linarith [hs i j]))
    simpa [nsmul_eq_mul] using this
  rw [softmax, div_le_div_iff₀ hZ hn]
  calc Real.exp (s i) * n = Real.exp D * (n * Real.exp (s i - D)) := by
        rw [Real.exp_sub]; field_simp
    _ ≤ Real.exp D * ∑ j, Real.exp (s j) := by gcongr

/-- **Dispersion, from below.**  Scores within a window of width `D` give every
token a weight of at least `e^{-D} / n` (arXiv:2410.01104, §2). -/
theorem le_softmax {D : ℝ} {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D) (i : Idx n) :
    Real.exp (-D) / n ≤ softmax s i := by
  have hZ := softmax_denom_pos s i
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hup : ∑ j, Real.exp (s j) ≤ n * Real.exp (s i + D) := by
    have := Finset.sum_le_card_nsmul Finset.univ (fun j => Real.exp (s j))
      (Real.exp (s i + D)) (fun j _ => Real.exp_le_exp.2 (by linarith [hs j i]))
    simpa [nsmul_eq_mul] using this
  rw [softmax, div_le_div_iff₀ hn hZ]
  calc Real.exp (-D) * ∑ j, Real.exp (s j) ≤ Real.exp (-D) * (n * Real.exp (s i + D)) := by
        gcongr
    _ = Real.exp (s i) * n := by
        rw [Real.exp_add, Real.exp_neg]; field_simp

/-- The hypotheses of `softmax_le` and `le_softmax` are satisfiable: constant
scores lie in a window of width `0`. -/
example : ∀ i j : Idx 3, (fun _ => (1 : ℝ)) i - (fun _ => (1 : ℝ)) j ≤ 0 := by
  intro i j; simp

/-- Round-to-nearest onto the grid `δ ℤ`: the fixed-point format of step `δ`.
Ties round up, as Mathlib's `round` does. -/
noncomputable def quantize (δ x : ℝ) : ℝ := δ * round (x / δ)

/-- **The dead zone.**  A nonnegative number below half a step rounds to `0`. -/
theorem quantize_eq_zero {δ x : ℝ} (hδ : 0 < δ) (hx0 : 0 ≤ x) (hx : x < δ / 2) :
    quantize δ x = 0 := by
  have h0 : 0 ≤ x / δ := div_nonneg hx0 hδ.le
  have h1 : x / δ < 1 / 2 := by rw [div_lt_iff₀ hδ]; linarith
  have : round (x / δ) = 0 := by
    rw [round_eq, Int.floor_eq_zero_iff]; constructor <;> linarith
  simp [quantize, this]

/-- **Outside the dead zone.**  A number at least half a step rounds to a
positive grid point. -/
theorem quantize_pos {δ x : ℝ} (hδ : 0 < δ) (hx : δ / 2 ≤ x) : 0 < quantize δ x := by
  have h1 : 1 / 2 ≤ x / δ := by rw [le_div_iff₀ hδ]; linarith
  have : (1 : ℤ) ≤ round (x / δ) := by
    rw [round_eq, Int.le_floor]; push_cast; linarith
  have : (0 : ℝ) < round (x / δ) := by exact_mod_cast this
  exact mul_pos hδ this

/-- The hypotheses of `quantize_eq_zero` and `quantize_pos` are satisfiable. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 : ℝ) / 4 < 1 / 2 ∧ (1 : ℝ) / 2 ≤ 1 / 2 := by
  norm_num

/-- Attention whose weights are stored through a rounding `Q`: the value of
token `i` enters with weight `Q (σ(s)_i)`.  With `Q = id` this is one row of
softmax attention (arXiv:2410.01104, §2). -/
noncomputable def qAttn {E : Type*} [AddCommGroup E] [Module ℝ E] (Q : ℝ → ℝ)
    (s : Idx n → ℝ) (v : Idx n → E) : E :=
  ∑ i, Q (softmax s i) • v i

end Precision
end Transformer
