/-
# Quantized attention has a finite context length

Store the softmax weights of an attention row in a format with `b` fractional
bits.  With scores confined to a window of width `D`, every weight is at most
`e^D / n` (`softmax_le`), and the format rounds everything below `2^{-(b+1)}`
to zero (`quantize_eq_zero`).  So once

  `n > 2^{b+1} · e^D`

every stored weight is `0`, and the head outputs `0` whatever the values are:
the signal is lost, not merely attenuated (`qAttn_bits_eq_zero`).  Below

  `n ≤ 2^{b+1} · e^{-D}`

every weight survives rounding, and each token's value is still read off
exactly (`qAttn_bits_injective`).  The critical length is therefore
`2^{b+1}` up to a factor `e^{±D}`: one bit fewer halves the context.

The bound on the score window is what makes this true, not a convenience:
scores whose window grows like `log n` keep one token visible at every length
(`exists_scores_qAttn_injective`).

The argument is the dispersion of softmax (Veličković, Perivolaropoulos,
Barbero, Pascanu, arXiv:2410.01104, §2) composed with the dead zone of
round-to-nearest; the threshold and its two-sided form are this repository's
own statements, not a paper's.  `qAttn_eq_zero` needs of the rounding only its
dead zone `[0, θ)`, so it applies to any format, floating point included, with
`θ` half its smallest positive number.
-/

import Transformer.Precision.Basic

open scoped BigOperators

namespace Transformer
namespace Precision

variable {n : ℕ} {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- **Signal loss, for any rounding with a dead zone.**  If `Q` sends `[0, θ)`
to `0` and the scores lie in a window of width `D`, then beyond `n θ > e^D`
the quantized head outputs `0` for every choice of values. -/
theorem qAttn_eq_zero {Q : ℝ → ℝ} {θ D : ℝ} (hQ : ∀ x, 0 ≤ x → x < θ → Q x = 0)
    {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D) (hn : Real.exp D < n * θ)
    (v : Idx n → E) : qAttn Q s v = 0 := by
  refine Finset.sum_eq_zero fun i _ => ?_
  have hn0 : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hlt : softmax s i < θ :=
    (softmax_le hs i).trans_lt (by rw [div_lt_iff₀ hn0]; linarith)
  rw [hQ _ (softmax_nonneg s i) hlt, zero_smul]

/-- The hypotheses of `qAttn_eq_zero` are satisfiable: the fixed-point rounding
of step `1` over four equal scores, where `4 · 1/2 > e^0`. -/
example : (∀ x : ℝ, 0 ≤ x → x < 1 / 2 → quantize 1 x = 0) ∧
    (∀ i j : Idx 4, (fun _ => (0 : ℝ)) i - (fun _ => (0 : ℝ)) j ≤ 0) ∧
    Real.exp 0 < (4 : ℕ) * (1 / 2 : ℝ) :=
  ⟨fun _ h0 h => quantize_eq_zero one_pos h0 (by linarith), by simp, by norm_num⟩

/-- A token whose stored weight is nonzero is read off exactly: changing its
value changes the output. -/
theorem qAttn_update_injective {Q : ℝ → ℝ} {s : Idx n → ℝ} (v : Idx n → E) (i : Idx n)
    (hi : Q (softmax s i) ≠ 0) : Function.Injective fun x => qAttn Q s (Function.update v i x) := by
  have key : ∀ x, qAttn Q s (Function.update v i x) =
      Q (softmax s i) • x + ∑ j ∈ Finset.univ.erase i, Q (softmax s j) • v j := by
    intro x
    rw [qAttn, ← Finset.add_sum_erase _ _ (Finset.mem_univ i), Function.update_self]
    congr 1
    exact Finset.sum_congr rfl fun j hj => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hj)]
  intro x y hxy
  simp only [key, add_left_inj] at hxy
  exact smul_right_injective E hi hxy

/-- The hypothesis of `qAttn_update_injective` is satisfiable: the identity
rounding of the single weight `1`. -/
example : (id : ℝ → ℝ) (softmax (fun _ : Idx 1 => 0) 0) ≠ 0 := by
  simp [softmax]

/-- `b` fractional bits: the grid `2^{-b} ℤ`, rounded to nearest. -/
theorem quantize_bits_eq_zero (b : ℕ) {x : ℝ} (hx0 : 0 ≤ x) (hx : x < 1 / 2 ^ (b + 1)) :
    quantize (1 / 2 ^ b) x = 0 :=
  quantize_eq_zero (by positivity) hx0 (by rw [pow_succ] at hx; linarith [hx, show
    (1 : ℝ) / (2 ^ b * 2) = 1 / 2 ^ b / 2 by ring])

/-- **Signal loss in `b` bits.**  With scores in a window of width `D` and
`n > 2^{b+1} e^D` tokens, every weight rounds to `0` in the `b`-bit fixed-point
format, and the head outputs `0` whatever the values are. -/
theorem qAttn_bits_eq_zero (b : ℕ) {D : ℝ} {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D)
    (hn : 2 ^ (b + 1) * Real.exp D < n) (v : Idx n → E) :
    qAttn (quantize (1 / 2 ^ b)) s v = 0 := by
  refine qAttn_eq_zero (θ := 1 / 2 ^ (b + 1)) (fun x h0 h => quantize_bits_eq_zero b h0 h) hs ?_ v
  rw [mul_one_div, lt_div_iff₀ (by positivity)]
  linarith

/-- The hypotheses of `qAttn_bits_eq_zero` are satisfiable: one bit, equal
scores, and `5 > 2^2 · e^0` tokens. -/
example : (2 : ℝ) ^ (1 + 1) * Real.exp 0 < (5 : ℕ) := by norm_num

/-- **Signal kept in `b` bits.**  With scores in a window of width `D` and
`n ≤ 2^{b+1} e^{-D}` tokens, every weight survives rounding in the `b`-bit
fixed-point format, and each token's value is read off exactly. -/
theorem qAttn_bits_injective (b : ℕ) {D : ℝ} {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D)
    (hn : (n : ℝ) ≤ 2 ^ (b + 1) * Real.exp (-D)) (v : Idx n → E) (i : Idx n) :
    Function.Injective fun x => qAttn (quantize (1 / 2 ^ b)) s (Function.update v i x) := by
  refine qAttn_update_injective v i (quantize_pos (by positivity) ?_).ne'
  have hn0 : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  refine le_trans ?_ (le_softmax hs i)
  rw [le_div_iff₀ hn0, pow_succ] at *
  calc 1 / 2 ^ b / 2 * n ≤ 1 / 2 ^ b / 2 * (2 ^ b * 2 * Real.exp (-D)) := by gcongr
    _ = Real.exp (-D) := by field_simp

/-- The hypotheses of `qAttn_bits_injective` are satisfiable: one bit, equal
scores, and `4 ≤ 2^2 · e^0` tokens. -/
example : ((4 : ℕ) : ℝ) ≤ 2 ^ (1 + 1) * Real.exp (-0) := by norm_num

/-- **Every format has a finite context length.**  For each number of bits and
each width of the score window there is a length beyond which the quantized
head outputs `0` on every input. -/
theorem exists_length_qAttn_eq_zero (b : ℕ) (D : ℝ) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ s : Idx n → ℝ, (∀ i j, s i - s j ≤ D) →
      ∀ v : Idx n → E, qAttn (quantize (1 / 2 ^ b)) s v = 0 := by
  refine ⟨⌈2 ^ (b + 1) * Real.exp D⌉₊ + 1, fun n hn s hs v => qAttn_bits_eq_zero b hs ?_ v⟩
  have : ((⌈2 ^ (b + 1) * Real.exp D⌉₊ + 1 : ℕ) : ℝ) ≤ n := by exact_mod_cast hn
  push_cast at this
  linarith [Nat.le_ceil (2 ^ (b + 1) * Real.exp D)]

/-- **The score window must be bounded.**  Without it no length is critical:
at every length, the scores `log n` on one token and `0` elsewhere keep that
token's weight at least `1/2`, which survives every format of at least one
step per unit, and its value is read off exactly. -/
theorem exists_scores_qAttn_injective (b : ℕ) (v : Idx n → E) (i : Idx n) :
    ∃ s : Idx n → ℝ,
      Function.Injective fun x => qAttn (quantize (1 / 2 ^ b)) s (Function.update v i x) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  set s : Idx n → ℝ := fun j => if j = i then Real.log n else 0
  refine ⟨s, qAttn_update_injective v i (quantize_pos (by positivity) ?_).ne'⟩
  have hsi : Real.exp (s i) = n := by simp [s, Real.exp_log hn0]
  have hrest : ∑ j ∈ Finset.univ.erase i, Real.exp (s j) ≤ n := by
    have h1 : ∑ j ∈ Finset.univ.erase i, Real.exp (s j) = ∑ j ∈ Finset.univ.erase i, (1 : ℝ) :=
      Finset.sum_congr rfl fun j hj => by simp [s, Finset.ne_of_mem_erase hj]
    rw [h1]
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
    exact_mod_cast (Finset.card_erase_le).trans (by simp)
  have hZ : ∑ j, Real.exp (s j) ≤ 2 * n := by
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i), hsi]; linarith
  have hhalf : 1 / 2 ≤ softmax s i := by
    rw [softmax, hsi, le_div_iff₀ (softmax_denom_pos s i)]; linarith
  have : 1 / 2 ^ b / 2 ≤ (1 : ℝ) / 2 := by
    gcongr; exact div_le_one_of_le₀ (one_le_pow₀ (by norm_num)) (by positivity)
  linarith

end Precision
end Transformer
