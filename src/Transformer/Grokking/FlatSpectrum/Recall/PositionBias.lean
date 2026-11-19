module

public import Transformer.Grokking.FlatSpectrum.Recall.Threshold

/-!
# No positional bias helps the worst position

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), items PB1 and
PB2, for the bias `−m |j − i|` that ALiBi adds to the scores (arXiv:2505.22842v4, §2.2, Lemma 2,
`Transformer.BAM.alibi`) and the scale `s log n` of SSMax (arXiv:2501.19399v1, §2).

A bias `b` of the positions adds `b j` to the score of the token `j`, so that the weights at the
scale `s log n` are `n^{s z_j} e^{b_j}` (`biasedLogScaleAttention`); ALiBi has `b = −m r` at the
distance `r` from the query. At a token of least bias the bias only adds to the miss: for all
scores, the value `1` there and `−1` elsewhere are answered at least as badly with the bias as
without it (`one_sub_softmaxAttention_indicator_le_add`). Every bias has a token of least bias,
the farthest token for ALiBi. With one-hot raw scores `±B` a key placed there is answered with an
error of at least `2m / (n^{2sB} + m)` (`exists_le_oneHot_biasedLogScaleAttention_error`), the
worst error without a bias at every position (`isGreatest_oneHot_logScaleAttention_error`). So
over the positions of the key and the dictionaries no bias lowers the worst error, and the bias
`0` attains it (`oneHot_logScaleAttention_error_le_biased`): for a key at any position the best
ALiBi slope is `0`, at every scale, and at the scale `s log n` the threshold `2 s B = 1` stays.

A key that leads by `g` is answered within `2 C n^{−sg}` times the sum of `e^{b_j − b_{j₀}}` over
the other tokens (`abs_sub_biasedLogScaleAttention_le`), and with raw scores bounded by `B` any
other token `j₁` leaves an error of at least `2 / (1 + n^{2sB} e^{b_{j₀} − b_{j₁}})`
(`le_one_sub_biasedLogScaleAttention`); `Recall.ALiBiWindow` applies both to ALiBi.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- **Attention at the scale `s log n` with a positional bias** (`review/flat_spectrum.md`, PB1):
over `n` tokens, the raw scores `z j` multiplied by `s log n`, plus the bias `b j` of the token
`j`; the weights are `n^{s z_j} e^{b_j}`. -/
noncomputable def biasedLogScaleAttention {ι : Type*} [Fintype ι] (s : ℝ) (b z v : ι → ℝ) : ℝ :=
  softmaxAttention (fun j => s * Real.log (Fintype.card ι) * z j + b j) v

/-- Without a bias it is attention at the scale `s log n`. -/
theorem biasedLogScaleAttention_zero {ι : Type*} [Fintype ι] (s : ℝ) (z v : ι → ℝ) :
    biasedLogScaleAttention s 0 z v = logScaleAttention s z v := by
  simp [biasedLogScaleAttention, logScaleAttention]

/-- **At a token of least bias the bias only adds to the miss** (`review/flat_spectrum.md`, PB1):
if `b j₀ ≤ b j` for every `j`, then for all scores `t` the value `1` at `j₀` and `−1` elsewhere
are answered by softmax attention with the scores `t + b` at least as badly as with `t`. -/
theorem one_sub_softmaxAttention_indicator_le_add {ι : Type*} [Fintype ι] [DecidableEq ι]
    (t : ι → ℝ) {b : ι → ℝ} {j₀ : ι} (hb : ∀ j, b j₀ ≤ b j) :
    1 - softmaxAttention t (fun j => if j = j₀ then 1 else -1) ≤
      1 - softmaxAttention (fun j => t j + b j) (fun j => if j = j₀ then 1 else -1) := by
  have hZ : ∑ j, Real.exp (t j) = Real.exp (t j₀) + ∑ j ∈ univ.erase j₀, Real.exp (t j) :=
    (add_sum_erase _ _ (mem_univ j₀)).symm
  have hZ' : ∑ j, Real.exp (t j + b j) =
      Real.exp (t j₀ + b j₀) + ∑ j ∈ univ.erase j₀, Real.exp (t j + b j) :=
    (add_sum_erase _ _ (mem_univ j₀)).symm
  have hX : Real.exp (b j₀) * ∑ j ∈ univ.erase j₀, Real.exp (t j) ≤
      ∑ j ∈ univ.erase j₀, Real.exp (t j + b j) := by
    rw [mul_sum]
    exact sum_le_sum fun j _ => by
      rw [Real.exp_add, mul_comm]
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (hb j)) (Real.exp_pos _).le
  have hX0 : 0 ≤ ∑ j ∈ univ.erase j₀, Real.exp (t j) := sum_nonneg fun j _ => (Real.exp_pos _).le
  have hA := Real.exp_pos (t j₀)
  have hB := Real.exp_pos (b j₀)
  rw [one_sub_softmaxAttention_indicator, one_sub_softmaxAttention_indicator, hZ, hZ',
    Real.exp_add (t j₀) (b j₀), div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hX hA.le]

/-- Two tokens, the second of least bias. -/
example := one_sub_softmaxAttention_indicator_le_add ![1, 0] (b := ![0, -1]) (j₀ := 1)
  fun j => by fin_cases j <;> norm_num

/-- **No bias helps the worst position** (`review/flat_spectrum.md`, PB1): with one-hot raw
scores `±B` at the scale `s log (m + 1)`, every bias `b` of the `m + 1` keys has a key, one of
least bias, and a dictionary of values `±1` answered with an error of at least
`2m / ((m + 1)^{2sB} + m)`, the worst error without a bias. -/
theorem exists_le_oneHot_biasedLogScaleAttention_error (s B : ℝ) {m : ℕ}
    (b : Fin (m + 1) → ℝ) :
    ∃ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      2 * m / (((m : ℝ) + 1) ^ (2 * s * B) + m) ≤ |(if π q then (1 : ℝ) else -1) -
        biasedLogScaleAttention s b (fun k => if k = q then B else -B)
          fun k => if π k then (1 : ℝ) else -1| := by
  obtain ⟨q, hq⟩ := Finite.exists_min b
  refine ⟨q, fun k => decide (k = q), ?_⟩
  have h := one_sub_softmaxAttention_indicator_le_add
    (fun k => s * Real.log (Fintype.card (Fin (m + 1))) * if k = q then B else -B) hq
  rw [one_sub_softmaxAttention_indicator, mul_div_assoc, oneHot_erase_div_sum,
    ← mul_div_assoc] at h
  dsimp only
  rw [biasedLogScaleAttention, abs_recall_error_decide]
  exact h.trans (le_abs_self _)

/-- ALiBi with the slope `1/2` over four keys, the bias `−r/2` at the key `r`. -/
example := exists_le_oneHot_biasedLogScaleAttention_error 1 1 (m := 3) fun r => -(r : ℝ) / 2

/-- **The bias `0` is best for a key at any position** (`review/flat_spectrum.md`, PB1): with
one-hot raw scores `±B` at the scale `s log (m + 1)`, the error at any key and any dictionary of
values `±1` without a bias is at most the error at some key and some dictionary with the bias `b`,
for every bias `b`. -/
theorem oneHot_logScaleAttention_error_le_biased (s B : ℝ) {m : ℕ} (b : Fin (m + 1) → ℝ)
    (q' : Fin (m + 1)) (π' : Fin (m + 1) → Bool) :
    ∃ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      |(if π' q' then (1 : ℝ) else -1) - logScaleAttention s (fun k => if k = q' then B else -B)
          fun k => if π' k then (1 : ℝ) else -1| ≤
        |(if π q then (1 : ℝ) else -1) - biasedLogScaleAttention s b
          (fun k => if k = q then B else -B) fun k => if π k then (1 : ℝ) else -1| := by
  obtain ⟨q, π, h⟩ := exists_le_oneHot_biasedLogScaleAttention_error s B b
  exact ⟨q, π, ((isGreatest_oneHot_logScaleAttention_error s B q').2
    (Set.mem_range_self π')).trans h⟩

/-- ALiBi with the slope `1/2` over four keys, against the first key without a bias. -/
example := oneHot_logScaleAttention_error_le_biased 1 1 (m := 3) (fun r => -(r : ℝ) / 2) 0
  fun _ => true

/-- **The bias against the lead** (`review/flat_spectrum.md`, PB2): if the raw score of `j₀` leads
every other by `g` and the values are bounded by `C`, attention at the scale `s log n`, `s ≥ 0`,
with the bias `b` answers the value of `j₀` within `2 C n^{−sg}` times the sum of `e^{b_j − b_{j₀}}`
over the other tokens. -/
theorem abs_sub_biasedLogScaleAttention_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s g C : ℝ}
    (hs : 0 ≤ s) (b : ι → ℝ) {z v : ι → ℝ} {j₀ : ι} (hz : ∀ j, j ≠ j₀ → z j + g ≤ z j₀)
    (hv : ∀ j, |v j| ≤ C) :
    |v j₀ - biasedLogScaleAttention s b z v| ≤
      2 * C * ((Fintype.card ι : ℝ) ^ (-(s * g)) *
        ∑ j ∈ univ.erase j₀, Real.exp (b j - b j₀)) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv j₀)
  have hlog : 0 ≤ s * Real.log (Fintype.card ι) :=
    mul_nonneg hs (Real.log_nonneg (by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩))
  have hng : (Fintype.card ι : ℝ) ^ (-(s * g)) =
      Real.exp (-(s * Real.log (Fintype.card ι) * g)) := by
    rw [Real.rpow_def_of_pos hn]
    ring_nf
  rw [biasedLogScaleAttention]
  refine (abs_sub_softmaxAttention_le hv j₀).trans (mul_le_mul_of_nonneg_left ?_ (by linarith))
  set c := s * Real.log (Fintype.card ι) with hc
  have hZ : Real.exp (c * z j₀ + b j₀) ≤ ∑ j, Real.exp (c * z j + b j) :=
    single_le_sum (f := fun j => Real.exp (c * z j + b j)) (fun j _ => (Real.exp_pos _).le)
      (mem_univ j₀)
  rw [div_le_iff₀ ((Real.exp_pos _).trans_le hZ)]
  refine le_trans ?_ (mul_le_mul_of_nonneg_left hZ (by positivity))
  rw [mul_sum, sum_mul]
  refine sum_le_sum fun j hj => ?_
  rw [hng, ← Real.exp_add, ← Real.exp_add]
  refine Real.exp_le_exp.2 ?_
  have h := mul_le_mul_of_nonneg_left (hz j (ne_of_mem_erase hj)) hlog
  rw [mul_add] at h
  linarith

/-- Two tokens, the first leading by `1` and of the greater bias. -/
example := abs_sub_biasedLogScaleAttention_le (s := 1) (g := 1) (C := 1) zero_le_one ![0, -1]
  (z := ![1, 0]) (v := ![1, -1]) (j₀ := 0) (fun j hj => by fin_cases j <;> simp_all)
  fun j => by fin_cases j <;> simp

/-- **One other token bounds the miss from below** (`review/flat_spectrum.md`, PB2): with raw
scores bounded by `B`, attention at the scale `s log n`, `s ≥ 0`, with the bias `b` answers the
value `1` at `j₀` and `−1` elsewhere with an error of at least
`2 / (1 + n^{2sB} e^{b_{j₀} − b_{j₁}})`, for every other token `j₁`. -/
theorem le_one_sub_biasedLogScaleAttention {ι : Type*} [Fintype ι] [DecidableEq ι] {s B : ℝ}
    (hs : 0 ≤ s) (b : ι → ℝ) {z : ι → ℝ} (hz : ∀ j, |z j| ≤ B) {j₀ j₁ : ι} (hj : j₁ ≠ j₀) :
    2 / (1 + (Fintype.card ι : ℝ) ^ (2 * s * B) * Real.exp (b j₀ - b j₁)) ≤
      1 - biasedLogScaleAttention s b z (fun j => if j = j₀ then 1 else -1) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hlog : 0 ≤ s * Real.log (Fintype.card ι) :=
    mul_nonneg hs (Real.log_nonneg (by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩))
  have hng : (Fintype.card ι : ℝ) ^ (2 * s * B) =
      Real.exp (2 * (s * Real.log (Fintype.card ι)) * B) := by
    rw [Real.rpow_def_of_pos hn]
    ring_nf
  rw [biasedLogScaleAttention, one_sub_softmaxAttention_indicator,
    ← add_sum_erase _ _ (mem_univ j₀), hng]
  set c := s * Real.log (Fintype.card ι) with hc
  have hX : Real.exp (c * z j₁ + b j₁) ≤ ∑ j ∈ univ.erase j₀, Real.exp (c * z j + b j) :=
    single_le_sum (f := fun j => Real.exp (c * z j + b j)) (fun j _ => (Real.exp_pos _).le)
      (mem_erase.2 ⟨hj, mem_univ _⟩)
  have hA : Real.exp (c * z j₀ + b j₀) ≤
      Real.exp (2 * c * B) * Real.exp (b j₀ - b j₁) * Real.exp (c * z j₁ + b j₁) := by
    rw [← Real.exp_add, ← Real.exp_add]
    refine Real.exp_le_exp.2 ?_
    have h1 := mul_le_mul_of_nonneg_left (le_of_abs_le (hz j₀)) hlog
    have h2 := mul_le_mul_of_nonneg_left (neg_le_of_abs_le (hz j₁)) hlog
    linarith
  have hK : 0 ≤ Real.exp (2 * c * B) * Real.exp (b j₀ - b j₁) := by positivity
  have hX0 := (Real.exp_pos (c * z j₁ + b j₁)).trans_le hX
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hX hK]

/-- Two tokens with raw scores `1` and `0`, the first of the greater bias. -/
example := le_one_sub_biasedLogScaleAttention (s := 1) (B := 1) zero_le_one ![0, -1]
  (z := ![1, 0]) (fun j => by fin_cases j <;> simp) (j₀ := 0) (j₁ := 1) (by decide)

end Transformer.Grokking.FlatSpectrum.Recall
