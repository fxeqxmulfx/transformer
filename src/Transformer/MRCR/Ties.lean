module

public import Transformer.MRCR.InnerNeedle
public import Transformer.MRCR.NearMiss

/-!
# Identical asks: attention averages tied needles, and ALiBi returns the last

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37: for a topic and a format
the context holds "a list of model outputs, in the order that they were presented in the
context", and the query asks for one of them by its index; MRCR-V2 of arXiv:2507.06261v6,
Appendix, line 75 of `8_appendix/1_evaluation_additional_details.tex`, has eight needles. The
attention is that of SSMax, arXiv:2501.19399v1, §2.2, at the scale `s log n`.

The asks of one key are the same text, so a head that scores them by content alone ties them.
If the needles `T` share the largest raw score and every other token trails by `g`, attention at
the scale `s log n` is within `2 C (n − |T|) n^{−s g} / |T|` of the average of their values
(`abs_logScaleAttention_sub_average_le`), not of any one of them.

A bias `−b d` at the distance `d`, that of ALiBi, breaks the tie towards the last needle. With
needles at distinct distances and the other tokens trailing the last needle by `g`, attention is
within `2 C (k n^{−s b} + (n − 1) n^{−s g})` of the value of the last of `k + 1` needles
(`abs_logScaleAttention_sub_last_le`), a bound that falls to `0` as `n` grows when `s b > 0` and
`s g > 1`. Here the bias enters the raw score, so the scale `s log n` multiplies it too. No raw
score of the distance returns a needle with a needle on each side
(`Transformer.MRCR.InnerNeedle`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

/-- Softmax attention minus a constant `c` is the average of `v j − c`. -/
lemma softmaxAttention_sub_const {ι : Type*} [Fintype ι] [Nonempty ι] (s v : ι → ℝ) (c : ℝ) :
    softmaxAttention s v - c = (∑ j, Real.exp (s j) * (v j - c)) / ∑ j, Real.exp (s j) := by
  have hZ : 0 < ∑ j, Real.exp (s j) := sum_pos (fun j _ => Real.exp_pos _) univ_nonempty
  rw [softmaxAttention, eq_div_iff hZ.ne', sub_mul, div_mul_cancel₀ _ hZ.ne', mul_sum,
    ← sum_sub_distrib]
  exact sum_congr rfl fun j _ => by ring

/-- **Tied needles are averaged** (MRCR, arXiv:2409.12640v2, `eval_framework.tex`): if the
needles `T` share the raw score `z₀` and every other token trails it by `g`, attention at the
scale `s log n` is within `2 C (n − |T|) n^{−s g} / |T|` of the average of the values of `T`, for
values bounded by `C`. -/
theorem abs_logScaleAttention_sub_average_le {ι : Type*} [Fintype ι] [DecidableEq ι]
    {s g C z₀ : ℝ} (hs : 0 ≤ s) {z v : ι → ℝ} (T : Finset ι) (hT : T.Nonempty)
    (htie : ∀ j ∈ T, z j = z₀) (hz : ∀ j ∉ T, z j + g ≤ z₀) (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s z v - (∑ j ∈ T, v j) / T.card| ≤
      2 * C * ((Fintype.card ι : ℝ) - T.card) * (Fintype.card ι : ℝ) ^ (-(s * g)) / T.card := by
  obtain ⟨j₀, hj₀⟩ := hT
  have : Nonempty ι := ⟨j₀⟩
  set n : ℝ := (Fintype.card ι : ℝ) with hn
  have hn1 : (1 : ℝ) ≤ n := by rw [hn]; exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hc : 0 ≤ s * Real.log n := mul_nonneg hs (Real.log_nonneg hn1)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv j₀)
  have hT0 : (0 : ℝ) < T.card := by exact_mod_cast card_pos.2 ⟨j₀, hj₀⟩
  set μ := (∑ j ∈ T, v j) / T.card with hμ
  set x : ι → ℝ := fun j => s * Real.log n * z j with hx
  obtain ⟨a, hadef⟩ : ∃ a, a = Real.exp (s * Real.log n * z₀) := ⟨_, rfl⟩
  have ha : 0 < a := hadef ▸ Real.exp_pos _
  have hrw : logScaleAttention s z v = softmaxAttention x v := rfl
  rw [hrw, softmaxAttention_sub_const, ← sum_add_sum_compl T, ← sum_add_sum_compl T
    (fun j => Real.exp (x j))]
  have hT1 (j : ι) (hj : j ∈ T) : Real.exp (x j) = a := by simp only [hx, htie j hj, hadef]
  have hzero : ∑ j ∈ T, Real.exp (x j) * (v j - μ) = 0 := by
    rw [sum_congr rfl fun j hj => by rw [hT1 j hj], ← mul_sum, sum_sub_distrib, sum_const,
      nsmul_eq_mul, hμ, mul_div_cancel₀ _ hT0.ne', sub_self, mul_zero]
  have hden : T.card * a ≤ ∑ j ∈ T, Real.exp (x j) + ∑ j ∈ Tᶜ, Real.exp (x j) := by
    have h1 : ∑ j ∈ T, Real.exp (x j) = T.card * a := by
      rw [sum_congr rfl hT1, sum_const, nsmul_eq_mul]
    rw [h1]
    linarith [sum_nonneg fun j (_ : j ∈ Tᶜ) => (Real.exp_pos (x j)).le]
  have hterm (j : ι) (hj : j ∈ Tᶜ) :
      |Real.exp (x j) * (v j - μ)| ≤ a * n ^ (-(s * g)) * (2 * C) := by
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
    refine mul_le_mul ?_ ?_ (abs_nonneg _) (by positivity)
    · rw [Real.rpow_def_of_pos (by linarith), hadef, ← Real.exp_add]
      refine Real.exp_le_exp.2 ?_
      have := mul_le_mul_of_nonneg_left (hz j (mem_compl.1 hj)) hc
      simp only [hx]
      linarith
    · have hμC : |μ| ≤ C := by
        rw [hμ, abs_div, abs_of_pos hT0, div_le_iff₀ hT0]
        refine (abs_sum_le_sum_abs _ _).trans ?_
        simpa [mul_comm] using sum_le_sum fun j (_ : j ∈ T) => hv j
      linarith [abs_sub (v j) μ, hv j]
  have hcompl : ((Tᶜ).card : ℝ) = n - T.card := by
    rw [card_compl, Nat.cast_sub (card_le_univ T)]
  have hD : 0 < ∑ j ∈ T, Real.exp (x j) + ∑ j ∈ Tᶜ, Real.exp (x j) := by
    nlinarith [mul_pos hT0 ha]
  rw [hzero, zero_add, abs_div, abs_of_pos hD, div_le_iff₀ hD]
  calc |∑ j ∈ Tᶜ, Real.exp (x j) * (v j - μ)|
      ≤ ∑ j ∈ Tᶜ, a * n ^ (-(s * g)) * (2 * C) := (abs_sum_le_sum_abs _ _).trans
        (sum_le_sum hterm)
    _ = 2 * C * (n - T.card) * n ^ (-(s * g)) / T.card * (T.card * a) := by
        rw [sum_const, nsmul_eq_mul, hcompl]
        field_simp
    _ ≤ _ := mul_le_mul_of_nonneg_left hden (by
        have := Real.rpow_nonneg (by linarith : (0 : ℝ) ≤ n) (-(s * g))
        have : (0 : ℝ) ≤ n - T.card := by rw [← hcompl]; positivity
        positivity)

/-- Two tied needles with values `±1` and a third token trailing by `1`. -/
example := abs_logScaleAttention_sub_average_le (s := 1) (g := 1) (C := 1) (z₀ := 1)
  (z := ![1, 1, 0]) (v := ![1, -1, 1]) zero_le_one {0, 1} ⟨0, by simp⟩
  (fun j hj => by fin_cases j <;> simp_all) (fun j hj => by fin_cases j <;> simp_all)
  fun j => by fin_cases j <;> simp

/-- **ALiBi returns the last needle**: the needles `e` at the strictly decreasing distances `d`
with the raw score `c − b d`, `b ≥ 0`, and every other token trailing the last needle by `g`,
leave attention at the scale `s log n` within `2 C (k n^{−s b} + (n − 1) n^{−s g})` of the value
of the last needle, for values bounded by `C`. The bias `−b d` is that of ALiBi. -/
theorem abs_logScaleAttention_sub_last_le {ι : Type*} [Fintype ι] [DecidableEq ι] {k : ℕ}
    {s b c g C : ℝ} (hs : 0 ≤ s) (hb : 0 ≤ b) {d : Fin (k + 1) → ℕ} (hd : StrictAnti d)
    {e : Fin (k + 1) → ι} (he : Function.Injective e) {z v : ι → ℝ}
    (hz : ∀ j, z (e j) = c - b * d j)
    (hrest : ∀ t, (∀ j, t ≠ e j) → z t + g ≤ c - b * d (Fin.last k)) (hv : ∀ t, |v t| ≤ C) :
    |logScaleAttention s z v - v (e (Fin.last k))| ≤
      2 * C * (k * (Fintype.card ι : ℝ) ^ (-(s * b)) +
        ((Fintype.card ι : ℝ) - 1) * (Fintype.card ι : ℝ) ^ (-(s * g))) := by
  set N := (univ.erase (Fin.last k)).image e
  have h₁ : ∀ t ∈ N, t ≠ e (Fin.last k) → z t + b ≤ z (e (Fin.last k)) := by
    intro t ht _
    obtain ⟨j, hj, rfl⟩ := mem_image.1 ht
    have hlt := hd (lt_of_le_of_ne (Fin.le_last j) (ne_of_mem_erase hj))
    have h1 : (d (Fin.last k) : ℝ) + 1 ≤ d j := by exact_mod_cast hlt
    have := mul_le_mul_of_nonneg_left h1 hb
    rw [hz, hz]
    linarith
  have h₂ : ∀ t ∉ N, t ≠ e (Fin.last k) → z t + g ≤ z (e (Fin.last k)) := by
    intro t ht htl
    rw [hz]
    refine hrest t fun j hj => ?_
    subst hj
    by_cases hjl : j = Fin.last k
    · exact htl (by rw [hjl])
    · exact ht (mem_image.2 ⟨j, mem_erase.2 ⟨hjl, mem_univ _⟩, rfl⟩)
  have h := abs_logScaleAttention_sub_le_two hs N h₁ h₂ hv
  rwa [card_image_of_injective _ he, card_erase_of_mem (mem_univ _), card_univ, Fintype.card_fin,
    Nat.add_sub_cancel] at h

/-- Two needles at the distances `2` and `1` with the slope `b = 1`, and a third token behind. -/
example := abs_logScaleAttention_sub_last_le (k := 1) (s := 1) (b := 1) (c := 0) (g := 2) (C := 1)
  zero_le_one zero_le_one (spaced_strictAnti (D := 1) one_pos) (Fin.castSucc_injective 2)
  (z := ![-2, -1, -3]) (v := ![1, -1, 1])
  (fun j => by fin_cases j <;> norm_num [spaced])
  (fun t ht => by
    fin_cases t
    · exact absurd rfl (ht 0)
    · exact absurd rfl (ht 1)
    · norm_num [spaced])
  fun t => by fin_cases t <;> simp

end Transformer.MRCR
