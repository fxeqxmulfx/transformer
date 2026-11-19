module

public import Transformer.Grokking.FlatSpectrum.Recall.Threshold

/-!
# Near misses: keys that share part of the queried key

Source: MRCR-V2 (8-needle) of the Gemini 2.5 report, arXiv:2507.06261v6, Appendix,
`8_appendix/1_evaluation_additional_details.tex`, line 75: the dictionary is nested to depth 3,
a topic, a format and a style ("write a poem about penguins in an archaic style"), where MRCR of
arXiv:2409.12640v2, `eval_framework.tex`, nests it to depth 2, a topic and a format. The
attention is that of SSMax, arXiv:2501.19399v1, §2.2: raw scores multiplied by `s log n` over
`n` tokens (`Transformer.Grokking.FlatSpectrum.Recall.logScaleAttention`).

Item SA5 of `review/flat_spectrum.md` gives every other key one gap `g` and recalls at every
length once `s g > 1`. A distractor that shares two of the three parts of the queried key is a
*near miss*: its raw score may trail by less than the others. Here each key has its own gap. The
miss is at most `2 C ∑_{j ≠ j₀} n^{−s g_j}` for values bounded by `C`
(`abs_logScaleAttention_sub_le_sum`); with near misses `N` trailing by `g₁` and the other keys by
`g₂`, at most `2 C (|N| n^{−s g₁} + (n − 1) n^{−s g₂})` (`abs_logScaleAttention_sub_le_two`).
`Transformer.MRCR.NearMissThreshold` turns the bound into a threshold, and
`Transformer.MRCR.PartScore` applies it to keys of several parts.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

/-- **Every key its own gap**: if the key `j₀` leads the raw score of each other key `j` by `g j`,
attention at the scale `s log n` misses the value of `j₀` by at most `2 C ∑_{j ≠ j₀} n^{−s g_j}`,
for values bounded by `C`. -/
theorem abs_logScaleAttention_sub_le_sum {ι : Type*} [Fintype ι] [DecidableEq ι] {s C : ℝ}
    (hs : 0 ≤ s) {z v g : ι → ℝ} {j₀ : ι} (hz : ∀ j, j ≠ j₀ → z j + g j ≤ z j₀)
    (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s z v - v j₀| ≤
      2 * C * ∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (-(s * g j)) := by
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hn0 : (0 : ℝ) < Fintype.card ι := by linarith
  have hc : 0 ≤ s * Real.log (Fintype.card ι) := mul_nonneg hs (Real.log_nonneg hn)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv j₀)
  set x : ι → ℝ := fun j => s * Real.log (Fintype.card ι) * z j with hx
  have hZ0 : Real.exp (x j₀) ≤ ∑ j, Real.exp (x j) :=
    single_le_sum (f := fun j => Real.exp (x j)) (fun j _ => (Real.exp_pos _).le) (mem_univ j₀)
  have hE := Real.exp_pos (x j₀)
  have h := abs_sub_softmaxAttention_le (s := x) hv j₀
  rw [abs_sub_comm] at h
  refine h.trans (mul_le_mul_of_nonneg_left ?_ (by linarith))
  rw [div_le_iff₀ (hE.trans_le hZ0)]
  have hterm (j : ι) (hj : j ∈ univ.erase j₀) :
      Real.exp (x j) ≤ (Fintype.card ι : ℝ) ^ (-(s * g j)) * Real.exp (x j₀) := by
    rw [Real.rpow_def_of_pos hn0, ← Real.exp_add]
    refine Real.exp_le_exp.2 ?_
    have := mul_le_mul_of_nonneg_left (hz j (ne_of_mem_erase hj)) hc
    simp only [hx]
    linarith
  calc ∑ j ∈ univ.erase j₀, Real.exp (x j)
      ≤ ∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (-(s * g j)) * Real.exp (x j₀) :=
        sum_le_sum hterm
    _ = (∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (-(s * g j))) * Real.exp (x j₀) :=
        (sum_mul _ _ _).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left hZ0 (sum_nonneg fun j _ => Real.rpow_nonneg hn0.le _)

/-- Two tokens with raw scores `1` and `0`, the second trailing by `1`, at `s = 1`. -/
example := abs_logScaleAttention_sub_le_sum (s := 1) (z := (![1, 0] : Fin 2 → ℝ))
  (v := ![1, -1]) (g := fun _ => 1) (j₀ := 0) (C := 1) zero_le_one
  (fun j hj => by fin_cases j <;> simp_all) fun j => by fin_cases j <;> simp

/-- **Near misses and the rest**: near misses `N` trailing the key `j₀` by `g₁` and the other
keys by `g₂` leave a miss of at most `2 C (|N| n^{−s g₁} + (n − 1) n^{−s g₂})`. -/
theorem abs_logScaleAttention_sub_le_two {ι : Type*} [Fintype ι] [DecidableEq ι]
    {s C g₁ g₂ : ℝ} (hs : 0 ≤ s) {z v : ι → ℝ} {j₀ : ι} (N : Finset ι)
    (h₁ : ∀ j ∈ N, j ≠ j₀ → z j + g₁ ≤ z j₀) (h₂ : ∀ j ∉ N, j ≠ j₀ → z j + g₂ ≤ z j₀)
    (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s z v - v j₀| ≤
      2 * C * (N.card * (Fintype.card ι : ℝ) ^ (-(s * g₁)) +
        ((Fintype.card ι : ℝ) - 1) * (Fintype.card ι : ℝ) ^ (-(s * g₂))) := by
  classical
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv j₀)
  have h := abs_logScaleAttention_sub_le_sum hs (g := fun j => if j ∈ N then g₁ else g₂)
    (fun j hj => by split_ifs with hN; exacts [h₁ j hN hj, h₂ j hN hj]) hv
  refine h.trans (mul_le_mul_of_nonneg_left ?_ (by linarith))
  set n : ℝ := (Fintype.card ι : ℝ)
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have ha := Real.rpow_nonneg hn0 (-(s * g₁))
  have hb := Real.rpow_nonneg hn0 (-(s * g₂))
  have hpt (j : ι) : n ^ (-(s * if j ∈ N then g₁ else g₂)) ≤
      (if j ∈ N then n ^ (-(s * g₁)) else 0) + n ^ (-(s * g₂)) := by
    split_ifs <;> linarith
  calc ∑ j ∈ univ.erase j₀, n ^ (-(s * if j ∈ N then g₁ else g₂))
      ≤ ∑ j ∈ univ.erase j₀, ((if j ∈ N then n ^ (-(s * g₁)) else 0) + n ^ (-(s * g₂))) :=
        sum_le_sum fun j _ => hpt j
    _ = ((univ.erase j₀).filter (· ∈ N)).card * n ^ (-(s * g₁)) +
          (univ.erase j₀).card * n ^ (-(s * g₂)) := by
        rw [sum_add_distrib, sum_ite, sum_const_zero, add_zero, sum_const, sum_const,
          nsmul_eq_mul, nsmul_eq_mul]
    _ ≤ _ := by
        have hf : (((univ.erase j₀).filter (· ∈ N)).card : ℝ) ≤ N.card := by
          exact_mod_cast card_le_card fun j hj => (mem_filter.1 hj).2
        have he : ((univ.erase j₀).card : ℝ) = n - 1 := by
          rw [card_erase_of_mem (mem_univ _), card_univ, Nat.cast_sub
            (Fintype.card_pos_iff.2 ⟨j₀⟩), Nat.cast_one]
        rw [he]
        nlinarith

/-- One near miss at the key `0`, trailing by `1/2`, and the rest trailing by `2`. -/
example := abs_logScaleAttention_sub_le_two (s := 1) (z := (![2, 3 / 2, 0] : Fin 3 → ℝ))
  (v := ![1, -1, -1]) (j₀ := 0) (C := 1) (g₁ := 1 / 2) (g₂ := 2) zero_le_one {1}
  (fun j hj _ => by obtain rfl := mem_singleton.1 hj; norm_num)
  (fun j hj hj₀ => by fin_cases j <;> simp_all) fun j => by fin_cases j <;> simp


end Transformer.MRCR
