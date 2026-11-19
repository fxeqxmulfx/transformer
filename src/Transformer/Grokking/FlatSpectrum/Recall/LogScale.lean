module

public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
public import Transformer.Grokking.FlatSpectrum.Recall.SoftmaxScale

/-!
# Softmax attention at the scale `s log n`

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*),
which that section attributes to long-context systems, SSMax among them, on the task of
arXiv:2312.04927v1, §3.

`logScaleAttention s z v` multiplies the raw scores `z j` of `n` tokens by `s log n`, so its
weights are `n^{s z_j} / ∑_k n^{s z_k}`, the weights of that section with `γ = s`
(`logScaleAttention_eq_rpow`). At one length it is softmax attention at the scale `s log n`,
and SA1 to SA4 hold there. Across lengths the scale grows by itself. If one token leads every
other raw score by `g`, the attention is within `2 C (n − 1) n^{−s g}` of its value, for values
bounded by `C` (`abs_logScaleAttention_sub_le`, from SA1). So with `s g > 1` one `s` and one
gap recall at every large number of keys to every accuracy
(`exists_forall_logScaleAttention_recall`): a fixed gap replaces the growth of the scale that
SA3 asks for.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **Softmax attention at the scale `s log n`** (`review/flat_spectrum.md`, SA5, for the scale
`β_n = γ log n` of arXiv:2512.01868v4, §6): over `n` tokens, the raw scores `z j` multiplied by
`s log n`. -/
noncomputable def logScaleAttention {ι : Type*} [Fintype ι] (s : ℝ) (z v : ι → ℝ) : ℝ :=
  softmaxAttention (fun j => s * Real.log (Fintype.card ι) * z j) v

/-- The weights of `logScaleAttention` are `n^{s z_j} / ∑_k n^{s z_k}`, those of
arXiv:2512.01868v4, §6, `A_ij = n^{γ ⟨x_i, x_j⟩} / ∑_k n^{γ ⟨x_i, x_k⟩}`, with `γ = s`. -/
theorem logScaleAttention_eq_rpow {ι : Type*} [Fintype ι] [Nonempty ι] (s : ℝ) (z v : ι → ℝ) :
    logScaleAttention s z v = (∑ j, (Fintype.card ι : ℝ) ^ (s * z j) * v j) /
      ∑ j, (Fintype.card ι : ℝ) ^ (s * z j) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have h (j : ι) : Real.exp (s * Real.log (Fintype.card ι) * z j) =
      (Fintype.card ι : ℝ) ^ (s * z j) := by
    rw [Real.rpow_def_of_pos hn]
    ring_nf
  simp only [logScaleAttention, softmaxAttention, h]

/-- Two tokens with raw scores `1` and `0` at `s = 1`. -/
example := logScaleAttention_eq_rpow (ι := Fin 2) 1 ![1, 0] ![1, -1]

/-- **A leading raw score concentrates the attention at every length**
(`review/flat_spectrum.md`, SA5, from SA1): if the token `j₀` leads every other raw score by
`g`, the attention at the scale `s log n` is within `2 C (n − 1) n^{−s g}` of the value of `j₀`,
for values bounded by `C`. -/
theorem abs_logScaleAttention_sub_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s g C : ℝ}
    (hs : 0 ≤ s) {z v : ι → ℝ} {j₀ : ι} (hz : ∀ j, j ≠ j₀ → z j + g ≤ z j₀)
    (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s z v - v j₀| ≤
      2 * C * (Fintype.card ι - 1) * (Fintype.card ι : ℝ) ^ (-(s * g)) := by
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hn0 : (0 : ℝ) < Fintype.card ι := by linarith
  have hc : 0 ≤ s * Real.log (Fintype.card ι) := mul_nonneg hs (Real.log_nonneg hn)
  calc |logScaleAttention s z v - v j₀|
      ≤ 2 * C * (Fintype.card ι - 1) * Real.exp (-(s * Real.log (Fintype.card ι) * g)) :=
        abs_softmaxAttention_sub_le
          (fun j hj => by linarith [mul_le_mul_of_nonneg_left (hz j hj) hc]) hv
    _ = _ := by
        rw [Real.rpow_def_of_pos hn0]
        ring_nf

/-- Two tokens with raw scores `1` and `0`, a gap `1`, at `s = 1`. -/
example := abs_logScaleAttention_sub_le (s := 1) (z := (![1, 0] : Fin 2 → ℝ)) (v := ![1, -1])
  (j₀ := 0) (g := 1) (C := 1) zero_le_one (fun j hj => by fin_cases j <;> simp_all)
  fun j => by fin_cases j <;> simp

/-- **Above the threshold one scale recalls at every length** (`review/flat_spectrum.md`, SA5):
if over every number `m + 1` of keys the queried key leads every other raw score by `g`, with
`s g > 1`, the attention at the scale `s log (m + 1)` recalls values bounded by `C` to every
accuracy `ε > 0` from some number of keys on. -/
theorem exists_forall_logScaleAttention_recall {ν : Type*} {s g C : ℝ} (hs : 0 ≤ s)
    (hsg : 1 < s * g) (z : (m : ℕ) → Fin (m + 1) → Fin (m + 1) → ℝ)
    (hz : ∀ m q k, k ≠ q → z m q k + g ≤ z m q q) {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |logScaleAttention s (z m q) (fun k => φ (π k)) - φ (π q)| ≤ ε := by
  have ht : Tendsto (fun m : ℕ => 2 * C * ((m : ℝ) + 1) ^ (1 - s * g)) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < s * g - 1)).comp
      (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)).const_mul (2 * C)
    rw [mul_zero] at h
    exact Tendsto.congr (fun m => by simp only [Function.comp_apply, neg_sub]) h
  obtain ⟨M, hM⟩ := eventually_atTop.1 (ht.eventually (ge_mem_nhds hε))
  refine ⟨M, fun m hm q π => ?_⟩
  have hx : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have h := abs_logScaleAttention_sub_le hs (hz m q) fun k => hφ (π k)
  have hcard : (Fintype.card (Fin (m + 1)) : ℝ) = (m : ℝ) + 1 := by simp
  rw [hcard] at h
  calc _ ≤ _ := h
    _ ≤ 2 * C * ((m : ℝ) + 1) * ((m : ℝ) + 1) ^ (-(s * g)) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) (by linarith))
          (Real.rpow_nonneg hx.le _)
    _ = 2 * C * ((m : ℝ) + 1) ^ (1 - s * g) := by
        rw [sub_eq_add_neg, Real.rpow_add hx, Real.rpow_one]
        ring
    _ ≤ ε := hM m hm

/-- One-hot raw scores `±1` lead by `2`; at `s = 1` they recall values `±1` to accuracy `1/10`
from some number of keys on. -/
example := exists_forall_logScaleAttention_recall (s := 1) (g := 2) zero_le_one (by norm_num)
  (fun _ q k => if k = q then (1 : ℝ) else -1)
  (fun _ q k hk => by
    show (if k = q then (1 : ℝ) else -1) + 2 ≤ if q = q then (1 : ℝ) else -1
    norm_num [hk])
  (φ := fun b : Bool => if b then (1 : ℝ) else -1) (C := 1) (fun b => by cases b <;> simp)
  (ε := 1 / 10) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
