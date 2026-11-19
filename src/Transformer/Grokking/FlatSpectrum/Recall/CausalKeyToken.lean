module

public import Transformer.Grokking.FlatSpectrum.Recall.CausalRecall

/-!
# Without a beginning-of-sequence token the first key is lost

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.CausalRecall` recalls a key `q` with `s₂ w > 1` whose key token copies it with less than
`w = copyWeight s₁ (b + 2q + 2)`, its copy at the value token. The second condition is needed.
If the key token copies `q` with `r > w`, attention at the scale `s₂ log n` goes to the key
token, whose value is `0`: the weight of one other token bounds the miss from below
(`rpow_div_le_one_sub_logScaleAttention`), and the indicator dictionary is answered at `q` with an
error at least

    1 / (n^{−s₂ (r − w)} + 1)        (`le_one_sub_causalInductionHead_of_key`),

which tends to `1` at every `s₂ > 0` (`exists_lt_causalInductionHead_error_of_key`). This happens
to the first key without a beginning-of-sequence token: its key token reads only itself and
copies the key with `1`, more than the `2^{s₁} / (2^{s₁} + 1)` of the value token, so the first
key is lost at every `s₁` and every `s₂ > 0` (`exists_lt_causalInductionHead_error_first`).
After one beginning-of-sequence token the key token of the first key reads that token instead,
and copies the key with `1 / (2^{s₁} + 1)`, less than the `3^{s₁} / (3^{s₁} + 2)` of the value
token if and only if `6^{s₁} > 2` (`causalCopy_key_lt_value_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **One other token bounds the miss from below** (`review/flat_spectrum.md`, SA5): if the
value of `j₀` is `1` and the other values are at most `0`, attention at the scale `s log n`
misses the value of `j₀` by at least the weight of any other token `j₁` against `j₀` alone. -/
theorem rpow_div_le_one_sub_logScaleAttention {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ℝ)
    {z v : ι → ℝ} {j₀ j₁ : ι} (hj : j₁ ≠ j₀) (hv₀ : v j₀ = 1) (hv : ∀ j, j ≠ j₀ → v j ≤ 0) :
    (Fintype.card ι : ℝ) ^ (s * z j₁) /
      ((Fintype.card ι : ℝ) ^ (s * z j₀) + (Fintype.card ι : ℝ) ^ (s * z j₁)) ≤
      1 - logScaleAttention s z v := by
  refine le_trans ?_ (sum_erase_div_le_one_sub_logScaleAttention s hv₀ hv)
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hx : (Fintype.card ι : ℝ) ^ (s * z j₁) ≤
      ∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (s * z j) :=
    single_le_sum (f := fun j => (Fintype.card ι : ℝ) ^ (s * z j))
      (fun j _ => (Real.rpow_pos_of_pos hn _).le) (mem_erase.2 ⟨hj, mem_univ _⟩)
  have ha := Real.rpow_pos_of_pos hn (s * z j₀)
  have hb := Real.rpow_pos_of_pos hn (s * z j₁)
  rw [← add_sum_erase _ _ (mem_univ j₀), div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- Two tokens with raw scores `0` and `1` and values `1` and `−1`. -/
example := rpow_div_le_one_sub_logScaleAttention 1 (z := ![0, 1]) (v := ![1, -1]) (j₀ := 0)
  (j₁ := 1) (by decide) rfl fun j hj => by fin_cases j <;> simp_all

/-- **The key token draws attention from the value token** (`review/flat_spectrum.md`, SA5): over
the `n = b + 2m + 2` tokens, the indicator dictionary of values `±1` is answered at the query `q`
with an error at least `1 / (n^{−s₂ (r − w)} + 1)`, where `r` is the copy of `q` at its key token
and `w = copyWeight s₁ (b + 2q + 2)` at its value token. -/
theorem le_one_sub_causalInductionHead_of_key (s₁ s₂ : ℝ) (b : ℕ) {m : ℕ} (q : Fin (m + 1)) :
    1 / ((b + 2 * m + 2 : ℝ) ^ (-(s₂ * (causalCopy s₁ b (b + 2 * (q : ℕ)) q -
        copyWeight s₁ (b + 2 * (q : ℕ) + 2)))) + 1) ≤
      1 - causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b (fun i => decide (i = q))
        q := by
  have h := rpow_div_le_one_sub_logScaleAttention s₂
    (z := fun t => causalCopy s₁ b (causalPos b t) q) (j₀ := .inr (q, true))
    (j₁ := .inr (q, false)) (by simp)
    (v := Sum.elim (fun _ => 0) fun t => if t.2 then (if decide (t.1 = q) then 1 else -1) else 0)
    (by simp) fun t ht => by
      rcases t with j | ⟨i, _ | _⟩
      · simp
      · simp
      · have hi : i ≠ q := fun hi => ht (by rw [hi])
        simp [hi]
  rw [card_causalTokens] at h
  simp only [causalPos, causalCopy_value] at h
  refine le_trans (le_of_eq ?_) h
  have hn : (0 : ℝ) < b + 2 * m + 2 := by positivity
  rw [show -(s₂ * (causalCopy s₁ b (b + 2 * (q : ℕ)) q - copyWeight s₁ (b + 2 * (q : ℕ) + 2))) =
      s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) - s₂ * causalCopy s₁ b (b + 2 * (q : ℕ)) q by ring,
    Real.rpow_sub hn, div_add_one (Real.rpow_pos_of_pos hn _).ne', one_div_div]

/-- **A key that its key token copies more is lost** (`review/flat_spectrum.md`, SA5): if the
key token of `q` copies it more than the value token does, then at every `s₂ > 0`, for every
`ε < 1`, from some number of keys on, a dictionary of values `±1` is answered at `q` with an
error above `ε`. -/
theorem exists_lt_causalInductionHead_error_of_key {s₁ s₂ ε : ℝ} (hs₂ : 0 < s₂) (hε : ε < 1)
    (b q : ℕ) (hr : copyWeight s₁ (b + 2 * q + 2) < causalCopy s₁ b (b + 2 * q) q) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ k : Fin (m + 1), (k : ℕ) = q → ∃ π : Fin (m + 1) → Bool,
      ε < |(if π k then (1 : ℝ) else -1) -
        causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b π k| := by
  have hlim := (tendsto_rpow_neg_atTop (mul_pos hs₂ (sub_pos.2 hr))).comp (tendsto_valuePrefix b)
  obtain ⟨M, hM⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < 1 - ε)))
  refine ⟨M, fun m hm k hk => ⟨fun i => decide (i = k), ?_⟩⟩
  have h := le_one_sub_causalInductionHead_of_key s₁ s₂ b k
  rw [hk] at h
  have hX := hM m hm
  simp only [Function.comp_apply] at hX
  simp only [decide_true, ite_true]
  refine lt_of_lt_of_le ?_ (h.trans (le_abs_self _))
  have hx0 := Real.rpow_nonneg (by positivity : (0 : ℝ) ≤ b + 2 * m + 2)
    (-(s₂ * (causalCopy s₁ b (b + 2 * q) q - copyWeight s₁ (b + 2 * q + 2))))
  generalize (b + 2 * m + 2 : ℝ) ^ (-(s₂ * (causalCopy s₁ b (b + 2 * q) q -
    copyWeight s₁ (b + 2 * q + 2)))) = x at hX hx0 ⊢
  rw [lt_div_iff₀ (by linarith)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith
  · nlinarith [mul_pos h0 (by linarith : 0 < 1 - ε - x), sq_nonneg (1 - ε)]

/-- The first key without a beginning-of-sequence token, at `s₁ = s₂ = 1` and `ε = 1/2`. -/
example := exists_lt_causalInductionHead_error_of_key (s₁ := 1) (s₂ := 1) (ε := 1 / 2)
  one_pos (by norm_num) 0 0 (by rw [causalCopy_key, copyWeight]; norm_num)

/-- **Without a beginning-of-sequence token the first key is lost** (`review/flat_spectrum.md`,
SA5): the key token of the first key reads only itself and copies the key with `1`, more than
the value token does, so at every `s₁` and every `s₂ > 0`, for every `ε < 1`, from some number
of keys on, a dictionary of values `±1` is answered at the first key with an error above `ε`. -/
theorem exists_lt_causalInductionHead_error_first (s₁ : ℝ) {s₂ ε : ℝ} (hs₂ : 0 < s₂)
    (hε : ε < 1) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ k : Fin (m + 1), (k : ℕ) = 0 → ∃ π : Fin (m + 1) → Bool,
      ε < |(if π k then (1 : ℝ) else -1) -
        causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) 0 π k| := by
  refine exists_lt_causalInductionHead_error_of_key hs₂ hε 0 0 ?_
  have h2 := Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) s₁
  rw [causalCopy_key, copyWeight, div_lt_iff₀ (by norm_num; linarith)]
  norm_num
  linarith

/-- At `s₁ = 2`, `s₂ = 100` and `ε = 9/10`. -/
example := exists_lt_causalInductionHead_error_first 2 (s₂ := 100) (ε := 9 / 10) (by norm_num)
  (by norm_num)

/-- **After one beginning-of-sequence token** (`review/flat_spectrum.md`, SA5): the key token of
the first key copies it with `1 / (2^{s} + 1)`, less than the `3^{s} / (3^{s} + 2)` of the value
token if and only if `6^{s} > 2`. -/
theorem causalCopy_key_lt_value_iff (s : ℝ) :
    causalCopy s 1 1 0 < copyWeight s 3 ↔ 2 < (6 : ℝ) ^ s := by
  have hk : causalCopy s 1 1 0 = 1 / ((2 : ℝ) ^ s + 1) := by
    rw [show causalCopy s 1 1 0 = causalCopy s 1 (1 + 2 * 0) 0 from rfl, causalCopy_key]
    norm_num
  have h6 : (6 : ℝ) ^ s = 2 ^ s * 3 ^ s := by
    rw [← Real.mul_rpow (by norm_num) (by norm_num)]
    norm_num
  have h2 := Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) s
  have h3 := Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 3) s
  rw [hk, copyWeight, h6, div_lt_div_iff₀ (by linarith) (by linarith)]
  constructor <;> intro h <;> nlinarith

end Transformer.Grokking.FlatSpectrum.Recall
