module

public import Transformer.Grokking.FlatSpectrum.Recall.CausalCopy

/-!
# The causal induction head fades at every key it copies too little

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.CausalCopy` computes the causal layer 1 over `b` tokens without a key and then
`k₀ v₀ … k_m v_m`: the value token of `q` copies its key with the weight
`w_q = L^{s₁} / (L^{s₁} + L − 1)`, `L = b + 2q + 2`, and every copy is at least `0`. Layer 2 at
the query `q` reads the `n = b + 2m + 2` tokens at the scale `s₂ log n`, with the copies of `q` as
raw scores (`causalInductionHead`). When the value of the target is `1` and the other values are
at most `0`, attention misses the target by at least the weight of the other tokens
(`sum_erase_div_le_one_sub_logScaleAttention`). Here the `n − 1` other tokens have raw scores at
least `0`, so the indicator dictionary of values `±1` is answered at `q` with an error at least

    (n − 1) / (n^{s₂ w_q} + n − 1)        (`le_one_sub_causalInductionHead`)

at every `m`, `q`, `s₁` and `s₂ ≥ 0`. So the head fades at every key with `s₂ w_q < 1`: for every
`ε < 1`, from some number of keys on, every key with `s₂ w_q ≤ σ < 1` has a dictionary answered
with an error above `ε` (`exists_lt_causalInductionHead_error`). The copy `w_q` is set by the
position of the key, not by the length of the context, so the condition `s₂ w_q > 1` of the
cyclic head now holds or fails key by key. Below `s₁ = 1` the copies of the late keys tend to `0`,
so at every `s₂` the head fades at every key from some key on
(`exists_lt_causalInductionHead_error_of_lt_one`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- The position of a token of the causal context: the `b` tokens without a key first, then the
key token `(i, false)` and the value token `(i, true)` of each key `i`. -/
def causalPos (b : ℕ) {m : ℕ} : Fin b ⊕ Fin (m + 1) × Bool → ℕ
  | .inl j => j
  | .inr (i, false) => b + 2 * i
  | .inr (i, true) => b + 2 * i + 1

/-- The causal context has `n = b + 2m + 2` tokens. -/
lemma card_causalTokens (b m : ℕ) :
    (Fintype.card (Fin b ⊕ Fin (m + 1) × Bool) : ℝ) = b + 2 * m + 2 := by
  simp only [Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_bool]
  push_cast
  ring

/-- **The causal induction head**, layer 2 on the causal layer 1 (`review/flat_spectrum.md`,
SA5): at the query `q`, attention over the `n = b + 2m + 2` tokens of the context at the scale
`s₂ log n`, with the raw score the copy of `q` at each token (`causalCopy`), reading the value
`φ (π i)` at the value token of `i` and `0` elsewhere. -/
noncomputable def causalInductionHead (s₁ s₂ : ℝ) {ν : Type*} (φ : ν → ℝ) (b : ℕ) {m : ℕ}
    (π : Fin (m + 1) → ν) (q : Fin (m + 1)) : ℝ :=
  logScaleAttention s₂ (fun t => causalCopy s₁ b (causalPos b t) q)
    (Sum.elim (fun _ => 0) fun t => if t.2 then φ (π t.1) else 0)

/-- **The other tokens bound the miss from below** (`review/flat_spectrum.md`, SA5): if the value
of `j₀` is `1` and the other values are at most `0`, attention at the scale `s log n` misses the
value of `j₀` by at least the weight of the other tokens. -/
theorem sum_erase_div_le_one_sub_logScaleAttention {ι : Type*} [Fintype ι] [DecidableEq ι]
    (s : ℝ) {z v : ι → ℝ} {j₀ : ι} (hv₀ : v j₀ = 1) (hv : ∀ j, j ≠ j₀ → v j ≤ 0) :
    (∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (s * z j)) /
      ∑ j, (Fintype.card ι : ℝ) ^ (s * z j) ≤ 1 - logScaleAttention s z v := by
  have : Nonempty ι := ⟨j₀⟩
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hZ : 0 < ∑ j, (Fintype.card ι : ℝ) ^ (s * z j) :=
    sum_pos (fun j _ => Real.rpow_pos_of_pos hn _) ⟨j₀, mem_univ _⟩
  have hsplit : (Fintype.card ι : ℝ) ^ (s * z j₀) + ∑ j ∈ univ.erase j₀,
      (Fintype.card ι : ℝ) ^ (s * z j) = ∑ j, (Fintype.card ι : ℝ) ^ (s * z j) :=
    add_sum_erase univ (fun j => (Fintype.card ι : ℝ) ^ (s * z j)) (mem_univ j₀)
  have hN : ∑ j, (Fintype.card ι : ℝ) ^ (s * z j) * v j ≤ (Fintype.card ι : ℝ) ^ (s * z j₀) := by
    rw [← add_sum_erase _ _ (mem_univ j₀), hv₀, mul_one, add_le_iff_nonpos_right]
    exact sum_nonpos fun j hj =>
      mul_nonpos_of_nonneg_of_nonpos (Real.rpow_nonneg hn.le _) (hv j (ne_of_mem_erase hj))
  rw [logScaleAttention_eq_rpow, le_sub_iff_add_le, ← add_div, div_le_one hZ]
  linarith

/-- Two tokens with raw scores `1` and `0` and values `1` and `−1`. -/
example := sum_erase_div_le_one_sub_logScaleAttention 1 (z := ![1, 0]) (v := ![1, -1]) (j₀ := 0)
  rfl fun j hj => by fin_cases j <;> simp_all

/-- **Other raw scores at least `0`** (`review/flat_spectrum.md`, SA5): then at `s ≥ 0` the miss
is at least `(n − 1) / (n^{s z_{j₀}} + n − 1)`. -/
theorem le_one_sub_logScaleAttention {ι : Type*} [Fintype ι] [DecidableEq ι] {s : ℝ}
    (hs : 0 ≤ s) {z v : ι → ℝ} {j₀ : ι} (hz : ∀ j, j ≠ j₀ → 0 ≤ z j) (hv₀ : v j₀ = 1)
    (hv : ∀ j, j ≠ j₀ → v j ≤ 0) :
    ((Fintype.card ι : ℝ) - 1) / ((Fintype.card ι : ℝ) ^ (s * z j₀) + Fintype.card ι - 1) ≤
      1 - logScaleAttention s z v := by
  refine le_trans ?_ (sum_erase_div_le_one_sub_logScaleAttention s hv₀ hv)
  have : Nonempty ι := ⟨j₀⟩
  have hn1 : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hR : (Fintype.card ι : ℝ) - 1 ≤ ∑ j ∈ univ.erase j₀, (Fintype.card ι : ℝ) ^ (s * z j) :=
    calc (Fintype.card ι : ℝ) - 1 = ∑ j ∈ univ.erase j₀, (1 : ℝ) := by
          rw [sum_const, card_erase_of_mem (mem_univ _), card_univ, nsmul_eq_mul, mul_one,
            Nat.cast_sub Fintype.card_pos, Nat.cast_one]
      _ ≤ _ := sum_le_sum fun j hj =>
          Real.one_le_rpow hn1 (mul_nonneg hs (hz j (ne_of_mem_erase hj)))
  have hW : 0 < (Fintype.card ι : ℝ) ^ (s * z j₀) := Real.rpow_pos_of_pos (by linarith) _
  rw [← add_sum_erase _ _ (mem_univ j₀), div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_le_mul_of_nonneg_left hR hW.le]

/-- Two tokens with raw scores `1` and `0` and values `1` and `−1`. -/
example := le_one_sub_logScaleAttention zero_le_one (z := ![1, 0]) (v := ![1, -1]) (j₀ := 0)
  (fun j _ => by fin_cases j <;> simp) rfl fun j hj => by fin_cases j <;> simp_all

/-- **The causal head misses a key it copies little** (`review/flat_spectrum.md`, SA5): over
the `n = b + 2m + 2` tokens, the indicator dictionary of values `±1` is answered at the query `q`
with an error at least `(n − 1) / (n^{s₂ w} + n − 1)`, where `w = copyWeight s₁ (b + 2q + 2)` is
the copy of `q` at its value token. -/
theorem le_one_sub_causalInductionHead (s₁ : ℝ) {s₂ : ℝ} (hs₂ : 0 ≤ s₂) (b : ℕ) {m : ℕ}
    (q : Fin (m + 1)) :
    (b + 2 * m + 1 : ℝ) / ((b + 2 * m + 2 : ℝ) ^ (s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2)) +
        b + 2 * m + 1) ≤
      1 - causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b (fun i => decide (i = q))
        q := by
  have h := le_one_sub_logScaleAttention hs₂ (z := fun t => causalCopy s₁ b (causalPos b t) q)
    (j₀ := .inr (q, true)) (fun t _ => causalCopy_nonneg _ _ _ _)
    (v := Sum.elim (fun _ => 0) fun t => if t.2 then (if decide (t.1 = q) then 1 else -1) else 0)
    (by simp) fun t ht => by
      rcases t with j | ⟨i, _ | _⟩
      · simp
      · simp
      · have hi : i ≠ q := fun hi => ht (by rw [hi])
        simp [hi]
  rw [card_causalTokens] at h
  refine le_trans (le_of_eq ?_) h
  simp only [causalPos]
  rw [causalCopy_value]
  ring_nf

/-- The first of two keys after a beginning-of-sequence token, at `s₁ = 2` and `s₂ = 1`. -/
example := le_one_sub_causalInductionHead 2 zero_le_one 1 (0 : Fin 2)

/-- **A key copied below `1 / s₂` fades** (`review/flat_spectrum.md`, SA5): if `s₂` times the
copy of a key is at most `σ < 1`, then for every `ε < 1`, from some number of keys on, every such
key has a dictionary of values `±1` answered with an error above `ε`. -/
theorem exists_lt_causalInductionHead_error {s₂ σ ε : ℝ} (hs₂ : 0 ≤ s₂) (hσ : σ < 1)
    (hε : ε < 1) (s₁ : ℝ) (b : ℕ) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) ≤ σ →
      ∃ π : Fin (m + 1) → Bool, ε < |(if π q then (1 : ℝ) else -1) -
        causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b π q| := by
  have hlim := (tendsto_rpow_neg_atTop (by linarith : 0 < 1 - σ)).comp (tendsto_valuePrefix b)
  obtain ⟨M, hM⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < (1 - ε) / 2)))
  refine ⟨M, fun m hm q hq => ⟨fun i => decide (i = q), ?_⟩⟩
  have h := le_one_sub_causalInductionHead s₁ hs₂ b q
  have hX := hM m hm
  simp only [Function.comp_apply] at hX
  simp only [decide_true, ite_true]
  have hb0 : (0 : ℝ) ≤ b := Nat.cast_nonneg b
  have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  have hn0 : (0 : ℝ) < b + 2 * m + 2 := by linarith
  rw [show -(1 - σ) = σ - 1 by ring, Real.rpow_sub_one hn0.ne', div_lt_iff₀ hn0] at hX
  have hD := Real.rpow_le_rpow_of_exponent_le (by linarith : (1 : ℝ) ≤ b + 2 * m + 2) hq
  have hlt : ε < (b + 2 * m + 1 : ℝ) / ((b + 2 * m + 2 : ℝ) ^ σ + b + 2 * m + 1) := by
    have hσ0 := Real.rpow_pos_of_pos hn0 σ
    rw [lt_div_iff₀ (by positivity)]
    rcases le_or_gt ε 0 with hε0 | hε0
    · nlinarith [mul_nonpos_of_nonpos_of_nonneg hε0 (by positivity :
        (0 : ℝ) ≤ (b + 2 * m + 2 : ℝ) ^ σ + b + 2 * m + 1)]
    · have h4 : 0 ≤ (1 - ε) * ((b + 2 * m + 1 : ℝ) - ε * (b + 2 * m + 2) / 2) :=
        mul_nonneg (by linarith) (by nlinarith)
      nlinarith [mul_lt_mul_of_pos_left hX hε0]
  refine hlt.trans_le (le_trans ?_ (h.trans (le_abs_self _)))
  exact div_le_div_of_nonneg_left (by positivity) (by positivity) (by linarith)

/-- At `σ = 1/2` and `ε = 1/2` after a beginning-of-sequence token. -/
example := exists_lt_causalInductionHead_error (s₂ := 1) (σ := 1 / 2) (ε := 1 / 2) zero_le_one
  (by norm_num) (by norm_num) 2 1

/-- **Below `s₁ = 1` the late keys fade at every `s₂`** (`review/flat_spectrum.md`, SA5): for
every `ε < 1` there is a key from which on, from some number of keys on, every key has a
dictionary of values `±1` answered with an error above `ε`. -/
theorem exists_lt_causalInductionHead_error_of_lt_one {s₁ s₂ ε : ℝ} (hs₁ : s₁ < 1)
    (hs₂ : 0 ≤ s₂) (hε : ε < 1) (b : ℕ) :
    ∃ Q M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), Q ≤ (q : ℕ) → ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b π q| := by
  have h := (tendsto_causalCopy_value_of_lt hs₁ b).const_mul s₂
  rw [mul_zero] at h
  obtain ⟨Q, hQ⟩ := eventually_atTop.1 (h.eventually (ge_mem_nhds (by norm_num : (0 : ℝ) < 1 / 2)))
  obtain ⟨M, hM⟩ := exists_lt_causalInductionHead_error hs₂ (by norm_num : (1 / 2 : ℝ) < 1) hε s₁ b
  refine ⟨Q, M, fun m hm q hq => hM m hm q ?_⟩
  have h' := hQ q hq
  rwa [causalCopy_value] at h'

/-- At `s₁ = 1/2`, `s₂ = 100` and `ε = 1/2` after a beginning-of-sequence token. -/
example := exists_lt_causalInductionHead_error_of_lt_one (s₁ := 1 / 2) (s₂ := 100) (ε := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) 1

end Transformer.Grokking.FlatSpectrum.Recall
