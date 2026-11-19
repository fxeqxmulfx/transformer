module

public import Transformer.Grokking.FlatSpectrum.Recall.Induction

/-!
# The induction head in closed form: one layer at the scale `s₂ a`

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.Induction` bounds the induction head through the error of its first layer. Layer 1 has
one-hot positional raw scores and copies one-hot key channels, so both layers can be computed
exactly. Raw scores `a` at one token and `b` at the other `n − 1` tokens give that token the
weight `n^{s (a − b)}` against `1` for each other token (`logScaleAttention_ite`). Over the
`n = 2m + 2` tokens of `k₀ v₀ … k_m v_m`:

* layer 1 copies the key `q` to the value token of `q` with the weight
  `n^{s₁} / (n^{s₁} + n − 1)`, and to every other token with `1 / (n^{s₁} + n − 1)`
  (`prevTokenHead_eq`);
* so layer 2 has two-valued raw scores with the lead `a = (n^{s₁} − 1) / (n^{s₁} + n − 1)`
  (`copyLead`), and the induction head is one-hot attention at the scale `s₂ a`
  (`inductionHead_eq_logScaleAttention`).

The error is `(n φ(π q) − ∑ᵢ φ(π i)) / (n^{s₂ a} + n − 1)` (`sub_inductionHead_eq`), which is
at most `(3m + 1) C / (n^{s₂ a} + 2m + 1)` for values bounded by `C`
(`abs_sub_inductionHead_le`). Over the dictionaries of values `±1` this bound is the greatest
error, attained by the indicator dictionary (`isGreatest_inductionHead_error`). Key tokens carry
the value `0`, so the bound tends to `3/2`, not to `2`, if `n^{s₂ a}` grows slower than `n`.
`Recall.InductionPhases` takes the limits in `n`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- **Two-valued raw scores** (`review/flat_spectrum.md`, SA5): with the raw score `a` at the
token `j₀` and `b` at the other `n − 1` tokens, attention at the scale `s log n` gives `j₀` the
weight `n^{s (a − b)}` against `1` for each other token. -/
theorem logScaleAttention_ite {ι : Type*} [Fintype ι] [DecidableEq ι] (s a b : ℝ) (v : ι → ℝ)
    (j₀ : ι) :
    logScaleAttention s (fun j => if j = j₀ then a else b) v =
      (((Fintype.card ι : ℝ) ^ (s * (a - b)) - 1) * v j₀ + ∑ j, v j) /
        ((Fintype.card ι : ℝ) ^ (s * (a - b)) + Fintype.card ι - 1) := by
  have : Nonempty ι := ⟨j₀⟩
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hn1 : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  rw [logScaleAttention_eq_rpow]
  set n : ℝ := (Fintype.card ι : ℝ)
  set E := n ^ (s * (a - b))
  set P := n ^ (s * b)
  have hw (j : ι) : n ^ (s * if j = j₀ then a else b) = P + if j = j₀ then (E - 1) * P else 0 := by
    split_ifs
    · rw [show P + (E - 1) * P = E * P by ring, ← Real.rpow_add hn]
      ring_nf
    · rw [add_zero]
  have hP : 0 < P := Real.rpow_pos_of_pos hn _
  have hE : 0 < E := Real.rpow_pos_of_pos hn _
  simp only [hw, add_mul, sum_add_distrib, ite_mul, zero_mul, Fintype.sum_ite_eq', ← mul_sum,
    sum_const, card_univ, nsmul_eq_mul]
  rw [div_eq_div_iff (by nlinarith) (by linarith)]
  ring

/-- Two tokens with raw scores `1` and `0` at `s = 1`. -/
example := logScaleAttention_ite (ι := Fin 2) 1 1 0 ![1, -1] 0

/-- The context `k₀ v₀ … k_m v_m` has `2m + 2` tokens. -/
lemma card_keyValueTokens (m : ℕ) : (Fintype.card (Fin (m + 1) × Bool) : ℝ) = 2 * m + 2 := by
  simp only [Fintype.card_prod, Fintype.card_fin, Fintype.card_bool]
  push_cast
  ring

/-- **The copy of the previous-token head** (`review/flat_spectrum.md`, SA5): over `n = 2m + 2`
tokens, layer 1 at the scale `s log n` copies the key `c` to the value token of `c` with the
weight `n^s / (n^s + n − 1)` and to every other token with `1 / (n^s + n − 1)`. -/
theorem prevTokenHead_eq (s : ℝ) {m : ℕ} (t : Fin (m + 1) × Bool) (c : Fin (m + 1)) :
    prevTokenHead s t c =
      (if t = (c, true) then (2 * m + 2 : ℝ) ^ s else 1) / ((2 * m + 2 : ℝ) ^ s + 2 * m + 1) := by
  have hsum : ∑ t' : Fin (m + 1) × Bool, keyChannel t' c = 1 := by
    simp only [keyChannel, Fintype.sum_ite_eq']
  have hk : keyChannel (prevToken t) c = if t = (c, true) then 1 else 0 := by
    simp only [keyChannel, prevToken_eq_iff]
  rw [prevTokenHead, logScaleAttention_ite, card_keyValueTokens, hsum, hk, sub_zero, mul_one]
  split_ifs <;> ring_nf

/-- The value token of `0` among two keys at `s = 2`. -/
example := prevTokenHead_eq 2 ((0 : Fin 2), true) 0

/-- **The lead of the copy** (`review/flat_spectrum.md`, SA5): over `n = 2m + 2` tokens, the
weight `(n^s − 1) / (n^s + n − 1)` by which layer 1 at the scale `s log n` copies a key more to
its value token than to any other token (`prevTokenHead_eq`). -/
noncomputable def copyLead (s : ℝ) (m : ℕ) : ℝ :=
  ((2 * m + 2 : ℝ) ^ s - 1) / ((2 * m + 2 : ℝ) ^ s + 2 * m + 1)

/-- **Two layers are one** (`review/flat_spectrum.md`, SA5): the induction head is one-hot
attention at the scale `s₂ a log n` at the value token of the query, where `a` is the lead of the
copy. -/
theorem inductionHead_eq_logScaleAttention (s₁ s₂ : ℝ) {ν : Type*} (φ : ν → ℝ) {m : ℕ}
    (π : Fin (m + 1) → ν) (q : Fin (m + 1)) :
    inductionHead s₁ s₂ φ π q = logScaleAttention (s₂ * copyLead s₁ m)
      (fun t => if t = (q, true) then 1 else 0) fun t => if t.2 then φ (π t.1) else 0 := by
  have h : (fun t => prevTokenHead s₁ t q) = fun t => if t = (q, true) then
      (2 * m + 2 : ℝ) ^ s₁ / ((2 * m + 2 : ℝ) ^ s₁ + 2 * m + 1) else
      1 / ((2 * m + 2 : ℝ) ^ s₁ + 2 * m + 1) := by
    funext t
    rw [prevTokenHead_eq, ite_div]
  rw [inductionHead, h, logScaleAttention_ite, logScaleAttention_ite, copyLead, ← sub_div,
    sub_zero, mul_one]

/-- Four keys, values `±1`, at `s₁ = s₂ = 2`. -/
example := inductionHead_eq_logScaleAttention 2 2 (fun b : Bool => if b then (1 : ℝ) else -1)
  (fun _ : Fin 4 => true) 0

/-- **The error of the induction head** (`review/flat_spectrum.md`, SA5): over `n = 2m + 2`
tokens it misses the value of the query by `(n φ(π q) − ∑ᵢ φ(π i)) / (n^{s₂ a} + n − 1)`, where
`a` is the lead of the copy. -/
theorem sub_inductionHead_eq (s₁ s₂ : ℝ) {ν : Type*} (φ : ν → ℝ) {m : ℕ}
    (π : Fin (m + 1) → ν) (q : Fin (m + 1)) :
    φ (π q) - inductionHead s₁ s₂ φ π q = ((2 * m + 2) * φ (π q) - ∑ i, φ (π i)) /
      ((2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + 2 * m + 1) := by
  have hV : ∑ t : Fin (m + 1) × Bool, (if t.2 then φ (π t.1) else 0) = ∑ i, φ (π i) := by
    simp only [Fintype.sum_prod_type, Fintype.sum_bool, ite_true, Bool.false_eq_true, ite_false,
      add_zero]
  have hD : 0 < (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) := Real.rpow_pos_of_pos (by positivity) _
  rw [inductionHead_eq_logScaleAttention, logScaleAttention_ite, card_keyValueTokens, hV,
    sub_zero, mul_one, show (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + (2 * m + 2) - 1 =
      (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + 2 * m + 1 by ring, ite_eq_left rfl]
  field_simp
  ring

/-- Four keys, values `±1`, at `s₁ = s₂ = 2`. -/
example := sub_inductionHead_eq 2 2 (fun b : Bool => if b then (1 : ℝ) else -1)
  (fun _ : Fin 4 => true) 0

/-- **The error bound of the induction head** (`review/flat_spectrum.md`, SA5): over
`n = 2m + 2` tokens it misses values bounded by `C` by at most
`(3m + 1) C / (n^{s₂ a} + 2m + 1)`, where `a` is the lead of the copy. -/
theorem abs_sub_inductionHead_le (s₁ s₂ : ℝ) {ν : Type*} {φ : ν → ℝ} {C : ℝ}
    (hφ : ∀ v, |φ v| ≤ C) {m : ℕ} (π : Fin (m + 1) → ν) (q : Fin (m + 1)) :
    |φ (π q) - inductionHead s₁ s₂ φ π q| ≤
      (3 * m + 1) * C / ((2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + 2 * m + 1) := by
  have hD : 0 < (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) := Real.rpow_pos_of_pos (by positivity) _
  rw [sub_inductionHead_eq, abs_div, abs_of_pos (by positivity : (0 : ℝ) < _ + 2 * m + 1)]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  rw [← add_sum_erase _ _ (mem_univ q)]
  have hR : |∑ i ∈ univ.erase q, φ (π i)| ≤ m * C := by
    refine (abs_sum_le_sum_abs _ _).trans ((sum_le_sum fun i _ => hφ (π i)).trans_eq ?_)
    rw [sum_const, card_erase_of_mem (mem_univ q), card_univ, Fintype.card_fin,
      Nat.add_sub_cancel, nsmul_eq_mul]
  calc |(2 * m + 2) * φ (π q) - (φ (π q) + ∑ i ∈ univ.erase q, φ (π i))|
      = |(2 * m + 1) * φ (π q) - ∑ i ∈ univ.erase q, φ (π i)| := by ring_nf
    _ ≤ (2 * m + 1) * |φ (π q)| + m * C := by
        refine (abs_sub _ _).trans (add_le_add (le_of_eq ?_) hR)
        rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 * m + 1)]
    _ ≤ (3 * m + 1) * C := by nlinarith [hφ (π q)]

/-- Four keys, values `±1`, at `s₁ = s₂ = 2`. -/
example := abs_sub_inductionHead_le 2 2 (φ := fun b : Bool => if b then (1 : ℝ) else -1)
  (C := 1) (fun b => by cases b <;> simp) (fun _ : Fin 4 => true) 0

/-- **The worst error of the induction head** (`review/flat_spectrum.md`, SA5): over the
dictionaries of values `±1` on `m + 1` keys, the greatest error is
`(3m + 1) / (n^{s₂ a} + 2m + 1)`, attained by the indicator dictionary. -/
theorem isGreatest_inductionHead_error (s₁ s₂ : ℝ) {m : ℕ} (q : Fin (m + 1)) :
    IsGreatest (Set.range fun π : Fin (m + 1) → Bool => |(if π q then (1 : ℝ) else -1) -
        inductionHead s₁ s₂ (fun b => if b then 1 else -1) π q|)
      ((3 * m + 1) / ((2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + 2 * m + 1)) := by
  have hD : 0 < (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) := Real.rpow_pos_of_pos (by positivity) _
  refine ⟨⟨fun i => decide (i = q), ?_⟩, ?_⟩
  · have hS : ∑ i : Fin (m + 1), (if decide (i = q) then (1 : ℝ) else -1) = 1 - m := by
      rw [← add_sum_erase _ _ (mem_univ q), sum_congr rfl fun i hi => by
        rw [decide_eq_false (ne_of_mem_erase hi)], sum_const, card_erase_of_mem (mem_univ q),
        card_univ, Fintype.card_fin, Nat.add_sub_cancel, nsmul_eq_mul]
      simp
      ring
    have h := sub_inductionHead_eq s₁ s₂ (fun b : Bool => if b then (1 : ℝ) else -1)
      (fun i => decide (i = q)) q
    beta_reduce at h
    dsimp only
    rw [h, hS, decide_eq_true rfl, ite_eq_left rfl,
      show (2 * m + 2 : ℝ) * 1 - (1 - m) = 3 * m + 1 by ring, abs_of_nonneg (by positivity)]
  · rintro _ ⟨π, rfl⟩
    simpa using abs_sub_inductionHead_le s₁ s₂ (φ := fun b : Bool => if b then (1 : ℝ) else -1)
      (C := 1) (fun b => by cases b <;> simp) π q

/-- At `s₁ = s₂ = 2` over four keys. -/
example := isGreatest_inductionHead_error 2 2 (3 : Fin 4)

end Transformer.Grokking.FlatSpectrum.Recall
