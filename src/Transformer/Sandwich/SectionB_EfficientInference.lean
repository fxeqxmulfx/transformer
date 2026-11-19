module

public import Transformer.SequenceModel.Attention.Grid.Separation
public import Transformer.Sandwich.Section5_Sinusoidal
public import Transformer.Sandwich.Section5_WindowLayer

/-!
# Appendix B: efficient inference with a cache window

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, Appendix B: "When it comes to generating the
`w̄ + 1`-th token, we simply discard the first cached `q_1`, `k_1`, and `v_1` and use the rest of
`w̄ − 1` tokens along with the newly added token to perform self-attention. If we want to generate
a length `L_ex` text snippet, the complexity is `O(w̄ × L_ex)` instead of `O(L_ex²)`. This
complexity is also better than that of an APE model, which is `O(w̄² × L_ex)` since an APE model
needs to completely re-encode the previous `w̄` vectors when generating new tokens following the
first `w̄` ones."

* `cutBias w̄ b`: discarding the keys at the distances `≥ w̄` is the bias `b` cut off at `w̄`,
  which masks them (`isWindowed_cutBias`); ALiBi or Sandwich with the cache are such models.
* `card_attended_le`: under a bias that masks the distances `≥ w̄`, the query `m` attends to at most
  `min(m + 1, w̄)` keys, and the `L` queries to at most `w̄ L` (`sum_card_attended_le`); without a
  window the query `m` attends to all `m + 1` keys (`attended_coe`), and the `L` queries to
  `L(L + 1)/2` (`two_mul_sum_card_attended_coe`).
* `apeCost w̄ L`: re-encoding the `w̄` most recent tokens, `w̄(w̄ + 1)/2` query–key pairs, for each
  token after the first `w̄`; between `(L − w̄) w̄(w̄ + 1)/2` and `L w̄(w̄ + 1)/2`
  (`le_two_mul_apeCost`, `two_mul_apeCost_le`).
* `sinusoidalAPE_injective`: re-encoding is needed because Sinusoidal's `p_m` determines the
  integer position `m`, so a cached token's input `e + p_m` changes when the window advances.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Transformer.BAM

variable {d k w : ℕ}

/-- The bias `b` with the keys at the distances `≥ w̄` discarded, as the cache window does. -/
noncomputable def cutBias (w : ℕ) (b : ℕ → EReal) (t : ℕ) : EReal := if t < w then b t else ⊥

/-- The cache window of size `w̄` makes every bias windowed. -/
theorem isWindowed_cutBias (b : ℕ → EReal) : IsWindowed w (cutBias w b) :=
  fun t ht => by simp [cutBias, Nat.not_lt.2 ht]

/-- The keys to which the query `m` gives a nonzero weight. -/
noncomputable def attended (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m : ℕ) : Finset ℕ :=
  (range (m + 1)).filter fun n => attnWeight A b x m n ≠ 0

/-- **The query `m` attends to at most `min(m + 1, w̄)` keys** when the bias masks the distances
`≥ w̄`: they lie in the window `m + 1 − w̄, …, m`. -/
theorem card_attended_le {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) (x : Embs d)
    (m : ℕ) : (attended A b x m).card ≤ min (m + 1) w := by
  have hsub : attended A b x m ⊆ Ico (m + 1 - w) (m + 1) := fun n hn => by
    simp only [attended, mem_filter, mem_range] at hn
    refine mem_Ico.2 ⟨?_, hn.1⟩
    by_contra hlt
    exact hn.2 (by rw [attnWeight, weight_score_eq_zero hb A x (by omega), zero_div])
  calc (attended A b x m).card ≤ (Ico (m + 1 - w) (m + 1)).card := card_le_card hsub
    _ = min (m + 1) w := by rw [Nat.card_Ico]; omega

/-- The hypothesis of `card_attended_le` holds for the cache window of size `2048`. -/
example (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m : ℕ) :
    (attended A (cutBias 2048 b) x m).card ≤ min (m + 1) 2048 :=
  card_attended_le (isWindowed_cutBias b) A x m

/-- **Generating `L` tokens costs at most `w̄ L`** query–key pairs with a cache window `w̄`. -/
theorem sum_card_attended_le {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) (x : Embs d)
    (L : ℕ) : ∑ m ∈ range L, (attended A b x m).card ≤ w * L := by
  calc ∑ m ∈ range L, (attended A b x m).card ≤ ∑ _m ∈ range L, w :=
        sum_le_sum fun m _ => (card_attended_le hb A x m).trans (min_le_right _ _)
    _ = w * L := by rw [sum_const, card_range, smul_eq_mul, mul_comm]

/-- The hypothesis of `sum_card_attended_le` holds for the cache window of size `2048`. -/
example (A : Head d k) (b : ℕ → EReal) (x : Embs d) (L : ℕ) :
    ∑ m ∈ range L, (attended A (cutBias 2048 b) x m).card ≤ 2048 * L :=
  sum_card_attended_le (isWindowed_cutBias b) A x L

/-- **Without a window, the query `m` attends to all `m + 1` keys**: a real bias, such as ALiBi's
or Sandwich's without the cache, masks none. -/
theorem attended_coe (A : Head d k) (r : ℕ → ℝ) (x : Embs d) (m : ℕ) :
    attended A (fun t => (r t : EReal)) x m = range (m + 1) := by
  have hpos : ∀ i, 0 < weight (score A (fun t => (r t : EReal)) x m i) := fun i => by
    simp only [score, ← EReal.coe_add, weight_coe]
    exact Real.exp_pos _
  rw [attended]
  exact filter_true_of_mem fun n _ =>
    (div_pos (hpos n) (sum_pos (fun i _ => hpos i) ⟨m, mem_range.2 (Nat.lt_succ_self m)⟩)).ne'

/-- `2 Σ_{m < L} (m + 1) = L(L + 1)`. -/
theorem two_mul_sum_range_succ (L : ℕ) : 2 * ∑ m ∈ range L, (m + 1) = L * (L + 1) := by
  induction L with
  | zero => simp
  | succ L ih =>
    rw [sum_range_succ, mul_add, ih]
    ring

/-- **Without a window, `L` queries attend to `L(L + 1)/2` keys**: `O(L_ex²)`. -/
theorem two_mul_sum_card_attended_coe (A : Head d k) (r : ℕ → ℝ) (x : Embs d) (L : ℕ) :
    2 * ∑ m ∈ range L, (attended A (fun t => (r t : EReal)) x m).card = L * (L + 1) := by
  simp only [attended_coe, card_range]
  exact two_mul_sum_range_succ L

/-- The query–key pairs of an APE model generating `L` tokens with a cache window `w̄`: the token
`m < w̄` attends to `m + 1` cached keys, and every later token re-encodes the `w̄` most recent
tokens, `w̄(w̄ + 1)/2` pairs. -/
def apeCost (w L : ℕ) : ℕ := ∑ m ∈ range L, if m < w then m + 1 else ∑ j ∈ range w, (j + 1)

/-- **An APE model costs at least `(L − w̄) w̄(w̄ + 1)/2`**: `Ω(w̄² L_ex)` for `L_ex ≥ 2w̄`. -/
theorem le_two_mul_apeCost (w L : ℕ) : (L - w) * (w * (w + 1)) ≤ 2 * apeCost w L := by
  have hcard : ((range L).filter fun m => w ≤ m).card = L - w := by
    rw [show (range L).filter (fun m => w ≤ m) = Ico w L by ext m; simp [and_comm]]
    exact Nat.card_Ico w L
  calc (L - w) * (w * (w + 1))
      = ∑ m ∈ range L, if w ≤ m then 2 * ∑ j ∈ range w, (j + 1) else 0 := by
        rw [← sum_filter, sum_const, smul_eq_mul, hcard, two_mul_sum_range_succ]
    _ ≤ ∑ m ∈ range L, 2 * (if m < w then m + 1 else ∑ j ∈ range w, (j + 1)) :=
        sum_le_sum fun m _ => by split_ifs <;> omega
    _ = 2 * apeCost w L := by rw [apeCost, mul_sum]

/-- **An APE model costs at most `L w̄(w̄ + 1)/2`**: `O(w̄² L_ex)`. -/
theorem two_mul_apeCost_le (w L : ℕ) : 2 * apeCost w L ≤ L * (w * (w + 1)) := by
  have hS := two_mul_sum_range_succ w
  have hw : w ≤ ∑ j ∈ range w, (j + 1) := by
    calc w = ∑ _j ∈ range w, 1 := by simp
      _ ≤ ∑ j ∈ range w, (j + 1) := sum_le_sum fun j _ => by omega
  calc 2 * apeCost w L
      = ∑ m ∈ range L, 2 * (if m < w then m + 1 else ∑ j ∈ range w, (j + 1)) := by
        rw [apeCost, mul_sum]
    _ ≤ ∑ _m ∈ range L, w * (w + 1) := sum_le_sum fun m _ => by split_ifs <;> omega
    _ = L * (w * (w + 1)) := by rw [sum_const, card_range, smul_eq_mul]

/-- **Sinusoidal's `p_m` determines the position `m`** for `d ≥ 2`: its frequency-`1` coordinate
`cos m` does, `cos` being injective on `ℕ` since `π` is irrational. A window re-indexed from `0`
moves every cached token from the position `m` to `m − 1`, so its input `e + p_m` changes and the
window is re-encoded. -/
theorem sinusoidalAPE_injective {D : ℕ} (hD : 1 ≤ D) : Function.Injective (sinusoidalAPE D) := by
  intro m m' h
  have h0 : apeFreq D ⟨0, hD⟩ = 1 := by simp [apeFreq]
  have hc := congrFun h (finSumFinEquiv (Sum.inr ⟨0, hD⟩))
  simp only [sinusoidalAPE, Equiv.symm_apply_apply, sinusoid, Sum.elim_inr, h0, mul_one] at hc
  exact SequenceModel.Attention.Grid.cos_nat_injective hc

/-- The hypothesis of `sinusoidalAPE_injective` holds for `d = 128`. -/
example : Function.Injective (sinusoidalAPE 64) := sinusoidalAPE_injective (by norm_num)

end Transformer.Sandwich
