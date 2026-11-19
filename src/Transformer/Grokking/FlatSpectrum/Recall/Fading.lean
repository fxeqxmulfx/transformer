module

public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
public import Transformer.Grokking.FlatSpectrum.Recall.SoftmaxScale

/-!
# Bounded scores answer the worst dictionary with the opposite value

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA3, for
the attention of arXiv:2506.16055v3, §2.1, on the task of arXiv:2312.04927v1, §3.

Bind the queried key to the value `1` and every other key to `−1`. The attention misses `1` by
twice the share of the other tokens in the sum of the weights
(`one_sub_softmaxAttention_indicator`). With scores bounded by `B` that share is at least
`ρ = (n − 1) / (e^{2B} + n − 1)` (`mul_sum_exp_le_sum_erase`), so the error is at least `2ρ`
(`two_mul_le_one_sub_softmaxAttention_indicator`). SA3 averages over the dictionaries and
finds `ρ`; the worst one doubles it. Recall to accuracy `ε` has `(n − 1)(2 − ε) ≤ ε e^{2B}`
(`sub_one_mul_two_sub_le_of_softmaxAttention_recall`), and recall by the sign, an error below
`1` at every dictionary, has `n − 1 < e^{2B}` (`sub_one_lt_exp_of_softmaxAttention_sign`).

As the number of keys grows at bounded scores, `2ρ` tends to `2`, the distance from `1` to
`−1`: for every `ε < 2`, from some number of keys on, every query has a dictionary answered with
an error above `ε` (`exists_lt_softmaxAttention_recall_error`). The answer tends to the
opposite value.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- With the value `1` at the token `j₀` and `−1` at every other token, softmax attention misses
`1` by twice the share of the other tokens in the sum of the weights. -/
theorem one_sub_softmaxAttention_indicator {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ι → ℝ)
    (j₀ : ι) :
    1 - softmaxAttention s (fun j => if j = j₀ then 1 else -1) =
      2 * (∑ j ∈ univ.erase j₀, Real.exp (s j)) / ∑ j, Real.exp (s j) := by
  have hZ : ∑ j, Real.exp (s j) = Real.exp (s j₀) + ∑ j ∈ univ.erase j₀, Real.exp (s j) :=
    (add_sum_erase _ _ (mem_univ j₀)).symm
  have hN : ∑ j, Real.exp (s j) * (if j = j₀ then 1 else -1) =
      Real.exp (s j₀) - ∑ j ∈ univ.erase j₀, Real.exp (s j) := by
    rw [← add_sum_erase _ _ (mem_univ j₀), ite_eq_left rfl, mul_one, sub_eq_add_neg,
      ← sum_neg_distrib]
    exact congrArg _ (sum_congr rfl fun j hj => by
      rw [ite_eq_right (ne_of_mem_erase hj), mul_neg_one])
  have hpos : 0 < ∑ j, Real.exp (s j) := sum_pos (fun j _ => Real.exp_pos _) ⟨j₀, mem_univ _⟩
  rw [softmaxAttention, hN, eq_div_iff hpos.ne', sub_mul, div_mul_cancel₀ _ hpos.ne', hZ]
  ring

example := one_sub_softmaxAttention_indicator (![1, 0] : Fin 2 → ℝ) 0

/-- **The worst dictionary doubles the error** (`review/flat_spectrum.md`, SA3): with scores
bounded by `B`, the value `1` at the token `j₀` and `−1` elsewhere are answered with an error of
at least `2 (n − 1) / (e^{2B} + n − 1)`. -/
theorem two_mul_le_one_sub_softmaxAttention_indicator {ι : Type*} [Fintype ι] [DecidableEq ι]
    {s : ι → ℝ} {B : ℝ} (hs : ∀ j, |s j| ≤ B) (j₀ : ι) :
    2 * (((Fintype.card ι : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card ι - 1)) ≤
      1 - softmaxAttention s (fun j => if j = j₀ then 1 else -1) := by
  have hpos : 0 < ∑ j, Real.exp (s j) := sum_pos (fun j _ => Real.exp_pos _) ⟨j₀, mem_univ _⟩
  rw [one_sub_softmaxAttention_indicator, mul_div_assoc]
  exact mul_le_mul_of_nonneg_left ((le_div_iff₀ hpos).2 (mul_sum_exp_le_sum_erase hs j₀))
    zero_le_two

example := two_mul_le_one_sub_softmaxAttention_indicator (s := (![1, 0] : Fin 2 → ℝ)) (B := 1)
  (fun j => by fin_cases j <;> simp) 0

/-- The dictionary that binds the key `q` to `true` and every other key to `false`, read as
`±1`, is the indicator of `one_sub_softmaxAttention_indicator`. -/
lemma abs_recall_error_decide {ι : Type*} [Fintype ι] [DecidableEq ι] (t : ι → ℝ) (q : ι) :
    |(if decide (q = q) then (1 : ℝ) else -1) -
      softmaxAttention t fun k => if decide (k = q) then (1 : ℝ) else -1| =
      |1 - softmaxAttention t fun k => if k = q then 1 else -1| := by
  simp

/-- **The worst dictionary prices the accuracy** (`review/flat_spectrum.md`, SA3): recall of
values `±1` to accuracy `ε` with scores bounded by `B` has `(n − 1)(2 − ε) ≤ ε e^{2B}`, where
`sub_one_mul_le_of_softmaxAttention_recall`, from the average over dictionaries, has `1 − ε`. -/
theorem sub_one_mul_two_sub_le_of_softmaxAttention_recall {κ : Type*} [Fintype κ]
    [DecidableEq κ] [Nonempty κ] {B ε : ℝ} {s : κ → κ → Bool → ℝ} (hs : ∀ q k b, |s q k b| ≤ B)
    (h : ∀ q (π : κ → Bool), |(if π q then (1 : ℝ) else -1) -
      softmaxAttention (fun k => s q k (π k)) fun k => if π k then (1 : ℝ) else -1| ≤ ε) :
    ((Fintype.card κ : ℝ) - 1) * (2 - ε) ≤ ε * Real.exp (2 * B) := by
  obtain ⟨q⟩ := ‹Nonempty κ›
  have hm : (1 : ℝ) ≤ Fintype.card κ := by exact_mod_cast Fintype.card_pos_iff.2 ⟨q⟩
  have hden : 0 < Real.exp (2 * B) + Fintype.card κ - 1 := by linarith [Real.exp_pos (2 * B)]
  have h1 := two_mul_le_one_sub_softmaxAttention_indicator
    (s := fun k => s q k (decide (k = q))) (fun k => hs q k _) q
  have h2 := h q fun k => decide (k = q)
  rw [abs_recall_error_decide] at h2
  have h3 := (h1.trans (le_abs_self _)).trans h2
  rw [← mul_div_assoc, div_le_iff₀ hden] at h3
  linarith

/-- One key: scores `0` recall it exactly. -/
example : ((Fintype.card (Fin 1) : ℝ) - 1) * (2 - 0) ≤ 0 * Real.exp (2 * 0) :=
  sub_one_mul_two_sub_le_of_softmaxAttention_recall (s := fun _ _ _ => 0)
    (fun _ _ _ => by simp) fun q π => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      simp [softmaxAttention]

/-- **Recall by the sign needs `n − 1 < e^{2B}`** (`review/flat_spectrum.md`, SA3): if softmax
attention with scores bounded by `B` answers every dictionary of values `±1` with an error below
`1`, so with the sign of the bound value, then `n − 1 < e^{2B}`. -/
theorem sub_one_lt_exp_of_softmaxAttention_sign {κ : Type*} [Fintype κ] [DecidableEq κ]
    [Nonempty κ] {B : ℝ} {s : κ → κ → Bool → ℝ} (hs : ∀ q k b, |s q k b| ≤ B)
    (h : ∀ q (π : κ → Bool), |(if π q then (1 : ℝ) else -1) -
      softmaxAttention (fun k => s q k (π k)) fun k => if π k then (1 : ℝ) else -1| < 1) :
    (Fintype.card κ : ℝ) - 1 < Real.exp (2 * B) := by
  obtain ⟨q⟩ := ‹Nonempty κ›
  have hm : (1 : ℝ) ≤ Fintype.card κ := by exact_mod_cast Fintype.card_pos_iff.2 ⟨q⟩
  have hden : 0 < Real.exp (2 * B) + Fintype.card κ - 1 := by linarith [Real.exp_pos (2 * B)]
  have h1 := two_mul_le_one_sub_softmaxAttention_indicator
    (s := fun k => s q k (decide (k = q))) (fun k => hs q k _) q
  have h2 := h q fun k => decide (k = q)
  rw [abs_recall_error_decide] at h2
  have h3 := (h1.trans (le_abs_self _)).trans_lt h2
  rw [← mul_div_assoc, div_lt_one hden] at h3
  linarith

/-- One key: scores `0` recall it exactly. -/
example : (Fintype.card (Fin 1) : ℝ) - 1 < Real.exp (2 * 0) :=
  sub_one_lt_exp_of_softmaxAttention_sign (s := fun _ _ _ => 0) (fun _ _ _ => by simp)
    fun q π => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      simp [softmaxAttention]

/-- The length beats a power of it below one: for `a < 1` and `d > 0`, `c (m + 1)^a < d m` from
some `m` on. -/
lemma eventually_mul_rpow_lt_mul (c : ℝ) {d a : ℝ} (hd : 0 < d) (ha : a < 1) :
    ∀ᶠ m : ℕ in atTop, c * ((m : ℝ) + 1) ^ a < d * m := by
  have hx : Tendsto (fun m : ℕ => (m : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have h0 : Tendsto (fun m : ℕ => ((m : ℝ) + 1) ^ (-(1 - a))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp hx
  filter_upwards [h0.eventually (gt_mem_nhds (show (0 : ℝ) < d / (2 * (|c| + 1)) by positivity)),
    hx.eventually_gt_atTop 2] with m hm h2
  have hpos : (0 : ℝ) < (m : ℝ) + 1 := by linarith
  have hsplit : ((m : ℝ) + 1) ^ a = ((m : ℝ) + 1) ^ (-(1 - a)) * ((m : ℝ) + 1) := by
    conv_lhs => rw [show a = -(1 - a) + 1 by ring]
    rw [Real.rpow_add hpos, Real.rpow_one]
  have hy0 : 0 ≤ ((m : ℝ) + 1) ^ (-(1 - a)) := Real.rpow_nonneg hpos.le _
  have hy := (lt_div_iff₀ (by positivity : (0 : ℝ) < 2 * (|c| + 1))).1 hm
  calc c * ((m : ℝ) + 1) ^ a ≤ |c| * ((m : ℝ) + 1) ^ a :=
        mul_le_mul_of_nonneg_right (le_abs_self c) (Real.rpow_nonneg hpos.le _)
    _ = |c| * ((m : ℝ) + 1) ^ (-(1 - a)) * ((m : ℝ) + 1) := by rw [hsplit]; ring
    _ < d * m := by
        nlinarith [abs_nonneg c, mul_lt_mul_of_pos_right hy hpos, mul_nonneg hy0 hpos.le,
          mul_lt_mul_of_pos_left h2 hd]

example := eventually_mul_rpow_lt_mul 3 one_pos (by norm_num : (1 / 2 : ℝ) < 1)

/-- **Bounded scores answer the worst dictionary with the opposite value**
(`review/flat_spectrum.md`, SA3): softmax attention with scores bounded by one `B` at every
number `m + 1` of keys, for every accuracy `ε < 2`, from some number of keys on answers some
dictionary of values `±1` at every query with an error above `ε`. -/
theorem exists_lt_softmaxAttention_recall_error {B ε : ℝ} (hε : ε < 2)
    (s : (m : ℕ) → Fin (m + 1) → Fin (m + 1) → Bool → ℝ) (hs : ∀ m q k b, |s m q k b| ≤ B) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        softmaxAttention (fun k => s m q k (π k)) fun k => if π k then (1 : ℝ) else -1| := by
  obtain ⟨M, hM⟩ := eventually_atTop.1 (eventually_mul_rpow_lt_mul (ε * Real.exp (2 * B))
    (by linarith : (0 : ℝ) < 2 - ε) zero_lt_one)
  refine ⟨M, fun m hm q => ⟨fun k => decide (k = q), ?_⟩⟩
  rw [abs_recall_error_decide]
  have h := two_mul_le_one_sub_softmaxAttention_indicator
    (s := fun k => s m q k (decide (k = q))) (fun k => hs m q k _) q
  refine lt_of_lt_of_le ?_ (h.trans (le_abs_self _))
  have hM' := hM m hm
  rw [Real.rpow_zero, mul_one] at hM'
  have hn : (Fintype.card (Fin (m + 1)) : ℝ) = (m : ℝ) + 1 := by simp
  have hden : 0 < Real.exp (2 * B) + ((m : ℝ) + 1) - 1 := by
    linarith [Real.exp_pos (2 * B), m.cast_nonneg' (α := ℝ)]
  rw [hn, ← mul_div_assoc, lt_div_iff₀ hden]
  linarith

/-- Scores `0`: from some number of keys on, every query has a dictionary answered with an error
above `1`. -/
example := exists_lt_softmaxAttention_recall_error (B := 0) (ε := 1) (by norm_num)
  (fun _ _ _ _ => 0) fun _ _ _ _ => by simp

end Transformer.Grokking.FlatSpectrum.Recall
