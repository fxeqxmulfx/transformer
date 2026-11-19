module

public import Transformer.MRCR.Ties

/-!
# Counting the earlier needles of a key with one head at the scale `s log n`

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37 (the entry of a key with
the desired index); MRCR-V2 of arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`; attention at the scale `s log n` over the `n`
visible tokens, as in SSMax, arXiv:2501.19399v1, §2.2. This is the first layer of the
real-softmax counterpart of the formula of `Transformer.MRCR.Ordinal`.

After a BOS the rows `t = 0, 1, …` carry the keys `κ t`, and `occ κ t` counts the rows `u ≤ t`
with the key of `t`, `t` included. At the row `t` the counting head (`countHead`) attends
causally to the BOS and the rows `u ≤ t`, `n = t + 2` tokens, with the raw score `1` at the BOS
and at the rows of the key of `t` and `0` elsewhere, and reads `1` at the BOS and `0` at the rows.
The raw score `[u = BOS] + ⟨e_{κ u}, e_{κ t}⟩` is bilinear in one-hot features. Its
`occ κ t + 1` leading tokens tie, so it returns `1 / (occ κ t + 1)` within `(t + 2)^{1 − s}` for
`s ≥ 1` (`abs_countHead_sub_le`), by `Transformer.MRCR.abs_logScaleAttention_sub_average_le`.

Rows of one key have distinct counts (`occ_lt_occ`), and for every count `o ≠ r + 1` the value
`1/(o + 1)` lies at least `1/((r + 2)(r + 3))` from `1/(r + 2)`, however large `o`
(`one_le_lookupWeight_mul_sq`). `Transformer.MRCR.CountLookup` tells the needle with the desired
index from the others by that margin.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

variable {c : ℕ}

/-- The rows `u ≤ t` with the key of the row `t`, `t` included. -/
def occ (κ : ℕ → Fin c) (t : ℕ) : ℕ := ((range (t + 1)).filter fun u => κ u = κ t).card

lemma one_le_occ (κ : ℕ → Fin c) (t : ℕ) : 1 ≤ occ κ t :=
  card_pos.2 ⟨t, mem_filter.2 ⟨self_mem_range_succ t, rfl⟩⟩

/-- **Rows of one key have distinct counts.** -/
lemma occ_lt_occ {κ : ℕ → Fin c} {t t' : ℕ} (h : t < t') (hk : κ t = κ t') :
    occ κ t < occ κ t' := by
  refine card_lt_card ((ssubset_iff_of_subset fun u hu => ?_).2 ⟨t', ?_, fun ht' => ?_⟩)
  · obtain ⟨hu1, hu2⟩ := mem_filter.1 hu
    exact mem_filter.2 ⟨mem_range.2 (by have := mem_range.1 hu1; omega), hu2.trans hk⟩
  · exact mem_filter.2 ⟨self_mem_range_succ t', rfl⟩
  · have := mem_range.1 (mem_filter.1 ht').1
    omega

/-- Two rows of one key. -/
example := occ_lt_occ (κ := fun _ => (0 : Fin 1)) (t := 0) (t' := 1) one_pos rfl

/-- **The counting head**, layer 1: at the row `t`, attention at the scale `s log n` over the BOS
(`none`) and the rows `u ≤ t`, with the raw score `1` at the BOS and at the rows of the key of
`t` and `0` elsewhere, reading `1` at the BOS and `0` at the rows. -/
noncomputable def countHead (s : ℝ) (κ : ℕ → Fin c) (t : ℕ) : ℝ :=
  logScaleAttention s
    (fun u : Option (Fin (t + 1)) => u.elim 1 fun u => if κ u = κ t then 1 else 0)
    fun u => u.elim 1 fun _ => 0

/-- **The counting head counts** (MRCR, arXiv:2409.12640v2, `eval_framework.tex`): at the row `t`
and the scale `s log n`, `s ≥ 1`, it is within `(t + 2)^{1 − s}` of `1 / (occ κ t + 1)`. -/
theorem abs_countHead_sub_le {s : ℝ} (hs : 1 ≤ s) (κ : ℕ → Fin c) (t : ℕ) :
    |countHead s κ t - 1 / ((occ κ t : ℝ) + 1)| ≤ ((t : ℝ) + 2) ^ (1 - s) := by
  obtain ⟨T, hTdef⟩ : ∃ T : Finset (Option (Fin (t + 1))),
      T = insertNone (univ.filter fun u : Fin (t + 1) => κ u = κ t) := ⟨_, rfl⟩
  have hmem (u : Fin (t + 1)) : some u ∈ T ↔ κ u = κ t := by
    rw [hTdef, some_mem_insertNone, mem_filter]
    simp
  have hnone : none ∈ T := hTdef ▸ none_mem_insertNone
  have hT : (T.card : ℝ) = occ κ t + 1 := by
    have h2 : (univ.filter fun u : Fin (t + 1) => κ u = κ t).card = occ κ t := by
      rw [occ, card_filter, card_filter]
      exact Fin.sum_univ_eq_sum_range (fun u => if κ u = κ t then 1 else 0) (t + 1)
    rw [hTdef, card_insertNone, h2]
    push_cast
    ring
  have hsum : ∑ u ∈ T, (u.elim 1 fun _ => 0 : ℝ) = 1 := by
    rw [sum_eq_single_of_mem none hnone]
    · rfl
    · rintro (_ | u) _ hu
      · exact absurd rfl hu
      · rfl
  have h := abs_logScaleAttention_sub_average_le (s := s) (g := 1) (C := 1) (z₀ := 1)
    (z := fun u : Option (Fin (t + 1)) => u.elim 1 fun u => if κ u = κ t then 1 else 0)
    (v := fun u => u.elim 1 fun _ => 0) (by linarith) T ⟨none, hnone⟩
    (by rintro (_ | u) hu; exacts [rfl, by simp [(hmem u).1 hu]])
    (by
      rintro (_ | u) hu
      exacts [absurd hnone hu, by simp [show ¬κ u = κ t from fun h => hu ((hmem u).2 h)]])
    (by rintro (_ | u) <;> simp)
  rw [hsum, hT, Fintype.card_option, Fintype.card_fin] at h
  have ho : (1 : ℝ) ≤ occ κ t := by exact_mod_cast one_le_occ κ t
  have hn : (0 : ℝ) < (t : ℝ) + 2 := by positivity
  have hle : (occ κ t : ℝ) ≤ t + 1 := by
    have := card_filter_le (range (t + 1)) fun u => κ u = κ t
    rw [card_range] at this
    exact_mod_cast this
  have hp := Real.rpow_nonneg hn.le (-(s * 1))
  have hsplit : ((t : ℝ) + 2) ^ (1 - s) = ((t : ℝ) + 2) * ((t : ℝ) + 2) ^ (-(s * 1)) := by
    rw [mul_one, sub_eq_add_neg, Real.rpow_add hn, Real.rpow_one]
  refine h.trans ?_
  rw [hsplit, div_le_iff₀ (by linarith)]
  push_cast
  rw [show (t : ℝ) + 1 + 1 = (t : ℝ) + 2 by ring]
  have hq : 0 ≤ ((t : ℝ) + 2) * ((occ κ t : ℝ) + 1) - 2 * ((t : ℝ) + 2 - ((occ κ t : ℝ) + 1)) := by
    nlinarith
  nlinarith [mul_nonneg hp hq]

/-- One key throughout, at the row `3` and the scale `2 log n`. -/
example := abs_countHead_sub_le (s := 2) (by norm_num) (fun _ => (0 : Fin 1)) 3

/-- The weight `K_r = ((r + 2)(r + 3))²` that the second layer gives the ordinal `r`. -/
def lookupWeight (r : ℕ) : ℝ := (((r : ℝ) + 2) * ((r : ℝ) + 3)) ^ 2

/-- **Counts stay apart**: for every count `o ≠ r + 1`, `K_r (1/(o + 1) − 1/(r + 2))² ≥ 1`. -/
theorem one_le_lookupWeight_mul_sq {o r : ℕ} (h : o ≠ r + 1) :
    1 ≤ lookupWeight r * (1 / ((o : ℝ) + 1) - 1 / ((r : ℝ) + 2)) ^ 2 := by
  have ho : (0 : ℝ) < (o : ℝ) + 1 := by positivity
  have hr : (0 : ℝ) < (r : ℝ) + 2 := by positivity
  have key : ((r : ℝ) + 2) * ((r : ℝ) + 3) * (1 / ((o : ℝ) + 1) - 1 / ((r : ℝ) + 2)) =
      ((r : ℝ) + 3) * ((r : ℝ) + 1 - o) / ((o : ℝ) + 1) := by
    field_simp
    ring
  rw [lookupWeight, ← mul_pow, key]
  rcases Nat.lt_or_gt_of_ne h with hlt | hgt
  · have h1 : (o : ℝ) ≤ r := by exact_mod_cast (by omega : o ≤ r)
    have h2 : 1 ≤ ((r : ℝ) + 3) * ((r : ℝ) + 1 - o) / ((o : ℝ) + 1) := by
      rw [le_div_iff₀ ho]
      nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ (r : ℝ) + 3) (by linarith : (0 : ℝ) ≤ r - o)]
    nlinarith
  · have h1 : (r : ℝ) + 2 ≤ o := by exact_mod_cast (by omega : r + 2 ≤ o)
    have h2 : ((r : ℝ) + 3) * ((r : ℝ) + 1 - o) / ((o : ℝ) + 1) ≤ -1 := by
      rw [div_le_iff₀ ho]
      nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ (r : ℝ) + 2)
        (by linarith : (0 : ℝ) ≤ o - r - 2)]
    nlinarith

/-- The first needle against the third count. -/
example := one_le_lookupWeight_mul_sq (o := 3) (r := 0) (by decide)

end Transformer.MRCR
