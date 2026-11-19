module

public import Transformer.Grokking.FlatSpectrum.Recall.Fading
public import Transformer.MRCR.NearMiss

/-!
# The threshold of near misses is the growth of their number

Source: MRCR-V2 (8-needle) of arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, at the scale `s log n` of SSMax,
arXiv:2501.19399v1, §2.2; see `Transformer.MRCR.NearMiss`.

Over every number `m + 1` of keys let near misses trail the queried key by `g₁` and the other
keys trail it by `g₂`.

* With at most `K (m + 1)^α` near misses, `s g₁ > α` and `s g₂ > 1` recall at every length
  (`exists_forall_logScaleAttention_recall_nearMiss`), from the bound
  `2 C (K n^{α − s g₁} + n^{1 − s g₂})`.
* With at least `c (m + 1)^α` near misses, `c > 0`, trailing by at most `g₁` and `s g₁ < α`, the
  indicator dictionary is answered with an error that tends to `2`
  (`exists_lt_logScaleAttention_nearMiss_error`): the near misses keep at least
  `y / (1 + y)` of the weight, `y = c n^{α − s g₁} → ∞`.

So the threshold of the near misses is the exponent `α` of their number, not `1`: a bounded
number of them needs only `s g₁ > 0`, a number linear in the length needs `s g₁ > 1`, as the
bulk does.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Filter Topology Transformer.Grokking.FlatSpectrum.Recall

/-- **Near misses below the threshold of their number recall at every length**: if over every
number `m + 1` of keys at most `K (m + 1)^α` near misses trail the queried key by `g₁` and the
other keys trail it by `g₂`, with `s g₁ > α` and `s g₂ > 1`, attention at the scale
`s log (m + 1)` recalls values bounded by `C` to every accuracy `ε > 0` from some number of keys
on. -/
theorem exists_forall_logScaleAttention_recall_nearMiss {ν : Type*} {s g₁ g₂ α K C : ℝ}
    (hs : 0 ≤ s) (h₁ : α < s * g₁) (h₂ : 1 < s * g₂)
    (z : (m : ℕ) → Fin (m + 1) → Fin (m + 1) → ℝ)
    (N : (m : ℕ) → Fin (m + 1) → Finset (Fin (m + 1)))
    (hN : ∀ m q, ((N m q).card : ℝ) ≤ K * ((m : ℝ) + 1) ^ α)
    (hz₁ : ∀ m q k, k ∈ N m q → k ≠ q → z m q k + g₁ ≤ z m q q)
    (hz₂ : ∀ m q k, k ∉ N m q → k ≠ q → z m q k + g₂ ≤ z m q q)
    {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |logScaleAttention s (z m q) (fun k => φ (π k)) - φ (π q)| ≤ ε := by
  have hx : Tendsto (fun m : ℕ => (m : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have ht : Tendsto (fun m : ℕ => 2 * C * (K * ((m : ℝ) + 1) ^ (-(s * g₁ - α)) +
      ((m : ℝ) + 1) ^ (-(s * g₂ - 1)))) atTop (𝓝 0) := by
    have h := ((((tendsto_rpow_neg_atTop (by linarith : 0 < s * g₁ - α)).comp hx).const_mul
      K).add ((tendsto_rpow_neg_atTop (by linarith : 0 < s * g₂ - 1)).comp hx)).const_mul (2 * C)
    simpa [Function.comp_def] using h
  obtain ⟨M, hM⟩ := eventually_atTop.1 (ht.eventually (ge_mem_nhds hε))
  refine ⟨M, fun m hm q π => ?_⟩
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have hpos : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have h := abs_logScaleAttention_sub_le_two hs (N m q) (hz₁ m q) (hz₂ m q) fun k => hφ (π k)
  have hcard : (Fintype.card (Fin (m + 1)) : ℝ) = (m : ℝ) + 1 := by simp
  rw [hcard] at h
  refine h.trans (le_trans (mul_le_mul_of_nonneg_left (add_le_add ?_ ?_) (by linarith))
    (hM m hm))
  · calc ((N m q).card : ℝ) * ((m : ℝ) + 1) ^ (-(s * g₁))
        ≤ K * ((m : ℝ) + 1) ^ α * ((m : ℝ) + 1) ^ (-(s * g₁)) :=
          mul_le_mul_of_nonneg_right (hN m q) (Real.rpow_nonneg hpos.le _)
      _ = K * ((m : ℝ) + 1) ^ (-(s * g₁ - α)) := by
          rw [mul_assoc, ← Real.rpow_add hpos, show α + -(s * g₁) = -(s * g₁ - α) by ring]
  · calc ((m : ℝ) + 1 - 1) * ((m : ℝ) + 1) ^ (-(s * g₂))
        ≤ ((m : ℝ) + 1) * ((m : ℝ) + 1) ^ (-(s * g₂)) :=
          mul_le_mul_of_nonneg_right (by linarith) (Real.rpow_nonneg hpos.le _)
      _ = ((m : ℝ) + 1) ^ (-(s * g₂ - 1)) := by
          rw [show -(s * g₂ - 1) = 1 + -(s * g₂) by ring, Real.rpow_add hpos, Real.rpow_one]

/-- One near miss, the key `0`, trailing by `1/2`: a bounded number needs only `s g₁ > 0`. -/
example := exists_forall_logScaleAttention_recall_nearMiss (ν := Bool) (s := 1) (g₁ := 1 / 2)
  (g₂ := 2) (α := 0) (K := 1) zero_le_one (by norm_num) (by norm_num)
  (fun _ q k => if k = q then 2 else if k = 0 then 3 / 2 else 0) (fun _ _ => {0})
  (fun _ _ => by simp)
  (fun _ q k hk hkq => by
    obtain rfl := mem_singleton.1 hk
    norm_num [hkq])
  (fun _ q k hk hkq => by
    norm_num [hkq, Finset.notMem_singleton.1 hk])
  (φ := fun b => if b then 1 else -1) (C := 1) (fun b => by cases b <;> simp) (ε := 1 / 10)
  (by norm_num)

/-- **Near misses above the threshold of their number fade**: if from some number of keys on at
least `c (m + 1)^α` near misses, `c > 0`, trail the queried key by at most `g₁`, with
`s g₁ < α`, attention at the scale `s log (m + 1)` answers the indicator dictionary, `1` at the
queried key and `−1` at the others, with an error above every `ε < 2` from some number of keys
on. -/
theorem exists_lt_logScaleAttention_nearMiss_error {s g₁ α c ε : ℝ} (hs : 0 ≤ s) (hc : 0 < c)
    (h₁ : s * g₁ < α) (hε : ε < 2)
    (z : (m : ℕ) → Fin (m + 1) → Fin (m + 1) → ℝ)
    (N : (m : ℕ) → Fin (m + 1) → Finset (Fin (m + 1)))
    (hN : ∀ᶠ m : ℕ in atTop, ∀ q, c * ((m : ℝ) + 1) ^ α ≤ ((N m q).erase q).card)
    (hz : ∀ m q k, k ∈ N m q → z m q q ≤ z m q k + g₁) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1),
      ε < 1 - logScaleAttention s (z m q) fun k => if k = q then 1 else -1 := by
  have hx : Tendsto (fun m : ℕ => (m : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have ht : Tendsto (fun m : ℕ => c * ((m : ℝ) + 1) ^ (α - s * g₁)) atTop atTop :=
    ((tendsto_rpow_atTop (by linarith)).comp hx).const_mul_atTop hc
  obtain ⟨M, hM⟩ := eventually_atTop.1 (hN.and (ht.eventually_gt_atTop (ε / (2 - ε))))
  refine ⟨M, fun m hm q => ?_⟩
  obtain ⟨hNm, hy⟩ := hM m hm
  set n : ℝ := (m : ℝ) + 1 with hn
  have hn0 : 0 < n := by positivity
  have hc0 : 0 ≤ s * Real.log n := mul_nonneg hs (Real.log_nonneg (by simp [hn]))
  have hrw : logScaleAttention s (z m q) (fun k => if k = q then 1 else -1) =
      softmaxAttention (fun k => s * Real.log n * z m q k) fun k => if k = q then 1 else -1 := by
    simp [logScaleAttention, hn]
  rw [hrw, one_sub_softmaxAttention_indicator, ← add_sum_erase _ _ (mem_univ q)]
  set A := Real.exp (s * Real.log n * z m q q)
  set S := ∑ k ∈ univ.erase q, Real.exp (s * Real.log n * z m q k)
  have hA : 0 < A := Real.exp_pos _
  have hnear (k : Fin (m + 1)) (hk : k ∈ (N m q).erase q) :
      A * n ^ (-(s * g₁)) ≤ Real.exp (s * Real.log n * z m q k) := by
    rw [Real.rpow_def_of_pos hn0, ← Real.exp_add]
    refine Real.exp_le_exp.2 ?_
    have := mul_le_mul_of_nonneg_left (hz m q k (mem_of_mem_erase hk)) hc0
    linarith
  have hSy : (((N m q).erase q).card : ℝ) * (A * n ^ (-(s * g₁))) ≤ S :=
    calc (((N m q).erase q).card : ℝ) * (A * n ^ (-(s * g₁)))
        = ∑ _k ∈ (N m q).erase q, A * n ^ (-(s * g₁)) := by rw [sum_const, nsmul_eq_mul]
      _ ≤ ∑ k ∈ (N m q).erase q, Real.exp (s * Real.log n * z m q k) := sum_le_sum hnear
      _ ≤ S := sum_le_sum_of_subset_of_nonneg (erase_subset_erase q (subset_univ _))
          fun _ _ _ => (Real.exp_pos _).le
  have hy' : ε / (2 - ε) < ((N m q).erase q).card * n ^ (-(s * g₁)) :=
    calc ε / (2 - ε) < c * n ^ (α - s * g₁) := hy
      _ = c * n ^ α * n ^ (-(s * g₁)) := by
          rw [mul_assoc, ← Real.rpow_add hn0, sub_eq_add_neg]
      _ ≤ _ := mul_le_mul_of_nonneg_right (hNm q) (Real.rpow_nonneg hn0.le _)
  have h2ε : 0 < 2 - ε := by linarith
  rw [div_lt_iff₀ h2ε] at hy'
  rw [lt_div_iff₀ (by positivity)]
  nlinarith [mul_lt_mul_of_pos_left hy' hA]

/-- Every key a near miss, trailing by `1/2` at `s = 1`: the error tends to `2`. -/
example := exists_lt_logScaleAttention_nearMiss_error (s := 1) (g₁ := 1 / 2) (α := 1)
  (c := 1 / 2) (ε := 1) zero_le_one (by norm_num) (by norm_num) (by norm_num)
  (fun _ q k => if k = q then 1 / 2 else 0) (fun _ _ => univ)
  (eventually_atTop.2 ⟨1, fun m hm q => by
    rw [card_erase_of_mem (mem_univ q), card_univ, Fintype.card_fin, Nat.add_sub_cancel,
      Real.rpow_one]
    have : (1 : ℝ) ≤ m := by exact_mod_cast hm
    linarith⟩)
  (fun _ q k _ => by split_ifs <;> norm_num)

end Transformer.MRCR
