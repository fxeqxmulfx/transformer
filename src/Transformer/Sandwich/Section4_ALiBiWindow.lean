module

public import Mathlib.Algebra.Order.Field.GeomSum
public import Mathlib.Analysis.SpecialFunctions.Exp
public import Transformer.Sandwich.Section3_Biases

/-!
# §4 ALiBi and windowed attention

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §4.2 (Slope Equalization): "only large slopes
(`1/2^h`), or equivalently small `h`, allow a model to extrapolate well. Large slopes implicitly
enforce a narrow windowed bias on the self-attention matrix such that distant tokens cannot
interact with each other."

ALiBi's bias `−t/(2^h √k)` at the distance `t` never masks a key (`attended_coe` in
`SectionB_EfficientInference`): the window is implicit. With the scaled logits
`q_m^⊤ k_n/√k` bounded by `C` and the decay rate `s = 1/(2^h √k)`:

* `attnWeight_alibi_le`: the weight at the distance `t` is at most `exp(2C − s t)`;
* `sum_attnWeight_alibi_far_le`: the keys at the distances `≥ w` together get at most
  `exp(2C) e^{−s w}/(1 − e^{−s})`;
* `tendsto_alibi_far_bound`: that bound tends to `0` as the slope grows, for every `w ≥ 1`. The
  weight of the distant tokens shrinks with the slope but stays positive.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Filter Topology Transformer.BAM

variable {d k : ℕ}

/-- **ALiBi's weight decays exponentially with the distance** (§4.2): with the scaled logits
bounded by `C`, the key `n` of the query `m` gets at most `exp(2C − s(m − n))`, `s = 1/(2^h √k)`. -/
theorem attnWeight_alibi_le {A : Head d k} {h C : ℝ} {x : Embs d} {m : ℕ}
    (hC : ∀ i ≤ m, |(A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x i) / Real.sqrt k| ≤ C) {n : ℕ} (hn : n ≤ m) :
    attnWeight A (alibiBias k h) x m n ≤
      Real.exp (2 * C - slope h / Real.sqrt k * (m - n : ℕ)) := by
  have hw : ∀ i, weight (score A (alibiBias k h) x m i) = Real.exp
      ((A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x i) / Real.sqrt k - slope h / Real.sqrt k * (m - i : ℕ)) :=
    fun i => by
      simp only [score, alibiBias, ← EReal.coe_add, weight_coe]
      congr 1
      ring
  have hden : Real.exp (-C) ≤ ∑ i ∈ range (m + 1), weight (score A (alibiBias k h) x m i) := by
    calc Real.exp (-C) ≤ weight (score A (alibiBias k h) x m m) := by
          rw [hw, Nat.sub_self, Nat.cast_zero, mul_zero, sub_zero]
          exact Real.exp_le_exp.2 (abs_le.1 (hC m le_rfl)).1
      _ ≤ _ := single_le_sum (fun i _ => weight_nonneg _) (mem_range.2 (Nat.lt_succ_self m))
  rw [attnWeight, div_le_iff₀ (lt_of_lt_of_le (Real.exp_pos _) hden)]
  calc weight (score A (alibiBias k h) x m n)
      ≤ Real.exp (2 * C - slope h / Real.sqrt k * (m - n : ℕ)) * Real.exp (-C) := by
        rw [hw, ← Real.exp_add]
        exact Real.exp_le_exp.2 (by linarith [(abs_le.1 (hC n hn)).2])
    _ ≤ _ := mul_le_mul_of_nonneg_left hden (Real.exp_pos _).le

/-- The hypotheses of `attnWeight_alibi_le` hold for zero queries and keys, `C = 0`. -/
example (x : Embs 2) (m : ℕ) :
    attnWeight (⟨0, 0, 1⟩ : Head 2 2) (alibiBias 2 1) x m 0 ≤
      Real.exp (2 * 0 - slope 1 / Real.sqrt (2 : ℕ) * (m - 0 : ℕ)) :=
  attnWeight_alibi_le (fun i _ => by simp) (Nat.zero_le m)

/-- **Large slopes enforce a window** (§4.2): with the scaled logits bounded by `C`, the keys at
the distances `w, …, m` together get at most `exp(2C) e^{−s w}/(1 − e^{−s})`,
`s = 1/(2^h √k)`. -/
theorem sum_attnWeight_alibi_far_le (hk : 0 < k) {A : Head d k} {h C : ℝ} {x : Embs d} {m : ℕ}
    (hC : ∀ i ≤ m, |(A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x i) / Real.sqrt k| ≤ C) (w : ℕ) :
    ∑ t ∈ Ico w (m + 1), attnWeight A (alibiBias k h) x m (m - t) ≤
      Real.exp (2 * C) * (Real.exp (-(slope h / Real.sqrt k)) ^ w /
        (1 - Real.exp (-(slope h / Real.sqrt k)))) := by
  have hs : 0 < slope h / Real.sqrt k :=
    div_pos (by simp only [slope]; positivity) (Real.sqrt_pos.2 (by exact_mod_cast hk))
  have hq : Real.exp (-(slope h / Real.sqrt k)) < 1 := Real.exp_lt_one_iff.2 (by linarith)
  calc ∑ t ∈ Ico w (m + 1), attnWeight A (alibiBias k h) x m (m - t)
      ≤ ∑ t ∈ Ico w (m + 1), Real.exp (2 * C) * Real.exp (-(slope h / Real.sqrt k)) ^ t :=
        sum_le_sum fun t ht => by
          have htm : t ≤ m := by rw [mem_Ico] at ht; omega
          have h1 := attnWeight_alibi_le (h := h) hC (Nat.sub_le m t)
          rw [Nat.sub_sub_self htm] at h1
          rw [← Real.exp_nat_mul, ← Real.exp_add]
          convert h1 using 2
          ring
    _ = Real.exp (2 * C) * ∑ t ∈ Ico w (m + 1), Real.exp (-(slope h / Real.sqrt k)) ^ t :=
        (mul_sum _ _ _).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left
        (geom_sum_Ico_le_of_lt_one (Real.exp_pos _).le hq) (Real.exp_pos _).le

/-- The hypotheses of `sum_attnWeight_alibi_far_le` hold for zero queries and keys, `C = 0`. -/
example (x : Embs 2) (m w : ℕ) :
    ∑ t ∈ Ico w (m + 1), attnWeight (⟨0, 0, 1⟩ : Head 2 2) (alibiBias 2 1) x m (m - t) ≤
      Real.exp (2 * 0) * (Real.exp (-(slope 1 / Real.sqrt (2 : ℕ))) ^ w /
        (1 - Real.exp (-(slope 1 / Real.sqrt (2 : ℕ))))) :=
  sum_attnWeight_alibi_far_le (by norm_num) (fun i _ => by simp) w

/-- **The far mass vanishes as the slope grows** (§4.2): for every `w ≥ 1` the bound of
`sum_attnWeight_alibi_far_le` tends to `0` as the decay rate `s` tends to `∞`. -/
theorem tendsto_alibi_far_bound (C : ℝ) {w : ℕ} (hw : 1 ≤ w) :
    Tendsto (fun s => Real.exp (2 * C) * (Real.exp (-s) ^ w / (1 - Real.exp (-s)))) atTop
      (𝓝 0) := by
  have h := Real.tendsto_exp_neg_atTop_nhds_zero
  have h1 : Tendsto (fun s : ℝ => 1 - Real.exp (-s)) atTop (𝓝 (1 - 0)) :=
    tendsto_const_nhds.sub h
  have := ((h.pow w).div h1 (by norm_num)).const_mul (Real.exp (2 * C))
  simpa [zero_pow (by omega : w ≠ 0)] using this

/-- The hypothesis of `tendsto_alibi_far_bound` holds for the distances `≥ 1`. -/
example : Tendsto (fun s => Real.exp (2 * 0) * (Real.exp (-s) ^ 1 / (1 - Real.exp (-s)))) atTop
    (𝓝 0) :=
  tendsto_alibi_far_bound 0 le_rfl

end Transformer.Sandwich
