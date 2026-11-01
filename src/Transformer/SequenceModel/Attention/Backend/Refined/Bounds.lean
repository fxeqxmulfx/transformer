module

public import Transformer.SequenceModel.Attention.Backend.Refined.Weights

/-!
# Verified recorded norm bounds and local normalizer gains

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M3.
Selectors preserve the source fused layout. The norm inequalities are
checked from the stored dyadic entries, without a supplied operator norm.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive
open scoped BigOperators
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

/-- M3: squared recorded denominator check for the pre RMS gain. -/
theorem gain_pre (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (21 / 10 : ℝ) ^ 2 * (‖raw_initial i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the qnorm RMS gain. -/
theorem gain_qnorm (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (6000 : ℝ) ^ 2 * (‖qPart i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the knorm RMS gain. -/
theorem gain_knorm (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (6000 : ℝ) ^ 2 * (‖kPart i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the attn RMS gain. -/
theorem gain_attn :
    (4 * (4 : ℝ)) ≤ (60 : ℝ) ^ 2 * (‖raw_projected 1‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the ffn_pre RMS gain. -/
theorem gain_ffn_pre (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (21 / 10 : ℝ) ^ 2 * (‖raw_mid i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the ffn_out RMS gain. -/
theorem gain_ffn_out (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (1500 : ℝ) ^ 2 * (‖raw_ffn_projected i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: squared recorded denominator check for the final RMS gain. -/
theorem gain_final (i : Fin 2) :
    (4 * (4 : ℝ)) ≤ (3 / 2 : ℝ) ^ 2 * (‖raw_output i‖ ^ 2 + 4 * (1 / 8388608)) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  fin_cases i <;> norm_num [raw_initial, qPart, kPart, raw_projected, raw_mid, raw_ffn_projected, raw_output, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded qnorm tensor lies in this verified Euclidean ball. -/
theorem bound_qnorm (i : Fin 2) : ‖raw_qnorm i‖ ≤ 2 / 5 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded knorm tensor lies in this verified Euclidean ball. -/
theorem bound_knorm (i : Fin 2) : ‖raw_knorm i‖ ≤ 2 / 5 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded query tensor lies in this verified Euclidean ball. -/
theorem bound_query (i : Fin 2) : ‖raw_query i‖ ≤ 2 / 5 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded key tensor lies in this verified Euclidean ball. -/
theorem bound_key (i : Fin 2) : ‖raw_key i‖ ≤ 2 / 5 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded value tensor lies in this verified Euclidean ball. -/
theorem bound_value (i : Fin 2) : ‖vPart i‖ ≤ 11 / 10 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the recorded average tensor lies in this verified Euclidean ball. -/
theorem bound_average (i : Fin 2) : ‖raw_average i‖ ≤ 11 / 10 := by
  apply norm_checked _ _ (by norm_num)
  fin_cases i <;> norm_num [raw_qnorm, raw_knorm, raw_query, raw_key, raw_average, vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: recorded own-values have norm at least one; clipping is inactive there. -/
theorem gain_direction (i : Fin 2) : 2 ≤ 2 * max ‖vPart i‖ (1 / 1000000000000) := by
  have hsq : (1 : ℝ) ≤ ‖vPart i‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    fin_cases i <;> norm_num [vPart, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]
  have hn : (1 : ℝ) ≤ ‖vPart i‖ := by nlinarith [norm_nonneg (vPart i)]
  linarith [le_max_left ‖vPart i‖ (1 / 1000000000000 : ℝ)]

/-- M3: verified coordinate multiplier bound at the recorded gate input. -/
theorem coordinates_gate (i : Fin 2) (c : Fin 11) : |raw_gate i c| ≤ 1 / 4 := by
  fin_cases i <;> fin_cases c <;> norm_num [raw_gate]

/-- M3: verified coordinate multiplier bound at the recorded value input. -/
theorem coordinates_value (i : Fin 2) (c : Fin 11) : |raw_value i c| ≤ 1 / 4 := by
  fin_cases i <;> fin_cases c <;> norm_num [raw_value]

/-- M3: verified coordinate multiplier bound at the recorded activated input. -/
theorem coordinates_activated (i : Fin 2) (c : Fin 11) : |raw_activated i c| ≤ 1 / 8 := by
  fin_cases i <;> fin_cases c <;> norm_num [raw_activated]

end Transformer.SequenceModel.Attention.Backend.Refined
