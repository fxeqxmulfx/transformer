module

public import Transformer.SequenceModel.Attention.Backend.Complete.Trace
public import Transformer.SequenceModel.Attention.Execution.Readout

/-!
# A lower bound on the existing global execution budget

Source: `review/flat_spectrum.md`, *Complete captured execution*, L4.
The global recurrence charges the same operation allowance at embedding and
every normalization. Positive residual gains cannot undo that initial charge.
This is a property of the sufficient bound, not a lower bound on model error.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend.Complete

open GPTMini Stability Execution

/-- L4: the global recurrence never reduces its uniform initial error charge. -/
theorem budget_ge_initial (cfg : Config) (θ : Attention.ModelParams cfg)
    (eps xeps δ τ : ℝ) (hx : 0 ≤ xeps) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ) (n : ℕ) :
    rmsLip eps * δ + δ ≤ modelBudget cfg θ eps xeps δ τ n := by
  have h0 : 0 ≤ rmsLip eps * δ + δ := by unfold rmsLip; positivity
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
      rw [modelBudget]
      split_ifs with hn
      · let B := θ.blocks ⟨n, hn⟩
        have hf : 0 ≤ rmsLip eps * ffnLip cfg B * rmsLip eps := by
          unfold ffnLip rmsLip
          positivity
        have ha : 0 ≤ attentionGain cfg B eps *
            (2 * averageLip cfg B eps + directionGain cfg B eps xeps) := by
          unfold attentionGain averageLip directionGain componentLip valueBound rmsLip
          positivity
        have hL : 1 ≤ blockLip cfg B eps xeps := by
          unfold blockLip ffnGain
          nlinarith [mul_nonneg hf ha]
        have hR : 0 ≤ blockRadius cfg B eps xeps δ τ := by
          dsimp [blockRadius, attentionBudget, ffnBudget, ffnGain, ffnLip, valueBound, rmsLip]
          positivity
        have hp := mul_le_mul_of_nonneg_right hL (h0.trans ih)
        change _ ≤ blockLip cfg B eps xeps * modelBudget cfg θ eps xeps δ τ n + _
        linarith
      · exact ih

/-- L4: every stored tied embedding has norm exactly one. -/
theorem embedding_unit (v : Fin cfg.vocab_size) : ‖weights.embedding v‖ = 1 := by
  fin_cases v <;> norm_num [weights, EuclideanSpace.norm_eq, Fin.sum_univ_succ]

/-- L4: the old bound gives radius greater than two for any allowance at least 2^-24. -/
theorem radius_gt_two (δ τ : ℝ) (hδ : (1 / 16777216 : ℝ) ≤ δ) (hτ : 0 ≤ τ)
    (v : Fin cfg.vocab_size) :
    2 < logitRadius cfg (1 / 8388608) (weights.embedding v)
      (modelBudget cfg weights (1 / 8388608) (1 / 1000000000000) δ τ cfg.n_layers) δ := by
  have hδ0 : 0 < δ := by linarith
  let L := rmsLip (1 / 8388608)
  have hL : 0 < L := by dsimp [L, rmsLip]; positivity
  have hL2 : L ^ 2 = 33554432 := by
    dsimp [L, rmsLip]
    rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 1 / 8388608)]
    norm_num
  have hB := budget_ge_initial cfg weights (1 / 8388608) (1 / 1000000000000)
    δ τ (by norm_num) hδ0.le hτ cfg.n_layers
  have hM := mul_le_mul_of_nonneg_left hB hL.le
  have he : L * (L * δ + δ) = 33554432 * δ + L * δ := by
    calc
      _ = L ^ 2 * δ + L * δ := by ring
      _ = _ := by rw [hL2]
  change L * (L * δ + δ) ≤ _ at hM
  rw [he] at hM
  unfold logitRadius
  rw [embedding_unit]
  norm_num [cfg]
  change 2 < (L * _ + δ) + (2 + δ) * δ + δ
  nlinarith [mul_pos hL hδ0]

/-- L4: no exact source margin with this tied unit table can exceed four. -/
theorem source_margin_le_four (x : EucSpace 4) (u v : Fin cfg.vocab_size) :
    inner ℝ (rmsNormEps (1 / 8388608) x) (weights.embedding u) -
      inner ℝ (rmsNormEps (1 / 8388608) x) (weights.embedding v) ≤ 4 := by
  have hn := rmsNormEps_norm_le (1 / 8388608) (by norm_num) (by decide : 0 < 4) x
  norm_num at hn
  have hu := abs_real_inner_le_norm (rmsNormEps (1 / 8388608) x) (weights.embedding u)
  have hv := abs_real_inner_le_norm (rmsNormEps (1 / 8388608) x) (weights.embedding v)
  rw [embedding_unit, mul_one] at hu hv
  have hu' := (abs_le.mp (hu.trans hn)).2
  have hv' := (abs_le.mp (hv.trans hn)).1
  linarith

example := budget_ge_initial cfg weights (1 / 8388608) (1 / 1000000000000)
  (1 / 10000) (2 / 1000000) (by norm_num) (by norm_num) (by norm_num) 1
example := radius_gt_two (1 / 10000) (2 / 1000000) (by norm_num) (by norm_num) 0

end Transformer.SequenceModel.Attention.Backend.Complete
