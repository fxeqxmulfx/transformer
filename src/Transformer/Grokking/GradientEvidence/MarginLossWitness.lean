module

public import Transformer.Grokking.GradientEvidence.JointLoss

/-!
# Increasing mean margin can worsen actual mean cross-entropy

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC4.
The old binary margins are log 8 on both queries. New margins are log 256
and log 2. Both predictions remain strictly correct, the mean margin grows,
and the actual mean CE grows because the difficult query deteriorates.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset Geometry Landscape.Neural

/-- Actual binary CE for a positive target-to-competitor odds ratio, JC4. -/
theorem binary_log_odds_loss (a : ℝ) (ha : 0 < a) :
    crossEntropy (fun c : Bool => if c then Real.log a else 0) true =
      Real.log (1 + 1 / a) := by
  rw [crossEntropy_eq_log_odds]
  simp [oddsEnvelope, Real.exp_neg, Real.exp_log ha, one_div]

example := binary_log_odds_loss 8 (by norm_num)

/-- A nonempty two-query counterexample to inferring lower CE from a
larger mean margin, JC4. Both new margins are positive; no error is removed. -/
theorem mean_margin_loss_counterexample :
    let old : Bool → Bool → ℝ := fun _ c => if c then Real.log 8 else 0
    let next : Bool → Bool → ℝ := fun i c =>
      if c then (if i then Real.log 256 else Real.log 2) else 0
    (∑ i : Bool, (old i true - old i false)) / 2 <
      (∑ i : Bool, (next i true - next i false)) / 2 ∧
      meanCrossEntropy univ old (fun _ => true) < meanCrossEntropy univ next (fun _ => true) ∧
      (∀ i, next i false < next i true) ∧ next false true < old false true := by
  dsimp only
  have hm : Real.log 8 + Real.log 8 < Real.log 256 + Real.log 2 := by
    rw [← Real.log_mul (by norm_num : (8 : ℝ) ≠ 0) (by norm_num : (8 : ℝ) ≠ 0),
      ← Real.log_mul (by norm_num : (256 : ℝ) ≠ 0) (by norm_num : (2 : ℝ) ≠ 0)]
    apply Real.log_lt_log <;> norm_num
  have hl : Real.log (1 + 1 / (8 : ℝ)) + Real.log (1 + 1 / (8 : ℝ)) <
      Real.log (1 + 1 / (256 : ℝ)) + Real.log (1 + 1 / (2 : ℝ)) := by
    rw [← Real.log_mul (by norm_num : (1 + 1 / (8 : ℝ)) ≠ 0) (by norm_num),
      ← Real.log_mul (by norm_num : (1 + 1 / (256 : ℝ)) ≠ 0) (by norm_num)]
    apply Real.log_lt_log <;> norm_num
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Fintype.sum_bool, ite_true, Bool.false_eq_true, ite_false, sub_zero]
    linarith
  · unfold meanCrossEntropy
    simp only [Fintype.sum_bool, card_univ, Fintype.card_bool, Nat.cast_ofNat,
      ite_true, Bool.false_eq_true, ite_false]
    rw [binary_log_odds_loss 8 (by norm_num), binary_log_odds_loss 256 (by norm_num),
      binary_log_odds_loss 2 (by norm_num)]
    linarith
  · intro i
    cases i <;> exact Real.log_pos (by norm_num)
  · exact Real.log_lt_log (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
