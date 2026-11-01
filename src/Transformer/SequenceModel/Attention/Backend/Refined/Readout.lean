module

public import Transformer.SequenceModel.Attention.Backend.Refined.FFN
public import Transformer.SequenceModel.Attention.Backend.Complete.Obstruction
public import Transformer.SequenceModel.Attention.Spectral.Decisions

/-!
# Certified agreement between the captured and exact source classifiers

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M4.
The margin is derived from the captured logits and the fully propagated
error. The earlier uniform-budget impossibility theorem remains intact.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive
open scoped Classical

/-- M4: every captured class logit is within 1/16 of the exact source score. -/
theorem error_logits (i : Fin 2) (v : Fin 3) :
    |captured.logits i v - sourceLogits i v| ≤ 1 / 16 := by
  have hi : |inner ℝ (raw_normalized i) (weights.embedding v) - sourceLogits i v| ≤ 55 / 1000 := by
    unfold sourceLogits
    rw [← inner_sub_left]
    exact (abs_real_inner_le_norm _ _).trans
      (by rw [embedding_unit, mul_one]; exact error_normalized i)
  have ht := (abs_sub_le (captured.logits i v) (inner ℝ (raw_normalized i) (weights.embedding v))
    (sourceLogits i v)).trans (add_le_add (local_logits i v) hi)
  exact ht.trans (by norm_num)

/-- The two captured strict winners, retaining the original token indices. -/
def winner : Fin 2 → Fin 3 := ![0, 2]

/-- M4: a rational positive gap read directly from the frozen logits. -/
theorem captured_gap (i : Fin 2) (v : Fin 3) (hv : v ≠ winner i) :
    (17 / 20 : ℝ) ≤ captured.logits i (winner i) - captured.logits i v := by
  fin_cases i <;> fin_cases v <;> norm_num [winner, captured, raw_logits] at *

/-- M4: the exact source margin follows from the local operation certificates. -/
theorem source_margin (i : Fin 2) (v : Fin 3) (hv : v ≠ winner i) :
    (7 / 10 : ℝ) ≤ sourceLogits i (winner i) - sourceLogits i v := by
  have h₁ := (abs_le.mp (error_logits i (winner i))).2
  have h₂ := (abs_le.mp (error_logits i v)).1
  have hgap := captured_gap i v hv
  linarith

/-- M4: every class decision, including every rejected class, agrees. -/
theorem decisions_agree (i : Fin 2) (v : Fin 3) :
    decide (∀ w, w ≠ v → sourceLogits i w < sourceLogits i v) = captured.predict i v := by
  have hs : ∀ w, w ≠ winner i → sourceLogits i w < sourceLogits i (winner i) := by
    intro w hw
    have h := source_margin i w hw
    linarith
  have hc : captured.predict i (winner i) = true := by
    apply decide_eq_true
    intro w hw
    have h := captured_gap i w hw
    linarith
  apply decide_eq_decide.mpr
  exact (decide_eq_decide.mp (Spectral.strict_tests_eq (sourceLogits i) (winner i) hs v)).trans
    (decide_eq_decide.mp (Spectral.executed_tests_eq cfg captured i (winner i) hc v)).symm

/-- M4: the new radius is usable on the same trace whose old global budget fails. -/
theorem improved_radius :
    (∀ i v, |captured.logits i v - sourceLogits i v| ≤ 1 / 16) ∧
    (∀ i v, v ≠ winner i → (7 / 10 : ℝ) ≤ sourceLogits i (winner i) - sourceLogits i v) :=
  ⟨error_logits, source_margin⟩

example := captured_gap 0 1 (by decide)
example := source_margin 1 0 (by decide)

end Transformer.SequenceModel.Attention.Backend.Refined
