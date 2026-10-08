import Transformer.Grokking.CircuitEfficiency.SectionC_GainSmallDecayConfidence

/-!
# Original small-decay absent-Gen control with permanent wrong test answers

Sources: Varma et al., arXiv:2309.02390v1, section 3's three ingredients
and appendix C product/test logits; unit-gain zero-pair invariant at
6c335bd, ported to the gained original path at lab commit bfd4d4f.

Set both Gen factors and their initialized buffers to zero. Mem starts
(0,1) and forms through the actual original clipped-CE feedback with
beta1=0.9, beta2=0.98, epsilon=1e-8, decay=0.1, rate=0.001 and cap=1.
Prove the entire actual Gen state at every clock, including zero buffers
and its increasing completed clock, rather than prescribing zero logits.
Zero current gradients alone would not suffice with stale first moments.

The efficient Gen pair stays absent. Its true gained score is zero,
while actual Mem score is positive at every successor. Every successor
therefore uniquely selects the wrong Mem held-out class; the true target
is never uniquely correct, including the initially tied zero clock.
True held-out CE is strictly worse than uniform initialization at all
successors, with every finite class included in the denominator.

On this same original path train answers are uniquely correct, train
confidence is above uniform and train CE below uniform-class loss.
Those positive training measurements coexist with permanently wrong
held-out answers. This is an actual native control against training-only
grokking detection, not an attracting trajectory supplied as a premise.

The missing positive Gen seed violates the source's slow-formation
initialization intentionally. It does not refute that positive-seed
simulation or model stochastic discovery of new learned heads. Fixed
tables and uniform decoupled native decay differ from coupled-cost GD
and learned floating-point GPTMini. No optimizer or buffer reset is used.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Original true held-out logits from both actual gained products.
Sources: appendix C distinct Gen/Mem test labels and original path
at bfd4d4f; class count, seed and clock all enter the numerical forward. -/
noncomputable def smallDecayGainHeldoutLogits (remaining : ℕ) (seed : ℝ) (n : ℕ) : Fin (remaining + 2) → ℝ :=
  heldoutTableLogits remaining
    (physicalCircuitScore 3 (smallDecayGainPath remaining seed n 0).parameter (smallDecayGainPath remaining seed n 1).parameter)
    (physicalCircuitScore 2 (smallDecayGainPath remaining seed n 2).parameter (smallDecayGainPath remaining seed n 3).parameter)

/-- True held-out full CE of that original forward. Sources: appendix C
test denominator and original path at bfd4d4f; the target stays zero,
and the Mem label is a distinct competitor rather than a successful label. -/
noncomputable def smallDecayGainHeldoutCE (remaining : ℕ) (seed : ℝ) (n : ℕ) : ℝ :=
  crossEntropy (smallDecayGainHeldoutLogits remaining seed n) 0

/-- Fully absent initial Gen has zero actual parameters/buffers and
its original growing clock forever. Sources: appendix C product inputs
and native zero-pair proof at 6c335bd, ported to actual gained feedback;
neither future zero gradients nor a buffer reset is assumed. -/
theorem small_decay_gain_absent_gen_history (remaining n : ℕ) :
    smallDecayGainPath remaining 0 n 0 = zeroScalarStateAt n ∧
      smallDecayGainPath remaining 0 n 1 = zeroScalarStateAt n := by
  induction n with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih =>
    have hg0 : appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining 0 n) 0 = 0 := by
      rw [gain_native_applied_gradient_scale]
      change -(gainCEGradientScale remaining 3 2 1 (smallDecayGainPath remaining 0 n) * 3 *
        (smallDecayGainPath remaining 0 n 1).parameter) = 0
      rw [ih.2]
      norm_num [zeroScalarStateAt]
    have hg1 : appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining 0 n) 1 = 0 := by
      rw [gain_native_applied_gradient_scale]
      change -(gainCEGradientScale remaining 3 2 1 (smallDecayGainPath remaining 0 n) * 3 *
        (smallDecayGainPath remaining 0 n 0).parameter) = 0
      rw [ih.1]
      norm_num [zeroScalarStateAt]
    constructor
    · change scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (smallDecayGainPath remaining 0 n 0)
        (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining 0 n) 0) = zeroScalarStateAt (n + 1)
      rw [hg0, ih.1]
      exact scalar_native_zero_step _ _ _ _ _ n
    · change scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (smallDecayGainPath remaining 0 n 1)
        (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining 0 n) 1) = zeroScalarStateAt (n + 1)
      rw [hg1, ih.2]
      exact scalar_native_zero_step _ _ _ _ _ n

/-- True gained Gen score is zero from its actual full retained
history. Sources: appendix C products and zero-pair proof at 6c335bd;
the output follows trained state invariance rather than a zero-logit input. -/
theorem small_decay_gain_absent_gen_score (remaining n : ℕ) :
    physicalCircuitScore 3 (smallDecayGainPath remaining 0 n 0).parameter
      (smallDecayGainPath remaining 0 n 1).parameter = 0 := by
  rw [(small_decay_gain_absent_gen_history remaining n).1,
    (small_decay_gain_absent_gen_history remaining n).2]
  norm_num [physicalCircuitScore, zeroScalarStateAt]

/-- Original Mem forms a positive product at every successor despite
absent Gen. Sources: section 3 formation, appendix C common training
labels and generated actual successor correctness at 722638e;
positive Mem output is derived, not supplied as a trajectory hypothesis. -/
theorem small_decay_gain_absent_mem_positive_successor (remaining n : ℕ) :
    0 < physicalCircuitScore 2 (smallDecayGainPath remaining 0 (n + 1) 2).parameter
      (smallDecayGainPath remaining 0 (n + 1) 3).parameter := by
  have hp := (train_table_strict_correct_iff remaining _ _).mp
    (small_decay_gain_train_correct_successors remaining 0 n le_rfl)
  rw [small_decay_gain_absent_gen_score, zero_add] at hp
  exact hp

/-- The actual wrong Mem held-out label is uniquely largest at every
successor. Sources: appendix C distinct test labels and original native
formation at 722638e; all other finite classes have strictly lower logits. -/
theorem small_decay_gain_absent_heldout_wrong_successors (remaining n : ℕ) :
    StrictCorrect (smallDecayGainHeldoutLogits remaining 0 (n + 1)) (0 : Fin (remaining + 1)).succ := by
  unfold smallDecayGainHeldoutLogits
  rw [small_decay_gain_absent_gen_score]
  have hp := small_decay_gain_absent_mem_positive_successor remaining n
  intro k hk
  have hw : heldoutTableLogits remaining 0
      (physicalCircuitScore 2 (smallDecayGainPath remaining 0 (n + 1) 2).parameter
        (smallDecayGainPath remaining 0 (n + 1) 3).parameter) (0 : Fin (remaining + 1)).succ =
      physicalCircuitScore 2 (smallDecayGainPath remaining 0 (n + 1) 2).parameter
        (smallDecayGainPath remaining 0 (n + 1) 3).parameter := by simp [heldoutTableLogits]
  rw [hw]
  by_cases hz : k = 0
  · rw [hz]
    simpa only [heldoutTableLogits, ite_true] using hp
  · simpa only [heldoutTableLogits, hz, hk, ite_false] using hp

/-- Original true held-out strict correctness is absent at every clock.
Sources: appendix C competing test labels and native zero-pair proof
at 6c335bd; initial ties and uniquely wrong trained successors are distinct. -/
theorem small_decay_gain_absent_never_heldout_correct (remaining n : ℕ) :
    ¬StrictCorrect (smallDecayGainHeldoutLogits remaining 0 n) 0 := by
  intro hcorrect
  cases n with
  | zero =>
    have hh := hcorrect (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
    norm_num [smallDecayGainHeldoutLogits, smallDecayGainPath, gainNativePath,
      seededNativeSubweights, seededScalarState, physicalCircuitScore, heldoutTableLogits] at hh
  | succ n =>
    have hw := small_decay_gain_absent_heldout_wrong_successors remaining n 0 (Ne.symm (Fin.succ_ne_zero _))
    have ht := hcorrect (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
    linarith only [hw, ht]

/-- True held-out CE is strictly worse than uniform at every successor.
Sources: appendix C full test CE and original native Mem formation at
722638e; a confident wrong label enlarges the actual complete denominator. -/
theorem small_decay_gain_absent_heldout_ce_worse_successors (remaining n : ℕ) :
    Real.log ((remaining : ℝ) + 2) < smallDecayGainHeldoutCE remaining 0 (n + 1) := by
  unfold smallDecayGainHeldoutCE smallDecayGainHeldoutLogits
  rw [small_decay_gain_absent_gen_score, table_heldout_ce_formula, Real.exp_zero, sub_zero]
  have he := Real.exp_lt_exp.mpr (small_decay_gain_absent_mem_positive_successor remaining n)
  rw [Real.exp_zero] at he
  exact Real.log_lt_log (by positivity) (by linarith only [he])

/-- Positive train decisions/confidence/CE measurements coexist with
permanently wrong actual test answers on one original native control.
Sources: section 3's two circuits, appendix C full logits/CE and actual
weak-decay observations at bfd4d4f; no successful future path is assumed. -/
theorem small_decay_gain_absent_training_control (remaining n : ℕ) :
    StrictCorrect (smallDecayGainTrainLogits remaining 0 (n + 1)) 0 ∧
      1 / ((remaining : ℝ) + 2) < smallDecayGainTrainProbability remaining 0 (n + 1) ∧
      smallDecayGainTrainCE remaining 0 (n + 1) < Real.log ((remaining : ℝ) + 2) ∧
      StrictCorrect (smallDecayGainHeldoutLogits remaining 0 (n + 1)) (0 : Fin (remaining + 1)).succ ∧
      ¬StrictCorrect (smallDecayGainHeldoutLogits remaining 0 (n + 1)) 0 ∧
      Real.log ((remaining : ℝ) + 2) < smallDecayGainHeldoutCE remaining 0 (n + 1) := by
  have ht := small_decay_gain_train_confident_successors remaining 0 n le_rfl
  exact ⟨small_decay_gain_train_correct_successors remaining 0 n le_rfl, ht.1, ht.2,
    small_decay_gain_absent_heldout_wrong_successors remaining n,
    small_decay_gain_absent_never_heldout_correct remaining (n + 1),
    small_decay_gain_absent_heldout_ce_worse_successors remaining n⟩

end Transformer.Grokking.CircuitEfficiency
