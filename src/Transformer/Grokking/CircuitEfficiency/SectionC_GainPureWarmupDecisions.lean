import Transformer.Grokking.CircuitEfficiency.SectionC_GainPureWarmupPaths
import Transformer.Grokking.CircuitEfficiency.SectionC_TrainProbability

/-!
# Opposite held-out decisions under the same original native settings

Sources: Varma et al., arXiv:2309.02390v1, appendix C Gen/Mem train
and test tables; original native recipe at 0033b1b and actual
balanced histories at 68402e2. Evaluate the physical logits of the
proved pure-circuit original-warmup trajectories at every clock.

Either pure path uniquely fits the training class forever. The Gen
path uniquely selects the correct held-out class forever; the Mem
path uniquely selects the distinct wrong held-out class forever.
Both paths have positive proved amplitudes with exactly the same
betas, epsilon, decay, shared cap and completed-update warmup.
No feedback, moment or successful-label assumption replaces their
actual complete CE, physical parameters and retained histories.

This removes the formerly chosen-epsilon Mem boundary witness:
the obstruction now holds at the fixed original numerical settings.
It does not establish attraction, explain delayed formation from
competing positive-partner source seeds or transfer to learned GPTMini.
The models use fixed physical readout tables and exact arithmetic;
source norm-cost GD, stochastic and numerical bridges remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW
open Transformer.Grokking.NaiveLoss

/-- Both physical circuit scores on the actual original-warmup
trajectory. Sources: appendix C products and native gained feedback
at c268c1f; every argument enters the actual generated state. -/
noncomputable def originalPureCircuitScores (remaining : ℕ) (side : Fin 2) (amplitude : ℝ) (clock : ℕ) : ℝ × ℝ :=
  let state := gainNativeWarmupPath remaining 3 2 1 (1 / 100000000) (originalPureCircuitPoint side amplitude) clock
  (physicalCircuitScore 3 (state 0).parameter (state 1).parameter,
    physicalCircuitScore 2 (state 2).parameter (state 3).parameter)

/-- Actual training logits with both generated circuit scores.
Source: appendix C sim-overall-logits; label zero is the task target. -/
noncomputable def originalPureCircuitTrainLogits (remaining : ℕ) (side : Fin 2) (amplitude : ℝ) (clock : ℕ) : Fin (remaining + 2) → ℝ :=
  let scores := originalPureCircuitScores remaining side amplitude clock
  trainTableLogits remaining scores.1 scores.2

/-- Actual held-out logits; the Mem class differs from the target.
Source: appendix C sim-gen/mem-logits, with all finite competitors. -/
noncomputable def originalPureCircuitHeldoutLogits (remaining : ℕ) (side : Fin 2) (amplitude : ℝ) (clock : ℕ) : Fin (remaining + 2) → ℝ :=
  let scores := originalPureCircuitScores remaining side amplitude clock
  heldoutTableLogits remaining scores.1 scores.2

/-- Actual original-warmup roots give their pure gained products
at every retained clock. Sources: appendix C product readouts and
native balanced histories at 68402e2; the output is derived from
all four actual parameters rather than assigned to the trajectory. -/
theorem original_pure_circuit_warmup_scores (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ n, originalPureCircuitScores remaining side amplitude n =
      if side = 0 then (3 * (amplitude * amplitude), 0) else (0, 2 * (amplitude * amplitude)) := by
  have hfields := original_pure_circuit_warmup_root_fields remaining side amplitude ha hr
  dsimp only at hfields
  intro n
  unfold originalPureCircuitScores
  dsimp only
  rw [(hfields n 0).1, (hfields n 1).1, (hfields n 2).1, (hfields n 3).1]
  fin_cases side <;> norm_num [originalPureCircuitPoint, seededNativeSubweights, seededScalarState, physicalCircuitScore]

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 95 0 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 95 0
  exact ⟨amplitude, ha, hr⟩

/-- Both actual pure paths are uniquely train-correct at every
clock, under identical original optimizer settings. Sources:
appendix C training table and original recipe at 0033b1b; no
argmax tie or zero-confidence-loss claim is introduced. -/
theorem original_pure_circuit_warmup_train_correct (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ n, StrictCorrect (originalPureCircuitTrainLogits remaining side amplitude n) 0 := by
  intro n
  unfold originalPureCircuitTrainLogits
  rw [original_pure_circuit_warmup_scores remaining side amplitude ha hr n]
  fin_cases side
  · change StrictCorrect (trainTableLogits remaining (3 * (amplitude * amplitude)) 0) 0
    exact (train_table_strict_correct_iff remaining _ _).mpr (by positivity)
  · change StrictCorrect (trainTableLogits remaining 0 (2 * (amplitude * amplitude))) 0
    exact (train_table_strict_correct_iff remaining _ _).mpr (by positivity)

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 111 1 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 111 1
  exact ⟨amplitude, ha, hr⟩

/-- Each actual pure path has its unique held-out prediction forever:
Gen chooses the target, Mem chooses its distinct wrong table class.
Sources: appendix C test logits and native histories at 68402e2;
these are strict comparisons against every finite competitor. -/
theorem original_pure_circuit_warmup_heldout_prediction (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ n, StrictCorrect (originalPureCircuitHeldoutLogits remaining side amplitude n)
      (if side = 0 then 0 else (0 : Fin (remaining + 1)).succ) := by
  intro n
  unfold originalPureCircuitHeldoutLogits
  rw [original_pure_circuit_warmup_scores remaining side amplitude ha hr n]
  fin_cases side
  · change StrictCorrect (heldoutTableLogits remaining (3 * (amplitude * amplitude)) 0) 0
    exact heldout_table_strict_correct remaining _ _ (by positivity) (by positivity)
  · change StrictCorrect (heldoutTableLogits remaining 0 (2 * (amplitude * amplitude))) (0 : Fin (remaining + 1)).succ
    have hpos : 0 < 2 * (amplitude * amplitude) := by positivity
    intro k hk
    have hw : heldoutTableLogits remaining 0 (2 * (amplitude * amplitude)) (0 : Fin (remaining + 1)).succ =
        2 * (amplitude * amplitude) := by simp [heldoutTableLogits]
    rw [hw]
    by_cases hz : k = 0
    · rw [hz]
      simpa only [heldoutTableLogits, ite_true] using hpos
    · simpa only [heldoutTableLogits, hz, hk, ite_false] using hpos

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 95 1 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 95 1
  exact ⟨amplitude, ha, hr⟩

/-- Both opposite decision outcomes exist with the same original
native configuration at every clock. Sources: appendix C tables
and fixed original recipe at 0033b1b; the Mem outcome also explicitly
excludes correct held-out prediction, without an argmax tie. -/
theorem original_pure_circuit_warmup_decisions_exist (remaining : ℕ) (side : Fin 2) :
    ∃ amplitude : ℝ, 0 < amplitude ∧ amplitude < 10 ∧ ∀ n,
      StrictCorrect (originalPureCircuitTrainLogits remaining side amplitude n) 0 ∧
      StrictCorrect (originalPureCircuitHeldoutLogits remaining side amplitude n)
        (if side = 0 then 0 else (0 : Fin (remaining + 1)).succ) ∧
      (side = 1 → ¬StrictCorrect (originalPureCircuitHeldoutLogits remaining side amplitude n) 0) := by
  obtain ⟨amplitude, ha, hlt, hr⟩ := original_pure_circuit_residual_root remaining side
  refine ⟨amplitude, ha, hlt, ?_⟩
  intro n
  refine ⟨original_pure_circuit_warmup_train_correct remaining side amplitude ha hr n,
    original_pure_circuit_warmup_heldout_prediction remaining side amplitude ha hr n, ?_⟩
  intro hside
  subst side
  unfold originalPureCircuitHeldoutLogits
  rw [original_pure_circuit_warmup_scores remaining 1 amplitude ha hr n]
  change ¬StrictCorrect (heldoutTableLogits remaining 0 (2 * (amplitude * amplitude))) 0
  exact heldout_table_incorrect remaining _ _ (by positivity)

/-- Both actual pure paths keep a fixed strictly positive full
training CE despite their permanently correct train decisions.
Sources: appendix C complete finite-class CE and original retained
histories at 68402e2; accuracy does not imply exact unit confidence. -/
theorem original_pure_circuit_warmup_train_ce_fixed (remaining : ℕ) (side : Fin 2) (amplitude : ℝ)
    (ha : 0 < amplitude) (hr : originalPureCircuitResidual remaining side amplitude = 0) :
    ∀ n, tableTrainCE remaining ((originalPureCircuitScores remaining side amplitude n).1 +
      (originalPureCircuitScores remaining side amplitude n).2) =
        tableTrainCE remaining ((if side = 0 then 3 else 2) * (amplitude * amplitude)) ∧
      0 < tableTrainCE remaining ((originalPureCircuitScores remaining side amplitude n).1 +
        (originalPureCircuitScores remaining side amplitude n).2) := by
  have hce (score : ℝ) : 0 < tableTrainCE remaining score := by
    rw [table_train_ce_probability]
    have hl := Real.log_neg (table_train_target_probability_positive remaining score)
      (table_train_target_probability_lt_one remaining score)
    linarith only [hl]
  intro n
  rw [original_pure_circuit_warmup_scores remaining side amplitude ha hr n]
  fin_cases side
  · change tableTrainCE remaining (3 * (amplitude * amplitude) + 0) =
      tableTrainCE remaining (3 * (amplitude * amplitude)) ∧ 0 < tableTrainCE remaining _
    exact ⟨by rw [add_zero], hce _⟩
  · change tableTrainCE remaining (0 + 2 * (amplitude * amplitude)) =
      tableTrainCE remaining (2 * (amplitude * amplitude)) ∧ 0 < tableTrainCE remaining _
    exact ⟨by rw [zero_add], hce _⟩

example : ∃ amplitude : ℝ, 0 < amplitude ∧ originalPureCircuitResidual 95 0 amplitude = 0 := by
  obtain ⟨amplitude, ha, _, hr⟩ := original_pure_circuit_residual_root 95 0
  exact ⟨amplitude, ha, hr⟩

end Transformer.Grokking.CircuitEfficiency
