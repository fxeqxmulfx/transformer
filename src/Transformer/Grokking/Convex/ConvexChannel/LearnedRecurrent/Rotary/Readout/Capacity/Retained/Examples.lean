module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Witness

/-!
# Inhabited complete-error and current-content premises

Source: `review/convex/moments.md`, CM106. Examples include changed nonzero
complex states, different learned frequencies, original indexed refreshes,
and strict corrected descent with a genuinely nonzero signal error.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair Optimality Estimate
open scoped BigOperators

/-- Nonzero complex states inhabit the complete state-error premise, CM106. -/
theorem imaginary_distance : ‖((Complex.I, 0) : State) - (0, Complex.I)‖ ≤ 2 := by
  norm_num [Prod.norm_def]

example : ‖(1 / 2 : ℝ) • ((Complex.I, 0) : State) - (3 / 8 : ℝ) • (0, Complex.I)‖ ≤
    |(1 / 2 : ℝ) - 3 / 8| + |(3 / 8 : ℝ)| * 2 :=
  signal_state_error _ _ _ _ _ (by simp [Prod.norm_def]) imaginary_distance

example : ‖(mean 0 (fun (_ : Fin 1) (_ : Fin 1) => (Complex.I, 0))
    (fun _ (_ : Bool) => (1 / 2 : ℝ)) -
      mean 0 (fun _ _ => (0, Complex.I)) (fun _ _ => (3 / 8 : ℝ)) : Fin 1 → Bool → State) 0 true‖ ≤
        (((0 : ℕ) : ℝ) + 1)⁻¹ * ∑ _i : Fin 1, (|(1 / 2 : ℝ) - 3 / 8| + |(3 / 8 : ℝ)| * 2) :=
  row_error 0 (fun (_ : Fin 1) (_ : Fin 1) => (Complex.I, 0))
    (fun _ _ => (0, Complex.I)) (fun _ (_ : Bool) => (1 / 2 : ℝ))
      (fun _ _ => (3 / 8 : ℝ)) (fun _ => 2) (by intros; simp [Prod.norm_def])
        (fun _ _ => imaginary_distance) 0 true

example : blockSupport ((mean 0 (fun (_ : Fin 1) (_ : Fin 1) => (Complex.I, 0))
    (fun _ (_ : Bool) => (1 / 2 : ℝ)) -
      mean 0 (fun _ _ => (0, Complex.I)) (fun _ _ => (3 / 8 : ℝ)) : Fin 1 → Bool → State) 0) ≤
        errorBudget 0 (fun _ (_ : Bool) => (1 / 2 : ℝ)) (fun _ _ => (3 / 8 : ℝ))
          (fun _ => 2) :=
  block_error _ _ _ _ _ _ (by intros; simp [Prod.norm_def]) (fun _ _ => imaginary_distance) _

example : peak (mean 0 (fun (_ : Fin 1) (_ : Fin 1) => (Complex.I, 0))
    (fun _ (_ : Bool) => (1 / 2 : ℝ)) -
      mean 0 (fun _ _ => (0, Complex.I)) (fun _ _ => (3 / 8 : ℝ))) ≤
        errorBudget 0 (fun _ (_ : Bool) => (1 / 2 : ℝ)) (fun _ _ => (3 / 8 : ℝ))
          (fun _ => 2) :=
  peak_error _ _ _ _ _ _ (by intros; simp [Prod.norm_def]) (fun _ _ => imaginary_distance)

example : (mean 2 (fun (_ : Fin 3) (_ : Fin 1) => (Complex.I, 1))
    (refresh {0} (fun _ (_ : Bool) => (1 / 4 : ℝ)) (fun _ _ => (1 / 2 : ℝ))) -
      mean 2 (fun _ _ => (Complex.I, 1)) (fun _ (_ : Bool) => (1 / 4 : ℝ)) : Fin 1 → Bool → State) 0 true =
        (1 / 12 : ℝ) • ((Complex.I, 1) : State) := by
  rw [mean_refresh]
  norm_num [smul_smul]
  ring

example : ‖states 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false]) 0 0‖ ≤ 1 :=
  states_bound _ _ _ (fun _ => position_unit) _ _

example : ‖states 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false]) 0 0 -
    historical 0 (fun _ _ => unrotated) (fun _ => [true, false]) 0 0‖ ≤
      contentDrift 0 (fun _ : Fin 1 => positionWitness) (fun _ _ => unrotated)
        (fun _ => [true, false]) 0 :=
  historical_change _ _ _ _ (fun _ => position_unit) (fun _ _ => unrotated_unit) _ _

example : peak (jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
    (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) -
      cached 0 (fun _ _ => unrotated) (fun _ => [true, false]) (fun _ => witnessSignal)) ≤
        currentError 0 (fun _ : Fin 1 => positionWitness) (fun _ _ => unrotated)
          (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows)
            (fun _ => witnessSignal) :=
  current_coefficient_error _ _ _ _ _ _ _ (fun _ => position_unit) (fun _ _ => unrotated_unit)

example : jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
    (fun _ => true) (fun _ => witnessRows) -
      estimatedGap 16 (fun _ : Fin 1 => witnessRows)
        (cached 0 (fun _ _ => unrotated) (fun _ => [true, false]) (fun _ => witnessSignal)) -
          2 * 16 * currentError 0 (fun _ : Fin 1 => positionWitness) (fun _ _ => unrotated)
            (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows)
              (fun _ => witnessSignal) ≤
      jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
        (fun _ => true) (fun _ => imaginaryTarget) :=
  cached_loss_lower _ _ _ _ _ _ _ _ _ (fun _ => position_unit) (fun _ _ => unrotated_unit)
    witness_mem target_mem

example : linearCost (proposal 16
    (cached 0 (fun (_ : Fin 1) (_ : Fin 1) => unrotated) (fun _ => [true, false])
      (fun _ => witnessSignal)) (selected (cached 0 (fun (_ : Fin 1) (_ : Fin 1) => unrotated)
        (fun _ => [true, false]) (fun _ => witnessSignal))) - (fun _ => witnessRows))
      (jointCoefficient 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
        (fun _ => true) (fun _ => witnessRows)) ≤
    -estimatedGap 16 (fun _ : Fin 1 => witnessRows)
      (cached 0 (fun _ _ => unrotated) (fun _ => [true, false]) (fun _ => witnessSignal)) +
        2 * 16 * currentError 0 (fun _ : Fin 1 => positionWitness) (fun _ _ => unrotated)
          (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) (fun _ => witnessSignal) :=
  cached_direction_upper _ _ _ _ _ _ _ _ (fun _ => position_unit) (fun _ _ => unrotated_unit)
    witness_mem

example : ∃ v ∈ jointDomain (J := Fin 1) (O := Bool) 16,
    linearCost (v - (fun _ => witnessRows))
      (jointCoefficient 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
        (fun _ => true) (fun _ => witnessRows)) < 0 :=
  cached_negative 0 16 (fun _ => positionWitness) (fun _ _ => positionWitness)
    (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) (fun _ => witnessSignal)
      (fun _ => position_unit) (fun _ _ => position_unit) witness_mem witness_strict

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained
