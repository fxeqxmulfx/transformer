module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Basic

/-!
# Ordinary cross-entropy and a forward-only fitting certificate

Source: `review/flat_spectrum.md`, CM2. The logits are the shared class probabilities.
Ordinary log-sum-exp CE is convex in every coordinate. The one-hot target loss is a lower
bound for every feasible model; any feasible table exactly matching the observed one-hot
labels is a global CE minimum. This certifies a successful forward-only feasibility solve,
not convergence of a particular finite-step algorithm or any missing-label prediction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open Finset ConvexRecall
open Landscape.Neural (crossEntropy)
open scoped Matrix.Norms.L2Operator

variable {m n C S F : Type*} [Fintype C] [Fintype S]

/-- Ordinary empirical CE of the common probability coordinates (CM2). -/
noncomputable def loss (cell : S → m × n) (labels : S → C) (P : Table m n C) : ℝ :=
  ∑ s, crossEntropy (P (cell s).1 (cell s).2) (labels s)

/-- The full loss remains jointly convex (CM2). -/
theorem loss_convex (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ Set.univ (loss cell labels) := by
  refine ⟨convex_univ, ?_⟩
  intro P _ Q _ a b ha hb hab
  simp only [loss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  exact (convex_crossEntropy (labels s)).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

variable [DecidableEq C]

/-- Ideal empirical CE, summed over the actual observed labels (CM2). -/
noncomputable def idealLoss (labels : S → C) : ℝ :=
  ∑ s, crossEntropy (targetLogits (labels s)) (labels s)

/-- No probability-vector logits improve on the ideal observed CE (CM2). -/
theorem idealLoss_le (cell : S → m × n) (labels : S → C) {P : Table m n C}
    (hP : Rows P) : idealLoss labels ≤ loss cell labels P := by
  apply Finset.sum_le_sum
  intro s _
  exact target_crossEntropy_le (labels s) (hP.1 (cell s).1 (cell s).2)
    (coordinate_le_one hP _ _ _)

/-- Exact fitting means equality with the ideal loss (CM2). -/
theorem loss_eq_ideal_of_fit (cell : S → m × n) (labels : S → C) {P : Table m n C}
    (hfit : ∀ s, P (cell s).1 (cell s).2 = targetLogits (labels s)) :
    loss cell labels P = idealLoss labels := by
  apply Finset.sum_congr rfl
  intro s _
  rw [hfit s]

variable [Fintype m] [Fintype n] [DecidableEq n]

/-- A feasible exact one-hot fit is a global minimum of the same ordinary CE (CM2).
Membership is explicit; `IsMinOn` alone would not require it. -/
theorem fit_minimum (φ : F → C → ℂ) (R : F → ℝ) (cell : S → m × n) (labels : S → C)
    {P : Table m n C} (hP : Feasible φ R P)
    (hfit : ∀ s, P (cell s).1 (cell s).2 = targetLogits (labels s)) :
    Feasible φ R P ∧ IsMinOn (loss cell labels) {Q | Feasible φ R Q} P := by
  refine ⟨hP, ?_⟩
  intro Q hQ
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels hQ.1

/-- Observed ideal-loss excess bounds excess over every feasible competitor (CM2).
This can be measured even without a finite-step convergence theorem. -/
theorem excess_le_ideal_excess (φ : F → C → ℂ) (R : F → ℝ) (cell : S → m × n)
    (labels : S → C) (P : Table m n C) {Q : Table m n C} (hQ : Feasible φ R Q) :
    loss cell labels P - loss cell labels Q ≤ loss cell labels P - idealLoss labels := by
  linarith [idealLoss_le cell labels hQ.1]

/-! A nonempty two-class example with a nonzero complex code. -/

private def witness : Table (Fin 1) (Fin 1) Bool := pointMass (fun _ _ => true)

private theorem witness_feasible :
    Feasible (fun (_ : Fin 1) (c : Bool) => if c then (1 : ℂ) else -1)
      (fun _ => 1) witness := by
  refine ⟨rows_pointMass _, ?_⟩
  intro f
  change ‖moment _ (pointMass _)‖ ≤ 1
  rw [moment_pointMass]
  have he : (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℂ)) = (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
    ext i j
    fin_cases i
    fin_cases j
    simp
  simp [he]

private theorem witness_fit : witness 0 0 = targetLogits (true : Bool) := by
  funext c
  simp [witness, pointMass, targetLogits]

example := idealLoss_le (fun (_ : Fin 1) => (0, 0)) (fun _ => true) witness_feasible.1
example := loss_eq_ideal_of_fit (fun (_ : Fin 1) => (0, 0)) (fun _ => true) (fun _ => witness_fit)
example := fit_minimum _ _ (fun (_ : Fin 1) => (0, 0)) (fun _ => true)
  witness_feasible (fun _ => witness_fit)
example := excess_le_ideal_excess _ _ (fun (_ : Fin 1) => (0, 0)) (fun _ => true)
  witness witness_feasible

end Transformer.Grokking.FlatSpectrum.CoupledMoments
