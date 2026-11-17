module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Error

/-!
# Actual CE coefficients compared with historical learned execution

Source: `review/flat_spectrum.md`, CM106. Each original occurrence may retain
a state from a different learned model. The state error comes from complete
initial/operator displacements; the full-class CE signal error stays explicit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair Optimality
variable {J O : Type*}

/-- Actual current states obey the learned capacity, CM106. -/
theorem states_bound (n : ℕ) (f : J → Parameters O) (xs : Fin (n + 1) → List O)
    (hf : ∀ j, Unit (f j)) (i : Fin (n + 1)) (j : J) : ‖states n f xs i j‖ ≤ 1 :=
  (run_norm_le (f j) (hf j).2.1 0 _ (xs i)).trans (hf j).1

/-- Complete cached coefficient, allowing historical content per occurrence,
CM106. No current-signal or current-state equality is assumed. -/
noncomputable def cached (n : ℕ) (g : Fin (n + 1) → J → Parameters O)
    (xs : Fin (n + 1) → List O) (s : Fin (n + 1) → O → ℝ) : J → O → State :=
  mean n (historical n g xs) s

variable [Fintype J] [Nonempty J]

/-- Largest complete state displacement across all learned components for
one original occurrence, CM106; zero-weight components still participate. -/
noncomputable def contentDrift (n : ℕ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O) (i : Fin (n + 1)) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun j => stateDrift (f j) (g i j) (xs i))

/-- Every component is included in the occurrence's content bound, CM106. -/
theorem state_drift_le (n : ℕ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (i : Fin (n + 1)) (j : J) : stateDrift (f j) (g i j) (xs i) ≤ contentDrift n f g xs i :=
  Finset.le_sup' (f := fun j => stateDrift (f j) (g i j) (xs i)) (Finset.mem_univ j)

/-- Actual historical/current state discrepancy follows from their changed
initial coordinates and transported operators, CM106. -/
theorem historical_change (n : ℕ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (hf : ∀ j, Unit (f j)) (hg : ∀ i j, Unit (g i j)) (i : Fin (n + 1)) (j : J) :
    ‖states n f xs i j - historical n g xs i j‖ ≤ contentDrift n f g xs i :=
  (state_change_le (f j) (g i j) (hf j) (hg i j) (xs i)).trans (state_drift_le n f g xs i j)

variable [Fintype O] [DecidableEq O]

/-- Explicit current-residual and historical-content error budget, CM106.
It is not a free stochastic or rounding certificate. -/
noncomputable def currentError (n : ℕ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (target : Fin (n + 1) → O) (u : J → O → State) (s : Fin (n + 1) → O → ℝ) : ℝ :=
  errorBudget n (fun i => residual (jointScore f u (xs i)) (target i)) s (contentDrift n f g xs)

/-- Complete actual CE coefficient error includes both stale signals and
actual learned-content displacement, CM106. No answer or optimality premise. -/
theorem current_coefficient_error (n : ℕ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (target : Fin (n + 1) → O) (u : J → O → State) (s : Fin (n + 1) → O → ℝ)
    (hf : ∀ j, Unit (f j)) (hg : ∀ i j, Unit (g i j)) :
    peak (jointCoefficient n f xs target u - cached n g xs s) ≤
      currentError n f g xs target u s := by
  rw [actual_coefficient]
  exact peak_error n (states n f xs) (historical n g xs)
    (fun i => residual (jointScore f u (xs i)) (target i)) s (contentDrift n f g xs)
    (states_bound n f xs hf) (historical_change n f g xs hf hg)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained
