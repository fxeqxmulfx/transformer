module

public import Transformer.Grokking.GradientEvidence.MovingInteraction

/-!
# Class contrasts retain two different adaptive preconditioners

Source: `review/flat_spectrum.md`, *Adaptive interaction control*, AI5.
A competitor-minus-target contrast is updated by the difference of the two
actual row directions. It cannot in general use one shared diagonal scale.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open FlatSpectrum.FeatureCodes

variable {C K : Type*}

/-- Difference of the actual buffered class-row directions, AI5. Each row
retains its own first moment and its own second-moment preconditioner. -/
def bufferedContrastDirection (b1 b2 eps : ℝ) (n : ℕ) (m v : C → K → ℝ)
    (c target : C) (k : K) : ℝ :=
  adamDiagonal b2 eps n (v c k) * correctedMoment b1 n (m c k) -
    adamDiagonal b2 eps n (v target k) * correctedMoment b1 n (m target k)

/-- The actual contrast update has the common decay factor and the
difference of row-specific adaptive directions, AI5. -/
theorem buffered_contrast_eq_directionStep (b1 b2 eps η ρ : ℝ) (n : ℕ)
    (E m v : C → K → ℝ) (c target : C) :
    (fun k => bufferedAdamUpdate b1 b2 eps η ρ n (E c k) (m c k) (v c k) -
      bufferedAdamUpdate b1 b2 eps η ρ n (E target k) (m target k) (v target k)) =
      directionStep (1 - η * ρ) η (fun k => E c k - E target k)
        (bufferedContrastDirection b1 b2 eps n m v c target) := by
  funext k
  simp only [bufferedAdamUpdate_factor, directionStep, bufferedContrastDirection]
  ring

variable [Fintype K]

/-- The moving-interaction bound for a genuine updated class contrast,
AI5. No common preconditioner for competitor and target is assumed. -/
theorem buffered_contrast_interaction_le (b1 b2 eps η ρ : ℝ) (n : ℕ)
    (Q Q' : K → K → ℝ) (E m v : C → K → ℝ) (c target : C) (x x' : K → ℝ) :
    |linearScore (matrixApply Q'
      (fun k => bufferedAdamUpdate b1 b2 eps η ρ n (E c k) (m c k) (v c k) -
        bufferedAdamUpdate b1 b2 eps η ρ n (E target k) (m target k) (v target k)))
      (matrixApply Q' x')| ≤
      |1 - η * ρ| * |linearScore (matrixApply Q (fun k => E c k - E target k))
        (matrixApply Q x)| +
      movingInteractionBudget Q Q' (1 - η * ρ) η (fun k => E c k - E target k)
        (bufferedContrastDirection b1 b2 eps n m v c target) x x' := by
  rw [buffered_contrast_eq_directionStep]
  exact moving_update_interaction_le Q Q' (1 - η * ρ) η _ _ x x'

example := buffered_contrast_eq_directionStep (9 / 10) (49 / 50) (1 / 100000000)
  (1 / 1000) 1 1
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else -1)
  (fun (c : Bool) (_ : Fin 1) => if c then (1 / 10 : ℝ) else -1 / 5)
  (fun (c : Bool) (_ : Fin 1) => if c then (1 / 50 : ℝ) else 2 / 25) true false

end Transformer.Grokking.GradientEvidence
