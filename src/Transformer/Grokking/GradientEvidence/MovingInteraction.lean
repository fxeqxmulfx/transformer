module

public import Transformer.Grokking.GradientEvidence.InteractionDrift

/-!
# A sufficient interaction-control condition for moving adaptive updates

Source: `review/flat_spectrum.md`, *Adaptive interaction control*, AI5.
The update direction can be the exact buffered Adam direction from AI1.
Both the new residual map and new feature displacement remain explicit.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes

variable {K : Type*} [Fintype K]

/-- Update with a specified scale and direction, AI5. For buffered Adam,
a=1-eta*rho and f=adamDiagonal*correctedMoment, by AI1. -/
def directionStep (a η : ℝ) (w f : K → ℝ) (k : K) : ℝ := a * w k - η * f k

/-- Explicit contribution of changing the map at the new readout, AI5. -/
def basisMovement (Q Q' : K → K → ℝ) (w' : K → ℝ) (k : K) : ℝ :=
  matrixApply Q' w' k - matrixApply Q w' k

/-- The actual new readout residual consists of scaled old residual,
projected forcing and basis movement, AI5. -/
theorem moving_update_residual (Q Q' : K → K → ℝ) (a η : ℝ)
    (w f : K → ℝ) (k : K) :
    matrixApply Q' (directionStep a η w f) k =
      a * matrixApply Q w k - η * matrixApply Q f k +
        basisMovement Q Q' (directionStep a η w f) k := by
  unfold basisMovement directionStep
  simp only [matrixApply_sub_smul]
  ring

/-- **Signed adaptive interaction with both kinds of drift**, AI5. Here x
and x' are the old/new normalized query displacements, including their centers. -/
theorem moving_update_interaction (Q Q' : K → K → ℝ) (a η : ℝ)
    (w f x x' : K → ℝ) :
    linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q' x') =
      a * linearScore (matrixApply Q w) (matrixApply Q x) -
      η * linearScore (matrixApply Q f) (matrixApply Q x) +
      linearScore (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x) +
      linearScore (matrixApply Q' (directionStep a η w f))
        (fun k => matrixApply Q' x' k - matrixApply Q x k) := by
  have hu := fun k => moving_update_residual Q Q' a η w f k
  have hr : linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q x) =
      a * linearScore (matrixApply Q w) (matrixApply Q x) -
      η * linearScore (matrixApply Q f) (matrixApply Q x) +
      linearScore (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x) := by
    simp only [linearScore, hu, add_mul, sub_mul, sum_add_distrib, sum_sub_distrib, mul_sum]
    congr 2 <;> apply sum_congr rfl <;> intro k hk <;> ring
  have he : linearScore (matrixApply Q' (directionStep a η w f))
      (fun k => matrixApply Q' x' k - matrixApply Q x k) =
      linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q' x') -
      linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q x) := by
    simp only [linearScore, mul_sub, sum_sub_distrib]
  linarith

/-- The forcing, basis and feature-change allowance, AI5. All are concrete
geometric quantities, independent of the desired new interaction bound. -/
def movingInteractionBudget (Q Q' : K → K → ℝ) (a η : ℝ) (w f x x' : K → ℝ) : ℝ :=
  |η| * pairBudget (matrixApply Q f) (matrixApply Q x) +
    pairBudget (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x) +
    pairBudget (matrixApply Q' (directionStep a η w f))
      (fun k => matrixApply Q' x' k - matrixApply Q x k)

/-- A sufficient update bound, AI5. AI3 controls part of the forcing energy;
it does not control the moment discrepancy, commutator or either drift term. -/
theorem moving_update_interaction_le (Q Q' : K → K → ℝ) (a η : ℝ)
    (w f x x' : K → ℝ) :
    |linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q' x')| ≤
      |a| * |linearScore (matrixApply Q w) (matrixApply Q x)| +
        movingInteractionBudget Q Q' a η w f x x' := by
  rw [moving_update_interaction]
  have h0 := abs_add_le
    (a * linearScore (matrixApply Q w) (matrixApply Q x))
    (-η * linearScore (matrixApply Q f) (matrixApply Q x))
  have h1 := abs_add_le
    (a * linearScore (matrixApply Q w) (matrixApply Q x) -
      η * linearScore (matrixApply Q f) (matrixApply Q x))
    (linearScore (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x))
  have h2 := abs_add_le
    (a * linearScore (matrixApply Q w) (matrixApply Q x) -
      η * linearScore (matrixApply Q f) (matrixApply Q x) +
      linearScore (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x))
    (linearScore (matrixApply Q' (directionStep a η w f))
      (fun k => matrixApply Q' x' k - matrixApply Q x k))
  have hf := mul_le_mul_of_nonneg_left
    (interaction_abs_le (matrixApply Q f) (matrixApply Q x)) (abs_nonneg η)
  have hb := interaction_abs_le
    (basisMovement Q Q' (directionStep a η w f)) (matrixApply Q x)
  have hx := interaction_abs_le (matrixApply Q' (directionStep a η w f))
    (fun k => matrixApply Q' x' k - matrixApply Q x k)
  simp only [abs_mul, abs_neg, neg_mul, ← sub_eq_add_neg] at h0
  unfold movingInteractionBudget
  linarith

/-- Strict contraction follows from an explicit sufficient drift/forcing
budget, AI5. It is not inferred from CE descent, and need not hold in data. -/
theorem moving_update_interaction_contracts (Q Q' : K → K → ℝ) (a η : ℝ)
    (w f x x' : K → ℝ)
    (h : movingInteractionBudget Q Q' a η w f x x' <
      (1 - |a|) * |linearScore (matrixApply Q w) (matrixApply Q x)|) :
    |linearScore (matrixApply Q' (directionStep a η w f)) (matrixApply Q' x')| <
      |linearScore (matrixApply Q w) (matrixApply Q x)| := by
  have hb := moving_update_interaction_le Q Q' a η w f x x'
  nlinarith

omit [Fintype K] in
/-- Buffered native Adam is precisely this specified-direction update,
AI1–AI5, with no replacement of saved moments by a chosen gradient. -/
theorem buffered_step_eq_directionStep (b1 b2 eps η ρ : ℝ) (t : ℕ)
    (w m v : K → ℝ) :
    (fun k => bufferedAdamUpdate b1 b2 eps η ρ t (w k) (m k) (v k)) =
      directionStep (1 - η * ρ) η w
        (fun k => adamDiagonal b2 eps t (v k) * correctedMoment b1 t (m k)) := by
  funext k
  rw [bufferedAdamUpdate_factor]
  unfold directionStep
  ring

/-- The sufficient moving-interaction estimate instantiated with the actual
buffered Adam equations, AI5. Encoder/time/minibatch effects are in m, not omitted. -/
theorem buffered_interaction_le (b1 b2 eps η ρ : ℝ) (t : ℕ)
    (Q Q' : K → K → ℝ) (w m v x x' : K → ℝ) :
    |linearScore (matrixApply Q'
      (fun k => bufferedAdamUpdate b1 b2 eps η ρ t (w k) (m k) (v k))) (matrixApply Q' x')| ≤
      |1 - η * ρ| * |linearScore (matrixApply Q w) (matrixApply Q x)| +
        movingInteractionBudget Q Q' (1 - η * ρ) η w
          (fun k => adamDiagonal b2 eps t (v k) * correctedMoment b1 t (m k)) x x' := by
  rw [buffered_step_eq_directionStep]
  exact moving_update_interaction_le Q Q' (1 - η * ρ) η w _ x x'

example := moving_update_interaction_contracts
  (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) (fun _ _ => 1) (1 / 2) (1 / 2)
  (fun _ => 4) (fun _ => 1) (fun _ => 3) (fun _ => 3)
  (by norm_num [movingInteractionBudget, pairBudget, basisMovement, directionStep,
    matrixApply, linearScore])

end Transformer.Grokking.GradientEvidence
