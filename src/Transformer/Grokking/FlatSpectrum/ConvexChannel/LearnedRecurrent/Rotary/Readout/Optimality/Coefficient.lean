module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Table
public import Transformer.Grokking.FlatSpectrum.Training.Logits

/-!
# Actual full-class CE coefficients on indexed learned recurrent states

Source: `review/convex/moments.md`, CM103. Coefficients use every original record,
class and component. The contraction identity is derived from complete executed
logits, not introduced as a gradient assumption or an identification oracle.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality

open Pair
open Geometry (probability)
open Landscape.Neural (crossEntropy)
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O]

omit [Fintype O] in
/-- Complete actual logits in row coordinates, CM103. -/
theorem scores_pairing (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (u : J → O → State) (xs : List O) (c : O) :
    scoreMap B w f u xs c =
      ∑ j, w j * B * pairing (u j c) (runFrom (f j) 0 (f j).content.initial xs) := by
  simp only [scoreMap, LinearMap.coe_mk, AddHom.coe_mk, Finset.sum_apply,
    Pi.smul_apply, smul_eq_mul, score, replace_complex, pairing]
  apply Finset.sum_congr rfl
  intro j hj
  ring

omit [Fintype O] in
/-- Every component's full row contributes to the actual logit difference, CM103. -/
theorem scores_change (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (u v : J → O → State) (xs : List O) (c : O) :
    scoreMap B w f v xs c - scoreMap B w f u xs c =
      ∑ j, w j * B * pairing (v j c - u j c)
        (runFrom (f j) 0 (f j).content.initial xs) := by
  rw [scores_pairing, scores_pairing, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  rw [pairing_sub]
  ring

variable [DecidableEq O]

/-- Actual ordinary-CE probability minus original target indicator, CM103. -/
noncomputable def residual (z : O → ℝ) (target c : O) : ℝ :=
  probability z c - if c = target then 1 else 0

/-- Full-class residual contraction is precisely ordinary CE's first-order
lower bound, CM103. -/
theorem residual_first_order (z v : O → ℝ) (target : O) :
    crossEntropy z target + ∑ c, residual z target c * (v c - z c) ≤
      crossEntropy v target := by
  have h := Training.crossEntropy_first_order z v target
  simpa [residual, sub_mul, Finset.sum_sub_distrib, ite_mul] using h

/-- Complete unconjugated CE coefficient from all indexed records, CM103. -/
noncomputable def coefficient (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (j : J) (c : O) : State :=
  ((n : ℝ) + 1)⁻¹ • ∑ i,
    (w j * B * residual (scoreMap B w f u (xs i)) (target i) c) •
      runFrom (f j) 0 (f j).content.initial (xs i)

/-- Derive the complete row contraction from actual logits and CE residuals,
including every index and both complex coordinates, CM103. -/
theorem coefficient_pairing (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u v : J → O → State) :
    linearCost (v - u) (coefficient n B w f xs target u) =
      ((n : ℝ) + 1)⁻¹ * ∑ i, ∑ c,
        residual (scoreMap B w f u (xs i)) (target i) c *
          (scoreMap B w f v (xs i) c - scoreMap B w f u (xs i) c) := by
  simp only [linearCost, coefficient, map_smul, map_sum, smul_eq_mul, Pi.sub_apply]
  simp_rw [← Finset.mul_sum]
  congr 1
  conv_lhs => arg 2; ext j; rw [Finset.sum_comm]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c hc
  rw [scores_change, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Actual nonempty indexed mean CE has the derived complete row first-order
bound, with no assumed optimum and no unseen labels, CM103. -/
theorem mean_first_order (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u v : J → O → State) :
    meanLoss n B w f xs target u +
      linearCost (v - u) (coefficient n B w f xs target u) ≤
        meanLoss n B w f xs target v := by
  have h := Finset.sum_le_sum (s := Finset.univ) (fun (i : Fin (n + 1)) _ =>
    residual_first_order (scoreMap B w f u (xs i)) (scoreMap B w f v (xs i)) (target i))
  have hs := mul_le_mul_of_nonneg_left h (by positivity : 0 ≤ ((n : ℝ) + 1)⁻¹)
  rw [coefficient_pairing]
  simpa only [meanLoss, Finset.sum_add_distrib, mul_add] using hs

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
