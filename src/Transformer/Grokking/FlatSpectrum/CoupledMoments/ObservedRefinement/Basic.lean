module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.DecodedHull

/-!
# Observed errors and ordinary one-hot cross-entropy

Source: `review/flat_spectrum.md`, CM23. The full vocabulary remains in the
softmax. At unit logit scale, each incorrect deterministic prediction adds
exactly one to the summed ordinary CE. No claim about unseen labels follows.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement

open Finset ConvexRecall
open Landscape.Neural (crossEntropy)

variable {C S I : Type*} [Fintype C] [DecidableEq C] [Fintype S]

/-- One unit of excess CE for an incorrect categorical prediction (CM23). -/
def mistake (prediction label : C) : ℝ := if prediction = label then 0 else 1

/-- Observed mismatch count, as a real number for comparison with CE (CM23). -/
noncomputable def errors (prediction labels : S → C) : ℝ :=
  ∑ t, mistake (prediction t) (labels t)

/-- The softmax denominator includes every output class, even unused ones (CM23). -/
theorem target_exp_sum (p : C) :
    (∑ c, Real.exp (targetLogits p c)) = Real.exp 1 + Fintype.card C - 1 := by
  have h (c : C) : Real.exp (targetLogits p c) =
      (if c = p then Real.exp 1 - 1 else 0) + 1 := by
    by_cases hc : c = p <;> simp [targetLogits, hc]
  simp_rw [h]
  rw [sum_add_distrib]
  simp only [Fintype.sum_ite_eq', sum_const, card_univ, nsmul_eq_mul, mul_one]
  ring

/-- Exact ordinary-CE excess for a one-hot prediction; this is not zero-one
loss substituted for a different training objective (CM23). -/
theorem row_ce (p y : C) :
    crossEntropy (targetLogits p) y =
      crossEntropy (targetLogits y) y + mistake p y := by
  unfold crossEntropy
  rw [target_exp_sum, target_exp_sum]
  by_cases h : p = y
  · simp [targetLogits, mistake, h]
  · simp [targetLogits, mistake, h, Ne.symm h]

/-- Deterministic tables have ideal summed CE plus their observed error count (CM23). -/
theorem pointMass_loss {m n : Type*} (cell : S → m × n) (labels : S → C)
    (prediction : m → n → C) :
    loss cell labels (pointMass prediction) =
      idealLoss labels + errors (fun t => prediction (cell t).1 (cell t).2) labels := by
  unfold loss idealLoss errors
  rw [← sum_add_distrib]
  apply sum_congr rfl
  intro t ht
  change crossEntropy (targetLogits (prediction (cell t).1 (cell t).2)) (labels t) = _
  exact row_ce _ _

/-- A comparison of actual observed mismatches is exactly a CE comparison (CM23). -/
theorem loss_le_iff {m n : Type*} (cell : S → m × n) (labels : S → C)
    (a b : m → n → C) :
    loss cell labels (pointMass a) ≤ loss cell labels (pointMass b) ↔
      errors (fun t => a (cell t).1 (cell t).2) labels ≤
        errors (fun t => b (cell t).1 (cell t).2) labels := by
  rw [pointMass_loss, pointMass_loss]
  exact add_le_add_iff_left _

variable [Fintype I] [DecidableEq I]

/-- Grouping arbitrary observations by an axis exactly recovers the original
sum, including repeated observations and unobserved axis values (CM23). -/
theorem group_sum (axis : S → I) (cost : S → I → ℝ) (phase : I → I) :
    (∑ i, ∑ t, if axis t = i then cost t (phase i) else 0) =
      ∑ t, cost t (phase (axis t)) := by
  rw [sum_comm]
  apply sum_congr rfl
  intro t ht
  simp

example := target_exp_sum (true : Bool)
example := row_ce (false : Bool) true
example := pointMass_loss (fun t : Bool => (t, ())) id (fun i _ => !i)
example := loss_le_iff (fun t : Bool => (t, ())) id (fun i _ => i) (fun i _ => !i)
example := group_sum (id : Bool → Bool) (fun i j => mistake i j) not

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement
