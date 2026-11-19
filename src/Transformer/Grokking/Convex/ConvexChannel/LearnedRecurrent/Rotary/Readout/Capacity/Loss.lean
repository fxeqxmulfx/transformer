module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Execution
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Support

/-!
# Actual full-class mean CE with variable component capacity

Source: `review/convex/moments.md`, CM104. Scores use weighted rows, so capacity
and native weights must not multiply their coefficients again. The full indexed
first-order contraction is the proved CM103 identity in these coordinates.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O]

/-- Actual ordinary full-class indexed mean CE on weighted rows, CM104. -/
noncomputable def jointLoss (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) : ℝ :=
  meanLoss n 1 (fun _ => 1) f xs target u

/-- Full-class ordinary CE is convex jointly in all weighted readout rows,
with the shared capacity free to redistribute, CM104. -/
theorem joint_loss_convex [Nonempty O] (B : ℝ) (f : J → Parameters O)
    (xs : List O) (target : O) : ConvexOn ℝ (jointDomain B)
      (fun u => Landscape.Neural.crossEntropy (jointScore f u xs) target) := by
  have h := (ConvexRecall.convex_crossEntropy target).comp_linearMap
    ((evaluation xs).comp (jointScore f))
  have hu : ConvexOn ℝ Set.univ
      (fun u => Landscape.Neural.crossEntropy (jointScore f u xs) target) := by
    simpa only [Set.preimage_univ, Function.comp_def, LinearMap.comp_apply,
      evaluation, LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (joint_convex B)

/-- Every indexed nonempty mean retains joint convexity, including duplicate
records and short minibatch tails, CM104. -/
theorem joint_mean_convex [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) :
    ConvexOn ℝ (jointDomain B) (jointLoss n f xs target) := by
  refine ⟨joint_convex B, ?_⟩
  intro u hu v hv a b ha hb hab
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun (i : Fin (n + 1)) _ =>
    (joint_loss_convex B f (xs i) (target i)).2 hu hv ha hb hab)
  have h := mul_le_mul_of_nonneg_left hs (by positivity : 0 ≤ ((n : ℝ) + 1)⁻¹)
  simpa only [jointLoss, meanLoss, jointScore, smul_eq_mul, Finset.sum_add_distrib,
    ← Finset.mul_sum, mul_add, mul_left_comm] using h

variable [DecidableEq O]

/-- Actual CE coefficient of each full weighted row; no extra native mass factor, CM104. -/
noncomputable def jointCoefficient (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) :=
  coefficient n 1 (fun _ => 1) f xs target u

/-- The complete coefficient uses every original class, state and indexed
record, including components at zero current weight, CM104. -/
theorem joint_coefficient_apply (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (j : J) (c : O) :
    jointCoefficient n f xs target u j c = ((n : ℝ) + 1)⁻¹ • ∑ i,
      residual (jointScore f u (xs i)) (target i) c •
        runFrom (f j) 0 (f j).content.initial (xs i) := by
  simp only [jointCoefficient, coefficient, jointScore, one_mul]

/-- CM103's proved complete contraction specializes to the actual weighted-row
scores without a gradient or optimizer assumption, CM104. -/
theorem joint_contraction (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u v : J → O → State) :
    linearCost (v - u) (jointCoefficient n f xs target u) = ((n : ℝ) + 1)⁻¹ * ∑ i, ∑ c,
      residual (jointScore f u (xs i)) (target i) c *
        (jointScore f v (xs i) c - jointScore f u (xs i) c) :=
  coefficient_pairing n 1 (fun _ => 1) f xs target u v

/-- Actual CE first-order lower bound on the enlarged weighted-row space,
derived from the complete executed logits, CM104. -/
theorem joint_first_order (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u v : J → O → State) :
    jointLoss n f xs target u + linearCost (v - u) (jointCoefficient n f xs target u) ≤
      jointLoss n f xs target v := mean_first_order n 1 (fun _ => 1) f xs target u v

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
