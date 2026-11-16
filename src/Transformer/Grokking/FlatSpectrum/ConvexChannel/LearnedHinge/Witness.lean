module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Lifting
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Sequence

/-!
# Learned complex representations change relative next-object predictions

Source: `review/flat_spectrum.md`, CM75. Two input objects read two learned
complex representation coordinates. Two feasible points have different
representation directions, different recovered readouts, and reversed
relative predictions. Both observed answer losses strictly improve.
These are comparison points, not a claimed native optimization trajectory.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Witness

open scoped BigOperators

noncomputable section

/-- One visible object selects its actual learned representation coordinate. -/
def query (x : Bool) : (ℂ × ℂ) →ₗ[ℝ] ℂ :=
  if x then LinearMap.snd ℝ ℂ ℂ else LinearMap.fst ℝ ℂ ℂ

/-- First full complex joint point; its imaginary coordinate is nonzero. -/
def first : ℂ × ℂ := (4, Complex.I)

/-- The second point has a different representation direction and norm. -/
def second : ℂ × ℂ := (Complex.I, 9)

/-- A nonzero positive neuron for one class; every other coordinate is kept. -/
def weights (u : ℂ × ℂ) : Parameters Unit Bool (ℂ × ℂ) :=
  fun _ c s => if c && s then u else 0

/-- Both stages have genuinely different recovered values, CM75.
The readout amplitude changes from two to three, and the representation
changes direction rather than just acquiring a common scale. -/
theorem recovered_stages_change :
    amplitude first = 2 ∧ amplitude second = 3 ∧
    representation first = ((2, Complex.I / 2) : ℂ × ℂ) ∧
    representation second = ((Complex.I / 3, 3) : ℂ × ℂ) := by
  have ha : amplitude first = 2 := by norm_num [amplitude, first]
  have hb : amplitude second = 3 := by norm_num [amplitude, second]
  refine ⟨ha, hb, ?_, ?_⟩
  · rw [representation, ha]
    apply Prod.ext <;> norm_num [first, div_eq_mul_inv, mul_comm]
  · rw [representation, hb]
    apply Prod.ext <;> norm_num [second, div_eq_mul_inv, mul_comm]

/-- The first point satisfies every real and imaginary activation constraint,
CM75. Both classes, both signs, and both inputs are checked. -/
theorem first_mem : weights first ∈ complexDomain query Set.univ (fun _ _ => true) := by
  intro j c s x hx
  rcases x with ⟨x, k⟩
  cases x <;> cases k <;> cases c <;> cases s <;>
    norm_num [weights, components, query, first, orient]

/-- The changed representation/readout point obeys the same cone, CM75. -/
theorem second_mem : weights second ∈ complexDomain query Set.univ (fun _ _ => true) := by
  intro j c s x hx
  rcases x with ⟨x, k⟩
  cases x <;> cases k <;> cases c <;> cases s <;>
    norm_num [weights, components, query, second, orient]

/-- The first actual full complex forward gives these literal real logits,
CM75. The negative logit comes from the imaginary representation coordinate. -/
theorem first_logits (x c : Bool) : complexLogits (weights first) (query x) c =
    if c then (if x then -1 else 4) else 0 := by
  cases x <;> cases c <;>
    norm_num [complexLogits, complexForward, complexNeuron_eq, weights, query, first, splitHinge, hinge]

/-- The changed actual forward reverses the object preference, CM75. -/
theorem second_logits (x c : Bool) : complexLogits (weights second) (query x) c =
    if c then (if x then 9 else -1) else 0 := by
  cases x <;> cases c <;>
    norm_num [complexLogits, complexForward, complexNeuron_eq, weights, query, second, splitHinge, hinge]

/-- Both relative preferences change sign, CM75. A positive confidence
rescaling of the first logits cannot produce the second predictions. -/
theorem predictions_reverse :
    complexLogits (weights first) (query false) false < complexLogits (weights first) (query false) true ∧
    complexLogits (weights second) (query false) true < complexLogits (weights second) (query false) false ∧
    complexLogits (weights first) (query true) true < complexLogits (weights first) (query true) false ∧
    complexLogits (weights second) (query true) false < complexLogits (weights second) (query true) true := by
  norm_num [first_logits, second_logits]

/-- Ordinary CE for the first object's actual next-object target decreases,
CM75. No hidden path, teacher feature, or auxiliary loss occurs. -/
theorem false_loss_improves :
    complexLoss (query false) false (weights second) < complexLoss (query false) false (weights first) := by
  unfold complexLoss
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  simp only [Fintype.sum_bool, first_logits, second_logits]
  norm_num only [Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte, sub_zero, sub_self, Real.exp_zero]
  apply Real.log_lt_log (by positivity)
  exact add_lt_add_left (Real.exp_lt_exp.mpr (by norm_num : (-1 : ℝ) < 4)) 1

/-- The second object's ordinary next-object CE also decreases, CM75. -/
theorem true_loss_improves :
    complexLoss (query true) true (weights second) < complexLoss (query true) true (weights first) := by
  unfold complexLoss
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  simp only [Fintype.sum_bool, first_logits, second_logits]
  norm_num only [Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte, sub_zero, sub_self, zero_sub,
    neg_neg, Real.exp_zero]
  apply Real.log_lt_log (by positivity)
  exact add_lt_add_right (Real.exp_lt_exp.mpr (by norm_num : (-9 : ℝ) < 1)) 1

example := complexLoss_convex (J := Unit) query Set.univ (fun _ _ => true) (Set.mem_univ false) true
example := complexLogits_on_cone query Set.univ (fun _ _ => true) first_mem (Set.mem_univ false)

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Witness
