module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Basic

/-!
# Actual learned complex weights realizing a singleton-prefix response

Source: `review/flat_spectrum.md`, CM77. A learned bias 1-T and one learned
unit weight per matching position/object pair produce the exact preactivation
count. Complex weights multiply these values by 2+i, retaining nonzero real
and imaginary components; the actual readout phase yields the gain 1+3i.
No prefix comparison is executed by the model's original raw forward.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen

open scoped BigOperators
open PatternReduction (realInput)

noncomputable section

variable {O : Type*} [DecidableEq O] {T : ℕ}

/-- A permitted learned raw bias and position/object weight vector, CM77. -/
def realWeights (needle : Fin T → O) : Coordinate O T → ℝ
  | none => 1 - T
  | some (i, o) => if o = needle i then 1 else 0

/-- Both components of the representation are retained in the actual
complex weights, CM77. The readout is recovered by the original model. -/
def complexWeights (needle : Fin T → O) : ObjectWeights O T :=
  fun d => (2 + Complex.I) * (realWeights needle d : ℂ)

/-- The count equals the full real indicator sum, CM77. -/
theorem matchCount_cast (needle : Fin T → O) (tokens : Prefix O T) :
    (matchCount needle tokens : ℝ) =
      ∑ i : Fin T, if tokens.val[i.val]? = some (needle i) then 1 else 0 := by
  unfold matchCount
  rw [Finset.card_filter, Nat.cast_sum]
  apply Finset.sum_congr rfl
  intro i hi
  split_ifs <;> norm_num

/-- The original raw lookup computes the asserted score from actual learned
weights, CM77. The score is not substituted into the forward definition. -/
theorem realInput_weights (needle : Fin T → O) (tokens : Prefix O T) :
    realInput tokens (realWeights needle) = score needle tokens := by
  rw [PatternReduction.realInput_apply, score, matchCount_cast]
  change 1 - (T : ℝ) + _ = _
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  cases tokens.val[i.val]? <;> simp [realWeights]

/-- The actual raw complex preactivation keeps the complete factor 2+i,
CM77. No imaginary coordinate is ignored. -/
theorem rawInput_weights (needle : Fin T → O) (tokens : Prefix O T) :
    rawInput tokens (complexWeights needle) = (2 + Complex.I) * (score needle tokens : ℂ) := by
  have hr : PatternReduction.project true (complexWeights needle) = (2 : ℝ) • realWeights needle := by
    funext d
    simp [PatternReduction.project, complexWeights]
  have hi : PatternReduction.project false (complexWeights needle) = realWeights needle := by
    funext d
    simp [PatternReduction.project, complexWeights]
  rw [PatternReduction.rawInput_complexInput]
  apply Complex.ext
  · change realInput tokens (PatternReduction.project true (complexWeights needle)) = _
    rw [hr, map_smul, realInput_weights]
    simp
  · change realInput tokens (PatternReduction.project false (complexWeights needle)) = _
    rw [hi, realInput_weights]
    simp

/-- Componentwise complex ReLU on the used direction preserves its phase,
CM77, for every real preactivation including negative values. -/
theorem split_direction (r : ℝ) :
    splitHinge ((2 + Complex.I) * (r : ℂ)) = (2 + Complex.I) * (hinge r : ℂ) := by
  apply Complex.ext <;>
    simp [splitHinge, hinge, mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 2)]

variable [Fintype O]

/-- The actual recovered complex neuron isolates the selected full prefix,
CM77. Every other admitted prefix, shorter ones included, has zero response. -/
theorem neuron_singleton (needle : Fin T → O) (tokens : Prefix O T) :
    complexNeuron (complexWeights needle) (rawInput tokens) =
      if tokens.val = List.ofFn needle then (⟨1, 3⟩ : ℂ) else 0 := by
  rw [complexNeuron_eq, rawInput_weights, split_direction, hinge_score]
  by_cases h : tokens.val = List.ofFn needle
  · simp only [h, ↓reduceIte, Complex.ofReal_one, mul_one]
    apply Complex.ext <;> norm_num
  · simp [h]

end

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen
