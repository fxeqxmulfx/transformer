module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness.Model
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Checker
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Observation

/-!
# Checked success and refusal on distinct genuinely unseen inputs

Source: `review/convex/moments.md`, CM87. The positive certificate is checked
in exact rational arithmetic. A different unseen sequence admits opposite
original-object outputs from two models fitting the same observations. Thus
even adequate representation capacity cannot settle that answer from these
data. This is a proof control, not a training or stress-test accuracy result.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness

open scoped BigOperators
open Restriction (cube)

/-- Exact rational weights for the full checked table, CM87. -/
def rationalWeights (i : Fin 3) (c : Bool) : ℚ := if c = labels i then 0 else 1

/-- The finite rational checker accepts the genuinely unseen face, CM87.
Checker correctness transfers the proof over every complete coordinate. -/
theorem checker_accepts : checkCertificate tokens labels query true false 3 rationalWeights = true := by
  apply (checkCertificate_iff tokens labels query true false 3 rationalWeights).mpr
  have he : (fun i c => (rationalWeights i c : ℝ)) = weights := by
    funext i c
    unfold rationalWeights weights
    split <;> norm_num
  rw [he]
  exact_mod_cast checked_certificate

/-- Another unseen raw input, distinguished by an unobserved middle object,
CM87. Both output classes remain feasible answers here. -/
def ambiguous : LearnedHinge.Prefix Bool (3 + 1) := cube false true false

/-- The ambiguous input is absent from every indexed observation, CM87. -/
theorem ambiguous_unseen (i : Fin 3) : ambiguous ≠ tokens i := by
  fin_cases i <;> decide

/-- Two fitting parameters predict opposite original objects on the same
unseen input, CM87. The family and training data are identical. -/
theorem ambiguous_predictions (witness : Bool) :
    nextObject (point 0) ambiguous witness = false ∧
      nextObject (point 2) ambiguous witness = true := by
  constructor
  · apply nextObject_of_unique
    intro c hc
    cases c
    · exact False.elim (hc rfl)
    · norm_num [ambiguous, (point_logits 0 false true false).1,
        (point_logits 0 false true false).2]
  · apply nextObject_of_unique
    intro c hc
    cases c
    · norm_num [ambiguous, (point_logits 2 false true false).1,
        (point_logits 2 false true false).2]
    · exact False.elim (hc rfl)

/-- Explicit nonidentifiability within the actual learned family, CM87.
These two models satisfy every observed unit margin. -/
theorem ambiguous_fitting_models : ∃ p q : SequenceParameters Bool 3,
    FitsObserved tokens labels p ∧ FitsObserved tokens labels q ∧
      nextObject p ambiguous false ≠ nextObject q ambiguous false := by
  refine ⟨point 0, point 2, point_fits 0, point_fits 2, ?_⟩
  rw [(ambiguous_predictions false).1, (ambiguous_predictions false).2]
  decide

/-- No positive certificate exists for either proposed answer on this
ambiguous input, CM87. Refusal follows from actual fitting countermodels. -/
theorem certificate_refusal (a : Bool) (bound : ℝ) (hpos : 0 < bound) :
    ¬ ∃ w, Certificate tokens labels ambiguous a (!a) bound w := by
  rintro ⟨w, hw⟩
  cases a
  · have h := certificate_sound tokens labels ambiguous false true bound w hw (point 2) (point_fits 2)
    norm_num [margin, ambiguous, (point_logits 2 false true false).1,
      (point_logits 2 false true false).2] at h
    linarith
  · have h := certificate_sound tokens labels ambiguous true false bound w hw (point 0) (point_fits 0)
    norm_num [margin, ambiguous, (point_logits 0 false true false).1,
      (point_logits 0 false true false).2] at h
    linarith

/-- The concrete feasible model attains the CE threshold on every record,
CM87. This uses ordinary full-vocabulary CE and nonconstant original labels. -/
theorem point_zero_loss (i : Fin 3) :
    sequenceLoss (tokens i) (labels i) (point 0) = Real.log (1 + Real.exp (-1)) := by
  fin_cases i <;>
    simp [sequenceLoss, ConvexRecall.crossEntropy_relative, tokens, labels,
      (point_logits 0 _ _ _).1, (point_logits 0 _ _ _).2] <;> norm_num [add_comm]

example : FitsObserved tokens labels (point 0) :=
  fits_of_record_losses tokens labels (point 0) (fun i => (point_zero_loss i).le)

example : 3 ≤ margin (point 0) query true false :=
  certificate_sound tokens labels query true false 3 weights checked_certificate (point 0) (point_fits 0)

example : form (coefficient query true false) 0 = 0 :=
  certificate_kills_kernel tokens labels query true false 3 weights checked_certificate 0 (map_zero _)

example : nextObject (point 0) query false = true := by
  apply nextObject_of_certificates tokens labels query true false ?_ (point 0) (point_fits 0)
  intro b hb
  cases b
  · exact ⟨3, weights, by norm_num, checked_certificate⟩
  · exact False.elim (hb rfl)

example : ¬ ∃ w, Certificate tokens labels ambiguous true false 1 w :=
  certificate_refusal true 1 (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Identification.Witness
