module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard.Basic

/-!
# Raw decision stability, including deterministic ties

Source: `review/convex/moments.md`, CM47. Positive margin retention preserves
all strict reference comparisons as well as non-strict ones. Consequently
the least maximizing class is unchanged. The reference can still be wrong.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard

variable {X C : Type*} {reference z : X → C → ℝ} {label : X → C} {fraction : ℝ}

/-- A nonnegative reference gap remains nonnegative under a nonnegative
retention fraction, CM47. -/
theorem weak_comparison (hz : z ∈ region reference label fraction)
    (hzero : 0 ≤ fraction) (x : X) (c : C)
    (hcmp : reference x c ≤ reference x (label x)) : z x c ≤ z x (label x) := by
  have h := hz x c
  have hp := mul_nonneg hzero (sub_nonneg.mpr hcmp)
  simp only [gap] at h
  linarith

/-- A strictly positive fraction keeps every strictly positive reference
margin strictly positive, CM47; no uniform lower bound is assumed. -/
theorem strict_comparison (hz : z ∈ region reference label fraction)
    (hpos : 0 < fraction) (x : X) (c : C)
    (hcmp : reference x c < reference x (label x)) : z x c < z x (label x) := by
  have h := hz x c
  have hp := mul_pos hpos (sub_pos.mpr hcmp)
  simp only [gap] at h
  linarith

/-- The score-maximizing class with all lower-index ties excluded, CM47.
This is the criterion for an ordinary first-index argmax. -/
def firstMax [LT C] (scores : C → ℝ) (label : C) : Prop :=
  (∀ c, scores c ≤ scores label) ∧ (∀ c, c < label → scores c < scores label)

/-- Positive retention preserves the deterministic first maximizer even
when higher-index reference classes tie it, CM47. -/
theorem firstMax_preserved [LT C] (hz : z ∈ region reference label fraction)
    (hpos : 0 < fraction) (x : X) (href : firstMax (reference x) (label x)) :
    firstMax (z x) (label x) := by
  constructor
  · intro c
    exact weak_comparison hz (by linarith) x c (href.1 c)
  · intro c hc
    exact strict_comparison hz hpos x c (href.2 c hc)

/-- Two classes meeting the first-argmax criterion must be identical,
CM47. Together with preservation this fixes the predicted raw label. -/
theorem firstMax_unique [LinearOrder C] {scores : C → ℝ} {a b : C}
    (ha : firstMax scores a) (hb : firstMax scores b) : a = b := by
  rcases lt_trichotomy a b with hab | hab | hba
  · have h1 := hb.2 a hab
    have h2 := ha.1 b
    linarith
  · exact hab
  · have h1 := ha.2 b hba
    have h2 := hb.1 a
    linarith

/-- Any deterministic first-argmax predictor equals the reference predictor
throughout the constrained query domain, CM47. -/
theorem predictions_eq [LinearOrder C] (hz : z ∈ region reference label fraction)
    (hpos : 0 < fraction) (prediction : X → C)
    (href : ∀ x, firstMax (reference x) (label x))
    (hprediction : ∀ x, firstMax (z x) (prediction x)) : prediction = label := by
  funext x
  exact firstMax_unique (hprediction x) (firstMax_preserved hz hpos x (href x))

/-- Correct-record counts are identical for any targets, repeated inputs and
finite record batch, CM47. A wrong reference label is preserved as well. -/
theorem batch_correct_eq [LinearOrder C] {I : Type*}
    (hz : z ∈ region reference label fraction) (hpos : 0 < fraction)
    (prediction : X → C) (href : ∀ x, firstMax (reference x) (label x))
    (hprediction : ∀ x, firstMax (z x) (prediction x))
    (inputs : I → X) (targets : I → C) (batch : Finset I) :
    (batch.filter (fun i => prediction (inputs i) = targets i)).card =
      (batch.filter (fun i => label (inputs i) = targets i)).card := by
  have h := predictions_eq hz hpos prediction href hprediction
  rw [h]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard
