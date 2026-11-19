module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Joint

/-!
# Exact recovery of positive encoder and decoder distributions

Source: `review/convex/moments.md`, CM74. For a single input, the positive joint
simplex represents exactly all positive stochastic encoders and conditional
decoders. The decoder has an independent row for every hidden state. The
result does not retain the counterexample's keep/flip parameter tie or impose
one decoder shared across distinct inputs. Those are additional constraints.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Joint

open scoped BigOperators

variable {H C : Type*} [Fintype H] [Fintype C]

/-- The actual joint table of a positive two-stage stochastic model, CM74. -/
def ofStages (E : H → ℝ) (D : H → C → ℝ) : H → C → ℝ :=
  fun h c => E h * D h c

/-- Every positive encoder/decoder pair has an admitted joint point, CM74.
This is class inclusion, not convexity in the two independent factors. -/
theorem ofStages_mem (E : H → ℝ) (D : H → C → ℝ)
    (hE : ∀ h, 0 < E h) (hEsum : ∑ h, E h = 1)
    (hD : ∀ h c, 0 < D h c) (hDsum : ∀ h, ∑ c, D h c = 1) :
    ofStages E D ∈ domain := by
  refine ⟨fun h c => mul_pos (hE h) (hD h c), ?_⟩
  simp only [ofStages, ← Finset.mul_sum, hDsum, mul_one]
  exact hEsum

omit [Fintype H] in
/-- The reconstructed encoder is the original one, CM74. -/
theorem encoder_ofStages (E : H → ℝ) (D : H → C → ℝ)
    (hDsum : ∀ h, ∑ c, D h c = 1) : encoder (ofStages E D) = E := by
  funext h
  simp only [encoder, ofStages, ← Finset.mul_sum, hDsum, mul_one]

omit [Fintype H] in
/-- Positive encoder rows also recover every original decoder entry, CM74. -/
theorem decoder_ofStages (E : H → ℝ) (D : H → C → ℝ)
    (hE : ∀ h, 0 < E h) (hDsum : ∀ h, ∑ c, D h c = 1) :
    decoder (ofStages E D) = D := by
  funext h c
  unfold decoder
  rw [encoder_ofStages E D hDsum]
  unfold ofStages
  field_simp [(hE h).ne']

/-- Every admitted point recovers its complete joint table from the two
learned stages, CM74. This is an exact inverse, not a low-rank approximation. -/
theorem ofStages_recovered [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain) :
    ofStages (encoder P) (decoder P) = P := by
  funext h c
  unfold ofStages decoder
  field_simp [(encoder_pos hP h).ne']

/-- The reconstructed encoder is a normalized distribution, CM74. -/
theorem encoder_sum {P : H → C → ℝ} (hP : P ∈ domain) :
    ∑ h, encoder P h = 1 := hP.2

/-- Every recovered decoder entry is positive, CM74. -/
theorem decoder_pos [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain)
    (h : H) (c : C) : 0 < decoder P h c :=
  div_pos (hP.1 h c) (encoder_pos hP h)

/-- Every recovered decoder row is normalized, CM74. -/
theorem decoder_sum [Nonempty C] {P : H → C → ℝ} (hP : P ∈ domain) (h : H) :
    ∑ c, decoder P h c = 1 := by
  simp only [decoder, ← Finset.sum_div]
  change encoder P h / encoder P h = 1
  exact div_self (encoder_pos hP h).ne'

end Transformer.Grokking.FlatSpectrum.ConvexChannel.HiddenPath.Joint
