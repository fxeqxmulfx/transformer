module

public import Transformer.Rotate90.ComplexProduct.ProbabilityTransfer
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.SignedProbability

/-!
# Signed directional transport for the executed complex-product source

Source: `review/flat_spectrum.md`, *Signed directional transport*, ST3.
The exact normalized source readout inherits the signed training bound.
Execution error stays explicit; neither a floating-point SVD certificate nor
optimizer discovery is inferred from the algebraic inequality.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural Grokking.Geometry Grokking.FlatSpectrum.FeatureCodes

variable {I J V K H : Type*} [Fintype I] [Fintype J] [Fintype K] [Fintype H]

/-- **The actual executed relative logit obeys signed transport**, ST3.
All four source normalizations enter through `readoutState`. -/
theorem executed_relative_le_signed (eps : ℝ) (w : Weights V K H)
    (T : I → Channels K → ℝ) (center : Channels K → ℝ) (left right t : V)
    (a : I → ℝ) (B : J → Channels K → ℝ) (b : V → J → ℝ)
    (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w left right c| ≤ η) (c : V) :
    executed c - executed t ≤
      signedLogOddsBound (fun v => realify (w.embedding v)) T center
        (readoutState eps w left right) a B b t c + 2 * η := by
  have h := relative_logit_le_signed (fun v => realify (w.embedding v)) T center
    (readoutState eps w left right) a B b t c
  rw [readoutState_logit, readoutState_logit] at h
  have hc := (abs_le.mp (herr c)).2
  have ht := (abs_le.mp (herr t)).1
  linarith

/-- Executed probability inherits the floor with its numerical allowance,
ST3. The right side of each contrast bound uses training scores and geometry. -/
theorem executed_probability_ge_signed [Fintype V] [DecidableEq V]
    (eps : ℝ) (w : Weights V K H) (T : I → Channels K → ℝ)
    (center : Channels K → ℝ) (left right t : V) (a : I → ℝ)
    (B : J → Channels K → ℝ) (b : V → J → ℝ) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w left right c| ≤ η) :
    1 / (1 + oddsEnvelope (fun c => signedLogOddsBound
      (fun v => realify (w.embedding v)) T center (readoutState eps w left right) a B b t c +
      2 * η) t) ≤ probability executed t := by
  apply probability_ge_of_contrast_bounds
  intro c hc
  exact executed_relative_le_signed eps w T center left right t a B b executed herr c

/-- Executed CE has the corresponding upper bound, ST3. -/
theorem executed_crossEntropy_le_signed [Fintype V] [DecidableEq V]
    (eps : ℝ) (w : Weights V K H) (T : I → Channels K → ℝ)
    (center : Channels K → ℝ) (left right t : V) (a : I → ℝ)
    (B : J → Channels K → ℝ) (b : V → J → ℝ) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w left right c| ≤ η) :
    crossEntropy executed t ≤ Real.log (1 + oddsEnvelope (fun c => signedLogOddsBound
      (fun v => realify (w.embedding v)) T center (readoutState eps w left right) a B b t c +
      2 * η) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact executed_relative_le_signed eps w T center left right t a B b executed herr c

example := executed_relative_le_signed (I := Fin 1) (J := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0) (readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (fun _ => readoutState 1 activeWeights 0 0) (fun _ _ => 0)
  (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp) 1
example := executed_probability_ge_signed (I := Fin 1) (J := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0) (readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (fun _ => readoutState 1 activeWeights 0 0) (fun _ _ => 0)
  (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp)
example := executed_crossEntropy_le_signed (I := Fin 1) (J := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0) (readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (fun _ => readoutState 1 activeWeights 0 0) (fun _ _ => 0)
  (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp)

end Transformer.Rotate90.ComplexProduct
