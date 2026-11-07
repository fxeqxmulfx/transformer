module

public import Transformer.Rotate90.ComplexProduct.TrainingTransport
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ProbabilityTransfer

/-!
# Probability transfer through the normalized source and numerical execution

Source: `review/flat_spectrum.md`, *Probability transfer*, PT4.
Training and query states retain the actual fourth RMS denominator before
the tied linear readout. Every logit execution error contributes to the
relative-logit budget. Task-preserving routes transfer observed labels
separately; no learned equivariance or optimizer discovery is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural Grokking.Geometry Grokking.FlatSpectrum.FeatureCodes
open scoped BigOperators

variable {I V K H : Type*} [Fintype K] [Fintype H]

/-- Actual residual state after the fourth RMS normalization, PT4. -/
def readoutState (eps : ℝ) (w : Weights V K H) (a b : V) : Channels K → ℝ :=
  fun k => realify (state eps w a b) k / denominator eps (state eps w a b)

/-- Linear readout of that state is exactly the original source logit, PT4. -/
theorem readoutState_logit (eps : ℝ) (w : Weights V K H) (a b c : V) :
    linearScore (realify (w.embedding c)) (readoutState eps w a b) = realLogit eps w a b c := by
  rw [realLogit_eq_readout_div, ← realify_readout]
  simp only [linearScore, readoutState, tiedReadout, ← mul_div_assoc, ← Finset.sum_div]
  congr 1
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Execution adds twice the per-logit error to the geometric log-odds bound,
PT4. References may be normalized training states or their computed centers;
the query uses the full normalized source. -/
theorem executed_relative_le_anchor (eps : ℝ) (w : Weights V K H)
    (references : I → Channels K → ℝ) (a b t : V) (anchor : V → I)
    (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w a b c| ≤ η) (c : V) :
    executed c - executed t ≤
      anchorLogOddsBound (fun v => realify (w.embedding v))
        references
        (readoutState eps w a b) t anchor c + 2 * η := by
  have h := relative_logit_le_anchor (fun v => realify (w.embedding v))
    references
    (readoutState eps w a b) t anchor c
  rw [readoutState_logit, readoutState_logit] at h
  have hc := (abs_le.mp (herr c)).2
  have ht := (abs_le.mp (herr t)).1
  linarith

/-- **An actual executed source probability is bounded from training anchors**,
PT4. The numerical allowance remains an explicit hypothesis. -/
theorem executed_probability_ge_anchor [Fintype V] [DecidableEq V]
    (eps : ℝ) (w : Weights V K H) (references : I → Channels K → ℝ)
    (a b t : V) (anchor : V → I) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w a b c| ≤ η) :
    1 / (1 + oddsEnvelope (fun c =>
      anchorLogOddsBound (fun v => realify (w.embedding v))
        references
        (readoutState eps w a b) t anchor c + 2 * η) t) ≤ probability executed t := by
  apply probability_ge_of_contrast_bounds
  intro c hc
  exact executed_relative_le_anchor eps w references a b t anchor executed herr c

/-- Actual query cross-entropy has the matching bound, PT4. -/
theorem executed_crossEntropy_le_anchor [Fintype V] [DecidableEq V]
    (eps : ℝ) (w : Weights V K H) (references : I → Channels K → ℝ)
    (a b t : V) (anchor : V → I) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ c, |executed c - realLogit eps w a b c| ≤ η) :
    crossEntropy executed t ≤ Real.log (1 + oddsEnvelope (fun c =>
      anchorLogOddsBound (fun v => realify (w.embedding v))
        references
        (readoutState eps w a b) t anchor c + 2 * η) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact executed_relative_le_anchor eps w references a b t anchor executed herr c

example := readoutState_logit 1 activeWeights 0 1 0
example := executed_relative_le_anchor (I := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp) 1
example := executed_probability_ge_anchor (I := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp)
example := executed_crossEntropy_le_anchor (I := Fin 1) 1 activeWeights
  (fun _ => readoutState 1 activeWeights 0 0)
  0 1 0 (fun _ => 0) (realLogit 1 activeWeights 0 1) (η := 0) (by intro c; simp)

end Transformer.Rotate90.ComplexProduct
