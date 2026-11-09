module

public import Transformer.Rotate90.FeatureMotion.Encoder
public import Transformer.Grokking.GradientEvidence.ProbabilityMotion

/-!
# Probability ceilings from the complete moving encoder

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM4–EM5 and PM3.
Feature motion is derived from parameter caps through the actual graph;
it is no longer a separate hypothesis. Execution-contrast change stays explicit.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Grokking.GradientEvidence Grokking.Geometry Grokking.FlatSpectrum.FeatureCodes
open Landscape.Neural Finset
open scoped BigOperators
variable {V K H : Type*} [Fintype K] [Fintype H] [Nonempty K]

/-- EM4: normalized source features use the same embedding for the readout. -/
def logit (eps : ℝ) (w : Weights V K H) (a b c : V) : ℝ :=
  linearScore (w.embedding c) (trace eps w a b).feature

/-- EM4–PM3: complete exact-real contrast motion from checkpoint caps. -/
def contrastRadius (eps : ℝ) (w : Weights V K H) (a b : V) (R : Caps V)
    (L : Operators) (u c : V) : ℝ :=
  let S := featureRadius eps w a b R L
  let r := R.row u + R.row c
  r * ‖(trace eps w a b).feature‖ + ‖w.embedding u - w.embedding c‖ * S + r * S

omit [Nonempty K] in
/-- EM4: a tied readout contrast moves by at most the sum of its row caps. -/
theorem readout_motion (E E' : V → EuclideanSpace ℝ (Channels K)) (r : V → ℝ)
    (hr : ∀ v, ‖E' v - E v‖ ≤ r v) (u c : V) :
    ‖(E' u - E' c) - (E u - E c)‖ ≤ r u + r c := by
  have he : (E' u - E' c) - (E u - E c) = (E' u - E u) - (E' c - E c) := by abel
  rw [he]
  exact (norm_sub_le _ _).trans (add_le_add (hr u) (hr c))

/-- EM4–PM3: actual executed contrast motion, with the feature radius
proved through every encoder layer. Both old and new execution errors enter. -/
theorem executed_contrast_motion (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b u c : V) (R : Caps V) (L : Operators) (e e' : V → ℝ) (δ : ℝ)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output) (he : |(e' u-e' c)-(e u-e c)| ≤ δ) :
    |((logit eps w' a b u+e' u)-(logit eps w' a b c+e' c))-
      ((logit eps w a b u+e u)-(logit eps w a b c+e c))| ≤
      contrastRadius eps w a b R L u c + δ := by
  have hf := feature_motion eps heps w w' a b R L hLG hLU hLO hE hG hU hO
  have hc := readout_motion w.embedding w'.embedding R.row hE u c
  have hS := (norm_nonneg _).trans hf
  have hR := (norm_nonneg _).trans hc
  apply executed_contrast_motion_le _ _ _ _ e e' u c _ _ _ _ δ
    (norm_nonneg _) (norm_nonneg _) hR hS
  · exact (EuclideanSpace.real_norm_sq_eq (w.embedding u-w.embedding c)).symm.le
  · exact (EuclideanSpace.real_norm_sq_eq (trace eps w a b).feature).symm.le
  · have h := (sq_le_sq₀ (norm_nonneg _) hR).mpr hc
    simpa only [EuclideanSpace.real_norm_sq_eq, PiLp.sub_apply] using h
  · have h := (sq_le_sq₀ (norm_nonneg _) hS).mpr hf
    simpa only [EuclideanSpace.real_norm_sq_eq, PiLp.sub_apply] using h
  · exact he

variable [Fintype V] [DecidableEq V]

/-- EM4–PM1: a checkpoint probability ceiling from full parameter caps.
Every rival is retained. No future feature displacement or probability cap
is assumed, and arithmetic execution remains a separately bounded residual. -/
theorem probability_le_encoder_motion (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b u : V) (R : Caps V) (L : Operators) (e e' δ : V → ℝ)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output)
    (he : ∀ c, c ≠ u → |(e' u-e' c)-(e u-e c)| ≤ δ c) :
    probability (fun c => logit eps w' a b c + e' c) u ≤
      1 / (1 + oddsEnvelope (fun c => (logit eps w a b c+e c)-(logit eps w a b u+e u)-
        (contrastRadius eps w a b R L u c+δ c)) u) := by
  apply probability_le_motion_envelope
  intro c hc
  exact (le_abs_self _).trans (executed_contrast_motion eps heps w w' a b u c R L e e' (δ c)
    hLG hLU hLO hE hG hU hO (he c hc))

example := readout_motion witnessWeights.embedding witnessWeights.embedding (fun _ => 1)
  (by intro v; simp) false true
example := executed_contrast_motion 1 (by norm_num) witnessWeights witnessWeights
  false true true false ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ (fun _ => 0) (fun _ => 0) 0
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp) (by norm_num)
example := probability_le_encoder_motion 1 (by norm_num) witnessWeights witnessWeights
  false true true ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ (fun _ => 0) (fun _ => 0) (fun _ => 0)
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp) (by intro c hc; norm_num)

end Transformer.Rotate90.FeatureMotion
