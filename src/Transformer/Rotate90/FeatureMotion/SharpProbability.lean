module

public import Transformer.Rotate90.FeatureMotion.SharpEncoder
public import Transformer.Rotate90.FeatureMotion.Probability

/-!
# Probability control with pair-amplitude encoder radii

Source: `review/flat_spectrum.md`, *Pairwise encoder amplification*, PA1.
The sharper complete-encoder bound enters the existing executed-contrast
and all-rival probability results. Execution error remains explicit.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Grokking.GradientEvidence Grokking.Geometry
open Landscape.Neural
variable {V K H : Type*} [Fintype K] [Fintype H] [Nonempty K]

/-- PA1–PM3: the old pair amplitudes enter the tied readout budget. -/
def sharpContrastRadius (eps : ℝ) (w : Weights V K H) (a b : V) (R : Caps V)
    (L Lp : Operators) (G U : ℝ) (u c : V) : ℝ :=
  let S := sharpFeatureRadius eps w a b R L Lp G U
  let r := R.row u + R.row c
  r * ‖(trace eps w a b).feature‖ + ‖w.embedding u - w.embedding c‖ * S + r * S

/-- PA1–PM3: the sharp feature radius bounds executed contrast motion. -/
theorem executed_contrast_motion_sharp (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b u c : V) (R : Caps V) (L Lp : Operators) (G U : ℝ) (e e' : V → ℝ) (δ : ℝ)
    (hG0 : 0 ≤ G) (hU0 : 0 ≤ U)
    (hgp : ∀ k, (trace eps w a b).gate (k, 0)^2 + (trace eps w a b).gate (k, 1)^2 ≤ G^2)
    (hup : ∀ k, (trace eps w a b).value (k, 0)^2 + (trace eps w a b).value (k, 1)^2 ≤ U^2)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hLpG : ‖w.gate‖ ≤ Lp.gate) (hLpU : ‖w.value‖ ≤ Lp.value)
    (hLpO : ‖w.output‖ ≤ Lp.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output) (he : |(e' u-e' c)-(e u-e c)| ≤ δ) :
    |((logit eps w' a b u+e' u)-(logit eps w' a b c+e' c))-
      ((logit eps w a b u+e u)-(logit eps w a b c+e c))| ≤
      sharpContrastRadius eps w a b R L Lp G U u c + δ := by
  have hf := feature_motion_sharp eps heps w w' a b R L Lp G U hG0 hU0 hgp hup
    hLG hLU hLO hLpG hLpU hLpO hE hG hU hO
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

/-- PA1–PM1: retain every rival with the sharper full-encoder radius.
No future feature or probability premise replaces the parameter caps. -/
theorem probability_le_sharp_encoder_motion (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b u : V) (R : Caps V) (L Lp : Operators) (G U : ℝ) (e e' δ : V → ℝ)
    (hG0 : 0 ≤ G) (hU0 : 0 ≤ U)
    (hgp : ∀ k, (trace eps w a b).gate (k, 0)^2 + (trace eps w a b).gate (k, 1)^2 ≤ G^2)
    (hup : ∀ k, (trace eps w a b).value (k, 0)^2 + (trace eps w a b).value (k, 1)^2 ≤ U^2)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hLpG : ‖w.gate‖ ≤ Lp.gate) (hLpU : ‖w.value‖ ≤ Lp.value)
    (hLpO : ‖w.output‖ ≤ Lp.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output)
    (he : ∀ c, c ≠ u → |(e' u-e' c)-(e u-e c)| ≤ δ c) :
    probability (fun c => logit eps w' a b c + e' c) u ≤
      1 / (1 + oddsEnvelope (fun c => (logit eps w a b c+e c)-(logit eps w a b u+e u)-
        (sharpContrastRadius eps w a b R L Lp G U u c+δ c)) u) := by
  apply probability_le_motion_envelope
  intro c hc
  exact (le_abs_self _).trans (executed_contrast_motion_sharp eps heps w w' a b u c R L Lp
    G U e e' (δ c) hG0 hU0 hgp hup hLG hLU hLO hLpG hLpU hLpO hE hG hU hO (he c hc))

example := executed_contrast_motion_sharp 1 (by norm_num) witnessWeights witnessWeights
  false true true false ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ ⟨1, 1, 1⟩
  ‖(trace 1 witnessWeights false true).gate‖ ‖(trace 1 witnessWeights false true).value‖
  (fun _ => 0) (fun _ => 0) 0 (norm_nonneg _) (norm_nonneg _)
  (pair_energy_le_norm_sq _) (pair_energy_le_norm_sq _)
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp) (by norm_num)
example := probability_le_sharp_encoder_motion 1 (by norm_num) witnessWeights witnessWeights
  false true true ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ ⟨1, 1, 1⟩
  ‖(trace 1 witnessWeights false true).gate‖ ‖(trace 1 witnessWeights false true).value‖
  (fun _ => 0) (fun _ => 0) (fun _ => 0) (norm_nonneg _) (norm_nonneg _)
  (pair_energy_le_norm_sq _) (pair_energy_le_norm_sq _)
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp) (by intro c hc; norm_num)

end Transformer.Rotate90.FeatureMotion
