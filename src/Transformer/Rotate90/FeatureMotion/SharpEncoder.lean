module

public import Transformer.Rotate90.FeatureMotion.SharpProduct
public import Transformer.Rotate90.FeatureMotion.Encoder

/-!
# Pair-amplitude propagation through the complete source encoder

Source: `review/flat_spectrum.md`, *Pairwise encoder amplification*, PA1.
Only the old product amplification changes. All parameter motion, mixed
terms, normalizations and the original conservative comparison are retained.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Landscape.Neural Grokking.GradientEvidence
variable {V K H : Type*} [Fintype K] [Fintype H] [Nonempty K]

/-- PA1: propagate two old pair-amplitude caps through the entire encoder. -/
def pairFeatureRadius (eps : ℝ) (w : Weights V K H) (a b : V)
    (R : Caps V) (L : Operators) (G U : ℝ) : ℝ :=
  let t := trace eps w a b
  let input := rmsRadius eps t.gather (R.row a + R.row b)
  let inner := rmsRadius eps t.input input
  let gate := (L.gate + R.gate) * inner + R.gate * ‖t.inner‖
  let value := (L.value + R.value) * inner + R.value * ‖t.inner‖
  let hidden := (gate * U + G * value + gate * value) / Real.sqrt 2
  let branch := (L.output + R.output) * hidden + R.output * ‖t.hidden‖
  rmsRadius eps t.residual (input + rmsRadius eps t.branch branch)

/-- PA1: the sharper graph bound has no future activations on its RHS. -/
theorem feature_motion_pair (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b : V) (R : Caps V) (L : Operators) (G U : ℝ)
    (hG0 : 0 ≤ G) (hU0 : 0 ≤ U)
    (hgp : ∀ k, (trace eps w a b).gate (k, 0)^2 +
      (trace eps w a b).gate (k, 1)^2 ≤ G^2)
    (hup : ∀ k, (trace eps w a b).value (k, 0)^2 +
      (trace eps w a b).value (k, 1)^2 ≤ U^2)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output) :
    ‖(trace eps w' a b).feature - (trace eps w a b).feature‖ ≤
      pairFeatureRadius eps w a b R L G U := by
  have h0 := input_motion (w.embedding a) (w.embedding b) (w'.embedding a) (w'.embedding b)
    (R.row a) (R.row b) (hE a) (hE b)
  have h1 := rms_motion_radius eps heps _ _ _ h0
  have h2 := rms_motion_radius eps heps _ _ _ h1
  have h3 := linear_motion_of_bound w.gate w'.gate _ _ _ _ _ hLG hG h2
  have h4 := linear_motion_of_bound w.value w'.value _ _ _ _ _ hLU hU h2
  have h5 := product_motion_pair _ _ _ _ G U _ _ hG0 hU0 hgp hup h3 h4
  have h6 := linear_motion_of_bound w.output w'.output _ _ _ _ _ hLO hO h5
  have h7 := rms_motion_radius eps heps _ _ _ h6
  have h8 := residual_forecast _ _ _ _ _ _ h1 h7
  exact rms_motion_radius eps heps _ _ _ h8

/-- PA1: preserve the previous bound as a valid competing enclosure. -/
def sharpFeatureRadius (eps : ℝ) (w : Weights V K H) (a b : V)
    (R : Caps V) (L Lp : Operators) (G U : ℝ) : ℝ :=
  min (featureRadius eps w a b R L) (pairFeatureRadius eps w a b R Lp G U)

/-- PA1: intersecting the two proved graph bounds never loses enclosure. -/
theorem feature_motion_sharp (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b : V) (R : Caps V) (L Lp : Operators) (G U : ℝ)
    (hG0 : 0 ≤ G) (hU0 : 0 ≤ U)
    (hgp : ∀ k, (trace eps w a b).gate (k, 0)^2 +
      (trace eps w a b).gate (k, 1)^2 ≤ G^2)
    (hup : ∀ k, (trace eps w a b).value (k, 0)^2 +
      (trace eps w a b).value (k, 1)^2 ≤ U^2)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hLpG : ‖w.gate‖ ≤ Lp.gate) (hLpU : ‖w.value‖ ≤ Lp.value)
    (hLpO : ‖w.output‖ ≤ Lp.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output) :
    ‖(trace eps w' a b).feature - (trace eps w a b).feature‖ ≤
      sharpFeatureRadius eps w a b R L Lp G U := by
  exact le_min (feature_motion eps heps w w' a b R L hLG hLU hLO hE hG hU hO)
    (feature_motion_pair eps heps w w' a b R Lp G U hG0 hU0 hgp hup hLpG hLpU hLpO hE hG hU hO)

/-- PA1: the intersection cannot exceed the original radius, even when
the pairwise branch uses different certified operator caps. -/
theorem sharpFeatureRadius_le (eps : ℝ) (w : Weights V K H) (a b : V)
    (R : Caps V) (L Lp : Operators) (G U : ℝ) :
    sharpFeatureRadius eps w a b R L Lp G U ≤ featureRadius eps w a b R L :=
  min_le_left _ _

example := feature_motion_pair 1 (by norm_num) witnessWeights witnessWeights false true
  ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩
  ‖(trace 1 witnessWeights false true).gate‖ ‖(trace 1 witnessWeights false true).value‖
  (norm_nonneg _) (norm_nonneg _) (pair_energy_le_norm_sq _) (pair_energy_le_norm_sq _)
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp)
example := feature_motion_sharp 1 (by norm_num) witnessWeights witnessWeights false true
  ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ ⟨1, 1, 1⟩
  ‖(trace 1 witnessWeights false true).gate‖ ‖(trace 1 witnessWeights false true).value‖
  (norm_nonneg _) (norm_nonneg _) (pair_energy_le_norm_sq _) (pair_energy_le_norm_sq _)
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by simp [witnessWeights]) (by simp [witnessWeights]) (by simp [witnessWeights])
  (by intro v; simp) (by simp) (by simp) (by simp)

end Transformer.Rotate90.FeatureMotion
