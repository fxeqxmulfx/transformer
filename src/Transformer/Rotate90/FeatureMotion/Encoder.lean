module

public import Transformer.Rotate90.FeatureMotion.Product

/-!
# Propagating parameter balls through the whole Rotate90 encoder

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM4;
`experiments/flat_approx/fixtures/reference.rs`, plain arm, at 62638521.
Real matrices allow the source's tied complex-linear matrices as a special
case. Logical hidden pairs replace its split real/imaginary storage only.
There are no learned normalization gains and no frozen encoder parameters.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Landscape.Neural Grokking.GradientEvidence
variable {V K H : Type*} [Fintype K] [Fintype H]

/-- EM4: shared input/readout embeddings and the three moving real maps.
No claim that arbitrary real maps preserve the source parameter ties is made. -/
structure Weights (V K H : Type*) [Fintype K] [Fintype H] where
  embedding : V → EuclideanSpace ℝ (Channels K)
  gate : EuclideanSpace ℝ (Channels K) →L[ℝ] EuclideanSpace ℝ (Channels H)
  value : EuclideanSpace ℝ (Channels K) →L[ℝ] EuclideanSpace ℝ (Channels H)
  output : EuclideanSpace ℝ (Channels H) →L[ℝ] EuclideanSpace ℝ (Channels K)

/-- EM4: actual intermediate values, never a future-bound hypothesis. -/
structure Trace (K H : Type*) where
  (gather input inner branch residual feature : EuclideanSpace ℝ (Channels K))
  (gate value hidden : EuclideanSpace ℝ (Channels H))

/-- EM4: the entire plain-arm source graph, including all four RMSNorms. -/
def trace (eps : ℝ) (w : Weights V K H) (a b : V) : Trace K H :=
  let gather := w.embedding a + turn (w.embedding b)
  let input := rms eps gather
  let inner := rms eps input
  let gate := w.gate inner
  let value := w.value inner
  let hidden := pairedProduct gate value
  let branch := w.output hidden
  let residual := input + rms eps branch
  { gather, input, inner, gate, value, hidden, branch, residual, feature := rms eps residual }

/-- EM4: parameter-displacement caps, not observations of future features. -/
structure Caps (V : Type*) where
  row : V → ℝ
  (gate value output : ℝ)

/-- EM4: certified checkpoint operator caps, computed before future motion. -/
structure Operators where
  (gate value output : ℝ)

/-- EM4: a checkpoint-only radius for the final normalized feature.
Every matrix-motion/input-motion product and every RMS layer is retained. -/
def featureRadius [Nonempty K] (eps : ℝ) (w : Weights V K H) (a b : V)
    (R : Caps V) (L : Operators) : ℝ :=
  let t := trace eps w a b
  let input := rmsRadius eps t.gather (R.row a + R.row b)
  let inner := rmsRadius eps t.input input
  let gate := (L.gate + R.gate) * inner + R.gate * ‖t.inner‖
  let value := (L.value + R.value) * inner + R.value * ‖t.inner‖
  let hidden := (gate * ‖t.value‖ + ‖t.gate‖ * value + gate * value) / Real.sqrt 2
  let branch := (L.output + R.output) * hidden + R.output * ‖t.hidden‖
  rmsRadius eps t.residual (input + rmsRadius eps t.branch branch)

variable [Nonempty K]

/-- EM4: full parameter motion implies final feature motion. The bound
contains only the checkpoint weights and declared parameter-displacement
caps; there are no next-step matrices, gradients or features on its RHS. -/
theorem feature_motion (eps : ℝ) (heps : 0 < eps) (w w' : Weights V K H)
    (a b : V) (R : Caps V) (L : Operators)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hE : ∀ v, ‖w'.embedding v - w.embedding v‖ ≤ R.row v)
    (hG : ‖w'.gate - w.gate‖ ≤ R.gate) (hU : ‖w'.value - w.value‖ ≤ R.value)
    (hO : ‖w'.output - w.output‖ ≤ R.output) :
    ‖(trace eps w' a b).feature - (trace eps w a b).feature‖ ≤ featureRadius eps w a b R L := by
  have h0 := input_motion (w.embedding a) (w.embedding b) (w'.embedding a) (w'.embedding b)
    (R.row a) (R.row b) (hE a) (hE b)
  have h1 := rms_motion_radius eps heps _ _ _ h0
  have h2 := rms_motion_radius eps heps _ _ _ h1
  have h3 := linear_motion_of_bound w.gate w'.gate _ _ _ _ _ hLG hG h2
  have h4 := linear_motion_of_bound w.value w'.value _ _ _ _ _ hLU hU h2
  have h5 := product_motion _ _ _ _ _ _ h3 h4
  have h6 := linear_motion_of_bound w.output w'.output _ _ _ _ _ hLO hO h5
  have h7 := rms_motion_radius eps heps _ _ _ h6
  have h8 := residual_forecast _ _ _ _ _ _ h1 h7
  exact rms_motion_radius eps heps _ _ _ h8

/-- EM4: the full final feature retains the architecture's energy cap. -/
theorem feature_norm_le (eps : ℝ) (heps : 0 < eps) (w : Weights V K H) (a b : V) :
    ‖(trace eps w a b).feature‖ ≤ Real.sqrt (Fintype.card (Channels K)) := by
  exact rms_norm_le eps heps _

/-- EM4: final normalization supplies PM4's ordinary feature-energy cap,
equal to the real width; it need not be assumed about an unknown future state. -/
theorem feature_energy_le (eps : ℝ) (heps : 0 < eps) (w : Weights V K H) (a b : V) :
    (∑ i, (trace eps w a b).feature i ^ 2) ≤ Fintype.card (Channels K) := by
  have h := (sq_le_sq₀ (norm_nonneg _) (Real.sqrt_nonneg _)).mpr
    (feature_norm_le eps heps w a b)
  rw [EuclideanSpace.real_norm_sq_eq, Real.sq_sqrt (by positivity)] at h
  exact h

/-- EM4: valid parameter caps give a nonnegative final radius, even when
some source intermediate values vanish. -/
theorem featureRadius_nonneg (eps : ℝ) (heps : 0 < eps) (w : Weights V K H)
    (a b : V) (R : Caps V) (L : Operators)
    (hLG : ‖w.gate‖ ≤ L.gate) (hLU : ‖w.value‖ ≤ L.value) (hLO : ‖w.output‖ ≤ L.output)
    (hE : ∀ v, 0 ≤ R.row v)
    (hG : 0 ≤ R.gate) (hU : 0 ≤ R.value) (hO : 0 ≤ R.output) :
    0 ≤ featureRadius eps w a b R L := by
  simpa only [sub_self, norm_zero] using feature_motion eps heps w w a b R L hLG hLU hLO
    (by simpa using hE) (by simpa using hG) (by simpa using hU) (by simpa using hO)

/-- A nonzero embedding and three active real maps for EM4's witnesses. -/
def witnessWeights : Weights Bool (Fin 1) (Fin 1) where
  embedding v := WithLp.toLp 2 (fun j => if j.2 = 0 then (if v then 2 else 1) else 0)
  gate := ContinuousLinearMap.id ℝ _
  value := ContinuousLinearMap.id ℝ _
  output := ContinuousLinearMap.id ℝ _

example := feature_motion 1 (by norm_num) witnessWeights witnessWeights false true
  ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ (by simp [witnessWeights]) (by simp [witnessWeights])
  (by simp [witnessWeights]) (by intro v; simp) (by simp) (by simp) (by simp)
example := feature_norm_le 1 (by norm_num) witnessWeights false true
example := feature_energy_le 1 (by norm_num) witnessWeights false true
example := featureRadius_nonneg 1 (by norm_num) witnessWeights false true
  ⟨fun _ => 1, 1, 1, 1⟩ ⟨1, 1, 1⟩ (by simp [witnessWeights]) (by simp [witnessWeights])
  (by simp [witnessWeights]) (by intro v; norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Rotate90.FeatureMotion
