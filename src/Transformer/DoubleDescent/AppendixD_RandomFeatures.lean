import Transformer.DoubleDescent.Section2_EffectiveComplexity
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
# Fixed random features: exact fitting and effective complexity

arXiv:1912.02292v1, Appendix D, "Random Features: A Case Study". The
statement `EMC = d` is not a distribution-independent consequence of having
`d` features. For a zero-target distribution, zero-initialized squared-loss
gradient flow remains at zero, fits every sample size, and has infinite EMC.
This refutes that unrestricted reading, not a numerical claim about Fashion
MNIST. This module treats real-valued features. The source's complex
exponential Fourier features are treated in `AppendixD_FourierFeatures`.
-/

namespace Transformer.DoubleDescent

open scoped ENNReal

variable {X : Type*} {d : ℕ}

/-- Appendix D: a fixed feature map followed by a trained linear output layer. -/
noncomputable def featurePrediction (features : X → Fin d → ℝ)
    (weight : Fin d → ℝ) (x : X) : ℝ := ∑ j, weight j * features x j

/-- Appendix D, counterexample: squared loss when the true target is zero. -/
noncomputable def zeroTargetFeatureLoss (features : X → Fin d → ℝ)
    (weight : Fin d → ℝ) (x : X) : ℝ≥0∞ :=
  ENNReal.ofReal (featurePrediction features weight x ^ 2)

/-- Appendix D: coordinate gradient of the empirical mean squared loss
for a zero-target sample, with the empty-sample convention zero. -/
noncomputable def zeroTargetGradient (features : X → Fin d → ℝ) {n : ℕ}
    (sample : Fin n → X) (weight : Fin d → ℝ) (j : Fin d) : ℝ :=
  (2 / (n : ℝ)) * ∑ i, featurePrediction features weight (sample i) * features (sample i) j

/-- Appendix D: zero-initialized squared-loss gradient descent on the
actual feature model. Every later iterate uses the sample and its gradient. -/
noncomputable def featureGD (features : X → Fin d → ℝ) (rate : ℝ)
    {n : ℕ} (sample : Fin n → X) : ℕ → (Fin d → ℝ)
  | 0 => 0
  | epoch + 1 => fun j => featureGD features rate sample epoch j -
      rate * zeroTargetGradient features sample (featureGD features rate sample epoch) j

/-- Appendix D: the finite-time training procedure implemented by these iterates. -/
noncomputable def zeroTargetFeatureTrain (features : X → Fin d → ℝ)
    (rate : ℝ) (epochs : ℕ) : TrainingProcedure X (Fin d → ℝ) :=
  fun _ sample => featureGD features rate sample epochs

/-- Appendix D: zero weights predict zero, for any fixed feature map. -/
theorem featurePrediction_zero (features : X → Fin d → ℝ) (x : X) :
    featurePrediction features 0 x = 0 := by
  simp [featurePrediction]

/-- Appendix D: at zero weights the squared-loss gradient is zero. -/
theorem zeroTargetGradient_zero (features : X → Fin d → ℝ)
    {n : ℕ} (sample : Fin n → X) (j : Fin d) :
    zeroTargetGradient features sample 0 j = 0 := by
  simp [zeroTargetGradient, featurePrediction_zero]

/-- Appendix D: this is also the stationary solution of the source's
gradient-flow equation, coordinate by coordinate, at every real time. -/
theorem zero_target_gradient_flow (features : X → Fin d → ℝ)
    {n : ℕ} (sample : Fin n → X) (j : Fin d) (time : ℝ) :
    HasDerivAt (fun _ : ℝ => (0 : ℝ))
      (-zeroTargetGradient features sample 0 j) time := by
  simpa [zeroTargetGradient_zero] using hasDerivAt_const time (0 : ℝ)

/-- Appendix D: every discrete iterate is zero as well, for every sample,
learning rate, fixed feature map, and number of training epochs. -/
theorem featureGD_zero (features : X → Fin d → ℝ) (rate : ℝ)
    {n : ℕ} (sample : Fin n → X) (epochs : ℕ) :
    featureGD features rate sample epochs = 0 := by
  induction epochs with
  | zero => rfl
  | succ epoch ih =>
    funext j
    simp [featureGD, ih, zeroTargetGradient_zero]

/-- Appendix D: the zero-target feature learner fits every sample exactly,
so the corrected Definition 1 gives infinite EMC for every tolerance. -/
theorem zero_target_randomFeatures_EMC {X : Type*} [MeasurableSpace X]
    {d : ℕ} (D : MeasureTheory.Measure X) (features : X → Fin d → ℝ)
    (rate : ℝ) (epochs : ℕ) (ε : ℝ≥0∞) :
    EMC D (zeroTargetFeatureLoss features) (zeroTargetFeatureTrain features rate epochs) ε = ⊤ := by
  apply EMC_eq_top_of_perfect_training
  intro n sample i
  simp [zeroTargetFeatureTrain, featureGD_zero, zeroTargetFeatureLoss, featurePrediction_zero]

/-- Appendix D: finite feature dimension alone does not imply the asserted
identity `EMC = d`. Zero-target data give a counterexample even with the
specified zero initialization and squared-loss stationary gradient flow. -/
theorem randomFeatures_dimension_is_not_universal_EMC {X : Type*} [MeasurableSpace X]
    {d : ℕ} (D : MeasureTheory.Measure X) (features : X → Fin d → ℝ)
    (rate : ℝ) (epochs : ℕ) (ε : ℝ≥0∞) :
    EMC D (zeroTargetFeatureLoss features) (zeroTargetFeatureTrain features rate epochs) ε ≠
      (d : ℕ∞) := by
  rw [zero_target_randomFeatures_EMC]
  simp

end Transformer.DoubleDescent
