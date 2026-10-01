import Transformer.DoubleDescent.Section2_EffectiveComplexity
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Basic

/-!
# The source's complex Fourier feature model

arXiv:1912.02292v1, Appendix D. The source uses activation `exp(-i*x)`,
fixed Gaussian first-layer weights, zero output-layer initialization,
and squared-loss gradient flow. The counterexample below uses precisely
complex exponential features. It holds for every realization of the first
layer, hence in particular for any Gaussian draw. The target is identically
zero; this refutes the distribution-independent identity `EMC=d`, not a
claim about the particular Fashion MNIST labels in Figures 14--15.
-/

namespace Transformer.DoubleDescent

open scoped ENNReal

variable {X : Type*} {d : ℕ}

/-- Appendix D: the complex exponential activation at each fixed frequency. -/
noncomputable def fourierFeatures {inputDim d : ℕ}
    (frequency : Fin d → Fin inputDim → ℝ) (x : Fin inputDim → ℝ) (j : Fin d) : ℂ :=
  Complex.exp (-Complex.I * ((∑ k, frequency j k * x k : ℝ) : ℂ))

/-- Appendix D: the complex linear output layer on fixed features. -/
noncomputable def complexFeaturePrediction (features : X → Fin d → ℂ)
    (weight : Fin d → ℂ) (x : X) : ℂ := ∑ j, weight j * features x j

/-- Appendix D, counterexample: actual squared modulus error for target zero. -/
noncomputable def complexZeroTargetLoss (features : X → Fin d → ℂ)
    (weight : Fin d → ℂ) (x : X) : ℝ≥0∞ :=
  ENNReal.ofReal (Complex.normSq (complexFeaturePrediction features weight x))

/-- Appendix D: the real Euclidean squared-loss gradient, represented as
a complex vector; the conjugate of the feature is essential. -/
noncomputable def complexZeroTargetGradient (features : X → Fin d → ℂ) {n : ℕ}
    (sample : Fin n → X) (weight : Fin d → ℂ) (j : Fin d) : ℂ :=
  (2 / (n : ℂ)) * ∑ i,
    complexFeaturePrediction features weight (sample i) * star (features (sample i) j)

/-- Appendix D: discrete gradient descent from the specified zero initialization. -/
noncomputable def complexFeatureGD (features : X → Fin d → ℂ) (rate : ℝ)
    {n : ℕ} (sample : Fin n → X) : ℕ → (Fin d → ℂ)
  | 0 => 0
  | epoch + 1 => fun j => complexFeatureGD features rate sample epoch j -
      (rate : ℂ) * complexZeroTargetGradient features sample
        (complexFeatureGD features rate sample epoch) j

/-- Appendix D: the actual finite-time complex feature training procedure. -/
noncomputable def complexFeatureTrain (features : X → Fin d → ℂ)
    (rate : ℝ) (epochs : ℕ) : TrainingProcedure X (Fin d → ℂ) :=
  fun _ sample => complexFeatureGD features rate sample epochs

/-- Appendix D: zero weights give zero prediction for complex features too. -/
theorem complexFeaturePrediction_zero (features : X → Fin d → ℂ) (x : X) :
    complexFeaturePrediction features 0 x = 0 := by
  simp [complexFeaturePrediction]

/-- Appendix D: the complex squared-loss gradient vanishes at zero weights. -/
theorem complexZeroTargetGradient_zero (features : X → Fin d → ℂ)
    {n : ℕ} (sample : Fin n → X) (j : Fin d) :
    complexZeroTargetGradient features sample 0 j = 0 := by
  simp [complexZeroTargetGradient, complexFeaturePrediction_zero]

/-- Appendix D: the zero trajectory solves the source's continuous-time
gradient-flow equation for complex output weights, at every real time. -/
theorem complex_zero_target_gradient_flow (features : X → Fin d → ℂ)
    {n : ℕ} (sample : Fin n → X) (j : Fin d) (time : ℝ) :
    HasDerivAt (fun _ : ℝ => (0 : ℂ))
      (-complexZeroTargetGradient features sample 0 j) time := by
  simpa [complexZeroTargetGradient_zero] using hasDerivAt_const time (0 : ℂ)

/-- Appendix D: zero-initialized discrete iterates also stay exactly at zero. -/
theorem complexFeatureGD_zero (features : X → Fin d → ℂ) (rate : ℝ)
    {n : ℕ} (sample : Fin n → X) (epochs : ℕ) :
    complexFeatureGD features rate sample epochs = 0 := by
  induction epochs with
  | zero => rfl
  | succ epoch ih =>
    funext j
    simp [complexFeatureGD, ih, complexZeroTargetGradient_zero]

/-- Appendix D: for actual `exp(-i*x)` features and the zero-target
distribution, every sample size is fitted exactly and EMC is infinite.
No property of the fixed frequencies, or their sampling law, is assumed. -/
theorem fourier_zero_target_EMC {inputDim d : ℕ}
    (D : MeasureTheory.Measure (Fin inputDim → ℝ))
    (frequency : Fin d → Fin inputDim → ℝ) (rate : ℝ) (epochs : ℕ) (ε : ℝ≥0∞) :
    EMC D (complexZeroTargetLoss (fourierFeatures frequency))
      (complexFeatureTrain (fourierFeatures frequency) rate epochs) ε = ⊤ := by
  apply EMC_eq_top_of_perfect_training
  intro n sample i
  simp [complexFeatureTrain, complexFeatureGD_zero, complexZeroTargetLoss,
    complexFeaturePrediction_zero]

/-- Appendix D: the asserted feature-count identity cannot be universal
even for the source's complex exponential activation and zero initialization. -/
theorem fourier_dimension_is_not_universal_EMC {inputDim d : ℕ}
    (D : MeasureTheory.Measure (Fin inputDim → ℝ))
    (frequency : Fin d → Fin inputDim → ℝ) (rate : ℝ) (epochs : ℕ) (ε : ℝ≥0∞) :
    EMC D (complexZeroTargetLoss (fourierFeatures frequency))
      (complexFeatureTrain (fourierFeatures frequency) rate epochs) ε ≠ (d : ℕ∞) := by
  rw [fourier_zero_target_EMC]
  simp

end Transformer.DoubleDescent
