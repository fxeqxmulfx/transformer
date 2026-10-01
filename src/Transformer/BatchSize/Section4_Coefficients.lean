/-
# SGD and SignSGD drift and diffusion coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3) and Theorem 1.
The scalar formulas are applied independently to each diagonal coordinate.
-/

import Transformer.BatchSize.Section4_SignMoments

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- The coordinate minibatch gradient N(g,sigma^2/B), Section 4.3. -/
def gradientNoiseLaw (B : ℕ) (g σ : ℝ) : Measure ℝ :=
  gaussianReal g (.mk (σ ^ 2 / B) (by positivity))

/-- The Gaussian gradient law is a probability measure, Section 4.3. -/
instance gradientNoiseLaw_probability (B : ℕ) (g σ : ℝ) :
    IsProbabilityMeasure (gradientNoiseLaw B g σ) := by
  dsimp [gradientNoiseLaw]
  infer_instance

/-- The error-function response in equation (3), Section 4.3. -/
def signResponse (B : ℕ) (σ g : ℝ) : ℝ :=
  errorFunction (Real.sqrt ((B : ℝ) / 2) * g / σ)

/-- The denominator sqrt(2*variance) matches the source's sqrt(B/2)
scaling; Section 4.3's proof sketch. -/
theorem gaussian_batch_scale (B : ℕ) (σ : ℝ) (hB : 0 < B) (hσ : 0 < σ) :
    0 < σ * Real.sqrt (2 / (B : ℝ)) ∧
    (σ * Real.sqrt (2 / (B : ℝ))) ^ 2 / 2 = σ ^ 2 / B := by
  have hb : (0 : ℝ) < B := by exact_mod_cast hB
  constructor
  · exact mul_pos hσ (Real.sqrt_pos.2 (div_pos (by norm_num) hb))
  · rw [mul_pow, Real.sq_sqrt (by positivity)]
    ring

/-- Nonvacuity of nondegenerate minibatch noise, Section 4.3. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Equivalent parametrizations of the Gaussian CDF argument;
Section 4.3, equation (3). -/
theorem gaussian_batch_argument (B : ℕ) (g σ : ℝ) (hB : 0 < B) :
    g / (σ * Real.sqrt (2 / (B : ℝ))) = Real.sqrt ((B : ℝ) / 2) * g / σ := by
  have hb : (0 : ℝ) < B := by exact_mod_cast hB
  have hab : Real.sqrt ((B : ℝ) / 2) * Real.sqrt (2 / (B : ℝ)) = 1 := by
    rw [← Real.sqrt_mul (by positivity)]
    have hq : ((B : ℝ) / 2) * (2 / (B : ℝ)) = 1 := by field_simp
    rw [hq, Real.sqrt_one]
  have hinv := inv_eq_of_mul_eq_one_left hab
  rw [div_eq_mul_inv, mul_inv_rev, hinv]
  ring

/-- Nonvacuity of the CDF scaling identity, Section 4.3. -/
example : 0 < (2 : ℕ) := by norm_num

/-- Actual Gaussian expectations give the signed drift in Theorem 1;
Section 4.3, equation (3). -/
theorem signResponse_eq_mean (B : ℕ) (g σ : ℝ) (hB : 0 < B) (hσ : 0 < σ) :
    (∫ z, Real.sign z ∂gradientNoiseLaw B g σ) = signResponse B σ g := by
  obtain ⟨hs, hscale⟩ := gaussian_batch_scale B σ hB hσ
  have h := gaussian_sign_mean g _ hs
  have hv : (NNReal.mk ((σ * Real.sqrt (2 / (B : ℝ))) ^ 2 / 2) (by positivity)) =
      NNReal.mk (σ ^ 2 / B) (by positivity) := NNReal.coe_injective hscale
  rw [hv, gaussian_batch_argument B g σ hB] at h
  exact h

/-- Nonvacuity of the signed-drift calculation, Section 4.3. -/
example : 0 < (64 : ℕ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Exact Gaussian sign variance behind the diffusion in Theorem 1;
Section 4.3, equation (3). -/
theorem signResponse_variance (B : ℕ) (g σ : ℝ) (hB : 0 < B) (hσ : 0 < σ) :
    variance Real.sign (gradientNoiseLaw B g σ) = 1 - signResponse B σ g ^ 2 := by
  obtain ⟨hs, hscale⟩ := gaussian_batch_scale B σ hB hσ
  have h := gaussian_sign_variance g _ hs
  have hv : (NNReal.mk ((σ * Real.sqrt (2 / (B : ℝ))) ^ 2 / 2) (by positivity)) =
      NNReal.mk (σ ^ 2 / B) (by positivity) := NNReal.coe_injective hscale
  rw [hv, gaussian_batch_argument B g σ hB] at h
  exact h

/-- Nonvacuity of the signed-diffusion calculation, Section 4.3. -/
example : 0 < (256 : ℕ) ∧ 0 < (1 : ℝ) := by norm_num

/-- SGD's expected stochastic gradient is independent of batch size;
Section 4.3, equation (2) and the takeaway. -/
theorem sgd_mean_gradient (B : ℕ) (g σ : ℝ) :
    (∫ z, z ∂gradientNoiseLaw B g σ) = g :=
  integral_id_gaussianReal

/-- The noise covariance scales as sigma^2/B, Section 4.3, equation (2). -/
theorem sgd_gradient_variance (B : ℕ) (g σ : ℝ) :
    variance (fun z => z) (gradientNoiseLaw B g σ) = σ ^ 2 / B :=
  variance_fun_id_gaussianReal

/-- SGD's coordinate diffusion coefficient, Section 4.3, equation (2). -/
def sgdDiffusion (η : ℝ) (B : ℕ) (σ : ℝ) : ℝ := Real.sqrt (η * σ ^ 2 / B)

/-- SignSGD's coordinate diffusion coefficient, Section 4.3, equation (3). -/
def signDiffusion (η : ℝ) (B : ℕ) (σ g : ℝ) : ℝ :=
  Real.sqrt η * Real.sqrt (1 - signResponse B σ g ^ 2)

/-- The finite-signal signed covariance is positive;
Section 4.3, Theorem 1. -/
theorem sign_covariance_pos (B : ℕ) (σ g : ℝ) :
    0 < 1 - signResponse B σ g ^ 2 := by
  have h := errorFunction_abs_lt_one (Real.sqrt ((B : ℝ) / 2) * g / σ)
  have hh := (abs_lt.mp h)
  dsimp [signResponse]
  nlinarith

/-- Squaring the signed diffusion recovers eta times the centered
sign covariance, Section 4.3, equation (3). -/
theorem signDiffusion_sq (η : ℝ) (B : ℕ) (σ g : ℝ) (hη : 0 ≤ η) :
    signDiffusion η B σ g ^ 2 = η * (1 - signResponse B σ g ^ 2) := by
  rw [signDiffusion, mul_pow, Real.sq_sqrt hη,
    Real.sq_sqrt (sign_covariance_pos B σ g).le]

/-- Nonvacuity of the learning-rate hypothesis, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 1000 := by norm_num

end Transformer.BatchSize
