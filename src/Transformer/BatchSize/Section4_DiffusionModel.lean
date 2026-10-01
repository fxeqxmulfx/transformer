/-
# Discrete updates and their diffusion generators

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The discrete law is the pushforward of independent standard Gaussian
innovations. The continuous model is specified by its infinitesimal generator.
-/

import Transformer.BatchSize.Section4_DiagonalNoise

open MeasureTheory ProbabilityTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The two momentum-free methods in Section 4.3. -/
inductive UpdateKind where
  | gradient
  | sign
  deriving DecidableEq

/-- The coordinate Gaussian gradient sampled at a state, Section 4.3:
the innovation has variance one and the minibatch noise has variance sigma^2/B. -/
def sampledGradient {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (z : Fin d → ℝ) : EucSpace d :=
  WithLp.toLp 2 (fun k => gradient f x k + σ x k / Real.sqrt B * z k)

/-- A stochastic SGD or SignSGD step, Section 4.3, equations (2)--(3). -/
def stochasticStep {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) (z : Fin d → ℝ) : EucSpace d :=
  let g := sampledGradient B f σ x z
  x - η • match method with
    | .gradient => g
    | .sign => WithLp.toLp 2 (fun k => Real.sign (g k))

/-- A finite sequence of independent standard Gaussian innovations,
Section 4.3, Theorem 1. Independence is across both steps and coordinates. -/
def innovationLaw (d n : ℕ) : Measure (Fin n → Fin d → ℝ) :=
  Measure.pi (fun _ => diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1))

/-- The innovation law is a probability measure, Section 4.3. -/
instance innovationLaw_probability (d n : ℕ) : IsProbabilityMeasure (innovationLaw d n) := by
  dsimp [innovationLaw]
  infer_instance

/-- The parameters after n successive fresh-noise steps, Section 4.3. -/
def discreteEndpoint {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x₀ : EucSpace d) (n : ℕ) (z : Fin n → Fin d → ℝ) : EucSpace d :=
  (List.ofFn z).foldl (stochasticStep method η B f σ) x₀

/-- The actual n-step distribution, rather than an assumed recursion for
moments; Section 4.3's discrete optimizer and Theorem 1. -/
def discreteLaw {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x₀ : EucSpace d) (n : ℕ) : Measure (EucSpace d) :=
  (innovationLaw d n).map (discreteEndpoint method η B f σ x₀ n)

/-- Drift of the SDEs in Section 4.3, equations (2)--(3). -/
def diffusionDrift {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (x : EucSpace d) : EucSpace d :=
  match method with
  | .gradient => -gradient f x
  | .sign => -WithLp.toLp 2 (fun k => signResponse B (σ x k) (gradient f x k))

/-- Diagonal entries of the SDE covariance, including the eta factor;
Section 4.3, equations (2)--(3). -/
def diffusionCovariance {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) (k : Fin d) : ℝ :=
  match method with
  | .gradient => η * (σ x k) ^ 2 / B
  | .sign => η * (1 - signResponse B (σ x k) (gradient f x k) ^ 2)

/-- The SDE generator b dot grad(phi) + trace(a Hessian(phi))/2,
Section 4.3, equations (2)--(3). This specifies the continuous law without
assuming a conclusion about optimizer expectations. -/
def diffusionGenerator {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (φ : EucSpace d → ℝ) (x : EucSpace d) : ℝ :=
  fderiv ℝ φ x (diffusionDrift method B f σ x) +
    (∑ k, diffusionCovariance method η B f σ x k *
      iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)) / 2

/-- The signed diffusion is nonnegative, Section 4.3, equation (3). -/
theorem diffusionCovariance_nonneg {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) (k : Fin d) (hη : 0 ≤ η) :
    0 ≤ diffusionCovariance method η B f σ x k := by
  cases method
  · dsimp [diffusionCovariance]
    positivity
  · exact mul_nonneg hη (sign_covariance_pos B (σ x k) (gradient f x k)).le

/-- Nonvacuity of the learning-rate condition, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 1000 := by norm_num

/-- SGD's drift is independent of batch size and noise, Section 4.3,
equation (2). Unlike the signed method, its mean direction is the full gradient. -/
theorem sgd_drift_batch_independent {d : ℕ} (B C : ℕ)
    (f : EucSpace d → ℝ) (σ τ : EucSpace d → EucSpace d) (x : EucSpace d) :
    diffusionDrift .gradient B f σ x = diffusionDrift .gradient C f τ x := rfl

end Transformer.BatchSize
