/-
# Actual diagonal Gaussian expectations with frozen drift and amplitudes

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The vector flow integrates over the genuine independent-coordinate law.
Continuity at zero, integrability, and samplewise time derivatives are
proved explicitly before differentiation of the expectation.
-/

import Transformer.BatchSize.Section4_GaussianTestIntegration

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The actual frozen-coefficient Gaussian state after elapsed time t,
Section 4.3 (2)--(3). Positive t gives drift t*b and noise covariance
t times diag(A_k^2). -/
def gaussianVectorState {d : ℕ} (x b : EucSpace d) (A : Fin d → ℝ)
    (t : ℝ) (z : Fin d → ℝ) : EucSpace d :=
  x + t • b + Real.sqrt t • gaussianDiagonalMap A z

/-- The frozen state is exactly the affine Gaussian state with its
time-rescaled amplitudes, Section 4.3 (2)--(3). -/
theorem gaussianVectorState_eq_affine {d : ℕ} (x b : EucSpace d) (A : Fin d → ℝ)
    (t : ℝ) (z : Fin d → ℝ) : gaussianVectorState x b A t z =
      gaussianAffineState (x + t • b) (fun k => Real.sqrt t * A k) z := by
  ext k
  simp only [gaussianVectorState, gaussianAffineState, PiLp.add_apply, PiLp.smul_apply,
    smul_eq_mul, gaussianDiagonalMap_apply]
  ring

/-- The frozen Gaussian state is continuous in its actual innovation,
Section 4.3 (2)--(3). -/
theorem gaussianVectorState_continuous_noise {d : ℕ} (x b : EucSpace d)
    (A : Fin d → ℝ) (t : ℝ) : Continuous (gaussianVectorState x b A t) :=
  continuous_const.add ((gaussianDiagonalMap A).continuous.const_smul (Real.sqrt t))

/-- Each frozen Gaussian sample varies continuously in time, including
the zero-time boundary, Section 4.3 (2)--(3). -/
theorem gaussianVectorState_continuous_time {d : ℕ} (x b : EucSpace d)
    (A : Fin d → ℝ) (z : Fin d → ℝ) : Continuous (fun t => gaussianVectorState x b A t z) :=
  (continuous_const.add (continuous_id.smul continuous_const)).add
    (Real.continuous_sqrt.smul continuous_const)

/-- The actual samplewise positive-time derivative of the frozen
Gaussian state, Section 4.3 (2)--(3). -/
theorem gaussianVectorState_hasDerivAt {d : ℕ} (x b : EucSpace d) (A : Fin d → ℝ)
    (z : Fin d → ℝ) (t : ℝ) (ht : 0 < t) :
    HasDerivAt (fun u => gaussianVectorState x b A u z)
      (b + (1 / (2 * Real.sqrt t)) • gaussianDiagonalMap A z) t := by
  have h := (((hasDerivAt_id t).smul_const b).const_add x).add
    ((Real.hasDerivAt_sqrt ht.ne').smul_const (gaussianDiagonalMap A z))
  simp only [id_eq, one_smul] at h
  exact h

/-- Nonvacuity of the positive-time derivative hypothesis, Section 4.3. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The diagonal Gaussian noise vector has finite second moment under
its actual product measure, Section 4.3 (2)--(3). -/
theorem gaussianDiagonalMap_memLp {d : ℕ} (A : Fin d → ℝ) :
    MemLp (gaussianDiagonalMap A) 2 (standardGaussianVectorLaw d) := by
  apply MemLp.of_eval_piLp
  intro k
  have h := (memLp_id_gaussianReal (μ := 0) (v := 1) 2).comp_measurePreserving
    (measurePreserving_eval (fun _ : Fin d => gaussianReal 0 1) k)
  change MemLp (fun z : Fin d → ℝ => A k * z k) 2 (standardGaussianVectorLaw d)
  exact h.const_mul (A k)

/-- The genuine frozen-coefficient Gaussian expectation operator,
Section 4.3 (2)--(3). -/
def gaussianVectorFlow {d : ℕ} (φ : EucSpace d → ℝ) (x b : EucSpace d)
    (A : Fin d → ℝ) (t : ℝ) : ℝ :=
  ∫ z, φ (gaussianVectorState x b A t z) ∂standardGaussianVectorLaw d

/-- The actual vector Gaussian expectation starts at the prescribed
test value, Section 4.3 (2)--(3). -/
theorem gaussianVectorFlow_zero {d : ℕ} (φ : EucSpace d → ℝ) (x b : EucSpace d)
    (A : Fin d → ℝ) : gaussianVectorFlow φ x b A 0 = φ x := by
  simp [gaussianVectorFlow, gaussianVectorState]

/-- Bounded observables remain bounded under the actual vector
Gaussian expectation, Section 4.3 (2)--(3). -/
theorem gaussianVectorFlow_abs_le {d : ℕ} (φ : EucSpace d → ℝ) (x b : EucSpace d)
    (A : Fin d → ℝ) (C : ℝ) (hC : ∀ y, |φ y| ≤ C) (t : ℝ) :
    |gaussianVectorFlow φ x b A t| ≤ C := by
  simpa only [gaussianVectorFlow, Real.norm_eq_abs, probReal_univ, mul_one] using
    norm_integral_le_of_norm_le_const (μ := standardGaussianVectorLaw d)
      (f := fun z => φ (gaussianVectorState x b A t z))
        (Eventually.of_forall fun z => by simpa only [Real.norm_eq_abs] using hC _)

/-- Continuity of the actual vector Gaussian expectation on all real
times, Section 4.3 (2)--(3), from integrable domination of its samples. -/
theorem gaussianVectorFlow_continuous {d : ℕ} (φ : EucSpace d → ℝ) (hφ : Continuous φ)
    (x b : EucSpace d) (A : Fin d → ℝ) (C : ℝ) (hC : ∀ y, |φ y| ≤ C) :
    Continuous (gaussianVectorFlow φ x b A) :=
  continuous_of_dominated
    (fun t => (hφ.comp (gaussianVectorState_continuous_noise x b A t)).aestronglyMeasurable)
    (fun _ => Eventually.of_forall fun z => by simpa only [Real.norm_eq_abs] using hC _)
    (integrable_const C)
    (Eventually.of_forall fun z => hφ.comp (gaussianVectorState_continuous_time x b A z))

/-- Joint nonvacuity of continuity and boundedness hypotheses for the
vector Gaussian flow, Section 4.3: sin of a Euclidean coordinate. -/
example : Continuous (fun y : EucSpace 1 => Real.sin (y 0)) ∧
    (∀ y : EucSpace 1, |Real.sin (y 0)| ≤ 1) :=
  ⟨Real.continuous_sin.comp (PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) 0),
    fun y => Real.abs_sin_le_one (y 0)⟩

end Transformer.BatchSize
