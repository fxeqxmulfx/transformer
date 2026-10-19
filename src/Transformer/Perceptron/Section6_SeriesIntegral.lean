module

public import Mathlib.Analysis.Analytic.ConvergenceRadius
public import Mathlib.Analysis.Normed.Algebra.Exponential
public import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
public import Mathlib.MeasureTheory.Integral.DominatedConvergence

/-!
# Integrating the actual multilinear Taylor series

ArXiv:2601.21366v2, §5.3, the Weierstrass M-test for the exponential
first variation, and §6, the analytic interaction in `prop: unified.log`.
Uniform summable coefficient bounds allow a probability integral to
retain the actual Taylor series of its pointwise integrands.

The coefficients below are actual integrals of continuous multilinear
maps. Their radius is bounded using their operator norms. Evaluation
commutes with integration, and summable integral norms justify passing
the series through the original measure. The resulting analyticity is
proved from the series equality, rather than assumed for the integral.
-/

@[expose] public section

open MeasureTheory Set
open scoped NNReal ENNReal BigOperators

namespace Transformer.Perceptron

variable {α E F : Type*} [MeasurableSpace α]
  [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- The actual integrated multilinear coefficients.
Source: arXiv:2601.21366v2, §5.3, integrating the exponential expansion. -/
noncomputable def integralFormalMultilinearSeries (μ : Measure α)
    (p : α → FormalMultilinearSeries ℝ E F) : FormalMultilinearSeries ℝ E F :=
  fun n => ∫ y, p y n ∂μ

/-- Uniform coefficient bounds survive the actual probability integral.
Source: arXiv:2601.21366v2, §5.3, the coefficient bounds in the M-test. -/
theorem norm_integralFormalMultilinearSeries_le (μ : Measure α) [IsProbabilityMeasure μ]
    (p : α → FormalMultilinearSeries ℝ E F) (b : ℕ → ℝ)
    (hb : ∀ n y, ‖p y n‖ ≤ b n) (n : ℕ) :
    ‖integralFormalMultilinearSeries μ p n‖ ≤ b n := by
  simpa [integralFormalMultilinearSeries] using
    norm_integral_le_of_norm_le_const (ae_of_all μ (hb n))

/-- Evaluating an actual integrated coefficient commutes with integration.
Source: arXiv:2601.21366v2, §5.3, the multilinear coefficients in
the exponential first variation. All coefficient arguments are retained. -/
theorem integralFormalMultilinearSeries_apply (μ : Measure α)
    (p : α → FormalMultilinearSeries ℝ E F) (n : ℕ)
    (hI : Integrable (fun y => p y n) μ) (m : Fin n → E) :
    integralFormalMultilinearSeries μ p n m = ∫ y, p y n m ∂μ :=
  ContinuousMultilinearMap.integral_apply hI m

/-- The norm series of the actual integrated coefficients is summable
at the supplied radius. Source: arXiv:2601.21366v2, §5.3, the M-test
bound controlling the integrated series' radius of convergence. -/
theorem summable_norm_integralFormalMultilinearSeries (μ : Measure α) [IsProbabilityMeasure μ]
    (p : α → FormalMultilinearSeries ℝ E F) (b : ℕ → ℝ)
    (hb : ∀ n y, ‖p y n‖ ≤ b n) (r : ℝ≥0)
    (hsum : Summable (fun n => b n * (r : ℝ) ^ n)) :
    Summable (fun n => ‖integralFormalMultilinearSeries μ p n‖ * (r : ℝ) ^ n) :=
  Summable.of_nonneg_of_le (fun n => by positivity)
    (fun n => mul_le_mul_of_nonneg_right
      (norm_integralFormalMultilinearSeries_le μ p b hb n) (by positivity)) hsum

omit [MeasurableSpace α] in
/-- A uniform operator-norm bound controls every actual diagonal
evaluation on the ball. Source: arXiv:2601.21366v2, §5.3,
the uniform bound on each power in the exponential expansion. -/
theorem norm_series_diagonal_le (p : α → FormalMultilinearSeries ℝ E F)
    (b : ℕ → ℝ) (hb : ∀ n y, ‖p y n‖ ≤ b n) (r : ℝ≥0)
    (h : E) (hh : ‖h‖ ≤ (r : ℝ)) (n : ℕ) (y : α) :
    ‖p y n (fun _ => h)‖ ≤ b n * (r : ℝ) ^ n := by
  calc
    ‖p y n (fun _ => h)‖ ≤ ‖p y n‖ * ‖h‖ ^ n := by
      simpa using (p y n).le_opNorm (fun _ => h)
    _ ≤ ‖p y n‖ * (r : ℝ) ^ n :=
      mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg h) hh n) (norm_nonneg _)
    _ ≤ b n * (r : ℝ) ^ n := mul_le_mul_of_nonneg_right (hb n y) (by positivity)

/-- The integral retains the actual pointwise Taylor series on the
same ball. Source: arXiv:2601.21366v2, §5.3, the M-test for the
first variation, and §6, analyticity of the weight in `H_log`.
The hypotheses supply actual coefficients and their actual series equality;
the series equality for the original integral is proved here. -/
theorem hasFPowerSeriesOnBall_integral_of_uniform_bound
    (μ : Measure α) [IsProbabilityMeasure μ] (f : E → α → F)
    (p : α → FormalMultilinearSeries ℝ E F) (x : E) (r : ℝ≥0) (hr : 0 < r)
    (hI : ∀ n, Integrable (fun y => p y n) μ)
    (b : ℕ → ℝ) (hb : ∀ n y, ‖p y n‖ ≤ b n)
    (hsum : Summable (fun n => b n * (r : ℝ) ^ n))
    (hpoint : ∀ y, HasFPowerSeriesOnBall (fun z => f z y) (p y) x (r : ℝ≥0∞)) :
    HasFPowerSeriesOnBall (fun z => ∫ y, f z y ∂μ)
      (integralFormalMultilinearSeries μ p) x (r : ℝ≥0∞) := by
  refine ⟨(integralFormalMultilinearSeries μ p).le_radius_of_summable
    (summable_norm_integralFormalMultilinearSeries μ p b hb r hsum), ?_, ?_⟩
  · exact_mod_cast hr
  · intro h hball
    have hh : ‖h‖ < (r : ℝ) := by
      have hne := mem_eball_zero_iff.mp hball
      change (‖h‖₊ : ℝ≥0∞) < (r : ℝ≥0∞) at hne
      exact_mod_cast ENNReal.coe_lt_coe.mp hne
    have hdiagI (n : ℕ) : Integrable (fun y => p y n (fun _ => h)) μ :=
      (ContinuousMultilinearMap.apply ℝ (fun _ : Fin n => E) F (fun _ => h)).integrable_comp (hI n)
    have hdiagSum : Summable (fun n => ∫ y, ‖p y n (fun _ => h)‖ ∂μ) := by
      apply Summable.of_nonneg_of_le (fun n => integral_nonneg (fun y => norm_nonneg _)) ?_ hsum
      intro n
      have hi : 0 ≤ ∫ y, ‖p y n (fun _ => h)‖ ∂μ := integral_nonneg (fun y => norm_nonneg _)
      have hle := norm_integral_le_of_norm_le_const
        (f := fun y => ‖p y n (fun _ => h)‖) (ae_of_all μ (fun y =>
        by simpa only [norm_norm] using norm_series_diagonal_le p b hb r h hh.le n y))
      simpa [Real.norm_of_nonneg hi] using hle
    have heval : (fun n => integralFormalMultilinearSeries μ p n (fun _ => h)) =
        fun n => ∫ y, p y n (fun _ => h) ∂μ := by
      funext n
      exact integralFormalMultilinearSeries_apply μ p n (hI n) (fun _ => h)
    have heq : (fun y => ∑' n, p y n (fun _ => h)) = fun y => f (x + h) y :=
      funext (fun y => ((hpoint y).hasSum hball).tsum_eq)
    rw [heval, ← heq]
    exact hasSum_integral_of_summable_integral_norm hdiagI hdiagSum

/-- Actual uniformly bounded pointwise Taylor series give analyticity
of the original integral. Source: arXiv:2601.21366v2, §5.3 and §6,
the analytic first variation. This uses the proved integrated series. -/
theorem analyticAt_integral_of_uniform_series_bound
    (μ : Measure α) [IsProbabilityMeasure μ] (f : E → α → F)
    (p : α → FormalMultilinearSeries ℝ E F) (x : E) (r : ℝ≥0) (hr : 0 < r)
    (hI : ∀ n, Integrable (fun y => p y n) μ)
    (b : ℕ → ℝ) (hb : ∀ n y, ‖p y n‖ ≤ b n)
    (hsum : Summable (fun n => b n * (r : ℝ) ^ n))
    (hpoint : ∀ y, HasFPowerSeriesOnBall (fun z => f z y) (p y) x (r : ℝ≥0∞)) :
    AnalyticAt ℝ (fun z => ∫ y, f z y ∂μ) x :=
  (hasFPowerSeriesOnBall_integral_of_uniform_bound μ f p x r hr hI b hb hsum hpoint).hasFPowerSeriesAt.analyticAt

/-- Every probability, actual-series, integrability, bound, summability
and positive-radius input holds jointly for the actual exponential,
its true multilinear coefficients and the Dirac measure on `Unit`.
Source: arXiv:2601.21366v2, §5.3, the exponential M-test.
This also realizes the diagonal bound for the actual vector `h=0`. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧ (0 : ℝ≥0) < 1 ∧
    (∀ n, Integrable (fun _ : Unit => NormedSpace.expSeries ℝ ℝ n) (Measure.dirac ())) ∧
    (∀ n, ∀ y : Unit, ‖(fun _ : Unit => NormedSpace.expSeries ℝ ℝ) y n‖ ≤
      ‖NormedSpace.expSeries ℝ ℝ n‖) ∧
    Summable (fun n => ‖NormedSpace.expSeries ℝ ℝ n‖ * (1 : ℝ) ^ n) ∧
    (∀ y : Unit, HasFPowerSeriesOnBall (fun z : ℝ => (fun t (_ : Unit) => NormedSpace.exp t) z y)
      (NormedSpace.expSeries ℝ ℝ) 0 1) ∧ ‖(0 : ℝ)‖ ≤ (1 : ℝ) := by
  refine ⟨inferInstance, one_pos, fun n => integrable_const _, fun _ _ => le_rfl, ?_, ?_, by norm_num⟩
  · exact (NormedSpace.expSeries ℝ ℝ).summable_norm_mul_pow (r := 1)
      (by rw [NormedSpace.expSeries_radius_eq_top]; simp)
  · intro _
    exact (NormedSpace.exp_hasFPowerSeriesOnBall (𝕂 := ℝ) (𝔸 := ℝ)).mono (by norm_num) le_top

end Transformer.Perceptron
