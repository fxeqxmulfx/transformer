/-
# Continuity and the actual one-sided Gaussian interpolation endpoints

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The corrected bounded-test identity is
carried on the full original interval `[0,1]`. Both the original test
expectation and the original covariance–Hessian expectation are
continuous, by actual dominated convergence and the original bounds.

The actual square-root blend is continuous even at the two original
endpoints. The covariance contraction is the genuine sum of original
matrix entries times the original coordinate Hessian entries.
An interior derivative equal to a continuous original contraction
therefore extends to each actual one-sided endpoint derivative.
This does not restrict the interpolation to the open interval or
require a nonsingular original Gaussian covariance matrix.
-/

import Transformer.Homogenized.Section7_GaussianBlendGradient
import Mathlib.Analysis.Calculus.FDeriv.Extend
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open Real MeasureTheory ProbabilityTheory Set Filter
open scoped BigOperators Topology

namespace Transformer.Homogenized

/-- The original covariance contraction with the original coordinate
Hessian. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`, `Tr(Sᵀ ∇²φ)` in its displayed identity. -/
noncomputable def gaussianCovarianceTrace {m : ℕ} (φ : EucSpace m → ℝ)
    (S : Matrix (Fin m) (Fin m) ℝ) (z : EucSpace m) : ℝ :=
  ∑ i : Fin m, ∑ j : Fin m, S i j * iteratedFDeriv ℝ 2 φ z
    ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]

/-- The original Hessian contraction is continuous for the original
`C²` test. Source: arXiv:2604.01978v1, §7,
the original derivative expectation in `lem:gaussian_interp`. -/
theorem continuous_gaussianCovarianceTrace {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (S : Matrix (Fin m) (Fin m) ℝ) :
    Continuous (gaussianCovarianceTrace φ S) := by
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := hφ.fderiv_right (by norm_num)
  have hD2 : Continuous (fderiv ℝ (fderiv ℝ φ)) := hD.continuous_fderiv (by decide)
  change Continuous (fun z : EucSpace m => ∑ i : Fin m, ∑ j : Fin m, S i j *
    iteratedFDeriv ℝ 2 φ z ![EuclideanSpace.single i 1, EuclideanSpace.single j 1])
  simp only [iteratedFDeriv_two_apply, Matrix.cons_val_zero, Matrix.cons_val_one]
  fun_prop

/-- The original coordinate bounds give a genuine uniform bound on
the original covariance trace. Source: arXiv:2604.01978v1, §7,
dominated convergence for the corrected `lem:gaussian_interp`. -/
theorem norm_gaussianCovarianceTrace_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (S : Matrix (Fin m) (Fin m) ℝ) (z : EucSpace m) :
    ‖gaussianCovarianceTrace φ S z‖ ≤ (∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C := by
  unfold gaussianCovarianceTrace
  calc
    _ ≤ ∑ i : Fin m, ‖∑ j : Fin m, S i j * iteratedFDeriv ℝ 2 φ z
        ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]‖ := norm_sum_le _ _
    _ ≤ ∑ i : Fin m, ∑ j : Fin m, ‖S i j‖ * C := by
      apply Finset.sum_le_sum
      intro i hi
      apply (norm_sum_le _ _).trans
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_mul]
      exact mul_le_mul_of_nonneg_left (norm_gaussian_test_hessian_entry_le φ C hbdd i j z)
        (norm_nonneg (S i j))
    _ = _ := by simp only [Finset.sum_mul]

/-- The original bounded test expectation is continuous along the
actual original square-root blend. Source: arXiv:2604.01978v1,
§7, dominated convergence in `lem:gaussian_interp`.
Actual measurability of both original random vectors is retained. -/
theorem continuous_expectation_gaussian_blend {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace m)
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (g : EucSpace m → ℝ) (hg : Continuous g) (C : ℝ) (hbound : ∀ z, ‖g z‖ ≤ C) :
    Continuous (fun r : ℝ => ∫ ω, g (Real.sqrt r • X ω + Real.sqrt (1 - r) • Y ω) ∂P) := by
  have hmeas (r : ℝ) : AEStronglyMeasurable
      (fun ω => g (Real.sqrt r • X ω + Real.sqrt (1 - r) • Y ω)) P := by
    have hc := hg.comp (gaussianBlendLinearMap (m := m) (Real.sqrt r) (Real.sqrt (1 - r))).continuous
    simpa only [Function.comp_def, gaussianBlendLinearMap_apply] using
      hc.aestronglyMeasurable.comp_aemeasurable (hX.prodMk hY)
  apply continuous_iff_continuousAt.mpr
  intro t
  apply tendsto_integral_filter_of_norm_le_const (Eventually.of_forall hmeas)
    ⟨C, Eventually.of_forall fun r => ae_of_all _ fun ω => hbound _⟩
  apply ae_of_all
  intro ω
  have hc : Continuous (fun r : ℝ => g (Real.sqrt r • X ω + Real.sqrt (1 - r) • Y ω)) := by fun_prop
  exact hc.continuousAt

/-- The original covariance–Hessian expectation is continuous at
both original endpoints and on the original interval.
Source: arXiv:2604.01978v1, §7, the original trace expectation
in the corrected bounded-test identity of `lem:gaussian_interp`. -/
theorem continuous_expectation_gaussianCovarianceTrace {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace m)
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (S : Matrix (Fin m) (Fin m) ℝ) :
    Continuous (fun r : ℝ => ∫ ω,
      gaussianCovarianceTrace φ S (Real.sqrt r • X ω + Real.sqrt (1 - r) • Y ω) ∂P) :=
  continuous_expectation_gaussian_blend P X Y hX hY (gaussianCovarianceTrace φ S)
    (continuous_gaussianCovarianceTrace φ hφ S) ((∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C)
    (norm_gaussianCovarianceTrace_le φ C hbdd S)

/-- An actual continuous interior derivative extends to both actual
one-sided endpoints of the original interpolation interval.
Source: arXiv:2604.01978v1, §7, the corrected endpoint reading
of `lem:gaussian_interp`. The original derivative is retained at both ends. -/
theorem interpolation_hasDerivWithinAt_endpoints (f g : ℝ → ℝ)
    (hf : Continuous f) (hg : Continuous g)
    (hderiv : ∀ r ∈ Ioo (0 : ℝ) 1, HasDerivAt f (g r) r)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) : HasDerivWithinAt f (g t) (Icc 0 1) t := by
  have hdiff : DifferentiableOn ℝ f (Ioo (0 : ℝ) 1) :=
    fun r hr => (hderiv r hr).differentiableAt.differentiableWithinAt
  by_cases h0 : t = 0
  · subst t
    have he : ∀ᶠ r in 𝓝[>] (0 : ℝ), deriv f r = g r :=
      Filter.Eventually.mono (Ioo_mem_nhdsGT (by norm_num : (0 : ℝ) < 1))
        (fun r hr => (hderiv r hr).deriv)
    have hlim : Tendsto (deriv f) (𝓝[>] (0 : ℝ)) (𝓝 (g 0)) :=
      Filter.Tendsto.congr' (Filter.EventuallyEq.symm he)
        (hg.continuousAt.tendsto.mono_left nhdsWithin_le_nhds)
    exact (hasDerivWithinAt_Ici_of_tendsto_deriv hdiff hf.continuousAt.continuousWithinAt
      (Ioo_mem_nhdsGT (by norm_num)) hlim).mono Icc_subset_Ici_self
  by_cases h1 : t = 1
  · subst t
    have he : ∀ᶠ r in 𝓝[<] (1 : ℝ), deriv f r = g r :=
      Filter.Eventually.mono (Ioo_mem_nhdsLT (by norm_num : (0 : ℝ) < 1))
        (fun r hr => (hderiv r hr).deriv)
    have hlim : Tendsto (deriv f) (𝓝[<] (1 : ℝ)) (𝓝 (g 1)) :=
      Filter.Tendsto.congr' (Filter.EventuallyEq.symm he)
        (hg.continuousAt.tendsto.mono_left nhdsWithin_le_nhds)
    exact (hasDerivWithinAt_Iic_of_tendsto_deriv hdiff hf.continuousAt.continuousWithinAt
      (Ioo_mem_nhdsLT (by norm_num)) hlim).mono Icc_subset_Iic_self
  exact (hderiv t ⟨by have h := ht.1; grind, by have h := ht.2; grind⟩).hasDerivWithinAt

/-- Actual measurable original vector arguments and the nonconstant
bounded original sine jointly witness the continuity hypotheses. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    AEMeasurable (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  exact ⟨φ, C, measurable_id.aemeasurable, measurable_const.aemeasurable, hφ, hbdd, hn⟩

/-- The genuine nonconstant sine and its actual derivative jointly
witness the endpoint extension inputs, including both original endpoints. -/
example : Continuous Real.sin ∧ Continuous Real.cos ∧
    (∀ r : ℝ, HasDerivAt Real.sin (Real.cos r) r) ∧
    (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 := by
  exact ⟨Real.continuous_sin, Real.continuous_cos, Real.hasDerivAt_sin, by norm_num, by norm_num⟩

end Transformer.Homogenized
