/-
# Differentiating the original Gaussian interpolation expectation

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The actual original path derivative has
an explicit uniform local bound in terms of the two original vector
norms. Their genuine Gaussian first moments make this bound integrable.

The original test is bounded and both original vectors are measurable,
so its actual original expectation is finite. The actual scalar path
derivative is also measurable, and it is the derivative at every time
in the same original neighborhood. Dominated differentiation therefore
proves the derivative of the actual original expectation, together with
finiteness of the actual original derivative expectation.
The analytic step below takes integrability explicitly, so the original
Gaussian laws supply that input rather than an assumed derivative formula.
-/

import Transformer.Homogenized.Section7_GaussianInterpolationPath
import Mathlib.Analysis.Calculus.ParametricIntegral

open Real MeasureTheory ProbabilityTheory Set Filter
open scoped Topology

namespace Transformer.Homogenized

/-- The actual original test along the original interpolation is
measurable on its original probability space.
Source: arXiv:2604.01978v1, §7, the expectation differentiated
in `lem:gaussian_interp`. Both original vector measurability inputs remain explicit. -/
theorem aestronglyMeasurable_gaussianInterpolation_test
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) (X Y : Ω → EucSpace m)
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (g : EucSpace m → ℝ) (hg : Continuous g) (t : ℝ) :
    AEStronglyMeasurable (fun ω => g (gaussianInterpolationPoint t (X ω) (Y ω))) P := by
  have hp : Continuous (fun p : EucSpace m × EucSpace m => gaussianInterpolationPoint t p.1 p.2) := by
    unfold gaussianInterpolationPoint
    fun_prop
  have hm : AEStronglyMeasurable (fun p : EucSpace m × EucSpace m =>
      g (gaussianInterpolationPoint t p.1 p.2)) (P.map (fun ω => (X ω, Y ω))) :=
    (hg.comp hp).aestronglyMeasurable
  simpa only [Function.comp_def] using hm.comp_aemeasurable (hX.prodMk hY)

/-- The actual original bounded test expectation is finite at every
original time. Source: arXiv:2604.01978v1, §7,
the corrected bounded-test expectation in `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_test {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace m)
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (g : EucSpace m → ℝ) (hg : Continuous g) (C : ℝ) (hbound : ∀ z, ‖g z‖ ≤ C) (t : ℝ) :
    Integrable (fun ω => g (gaussianInterpolationPoint t (X ω) (Y ω))) P :=
  Integrable.of_bound (aestronglyMeasurable_gaussianInterpolation_test P X Y hX hY g hg t)
    C (ae_of_all _ fun ω => hbound (gaussianInterpolationPoint t (X ω) (Y ω)))

/-- The genuine original first moments make the explicit original
local derivative dominator integrable.
Source: arXiv:2604.01978v1, §7, dominated differentiation
before the Gaussian integration-by-parts step of `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_derivative_dominator
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) (X Y : Ω → EucSpace m)
    (hX : Integrable X P) (hY : Integrable Y P) (C u v : ℝ) :
    Integrable (fun ω => C * (u * ‖X ω‖ + v * ‖Y ω‖)) P :=
  ((hX.norm.const_mul u).add (hY.norm.const_mul v)).const_mul C

/-- The derivative of the original interpolation expectation is the
actual original scalar path derivative expectation, which is finite.
Source: arXiv:2604.01978v1, §7, the differentiation under the
integral in `lem:gaussian_interp`. The original Gaussian first moments
will supply the integrability inputs when proving its covariance formula. -/
theorem gaussianInterpolation_differentiate_integral
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) [IsProbabilityMeasure P]
    (X Y : Ω → EucSpace m) (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (hXint : Integrable X P) (hYint : Integrable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    Integrable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (gaussianInterpolationVelocity t (X ω) (Y ω))) P ∧
    HasDerivAt (fun r : ℝ => ∫ ω, φ (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P)
      (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
        (gaussianInterpolationVelocity t (X ω) (Y ω)) ∂P) t := by
  have hs : Ioo (t / 2) ((t + 1) / 2) ∈ 𝓝 t :=
    isOpen_Ioo.mem_nhds ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have hmeas (r : ℝ) := aestronglyMeasurable_gaussianInterpolation_test P X Y hX hY φ hφ.continuous r
  have htestint := integrable_gaussianInterpolation_test P X Y hX hY φ hφ.continuous C
    (norm_gaussian_test_le φ C hbdd) t
  have hderivmeas := aestronglyMeasurable_gaussianInterpolation_test_derivative P X Y hX hY φ hφ t
  have hbound : ∀ᵐ ω ∂P, ∀ r ∈ Ioo (t / 2) ((t + 1) / 2),
      ‖fderiv ℝ φ (gaussianInterpolationPoint r (X ω) (Y ω))
        (gaussianInterpolationVelocity r (X ω) (Y ω))‖ ≤
      C * ((1 / (2 * Real.sqrt (t / 2))) * ‖X ω‖ +
        (1 / (2 * Real.sqrt ((1 - t) / 2))) * ‖Y ω‖) :=
    ae_of_all _ fun ω r hr => norm_gaussianInterpolation_test_derivative_local_le φ C hbdd t ht r hr _ _
  have hdom := integrable_gaussianInterpolation_derivative_dominator P X Y hXint hYint C
    (1 / (2 * Real.sqrt (t / 2))) (1 / (2 * Real.sqrt ((1 - t) / 2)))
  have hdiff : ∀ᵐ ω ∂P, ∀ r ∈ Ioo (t / 2) ((t + 1) / 2),
      HasDerivAt (fun q => φ (gaussianInterpolationPoint q (X ω) (Y ω)))
        (fderiv ℝ φ (gaussianInterpolationPoint r (X ω) (Y ω))
          (gaussianInterpolationVelocity r (X ω) (Y ω))) r := by
    apply ae_of_all
    intro ω r hr
    apply hasDerivAt_gaussianInterpolation_test φ hφ (X ω) (Y ω) r
    exact ⟨by linarith [ht.1, hr.1], by linarith [ht.2, hr.2]⟩
  exact hasDerivAt_integral_of_dominated_loc_of_deriv_le hs (Eventually.of_forall hmeas)
    htestint hderivmeas hbound hdom hdiff

/-- Both original Gaussian laws supply their genuine first moments
for differentiation of the actual original expectation.
Source: arXiv:2604.01978v1, §7, the analytic step in
`lem:gaussian_interp`. No derivative identity is supplied as an input;
measurability and integrability follow from the actual original laws. -/
theorem gaussianInterpolation_differentiate_gaussian_integral
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) (X Y : Ω → EucSpace m)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    Integrable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (gaussianInterpolationVelocity t (X ω) (Y ω))) P ∧
    HasDerivAt (fun r : ℝ => ∫ ω, φ (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P)
      (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
        (gaussianInterpolationVelocity t (X ω) (Y ω)) ∂P) t := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  exact gaussianInterpolation_differentiate_integral P X Y hX.aemeasurable hY.aemeasurable
    hX.integrable hY.integrable φ hφ C hbdd t ht

/-- Genuine original standard Gaussian first moments, an independent
constant original vector and the nonconstant bounded original sine
jointly witness the differentiability, measurability and bound inputs. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    AEMeasurable (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧
    Integrable (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    Integrable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  exact ⟨φ, C, measurable_id.aemeasurable, measurable_const.aemeasurable,
    IsGaussian.integrable_id, integrable_const _, hφ, hbdd, by norm_num, hn⟩

/-- The genuine original standard law supplies both Gaussian-law
inputs, together with the nonconstant bounded original sine and an
actual interior time, for the Gaussian differentiation step. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  exact ⟨φ, C, IsGaussian.hasGaussianLaw_id, hφ, hbdd, by norm_num, hn⟩

end Transformer.Homogenized
