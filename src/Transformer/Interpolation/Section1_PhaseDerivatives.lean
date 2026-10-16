/-
# Derivatives inside a complete original control phase

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`.
The parameter curve can switch outside the time under consideration.
If its complete tuple is constant on a neighborhood of that time,
continuity of the fixed-tuple flux gives the actual weak-test derivative.
The vector barycenter and spherical variance therefore differentiate
there without any nonlinear uniqueness or atomicity hypothesis.

All five original blocks are retained. The variance growth constant
uses the actual tuple on this phase. No differentiability is asserted
at its switching endpoints; the weak barycenter is continuous there.
-/

import Transformer.Interpolation.Section1_FullWeakContinuity
import Transformer.Interpolation.Section1_VarianceFlux
import Transformer.Interpolation.Section1_SpacetimeTests

open Real MeasureTheory Filter Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original weak test differentiates inside a constant complete
control phase. Source auxiliary: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, with the source's piecewise constant parameters. -/
theorem hasDerivAt_cauchyPB_locally_constant_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (A : Params d) (t : ℝ) (hθ : θ =ᶠ[𝓝 t] (fun _ => A))
    (f : EucSpace d → ℝ) (hf : ContDiff ℝ 1 f) :
    HasDerivAt (fun s => ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner ℝ (gradient f (x : EucSpace d))
        (fullVF d θ (μ t) t x) ∂(μ t : Measure (SSphere d))) t := by
  apply hasDerivAt_cauchyPB_test_at θ μ hμ f hf t
  apply (continuous_fullVF_fixed_params_test_flux A θ μ hμ f hf).continuousAt.congr
  filter_upwards [hθ] with s hs
  simp only [fullVF, hs]

/-- The actual original vector barycenter differentiates on the interior
of any constant complete phase. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_locally_constant_barycenter (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (A : Params d) (t : ℝ) (hθ : θ =ᶠ[𝓝 t] (fun _ => A)) :
    HasDerivAt (fun s => barycenter d (μ s))
      (∫ x : SSphere d, fullVF d θ (μ t) t x ∂(μ t : Measure (SSphere d))) t := by
  let v := ∫ x : SSphere d, fullVF d θ (μ t) t x ∂(μ t : Measure (SSphere d))
  have hi : Integrable (fun x : SSphere d => fullVF d θ (μ t) t x)
      (μ t : Measure (SSphere d)) :=
    (continuous_fullVF_sphere θ (μ t) t).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hc (i : Fin d) : HasDerivAt (fun s => (barycenter d (μ s)).ofLp i) (v.ofLp i) t := by
    have hd := hasDerivAt_cauchyPB_locally_constant_test θ μ hμ A t hθ
      (fun x => inner ℝ (EuclideanSpace.single i 1) x) (contDiff_const.inner ℝ contDiff_id)
    simp only [gradient_inner_linear] at hd
    convert hd using 1
    · funext s
      rw [integral_inner (integrable_spherical_coe (μ s))]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
    · rw [integral_inner hi]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
  have hpi := hasDerivAt_pi.mpr hc
  let L := (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.toContinuousLinearMap
  exact L.hasFDerivAt.comp_hasDerivAt t hpi

/-- The spherical variance differentiates at every interior phase time.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: cauchy.pb`.
The derivative is expressed using the complete original field. -/
theorem hasDerivAt_cauchyPB_locally_constant_variance (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (A : Params d) (t : ℝ) (hθ : θ =ᶠ[𝓝 t] (fun _ => A)) :
    HasDerivAt (fun s => 1 - ‖barycenter d (μ s)‖ ^ 2)
      (-2 * inner ℝ (barycenter d (μ t))
        (∫ x : SSphere d, fullVF d θ (μ t) t x ∂(μ t : Measure (SSphere d)))) t := by
  convert ((hasDerivAt_cauchyPB_locally_constant_barycenter θ μ hμ A t hθ).norm_sq).const_sub 1
    using 1
  ring

/-- The original weak law's variance is continuous also at switches.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: cauchy.pb`.
This follows from the original linear weak tests, without a topology
or additional continuity assumption on the measure-valued curve. -/
theorem continuous_cauchyPB_variance (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    Continuous (fun s => 1 - ‖barycenter d (μ s)‖ ^ 2) :=
  continuous_const.sub ((continuous_cauchyPB_mean θ μ hμ).norm.pow 2)

/-- The variance of every original spherical probability law is
nonnegative. Source auxiliary: arXiv:2411.04551v3, §1,
the barycenter in `eq: average.vf`. -/
theorem spherical_variance_nonneg (ν : Perspective.ProbSphere d) :
    0 ≤ 1 - ‖barycenter d ν‖ ^ 2 := by
  have hn : ‖barycenter d ν‖ ≤ 1 := norm_spherical_mean_le_one ν
  have hp := norm_nonneg (barycenter d ν)
  nlinarith

/-- The actual variance flux is bounded by variance times the exact
full-field spatial constant. Source auxiliary: arXiv:2411.04551v3,
§1, `eq: vf`. This estimate also holds at switching times. -/
theorem cauchyPB_variance_flux_le (θ : TimeParams d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    -2 * inner ℝ (barycenter d ν)
      (∫ x : SSphere d, fullVF d θ ν t x ∂(ν : Measure (SSphere d))) ≤
      2 * (‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) +
        ‖(θ t).W‖ * ‖(θ t).U‖ +
        2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖))) *
          (1 - ‖barycenter d ν‖ ^ 2) := by
  have h := neg_inner_mean_flux_le_variance ν
    (fun x : SSphere d => fullVF d θ ν t x)
    ⟨‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) +
      ‖(θ t).W‖ * ‖(θ t).U‖ +
      2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖)), by positivity⟩
    (lipschitzWith_fullVF_sphere θ ν t)
    (fun x => inner_fullVF_eq_zero θ ν t x (mem_sphere_zero_iff_norm.mp x.2))
  change -inner ℝ (barycenter d ν)
      (∫ x : SSphere d, fullVF d θ ν t x ∂(ν : Measure (SSphere d))) ≤
    (‖(θ t).V‖ * (2 * exp (2 * ‖(θ t).B‖) * ‖(θ t).B‖) +
      ‖(θ t).W‖ * ‖(θ t).U‖ +
      2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖))) *
        (1 - ‖barycenter d ν‖ ^ 2) at h
  linarith

/-- Constancy on a half-open phase supplies local constancy at every
strictly interior time. Source auxiliary: arXiv:2411.04551v3, §1,
the source's piecewise constant controls in `eq: cauchy.pb`. -/
theorem eventuallyEq_const_of_constantOn_Ico (θ : TimeParams d) (a b t : ℝ)
    (hθ : ∀ s ∈ Ico a b, ∀ r ∈ Ico a b, θ s = θ r) (ht : t ∈ Ioo a b) :
    θ =ᶠ[𝓝 t] (fun _ => θ t) := by
  filter_upwards [Ioo_mem_nhds ht.1 ht.2] with s hs
  exact hθ s ⟨hs.1.le, hs.2⟩ t ⟨ht.1.le, ht.2⟩

/-- One nonzero original constant-attention tuple, its actual stationary
unit Dirac law, a nonconstant smooth test and an interior time jointly
satisfy every weak-law, local constancy and interval hypothesis. -/
example : ∃ (θ : TimeParams 2) (μ : ℝ → Perspective.ProbSphere 2)
    (A : Params 2) (f : EucSpace 2 → ℝ),
    cauchyPB 2 θ μ ∧ θ =ᶠ[𝓝 (1 / 2 : ℝ)] (fun _ => A) ∧ ContDiff ℝ 1 f ∧
    (∀ s ∈ Ico (0 : ℝ) 1, ∀ r ∈ Ico (0 : ℝ) 1, θ s = θ r) ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let A : Params 2 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 2 := basePoint 1
  have hθ : pureAttentionParams 2 1 = (fun _ => A) := by
    funext s
    simp only [pureAttentionParams, one_smul]
    rfl
  have hμ : cauchyPB 2 (fun _ => A) (fun _ => Perspective.diracProb 2 p) := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  exact ⟨(fun _ => A), (fun _ => Perspective.diracProb 2 p), A,
    (fun x => inner ℝ (p : EucSpace 2) x), hμ, Filter.EventuallyEq.rfl,
    contDiff_const.inner ℝ contDiff_id, (fun _ _ _ _ => rfl), by norm_num⟩

end Transformer.Interpolation
