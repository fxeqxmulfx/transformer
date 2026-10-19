module

public import Transformer.Interpolation.Section1_CompactTests
public import Transformer.Interpolation.Section3_ProjectionDynamics

/-!
# Ambient spacetime tests for the original weak equation

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. The time derivative of a `C²` ambient spacetime
test is `C¹`, including in its spatial variable. Compactness then supplies
the supremum-norm derivative proved in `Section1_CompactTests`.
For a constant tangent projection field, its fixed-test flux is also a
`C¹` ambient test. Its integral against an actual weak solution is therefore
continuous directly from the original weak equation.
-/

@[expose] public section

open Real MeasureTheory Filter Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original integrated equation differentiates at every time where
its actual flux is continuous. Global continuity of the flux is unnecessary,
so this applies inside each phase of piecewise constant controls.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_test_at (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ)
    (hflux : ContinuousAt (fun s => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d θ (μ s) s (x : EucSpace d)) ∂(μ s : Measure (SSphere d))) t) :
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
        (fullVF d θ (μ t) t (x : EucSpace d)) ∂(μ t : Measure (SSphere d))) t := by
  let g (s : ℝ) := ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
    (fullVF d θ (μ s) s (x : EucSpace d)) ∂(μ s : Measure (SSphere d))
  have hi (a b : ℝ) : IntervalIntegrable g volume a b :=
    ((hμ φ hφ a).1.symm).trans (hμ φ hφ b).1
  have hmeas : StronglyMeasurableAtFilter g (nhds t) volume :=
    AEStronglyMeasurable.stronglyMeasurableAtFilter_of_mem
      (hi (t - 1) (t + 1)).1.aestronglyMeasurable
      (Ioc_mem_nhds (by linarith) (by linarith))
  have heq : (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d))) =
      (fun s => (∫ x, φ (x : EucSpace d) ∂(μ 0 : Measure (SSphere d))) +
        ∫ r in (0 : ℝ)..s, g r) := funext fun s => (hμ φ hφ s).2
  rw [heq]
  exact (intervalIntegral.integral_hasDerivAt_right (hi 0 t) hmeas hflux).const_add _

/-- The actual time derivative of an ambient spacetime test.
Source: arXiv:2411.04551v3, §1, the time-dependent tests for
`eq: cauchy.pb`. -/
noncomputable def timeTestDerivative (F : ℝ × EucSpace d → ℝ) (q : ℝ × EucSpace d) : ℝ :=
  fderiv ℝ F q ((1 : ℝ), (0 : EucSpace d))

/-- The actual time derivative of a `C²` spacetime test is jointly `C¹`.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_timeTestDerivative (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F) :
    ContDiff ℝ 1 (timeTestDerivative F) := by
  exact (hF.contDiff_fderiv_apply (m := 1) (by norm_num)).comp
    (contDiff_id.prodMk contDiff_const)

/-- Restricting a spacetime test to one time retains its spatial `C²`
regularity. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_spacetime_slice (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F) (t : ℝ) :
    ContDiff ℝ 2 (fun x : EucSpace d => F (t, x)) :=
  ContDiff.comp (f := fun x : EucSpace d => (t, x)) (g := F)
    hF (contDiff_const.prodMk contDiff_id)

/-- The fixed-time derivative is an admissible `C¹` ambient weak test.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_timeTestDerivative_slice (F : ℝ × EucSpace d → ℝ)
    (hF : ContDiff ℝ 2 F) (t : ℝ) :
    ContDiff ℝ 1 (fun x : EucSpace d => timeTestDerivative F (t, x)) :=
  ContDiff.comp (f := fun x : EucSpace d => (t, x)) (g := timeTestDerivative F)
    (contDiff_timeTestDerivative F hF) (contDiff_const.prodMk contDiff_id)

/-- The ambient derivative is the original pointwise derivative at every
time and point. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_spacetime_slice (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F)
    (t : ℝ) (x : EucSpace d) :
    HasDerivAt (fun s : ℝ => F (s, x)) (timeTestDerivative F (t, x)) t := by
  have hd := (hF.differentiable (by decide) (t, x)).hasFDerivAt
  exact hd.comp_hasDerivAt t ((hasDerivAt_id t).prodMk (hasDerivAt_const t x))

/-- The original weak equation gives continuity of the time-derivative
test integral, without assuming weak regularity of the measure curve.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_timeTestDerivative (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F) (t : ℝ) :
    Continuous (fun s => ∫ x, timeTestDerivative F (t, (x : EucSpace d))
      ∂(μ s : Measure (SSphere d))) :=
  continuous_cauchyPB_test θ μ hμ _ (contDiff_timeTestDerivative_slice F hF t)

/-- The integral of an actual smooth spacetime test is continuous at every
time, including switches of the original controls.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_spacetime_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F) :
    Continuous (fun s => ∫ x, F (s, (x : EucSpace d)) ∂(μ s : Measure (SSphere d))) := by
  have hcoe : Continuous (fun q : ℝ × SSphere d => (q.1, (q.2 : EucSpace d))) :=
    continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd)
  apply continuous_iff_continuousAt.mpr
  intro t
  exact continuousAt_spherical_compact_test μ
    (fun s x => F (s, (x : EucSpace d)))
    (fun s x => timeTestDerivative F (s, (x : EucSpace d)))
    (hF.continuous.comp hcoe) ((contDiff_timeTestDerivative F hF).continuous.comp hcoe)
    (fun s x => hasDerivAt_spacetime_slice F hF s (x : EucSpace d)) t
    (continuous_cauchyPB_test θ μ hμ _
      ((contDiff_spacetime_slice F hF t).of_le (by norm_num))).continuousAt

/-- The moving-test rule for a `C²` ambient test now obtains its derivative
and weak-continuity inputs from compactness and the original equation.
Only the fixed-test derivative at the specified time remains as an input.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_spacetime_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (F : ℝ × EucSpace d → ℝ) (hF : ContDiff ℝ 2 F) (t a : ℝ)
    (hfixed : HasDerivAt
      (fun s => ∫ x, F (t, (x : EucSpace d)) ∂(μ s : Measure (SSphere d))) a t) :
    HasDerivAt (fun s => ∫ x, F (s, (x : EucSpace d)) ∂(μ s : Measure (SSphere d)))
      (a + ∫ x, timeTestDerivative F (t, (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t := by
  have hcoe : Continuous (fun q : ℝ × SSphere d => (q.1, (q.2 : EucSpace d))) :=
    continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd)
  exact hasDerivAt_spherical_compact_test μ
    (fun s x => F (s, (x : EucSpace d)))
    (fun s x => timeTestDerivative F (s, (x : EucSpace d)))
    (hF.continuous.comp hcoe) ((contDiff_timeTestDerivative F hF).continuous.comp hcoe)
    (fun s x => hasDerivAt_spacetime_slice F hF s (x : EucSpace d)) t a hfixed
    (continuous_cauchyPB_timeTestDerivative θ μ hμ F hF t).continuousAt

/-- The actual fixed-test flux for the constant tangent projection field.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. -/
noncomputable def projectionTestFlux (φ : EucSpace d → ℝ) (w x : EucSpace d) : ℝ :=
  fderiv ℝ φ x (proj d x w)

/-- This is the original gradient pairing in the weak equation.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem projectionTestFlux_eq_inner (φ : EucSpace d → ℝ) (w x : EucSpace d) :
    projectionTestFlux φ w x = inner (𝕜 := ℝ) (gradient φ x) (proj d x w) := by
  rw [inner_gradient_left]
  rfl

/-- A `C¹` test has a continuous actual projection flux.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem continuous_projectionTestFlux (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ)
    (w : EucSpace d) : Continuous (projectionTestFlux φ w) := by
  have hc : Continuous (fun x : EucSpace d => proj d x w) := by
    unfold proj
    exact continuous_const.sub ((Continuous.inner (𝕜 := ℝ)
      (f := fun x : EucSpace d => x) (g := fun _ : EucSpace d => w)
      continuous_id continuous_const).smul continuous_id)
  exact (hφ.continuous_fderiv_apply (by decide)).comp (continuous_id.prodMk hc)

/-- A `C²` test has a `C¹` projection flux on the whole ambient space.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem contDiff_projectionTestFlux (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 2 φ)
    (w : EucSpace d) : ContDiff ℝ 1 (projectionTestFlux φ w) := by
  have hfield : ContDiff ℝ 1 (fun x : EucSpace d => proj d x w) := by
    exact contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  exact (hφ.contDiff_fderiv_apply (m := 1) (by norm_num)).comp
    (contDiff_id.prodMk hfield)

/-- Its integral against any original weak solution is continuous.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_projectionTestFlux (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 2 φ) (w : EucSpace d) :
    Continuous (fun s => ∫ x, projectionTestFlux φ w (x : EucSpace d)
      ∂(μ s : Measure (SSphere d))) :=
  continuous_cauchyPB_test θ μ hμ _ (contDiff_projectionTestFlux φ hφ w)

/-- Nonconstant spacetime and spatial tests satisfy all differentiability
inputs; the actual nonzero Dirac curve solves the original weak equation.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
example :
    ContDiff ℝ 2 (fun q : ℝ × EucSpace 2 => q.1 * inner (𝕜 := ℝ)
      (basePoint 1 : EucSpace 2) q.2) ∧
      ContDiff ℝ 2 (fun x : EucSpace 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) ∧
      cauchyPB 2 (pureAttentionParams 2 0) (fun _ => Perspective.diracProb 2 (basePoint 1)) := by
  refine ⟨contDiff_fst.mul (contDiff_const.inner ℝ contDiff_snd),
    contDiff_const.inner ℝ contDiff_id, cauchyPB_pureAttention_dirac 0 (basePoint 1)⟩

/-- The local derivative inputs also hold simultaneously for an actual
nonzero weak solution and a nonzero constant test. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 2 => (1 : ℝ)) ∧
    cauchyPB 2 (pureAttentionParams 2 0) (fun _ => Perspective.diracProb 2 (basePoint 1)) ∧
    ContinuousAt (fun s => ∫ x : SSphere 2,
      inner (𝕜 := ℝ) (gradient (fun _ : EucSpace 2 => (1 : ℝ)) (x : EucSpace 2))
        (fullVF 2 (pureAttentionParams 2 0) (Perspective.diracProb 2 (basePoint 1)) s
          (x : EucSpace 2)) ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) 0 := by
  refine ⟨contDiff_const, cauchyPB_pureAttention_dirac 0 (basePoint 1), ?_⟩
  simp only [gradient_fun_const, inner_zero_left, integral_zero]
  exact continuousAt_const

end Transformer.Interpolation
