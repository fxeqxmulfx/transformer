/-
# Attention continuity along arbitrary original weak laws

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`.
The original integrated equation gives continuous integrals of every
continuous ambient test, even with arbitrary controls. Consequently
an attention partition, numerator and average evaluated along that law
are continuous in time for any fixed score matrix and query.

The uniform spatial estimate gives joint continuity in time and sphere
query. These statements impose no attention-only or autonomous-control
restriction on the original weak equation. This distinction is needed
inside individual constant phases of the complete network.
-/

import Transformer.Interpolation.Section1_ContinuousWeakTests
import Transformer.Interpolation.Section1_FullFieldLipschitz

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual partition is continuous in time at a fixed query.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem continuous_attention_partition_of_cauchyPB (B : ParamMatrix d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (x : EucSpace d) : Continuous (fun t => Perspective.partitionMu d 1 (μ t) (B x)) := by
  simpa only [Perspective.partitionMu, one_mul] using
    continuous_cauchyPB_continuous_test θ μ hμ (fun y => exp (inner ℝ (B x) y))
      (continuous_const.inner continuous_id).rexp

/-- Each coordinate of the original vector numerator is a continuous
test integral. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_attention_numerator_of_cauchyPB (B : ParamMatrix d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (x : EucSpace d) :
    Continuous (fun t => ∫ y : SSphere d,
      exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d) ∂(μ t : Measure (SSphere d))) := by
  let m (t : ℝ) := ∫ y : SSphere d,
    exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d) ∂(μ t : Measure (SSphere d))
  have hc (i : Fin d) : Continuous (fun t => (m t).ofLp i) := by
    have heq : (fun t => (m t).ofLp i) =
        (fun t => ∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) *
          inner ℝ (EuclideanSpace.single i 1) (y : EucSpace d) ∂(μ t : Measure (SSphere d))) := by
      funext t
      have hi : Integrable (fun y : SSphere d => exp (inner ℝ (B x) (y : EucSpace d)) •
          (y : EucSpace d)) (μ t : Measure (SSphere d)) := by
        simpa only [one_mul] using Perspective.integrable_expInner_smul d 1 (μ t) (B x)
      have he := (integral_inner (𝕜 := ℝ) hi (EuclideanSpace.single i 1)).symm
      simpa only [real_inner_smul_right, EuclideanSpace.inner_single_left,
        map_one, one_mul, PiLp.smul_apply, smul_eq_mul] using he
    rw [heq]
    exact continuous_cauchyPB_continuous_test θ μ hμ _
      (((continuous_const.inner continuous_id).rexp).mul (continuous_const.inner continuous_id))
  exact (PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).comp (continuous_pi_iff.mpr hc)

/-- The actual attention average is continuous in time at a fixed query.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem continuous_attentionAverage_time_of_cauchyPB (B : ParamMatrix d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (x : EucSpace d) : Continuous (fun t => attentionAverage B (μ t) x) := by
  exact ((continuous_attention_partition_of_cauchyPB B θ μ hμ x).inv₀
    (fun t => (Perspective.partitionMu_pos d 1 (μ t) (B x)).ne')).smul
      (continuous_attention_numerator_of_cauchyPB B θ μ hμ x)

/-- The original average is jointly continuous in time and spherical
query. Auxiliary for arXiv:2411.04551v3, §1, the characteristic flow. -/
theorem continuous_attentionAverage_joint_sphere_of_cauchyPB (B : ParamMatrix d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    Continuous (fun q : ℝ × SSphere d => attentionAverage B (μ q.1) q.2) := by
  exact continuous_prod_of_continuous_lipschitzWith' _
    ⟨2 * exp (2 * ‖B‖) * ‖B‖, by positivity⟩
    (fun t => lipschitzWith_attentionAverage_sphere B (μ t))
    (fun x => continuous_attentionAverage_time_of_cauchyPB B θ μ hμ x)


/-- Every jointly continuous moving test on the sphere pairs
continuously with an arbitrary original weak law. Ambient continuity
of each slice supplies its weak test. Source: arXiv:2411.04551v3, §1. -/
theorem continuous_cauchyPB_moving_integral (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (F : ℝ → EucSpace d → ℝ) (hs : ∀ t, Continuous (F t))
    (hc : Continuous (fun q : ℝ × SSphere d => F q.1 q.2)) :
    Continuous (fun t => ∫ x : SSphere d, F t x ∂(μ t : Measure (SSphere d))) := by
  apply continuous_iff_continuousAt.mpr
  intro t
  exact continuousAt_bounded_weak_pairing (fun s => sphericalIntegralCLM (μ s))
    (fun s => norm_sphericalIntegralCLM_le (μ s))
    (boundedTestCurve (fun s (x : SSphere d) => F s x) hc) t
    (continuous_boundedTestCurve_of_continuous _ hc).continuousAt
    (continuous_cauchyPB_continuous_test θ μ hμ (F t) (hs t)).continuousAt

/-- The complete original field at any fixed parameter tuple is
jointly continuous along an arbitrary original weak law. Source:
arXiv:2411.04551v3, §1, the field within one constant control phase. -/
theorem continuous_fullVF_fixed_params_joint (A : Params d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    Continuous (fun q : ℝ × SSphere d => fullVF d (fun _ => A) (μ q.1) 0 q.2) := by
  have htime (x : SSphere d) : Continuous (fun t => fullVF d (fun _ => A) (μ t) 0 x) := by
    have hc := A.V.continuous.comp (continuous_attentionAverage_time_of_cauchyPB A.B θ μ hμ x)
    have hp : Continuous (fun t => proj d (x : EucSpace d) (A.V (attentionAverage A.B (μ t) x))) :=
      hc.sub ((continuous_const.inner (𝕜 := ℝ) hc).smul continuous_const)
    convert hp.add (continuous_const : Continuous (fun _ : ℝ =>
      neuralVF d (fun _ => A.W) (fun _ => A.U) (fun _ => A.b) 0 x)) using 1
    funext t
    exact fullVF_eq_proj_attention_add_neural (fun _ => A) (μ t) 0 x
  exact continuous_prod_of_continuous_lipschitzWith' _
    ⟨‖A.V‖ * (2 * exp (2 * ‖A.B‖) * ‖A.B‖) + ‖A.W‖ * ‖A.U‖ +
      2 * (‖A.V‖ + ‖A.W‖ * (‖A.U‖ + ‖A.b‖)), by positivity⟩
    (fun t => lipschitzWith_fullVF_sphere (fun _ => A) (μ t) 0) htime

/-- Every C¹ weak flux for a fixed complete parameter tuple is
continuous along any original weak law. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_fullVF_fixed_params_test_flux (A : Params d) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (f : EucSpace d → ℝ) (hf : ContDiff ℝ 1 f) :
    Continuous (fun t => ∫ x : SSphere d, inner ℝ (gradient f (x : EucSpace d))
      (fullVF d (fun _ => A) (μ t) 0 x) ∂(μ t : Measure (SSphere d))) := by
  simp_rw [inner_gradient_left]
  refine continuous_cauchyPB_moving_integral θ μ hμ
    (fun t x => fderiv ℝ f x (fullVF d (fun _ => A) (μ t) 0 x)) ?_ ?_
  · intro t
    exact (hf.continuous_fderiv_apply (by norm_num)).comp
      (continuous_id.prodMk (continuous_fullVF (fun _ => A) (μ t) 0))
  · exact (hf.continuous_fderiv_apply (by norm_num)).comp
      ((continuous_subtype_val.comp continuous_snd).prodMk
        (continuous_fullVF_fixed_params_joint A θ μ hμ))

/-- The original integrated equation differentiates every test for
constant complete controls. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`. Nonlinear weak uniqueness is not assumed. -/
theorem hasDerivAt_cauchyPB_constant_params_test (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ)
    (f : EucSpace d → ℝ) (hf : ContDiff ℝ 1 f) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner ℝ (gradient f (x : EucSpace d))
        (fullVF d (fun _ => A) (μ t) t x) ∂(μ t : Measure (SSphere d))) t :=
  hasDerivAt_cauchyPB_test (fun _ => A) μ hμ f hf
    (continuous_fullVF_fixed_params_test_flux A (fun _ => A) μ hμ f hf) t

/-- The same nonzero constant attention, actual unit Dirac weak law,
nonconstant smooth scalar test and moving linear test satisfy all
hypotheses jointly, including the constant-control equation. -/
example : ∃ (A : Params 2) (θ : TimeParams 2) (μ : ℝ → Perspective.ProbSphere 2)
    (B : ParamMatrix 2) (x : EucSpace 2) (f : EucSpace 2 → ℝ) (F : ℝ → EucSpace 2 → ℝ),
    θ = (fun _ => A) ∧ cauchyPB 2 θ μ ∧ B x = x ∧ ‖x‖ = 1 ∧ ContDiff ℝ 1 f ∧
    (∀ t, Continuous (F t)) ∧ Continuous (fun q : ℝ × SSphere 2 => F q.1 q.2) := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let A : Params 2 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 2 := basePoint 1
  let f (x : EucSpace 2) := inner ℝ (p : EucSpace 2) x
  let F (t : ℝ) (x : EucSpace 2) := sin t * f x
  have hθ : pureAttentionParams 2 1 = (fun _ => A) := by
    funext t
    simp only [pureAttentionParams, one_smul]
    rfl
  have hμ : cauchyPB 2 (fun _ => A) (fun _ => Perspective.diracProb 2 p) := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  refine ⟨A, (fun _ => A), (fun _ => Perspective.diracProb 2 p), I, p, f, F,
    rfl, hμ, rfl, mem_sphere_zero_iff_norm.mp p.2, contDiff_const.inner ℝ contDiff_id,
    fun t => continuous_const.mul (continuous_const.inner continuous_id), ?_⟩
  exact (continuous_sin.comp continuous_fst).mul
    (continuous_const.inner (continuous_subtype_val.comp continuous_snd))

end Transformer.Interpolation
