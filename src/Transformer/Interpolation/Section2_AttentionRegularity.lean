module

public import Transformer.Interpolation.Section2_AttentionLipschitz
public import Transformer.Interpolation.Section1_AutonomousWeakTests

/-!
# Time regularity along original attention weak solutions

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §2, `prop: targets.atoms`.

The original attention velocity has norm at most one. The weak equation
therefore gives continuity of every continuous test integral. In particular,
the partition function and the vector numerator are continuous in time at
every fixed query. Their quotient is continuous because the partition is
strictly positive.

The uniform spatial Lipschitz bound upgrades this to joint continuity on
time times the sphere. Pairing a jointly continuous moving test with the
actual probability curve is continuous in time. Hence every original C1
test flux is continuous, and the integrated weak equation supplies its
pointwise derivative at every time. No independent continuity assumption
on the probability curve is used.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every continuous ambient test is weakly continuous along the actual
attention solution. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_attention_cauchyPB_test (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (f : EucSpace d → ℝ) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) := by
  exact continuous_cauchyPB_bounded_test _ 1 zero_le_one
    (fun ν t x => norm_fullVF_attention_le_one B ν t x) μ hμ f hf

/-- The actual partition is continuous in time at a fixed query.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem continuous_attention_partition (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : EucSpace d) : Continuous (fun t => Perspective.partitionMu d 1 (μ t) (B x)) := by
  simpa only [Perspective.partitionMu, one_mul] using
    continuous_attention_cauchyPB_test B μ hμ (fun y => exp (inner ℝ (B x) y))
      (continuous_const.inner continuous_id).rexp

/-- Each coordinate of the original vector numerator is a continuous
test integral. Auxiliary for arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_attention_numerator (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
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
    exact continuous_attention_cauchyPB_test B μ hμ _
      (((continuous_const.inner continuous_id).rexp).mul (continuous_const.inner continuous_id))
  exact (PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).comp (continuous_pi_iff.mpr hc)

/-- The actual attention average is continuous in time at a fixed query.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem continuous_attentionAverage_time (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : EucSpace d) : Continuous (fun t => attentionAverage B (μ t) x) := by
  exact ((continuous_attention_partition B μ hμ x).inv₀
    (fun t => (Perspective.partitionMu_pos d 1 (μ t) (B x)).ne')).smul
      (continuous_attention_numerator B μ hμ x)

/-- The original average is jointly continuous in time and spherical
query. Auxiliary for arXiv:2411.04551v3, §1, the characteristic flow. -/
theorem continuous_attentionAverage_joint_sphere (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ) :
    Continuous (fun q : ℝ × SSphere d => attentionAverage B (μ q.1) q.2) := by
  exact continuous_prod_of_continuous_lipschitzWith' _
    ⟨2 * exp (2 * ‖B‖) * ‖B‖, by positivity⟩
    (fun t => lipschitzWith_attentionAverage_sphere B (μ t))
    (fun x => continuous_attentionAverage_time B μ hμ x)

/-- The actual field along the evolving law is jointly continuous on
time times the sphere. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem continuous_fullVF_attention_joint_sphere (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ) :
    Continuous (fun q : ℝ × SSphere d =>
      fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ q.1) q.1 q.2) := by
  simp_rw [fullVF_eq_proj_attentionAverage, proj]
  have hm := continuous_attentionAverage_joint_sphere B μ hμ
  have hx : Continuous (fun q : ℝ × SSphere d => (q.2 : EucSpace d)) :=
    continuous_subtype_val.comp continuous_snd
  exact hm.sub ((hx.inner (𝕜 := ℝ) hm).smul hx)

/-- A jointly continuous moving test pairs continuously with the actual
weak solution. Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_attention_moving_integral (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (F : ℝ → EucSpace d → ℝ) (hs : ∀ t, Continuous (F t))
    (hc : Continuous (fun q : ℝ × SSphere d => F q.1 q.2)) :
    Continuous (fun t => ∫ x : SSphere d, F t x ∂(μ t : Measure (SSphere d))) := by
  apply continuous_iff_continuousAt.mpr
  intro t
  exact continuousAt_bounded_weak_pairing (fun s => sphericalIntegralCLM (μ s))
    (fun s => norm_sphericalIntegralCLM_le (μ s))
    (boundedTestCurve (fun s (x : SSphere d) => F s x) hc) t
    (continuous_boundedTestCurve_of_continuous _ hc).continuousAt
    (continuous_attention_cauchyPB_test B μ hμ (F t) (hs t)).continuousAt

/-- The original flux of every C1 ambient test is continuous in time.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_attention_test_flux (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (f : EucSpace d → ℝ) (hf : ContDiff ℝ 1 f) :
    Continuous (fun s => ∫ x : SSphere d, inner ℝ (gradient f (x : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s x)
      ∂(μ s : Measure (SSphere d))) := by
  simp_rw [inner_gradient_left]
  refine continuous_attention_moving_integral B μ hμ
    (fun s x => fderiv ℝ f x
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s x)) ?_ ?_
  · intro s
    exact (hf.continuous_fderiv_apply (by norm_num)).comp
      (continuous_id.prodMk (continuous_fullVF_attention B (μ s) s))
  · exact (hf.continuous_fderiv_apply (by norm_num)).comp
      ((continuous_subtype_val.comp continuous_snd).prodMk
        (continuous_fullVF_attention_joint_sphere B μ hμ))

/-- The original integrated attention equation gives its pointwise weak
test derivative at every time. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, used for characteristics in §2, `prop: targets.atoms`. -/
theorem hasDerivAt_attention_cauchyPB_test (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (f : EucSpace d → ℝ) (hf : ContDiff ℝ 1 f) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner ℝ (gradient f (x : EucSpace d))
        (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t x)
          ∂(μ t : Measure (SSphere d))) t :=
  hasDerivAt_cauchyPB_test _ μ hμ f hf (continuous_attention_test_flux B μ hμ f hf) t

/-- A nonzero actual attention matrix, a genuine weak probability curve,
a nonconstant linear test and a test varying in both time and space
jointly realize every regularity and weak-equation hypothesis. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2)
    (f : EucSpace 2 → ℝ) (F : ℝ → EucSpace 2 → ℝ),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧
    Continuous f ∧ ContDiff ℝ 1 f ∧ (∀ t, Continuous (F t)) ∧
    Continuous (fun q : ℝ × SSphere 2 => F q.1 q.2) := by
  let p : EucSpace 2 := basePoint 1
  have hw := cauchyPB_pureAttention_dirac (1 : ℝ) (basePoint 1 : SSphere 2)
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 2),
    (fun _ => Perspective.diracProb 2 (basePoint 1)), (fun x => inner ℝ p x),
    (fun t x => sin t * inner ℝ p x), ?_, continuous_const.inner continuous_id,
    contDiff_const.inner ℝ contDiff_id, fun t => continuous_const.mul
      (continuous_const.inner continuous_id), ?_⟩
  · have hθ : pureAttentionParams 2 1 =
        (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2),
          ContinuousLinearMap.id ℝ (EucSpace 2), 0, 0, 0⟩) := by
      funext t
      simp only [pureAttentionParams, one_smul]
    rw [← hθ]
    exact hw
  · exact (continuous_sin.comp continuous_fst).mul
      (continuous_const.inner (continuous_subtype_val.comp continuous_snd))

end Transformer.Interpolation
