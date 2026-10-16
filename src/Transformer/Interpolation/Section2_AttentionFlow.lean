/-
# Regularity of the actual attention characteristic flow

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`, and §2,
`prop: targets.atoms`. The characteristic through each unit initial
point is chosen from the proved existence theorem. Its defining
properties are the original particle equation and the initial value.

Uniqueness identifies every other actual trajectory with this one.
The original spherical Lipschitz estimate gives the initial-point
modulus in both time directions. The original speed bound is one,
so the flow has a uniform time modulus and is jointly continuous.
The finite smooth-cone comparison applies to its genuine particle
centers and arbitrary Lipschitz terminal tests.
-/

import Transformer.Interpolation.Section2_AttentionCharacteristics
import Transformer.Interpolation.Section1_TimeDependentFlowBounds

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The unique complete original particle through an initial unit point,
for the given actual weak law. Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
noncomputable def attentionFlow (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) : ℝ → SSphere d := (exists_attention_particle B μ hμ x).choose

/-- The actual initial value of the attention characteristic.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem attentionFlow_zero (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) : attentionFlow B μ hμ x 0 = x :=
  (exists_attention_particle B μ hμ x).choose_spec.1

/-- The chosen characteristic solves the original evolving-law equation.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: vf`. -/
theorem hasDerivAt_attentionFlow (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) (t : ℝ) : HasDerivAt (fun s => (attentionFlow B μ hμ x s : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩)
        (μ t) t (attentionFlow B μ hμ x t)) t :=
  (exists_attention_particle B μ hμ x).choose_spec.2 t

/-- Every other actual particle agrees with the constructed characteristic.
Source: arXiv:2411.04551v3, §1, uniqueness for `eq: ips`. -/
theorem attentionFlow_eq_of_particle (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) (Y : ℝ → SSphere d) (h0 : Y 0 = x)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t (Y t)) t) :
    attentionFlow B μ hμ x = Y :=
  unique_attention_particle B μ _ Y (hasDerivAt_attentionFlow B μ hμ x) hY
    ((attentionFlow_zero B μ hμ x).trans h0.symm)

/-- Each original attention path is C1, despite only weak law regularity.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem contDiff_attentionFlow (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) : ContDiff ℝ 1 (fun s => (attentionFlow B μ hμ x s : EucSpace d)) :=
  contDiff_attention_particle B μ hμ _ (hasDerivAt_attentionFlow B μ hμ x)

/-- The original attention speed gives a uniform time modulus of one.
Source: arXiv:2411.04551v3, §1, `eq: vf`, with `V = I` and `W = 0`. -/
theorem lipschitzWith_attentionFlow_time (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x : SSphere d) : LipschitzWith 1 (attentionFlow B μ hμ x) :=
  lipschitzWith_timeDependent_spherical_trajectory_time _ 1
    (fun t z => norm_fullVF_attention_le_one B (μ t) t z) _ (hasDerivAt_attentionFlow B μ hμ x)

/-- The initial-point modulus of the actual attention flow at every real
time. Source: arXiv:2411.04551v3, §1, Lipschitz characteristics. -/
theorem lipschitzWith_attentionFlow_initial (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (t : ℝ) : LipschitzWith (Real.toNNReal (exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * |t|)))
      (fun x => attentionFlow B μ hμ x t) :=
  lipschitzWith_timeDependent_sphericalFlow_initial _ ⟨_, by positivity⟩
    (fun s => lipschitzOnWith_fullVF_attention_sphere B (μ s) s)
    (attentionFlow B μ hμ) (hasDerivAt_attentionFlow B μ hμ) (attentionFlow_zero B μ hμ) t

/-- Forward Grönwall bound between the actual original particles.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: ips`. -/
theorem dist_attentionFlow_initial_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (x y : SSphere d) (t : ℝ) (ht : 0 ≤ t) :
    dist (attentionFlow B μ hμ x t) (attentionFlow B μ hμ y t) ≤
      dist x y * exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * t) := by
  have hh := dist_timeDependent_spherical_trajectories_le_exp _ ⟨_, by positivity⟩
      (fun s => lipschitzOnWith_fullVF_attention_sphere B (μ s) s)
      (attentionFlow B μ hμ x) (attentionFlow B μ hμ y)
      (hasDerivAt_attentionFlow B μ hμ x) (hasDerivAt_attentionFlow B μ hμ y) t ht
  change dist (attentionFlow B μ hμ x t) (attentionFlow B μ hμ y t) ≤
    dist (attentionFlow B μ hμ x 0) (attentionFlow B μ hμ y 0) *
      exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * t) at hh
  simpa only [attentionFlow_zero] using hh

/-- Joint continuity of the actual characteristic flow.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem continuous_attentionFlow (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ) :
    Continuous (fun q : ℝ × SSphere d => attentionFlow B μ hμ q.2 q.1) :=
  continuous_timeDependent_sphericalFlow _ ⟨_, by positivity⟩ 1
    (fun s => lipschitzOnWith_fullVF_attention_sphere B (μ s) s)
    (fun s x => norm_fullVF_attention_le_one B (μ s) s x)
    (attentionFlow B μ hμ) (hasDerivAt_attentionFlow B μ hμ) (attentionFlow_zero B μ hμ)

/-- Finite smooth cones bound arbitrary Lipschitz terminal tests along
the original weak law, using actual attention-flow centers.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem integral_attention_test_le_finiteTransport_initial {ι : Type*} [Fintype ι] [Nonempty ι]
    (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (L : ℝ≥0) (ε β T : ℝ) (hε : 0 < ε) (hβ : 0 < β) (hT : 0 ≤ T)
    (z : ι → SSphere d) (f : SSphere d → ℝ) (hf : LipschitzWith L f) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      ∫ x : SSphere d, finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T
        (fun i => attentionFlow B μ hμ (z i)) f (0, (x : EucSpace d)) ∂(μ 0 : Measure (SSphere d)) := by
  let Y (i : ι) := attentionFlow B μ hμ (z i)
  have hc := (contDiff_finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T hε Y
    (fun i => contDiff_attentionFlow B μ hμ (z i)) f).continuous
  have hi := integral_mono_ae (integrable_spherical_continuous (μ T) f hf.continuous)
    (integrable_spherical_continuous (μ T) _ (hc.comp (continuous_const.prodMk continuous_subtype_val)))
    (ae_of_all _ fun x => le_finiteTransportTest_terminal (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T hβ Y f hf x)
  exact hi.trans (integral_attention_finiteTransportTest_le B μ hμ L ε β T L.coe_nonneg hε hβ Y
    (fun i => hasDerivAt_attentionFlow B μ hμ (z i)) f 0 T hT)

/-- A genuine Dirac attention weak law, a unit initial point and its
actual particle jointly witness the weak, trajectory and initial-value
hypotheses. A two-center family and a nonconstant distance test witness
all finite comparison premises with positive scales. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2) (Y : ℝ → SSphere 2)
    (x : SSphere 2) (z : Fin 2 → SSphere 2) (f : SSphere 2 → ℝ),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧ Y 0 = x ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2))
      (fullVF 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) (μ t) t (Y t)) t) ∧
    LipschitzWith 1 f ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ z 0 = x := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨B, B, 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  refine ⟨B, μ, attentionFlow B μ hμ p, p, (fun _ => p), (fun x => dist x p),
    hμ, attentionFlow_zero B μ hμ p, hasDerivAt_attentionFlow B μ hμ p,
    LipschitzWith.dist_left _, zero_le_one, zero_lt_one, rfl⟩

end Transformer.Interpolation
