module

public import Transformer.Interpolation.Section1_FiniteNets
public import Transformer.Interpolation.Section1_TimeRescaling
public import Transformer.Interpolation.Section3_ReLURidges
public import Transformer.Interpolation.HypPropagationFalse

/-!
# Actual characteristic transport for arbitrary spherical weak laws

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: cauchy.pb`, and
the Lipschitz well-posedness argument, for autonomous law-independent
original fields. The single-neuron ReLU fields are included.

The finite-net comparison applies to both a Lipschitz test and its
negative, hence gives equality with the actual transported initial test.
Affine ReLU tests are Lipschitz and determine every spherical probability,
so this identifies the entire weak law as the original flow pushforward.
Reversing the original equation gives the same identity at negative times.

This proves weak uniqueness for arbitrary initial probabilities. The
original flow solves the actual particle equation, and its Lipschitz
regularity is explicit; weak uniqueness is proved, never hypothesized.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every Lipschitz test has the original characteristic identity along
every actual weak solution, for nonnegative time.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem integral_lipschitz_test_eq_transport (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (M L : ℝ≥0) (T : ℝ) (hT : 0 ≤ T)
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦT : LipschitzWith M (fun x => Φ x T))
    (f : SSphere d → ℝ) (hf : LipschitzWith L f)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) =
      ∫ x : SSphere d, f (Φ x T) ∂(μ 0 : Measure (SSphere d)) := by
  apply le_antisymm (integral_lipschitz_test_le_transport θ F hF hfield K hK M L T hT Φ hΦ hΦ₀ hΦT f hf μ hμ)
  have hn := integral_lipschitz_test_le_transport θ F hF hfield K hK M L T hT Φ hΦ hΦ₀ hΦT (-f) hf.neg μ hμ
  simpa only [Pi.neg_apply, integral_neg, neg_le_neg_iff] using hn

/-- Every determining affine ReLU test is Lipschitz on the original sphere.
Auxiliary for arXiv:2411.04551v3, §1, the weak transport identity. -/
theorem exists_lipschitz_spherical_ridge (u : EucSpace d) (b : ℝ) :
    ∃ L : ℝ≥0, LipschitzWith L (fun x : SSphere d => max (b - inner ℝ (x : EucSpace d) u) 0) := by
  have hu := (innerSL ℝ u).lipschitzWith.comp (LipschitzWith.subtype_val (Sphere d))
  have hh := ((LipschitzWith.const b).sub hu).max_const 0
  refine ⟨‖innerSL ℝ u‖₊, ?_⟩
  simpa only [mul_one, zero_add, Function.comp_def, coe_innerSL_apply, real_inner_comm] using hh

/-- The entire original weak law is its actual characteristic pushforward,
with no atomicity assumption. Source: arXiv:2411.04551v3,
§1, `eq: ips` and `eq: cauchy.pb`, for autonomous original fields. -/
theorem cauchyPB_autonomous_transport_nonneg (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (M : ℝ≥0) (T : ℝ) (hT : 0 ≤ T)
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦT : LipschitzWith M (fun x => Φ x T))
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    (μ T : Measure (SSphere d)) = Measure.map (fun x => Φ x T) (μ 0 : Measure (SSphere d)) := by
  apply measure_eq_of_sphere_relu_ridges
  intro u b
  obtain ⟨L, hL⟩ := exists_lipschitz_spherical_ridge u b
  rw [integral_map hΦT.continuous.measurable.aemeasurable hL.continuous.measurable.aestronglyMeasurable]
  exact integral_lipschitz_test_eq_transport θ F hF hfield K hK M L T hT Φ hΦ hΦ₀ hΦT _ hL μ hμ

/-- Exact original characteristic transport holds for every real time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` and its time reversal. -/
theorem cauchyPB_autonomous_transport (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d))
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦLip : ∀ s, ∃ M : ℝ≥0, LipschitzWith M (fun x => Φ x s))
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) :
    (μ t : Measure (SSphere d)) = Measure.map (fun x => Φ x t) (μ 0 : Measure (SSphere d)) := by
  obtain ⟨M, hM⟩ := hΦLip t
  by_cases ht : 0 ≤ t
  · exact cauchyPB_autonomous_transport_nonneg θ F hF hfield K hK M t ht Φ hΦ hΦ₀ hM μ hμ
  have hfield' (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d) :
      fullVF d (rescaleTimeParams (-1) θ) ρ s x = -F x := by
    rw [fullVF_rescaleTimeParams, hfield, neg_one_smul]
  have hμ' : cauchyPB d (rescaleTimeParams (-1) θ) (fun s => μ (-s)) := by
    simpa only [neg_one_mul] using cauchyPB_rescaleTimeParams (-1) (by norm_num) θ μ hμ
  have hΦ' (x : SSphere d) (s : ℝ) : HasDerivAt (fun r => (Φ x (-r) : EucSpace d)) (-F (Φ x (-s))) s := by
    simpa only [Function.comp_def, neg_smul, one_smul] using
      (hΦ x (-s)).scomp s ((hasDerivAt_id s).neg)
  have hM' : LipschitzWith M (fun x => Φ x (-(-t))) := by simpa only [neg_neg] using hM
  have he := cauchyPB_autonomous_transport_nonneg (rescaleTimeParams (-1) θ) (fun x => -F x)
    hF.neg hfield' K hK.neg M (-t) (by linarith) (fun x s => Φ x (-s)) hΦ'
    (by simpa only [neg_zero] using hΦ₀) hM' (fun s => μ (-s)) hμ'
  simpa only [neg_neg, neg_zero] using he


/-- All actual weak solutions with the same arbitrary initial law agree.
Source: arXiv:2411.04551v3, §1, weak uniqueness for `eq: cauchy.pb`,
restricted to the autonomous law-independent original fields. -/
theorem cauchyPB_autonomous_unique (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d))
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦLip : ∀ s, ∃ M : ℝ≥0, LipschitzWith M (fun x => Φ x s))
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (h0 : μ 0 = ν 0) : μ = ν := by
  funext t
  apply Subtype.ext
  change (μ t : Measure (SSphere d)) = (ν t : Measure (SSphere d))
  rw [cauchyPB_autonomous_transport θ F hF hfield K hK Φ hΦ hΦ₀ hΦLip μ hμ t,
    cauchyPB_autonomous_transport θ F hF hfield K hK Φ hΦ hΦ₀ hΦLip ν hν t, h0]

/-- A nonzero projection neuron, its complete flow and an actual
transported two-atom law jointly realize all original hypotheses.
The determining scalar test is a nonconstant distance function. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2) (K M : ℝ≥0)
    (Φ : SSphere 2 → ℝ → SSphere 2) (μ ν : ℝ → Perspective.ProbSphere 2) (f : SSphere 2 → ℝ),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    (∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace 2)) (F (Φ x s)) s) ∧
    (∀ x, Φ x 0 = x) ∧ (∀ s, ∃ M : ℝ≥0, LipschitzWith M (fun x => Φ x s)) ∧
    LipschitzWith M (fun x => Φ x 1) ∧ LipschitzWith 1 f ∧
    cauchyPB 2 θ μ ∧ cauchyPB 2 θ ν ∧ μ 0 = ν 0 ∧ (0 : ℝ) ≤ 1 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let Φ := ridgeFlow a 0 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let μ := ridgeLaw a 0 1 (halfDirac 2 z (antipode 2 z))
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  obtain ⟨M, hM⟩ := exists_lipschitzWith_ridgeFlow a 0 1 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  have hμ := cauchyPB_ridgeLaw (0 : Fin 2) a 0 1 (halfDirac 2 z (antipode 2 z))
  refine ⟨θ, F, K, M, Φ, μ, μ, fun x => dist x (basePoint 1),
    hF.continuous, hK.mono hsphere, ?_, ?_, ridgeFlow_zero _ _ _,
    exists_lipschitzWith_ridgeFlow _ _ _, hM, LipschitzWith.dist_left _, hμ, hμ, rfl, zero_le_one⟩
  · intro ρ s x
    rw [fullVF_reluRidgeParams]
    simp [F]
  · intro x t
    simpa [F, Φ] using hasDerivAt_ridgeFlow a 0 1 x t

end Transformer.Interpolation
