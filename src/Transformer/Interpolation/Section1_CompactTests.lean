module

public import Transformer.Interpolation.Section1_MovingTests
public import Mathlib.Topology.UniformSpace.HeineCantor
public import Mathlib.Topology.ContinuousMap.Compact

/-!
# Uniform differentiation of tests on the compact sphere

Auxiliary to arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and
§3.1, `lem: first.quadrant`. A jointly continuous time derivative gives
differentiability in the supremum norm on a compact spatial domain.
Uniform continuity on a compact time strip and the mean value theorem
prove the required uniform remainder estimate. This supplies the strong
test derivative needed by the weak product rule, rather than assuming it
as a new condition on the original measure-valued solution.
-/

@[expose] public section

open Real MeasureTheory Filter Topology Asymptotics

namespace Transformer.Interpolation

section CompactTests

variable {X : Type*} [MetricSpace X] [CompactSpace X]

/-- A continuous test on the original compact spatial domain, regarded
with its actual supremum norm. Source: arXiv:2411.04551v3, §1,
the tests in `eq: cauchy.pb`. -/
noncomputable def boundedTestCurve (F : ℝ → X → ℝ)
    (hF : Continuous (Function.uncurry F)) (t : ℝ) : BoundedContinuousFunction X ℝ :=
  BoundedContinuousFunction.mkOfCompact
    ⟨F t, hF.comp (continuous_const.prodMk continuous_id)⟩

/-- Evaluation retains the actual original test.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
@[simp] theorem boundedTestCurve_apply (F : ℝ → X → ℝ)
    (hF : Continuous (Function.uncurry F)) (t : ℝ) (x : X) :
    boundedTestCurve F hF t x = F t x := rfl

/-- Joint continuity alone gives continuity in the actual supremum norm
on the compact spatial domain. Source: arXiv:2411.04551v3, §1,
the weak tests in `eq: cauchy.pb`. -/
theorem continuous_boundedTestCurve_of_continuous (F : ℝ → X → ℝ)
    (hF : Continuous (Function.uncurry F)) : Continuous (boundedTestCurve F hF) := by
  have hi : Isometry (BoundedContinuousFunction.mkOfCompact : C(X, ℝ) →
      BoundedContinuousFunction X ℝ) :=
    isometry_iff_dist_eq.mpr fun f g => BoundedContinuousFunction.dist_mkOfCompact f g
  exact hi.continuous.comp (ContinuousMap.curry ⟨Function.uncurry F, hF⟩).continuous

/-- Joint continuity of the pointwise time derivative gives the derivative
in the supremum norm on a compact domain. Source: arXiv:2411.04551v3,
§1, the time-dependent weak tests used for characteristics in §3.1. -/
theorem hasDerivAt_boundedTestCurve (F G : ℝ → X → ℝ)
    (hF : Continuous (Function.uncurry F)) (hG : Continuous (Function.uncurry G))
    (hD : ∀ t x, HasDerivAt (fun s => F s x) (G t x) t) (t : ℝ) :
    HasDerivAt (boundedTestCurve F hF) (boundedTestCurve G hG t) t := by
  apply HasDerivAt.of_isLittleO
  rw [isLittleO_iff]
  intro ε hε
  have hcompact : IsCompact (Set.Icc (t - 1) (t + 1) ×ˢ (Set.univ : Set X)) :=
    isCompact_Icc.prod isCompact_univ
  obtain ⟨δ, hδ, hδG⟩ := Metric.uniformContinuousOn_iff.mp
    (hcompact.uniformContinuousOn_of_continuous hG.continuousOn) ε hε
  let ρ := min δ 1
  have hρ : 0 < ρ := lt_min hδ zero_lt_one
  have hbound : ∀ r ∈ Metric.ball t ρ, ∀ x : X, ‖G r x - G t x‖ ≤ ε := by
    intro r hr x
    have hrabs : |r - t| < ρ := by simpa only [Metric.mem_ball, Real.dist_eq] using hr
    have hrI : r ∈ Set.Icc (t - 1) (t + 1) := by
      have hρ1 : ρ ≤ 1 := min_le_right δ 1
      have hh := abs_lt.mp hrabs
      constructor <;> linarith
    have htI : t ∈ Set.Icc (t - 1) (t + 1) := ⟨by linarith, by linarith⟩
    have hdist : dist (r, x) (t, x) < δ := by
      rw [Prod.dist_eq, dist_self, max_lt_iff]
      exact ⟨by simpa only [Real.dist_eq] using hrabs.trans_le (min_le_left δ 1), hδ⟩
    have hh := hδG (r, x) ⟨hrI, Set.mem_univ x⟩ (t, x) ⟨htI, Set.mem_univ x⟩ hdist
    exact (by simpa only [Function.uncurry_apply_pair, Real.dist_eq, Real.norm_eq_abs] using hh.le)
  filter_upwards [Metric.ball_mem_nhds t hρ] with s hs
  apply (BoundedContinuousFunction.norm_le (mul_nonneg hε.le (norm_nonneg (s - t)))).mpr
  intro x
  change ‖F s x - F t x - (s - t) * G t x‖ ≤ ε * ‖s - t‖
  have hd : ∀ r ∈ Metric.ball t ρ,
      HasDerivWithinAt (fun u => F u x - (u - t) * G t x)
        (G r x - G t x) (Metric.ball t ρ) r := by
    intro r hr
    simpa only [id_eq, one_mul] using ((hD r x).fun_sub
      (((hasDerivAt_id r).sub_const t).mul_const (G t x))).hasDerivWithinAt
  have hm := (convex_ball t ρ).norm_image_sub_le_of_norm_hasDerivWithin_le hd
    (fun r hr => hbound r hr x) (Metric.mem_ball_self hρ) hs
  calc
    _ = ‖(F s x - (s - t) * G t x) - (F t x - (t - t) * G t x)‖ := by
      congr 1
      ring
    _ ≤ _ := hm

/-- The original continuous test curve is strongly continuous in its
supremum norm. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_boundedTestCurve (F G : ℝ → X → ℝ)
    (hF : Continuous (Function.uncurry F)) (hG : Continuous (Function.uncurry G))
    (hD : ∀ t x, HasDerivAt (fun s => F s x) (G t x) t) :
    Continuous (boundedTestCurve F hF) :=
  continuous_iff_continuousAt.mpr fun t =>
    (hasDerivAt_boundedTestCurve F G hF hG hD t).continuousAt

end CompactTests

/-- A genuine spacetime test and derivative satisfy all uniform-derivative
inputs on a nonempty compact spatial interval; both variables are used. -/
example :
    Continuous (fun q : ℝ × Set.Icc (0 : ℝ) 1 => sin q.1 * (q.2 : ℝ)) ∧
      Continuous (fun q : ℝ × Set.Icc (0 : ℝ) 1 => cos q.1 * (q.2 : ℝ)) ∧
      ∀ t : ℝ, ∀ x : Set.Icc (0 : ℝ) 1,
        HasDerivAt (fun s => sin s * (x : ℝ)) (cos t * (x : ℝ)) t := by
  refine ⟨(continuous_sin.comp continuous_fst).mul
    (continuous_subtype_val.comp continuous_snd),
    (continuous_cos.comp continuous_fst).mul
      (continuous_subtype_val.comp continuous_snd), ?_⟩
  exact fun t x => (hasDerivAt_sin t).mul_const (x : ℝ)

variable {d : ℕ}

/-- Integration of a jointly continuous compact test against one original
probability law is continuous in time. Source: arXiv:2411.04551v3,
§1, `eq: cauchy.pb`. -/
theorem continuous_spherical_integral_of_joint_continuous (μ : Perspective.ProbSphere d)
    (F : ℝ → SSphere d → ℝ) (hF : Continuous (Function.uncurry F)) :
    Continuous (fun s => ∫ x, F s x ∂(μ : Measure (SSphere d))) :=
  (sphericalIntegralCLM μ).continuous.comp (continuous_boundedTestCurve_of_continuous F hF)

/-- The actual integral of a moving compact test is continuous whenever
the fixed test is weakly continuous. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`; this also retains continuity at switching times. -/
theorem continuousAt_spherical_compact_test (μ : ℝ → Perspective.ProbSphere d)
    (F G : ℝ → SSphere d → ℝ) (hF : Continuous (Function.uncurry F))
    (hG : Continuous (Function.uncurry G))
    (hD : ∀ t x, HasDerivAt (fun s => F s x) (G t x) t) (t : ℝ)
    (hfixed : ContinuousAt (fun s => ∫ x, F t x ∂(μ s : Measure (SSphere d))) t) :
    ContinuousAt (fun s => ∫ x, F s x ∂(μ s : Measure (SSphere d))) t :=
  continuousAt_bounded_weak_pairing (fun s => sphericalIntegralCLM (μ s))
    (fun s => norm_sphericalIntegralCLM_le (μ s)) (boundedTestCurve F hF) t
    (hasDerivAt_boundedTestCurve F G hF hG hD t).continuousAt hfixed

/-- The moving-test derivative for actual spherical probabilities now
needs only the pointwise derivative and its joint continuity; compactness
supplies the required supremum-norm derivative.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. -/
theorem hasDerivAt_spherical_compact_test (μ : ℝ → Perspective.ProbSphere d)
    (F G : ℝ → SSphere d → ℝ) (hF : Continuous (Function.uncurry F))
    (hG : Continuous (Function.uncurry G))
    (hD : ∀ t x, HasDerivAt (fun s => F s x) (G t x) t) (t a : ℝ)
    (hfixed : HasDerivAt (fun s => ∫ x, F t x ∂(μ s : Measure (SSphere d))) a t)
    (hweak : ContinuousAt (fun s => ∫ x, G t x ∂(μ s : Measure (SSphere d))) t) :
    HasDerivAt (fun s => ∫ x, F s x ∂(μ s : Measure (SSphere d)))
      (a + ∫ x, G t x ∂(μ t : Measure (SSphere d))) t :=
  hasDerivAt_spherical_moving_test μ (boundedTestCurve F hF) (boundedTestCurve G hG t)
    t a (hasDerivAt_boundedTestCurve F G hF hG hD t) hfixed hweak

/-- The fixed-test and weak-continuity inputs are simultaneously realized
by a nonzero spherical probability law, at the nonconstant time test
`F(t,x)=sin(t)`. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
example :
    HasDerivAt (fun s : ℝ => ∫ _x : SSphere 2, sin s
      ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) 1 0 := by
  simpa using hasDerivAt_spherical_compact_test (fun _ => Perspective.diracProb 2 (basePoint 1))
    (fun s _ => sin s) (fun s _ => cos s)
    (continuous_sin.comp continuous_fst) (continuous_cos.comp continuous_fst)
    (fun t _ => hasDerivAt_sin t) 0 0
    (by simpa using hasDerivAt_const (0 : ℝ) (0 : ℝ)) continuousAt_const

end Transformer.Interpolation
