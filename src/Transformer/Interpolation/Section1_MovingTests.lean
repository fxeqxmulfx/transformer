/-
# Differentiating a moving test against a weakly varying measure

Auxiliary calculus for arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and
§3.1, `lem: first.quadrant`. A weak equation differentiates fixed tests.
The test pulled back by an inverse characteristic also varies in time.
The product rule below needs weak continuity of only its time derivative,
and the probability bound on the integral functional. It never assumes
continuity in the operator norm of the measure-valued curve.
-/

import Transformer.Interpolation.Section1_WeakContinuity
import Mathlib.Analysis.Asymptotics.Ring

open Real MeasureTheory Filter Topology Asymptotics

namespace Transformer.Interpolation

/-- Strong continuity of a test and weak continuity of its fixed value give
continuity of their pairing. Source: arXiv:2411.04551v3, §1,
the moving weak tests used in `eq: cauchy.pb`. -/
theorem continuousAt_bounded_weak_pairing {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (L : ℝ → E →L[ℝ] ℝ) (hL : ∀ s, ‖L s‖ ≤ 1)
    (f : ℝ → E) (t : ℝ) (hf : ContinuousAt f t)
    (hfixed : ContinuousAt (fun s => L s (f t)) t) :
    ContinuousAt (fun s => L s (f s)) t := by
  have herr : Tendsto (fun s => L s (f s - f t)) (nhds t) (nhds 0) := by
    apply squeeze_zero_norm (fun s => (L s).le_opNorm _ |>.trans
      (mul_le_of_le_one_left (norm_nonneg _) (hL s)))
    have hconst : Tendsto (fun _ : ℝ => f t) (nhds t) (nhds (f t)) := tendsto_const_nhds
    simpa only [sub_self, norm_zero] using (hf.tendsto.sub hconst).norm
  have ht := herr.add hfixed.tendsto
  change Tendsto (fun s => L s (f s)) (nhds t) (nhds (L t (f t)))
  simpa only [map_sub, sub_add_cancel, zero_add] using ht

/-- A product rule for uniformly bounded, weakly varying functionals.
The first term differentiates the fixed test; the second differentiates
the moving test. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`,
as used for characteristic transport in §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_bounded_weak_pairing {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (L : ℝ → E →L[ℝ] ℝ) (hL : ∀ s, ‖L s‖ ≤ 1)
    (f : ℝ → E) (v : E) (t a : ℝ) (hf : HasDerivAt f v t)
    (hfixed : HasDerivAt (fun s => L s (f t)) a t)
    (hweak : ContinuousAt (fun s => L s v) t) :
    HasDerivAt (fun s => L s (f s)) (a + L t v) t := by
  let R (s : ℝ) := f s - f t - (s - t) • v
  have hR : (fun s => L s (R s)) =o[nhds t] fun s => s - t := by
    apply (IsBigO.of_bound' (Filter.Eventually.of_forall fun s =>
      (L s).le_opNorm (R s) |>.trans
        (mul_le_of_le_one_left (norm_nonneg _) (hL s)))).trans_isLittleO
    exact hf.isLittleO
  have hsmall : (fun s => L s v - L t v) =o[nhds t] fun _ => (1 : ℝ) :=
    (isLittleO_one_iff ℝ).mpr (by
      have hconst : Tendsto (fun _ : ℝ => L t v) (nhds t) (nhds (L t v)) :=
        tendsto_const_nhds
      simpa only [sub_self] using (hweak.tendsto.sub hconst))
  have hcross : (fun s => (s - t) * (L s v - L t v)) =o[nhds t] fun s => s - t := by
    simpa only [mul_one] using (isBigO_refl (fun s : ℝ => s - t) (nhds t)).mul_isLittleO hsmall
  apply HasDerivAt.of_isLittleO
  apply ((hR.add hfixed.isLittleO).add hcross).congr_left
  intro s
  simp only [R, map_sub, map_smul, smul_eq_mul]
  ring

/-- Both the test and the bounded functional may vary: multiplication by
`cos(s)` acts on the test `f(s)=s` at time zero. This realizes every input
of the weak product rule with nonconstant data. -/
example :
    (∀ s : ℝ, ‖cos s • ContinuousLinearMap.id ℝ ℝ‖ ≤ 1) ∧
      HasDerivAt (fun s : ℝ => s) 1 0 ∧
      HasDerivAt (fun s : ℝ => (cos s • ContinuousLinearMap.id ℝ ℝ) (0 : ℝ)) 0 0 ∧
      ContinuousAt (fun s : ℝ => (cos s • ContinuousLinearMap.id ℝ ℝ) (1 : ℝ)) 0 := by
  refine ⟨fun s => ?_, hasDerivAt_id 0, ?_, ?_⟩
  · apply ContinuousLinearMap.opNorm_le_bound _ zero_le_one
    intro x
    change |cos s * x| ≤ 1 * |x|
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right (abs_cos_le_one s) (abs_nonneg x)
  · simpa using hasDerivAt_const (0 : ℝ) (0 : ℝ)
  · simpa using continuous_cos.continuousAt (x := (0 : ℝ))

variable {d : ℕ}

/-- Integration against the original spherical probability law is a
bounded linear functional on bounded continuous tests.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def sphericalIntegralCLM (μ : Perspective.ProbSphere d) :
    BoundedContinuousFunction (SSphere d) ℝ →L[ℝ] ℝ :=
  LinearMap.mkContinuous
    { toFun := fun f : BoundedContinuousFunction (SSphere d) ℝ =>
        ∫ x, f x ∂(μ : Measure (SSphere d))
      map_add' := fun f g => integral_add (μ := (μ : Measure (SSphere d)))
        (f.integrable (μ := (μ : Measure (SSphere d))))
        (g.integrable (μ := (μ : Measure (SSphere d))))
      map_smul' := fun c f => by
        simpa only [BoundedContinuousFunction.coe_smul, smul_eq_mul, RingHom.id_apply]
          using integral_smul c (f : SSphere d → ℝ) }
    1 (fun f => by
      change ‖∫ x, f x ∂(μ : Measure (SSphere d))‖ ≤ 1 * ‖f‖
      simpa only [probReal_univ, mul_one, one_mul] using
        norm_integral_le_of_norm_le_const
          (ae_of_all (μ : Measure (SSphere d)) fun x => f.norm_coe_le_norm x))

/-- The functional evaluates to the actual original measure integral.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
@[simp] theorem sphericalIntegralCLM_apply (μ : Perspective.ProbSphere d)
    (f : BoundedContinuousFunction (SSphere d) ℝ) :
    sphericalIntegralCLM μ f = ∫ x, f x ∂(μ : Measure (SSphere d)) := rfl

/-- Constant tests retain their value under the original probability law.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
@[simp] theorem sphericalIntegralCLM_const (μ : Perspective.ProbSphere d) (c : ℝ) :
    sphericalIntegralCLM μ (BoundedContinuousFunction.const (SSphere d) c) = c := by
  simp [sphericalIntegralCLM_apply]

/-- Its norm is at most one, uniformly over the original probabilities.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem norm_sphericalIntegralCLM_le (μ : Perspective.ProbSphere d) :
    ‖sphericalIntegralCLM μ‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ zero_le_one
  intro f
  simpa only [sphericalIntegralCLM_apply, probReal_univ, mul_one, one_mul] using
    norm_integral_le_of_norm_le_const
      (ae_of_all (μ : Measure (SSphere d)) fun x => f.norm_coe_le_norm x)

/-- The moving-test formula for actual spherical probabilities.
No differentiability of the measure in total variation is required.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the characteristic transport in `lem: first.quadrant`. -/
theorem hasDerivAt_spherical_moving_test (μ : ℝ → Perspective.ProbSphere d)
    (f : ℝ → BoundedContinuousFunction (SSphere d) ℝ)
    (v : BoundedContinuousFunction (SSphere d) ℝ) (t a : ℝ) (hf : HasDerivAt f v t)
    (hfixed : HasDerivAt (fun s => ∫ x, f t x ∂(μ s : Measure (SSphere d))) a t)
    (hweak : ContinuousAt (fun s => ∫ x, v x ∂(μ s : Measure (SSphere d))) t) :
    HasDerivAt (fun s => ∫ x, f s x ∂(μ s : Measure (SSphere d)))
      (a + ∫ x, v x ∂(μ t : Measure (SSphere d))) t :=
  hasDerivAt_bounded_weak_pairing (fun s => sphericalIntegralCLM (μ s))
    (fun s => norm_sphericalIntegralCLM_le (μ s)) f v t a hf hfixed hweak

/-- The moving-test hypotheses hold for a genuinely varying scalar test
against a nonzero actual probability law. Source: arXiv:2411.04551v3,
§1, `eq: cauchy.pb`. -/
example :
    HasDerivAt (fun s : ℝ => sin s • BoundedContinuousFunction.const (SSphere 2) (1 : ℝ))
      (BoundedContinuousFunction.const (SSphere 2) (1 : ℝ)) 0 ∧
      HasDerivAt (fun _ : ℝ => ∫ _x : SSphere 2, (0 : ℝ)
        ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) 0 0 ∧
      ContinuousAt (fun _ : ℝ => ∫ _x : SSphere 2, (1 : ℝ)
        ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) 0 := by
  refine ⟨?_, ?_, continuousAt_const⟩
  · simpa using (hasDerivAt_sin 0).smul_const
      (BoundedContinuousFunction.const (SSphere 2) (1 : ℝ))
  · simpa using hasDerivAt_const (0 : ℝ) (0 : ℝ)

end Transformer.Interpolation
