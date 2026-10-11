/-
# Actual empirical solutions for the generic pure-attention parameters

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: vf` and `eq: cauchy.pb`,
and §5, `prop:generic-discrete`; arXiv:2312.10794v5, §3.1,
`eq: mean.field.ips`.

For `V = I`, `B = β I` and `W = 0`, the full interpolation field is
exactly the spherical attention field. A finite average of the actual
particle trajectories satisfies the weak continuity equation.
The test-function flux is continuous in time, so it is integrable on
every finite interval, and the fundamental theorem of calculus gives
the integrated equation used by the interpolation development.

Thus the conditional cluster hypothesis can be applied to an actual
probability-measure solution from every nonempty initial tuple.
-/

import Transformer.Interpolation.Basic
import Transformer.Perspective.Section3_EmpiricalMeasures
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Interpolation

variable {d n : ℕ}

/-- The parameters in arXiv:2411.04551v3, §5, `prop:generic-discrete`.
The unused perceptron parameters `U` and `b` are chosen to be zero;
their values do not affect the source's field because `W = 0`. -/
noncomputable def pureAttentionParams (d : ℕ) (β : ℝ) : TimeParams d :=
  fun _ =>
    { V := ContinuousLinearMap.id ℝ (EucSpace d)
      B := β • ContinuousLinearMap.id ℝ (EucSpace d)
      W := 0
      U := 0
      b := 0 }

/-- The generic pure-attention specialization of `eq: vf` is exactly the
mean-field attention velocity of arXiv:2312.10794v5, §3.1, `eq: vfSd`.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem fullVF_pureAttentionParams (β : ℝ) (μ : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) :
    fullVF d (pureAttentionParams d β) μ t x = Perspective.vectorField d β μ x := by
  simp only [fullVF, pureAttentionParams, smul_apply, ContinuousLinearMap.id_apply,
    zero_apply, add_zero, real_inner_smul_left, Perspective.vectorField,
    Perspective.partitionMu]

/-- The empirical average of a `C¹` test function has the actual weak
continuity-equation derivative along the attention particles.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §5,
`prop:generic-discrete`; arXiv:2312.10794v5, §3.1, `eq: mean.field.ips`. -/
theorem hasDerivAt_empirical_test (hn : 0 < n) (β : ℝ)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d)
      ∂(Perspective.empiricalProb hn (X s) : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
        (fullVF d (pureAttentionParams d β) (Perspective.empiricalProb hn (X t)) t
          (x : EucSpace d))
        ∂(Perspective.empiricalProb hn (X t) : Measure (SSphere d))) t := by
  have htoken (i : Idx n) : HasDerivAt (fun s => φ (X s i : EucSpace d))
      (inner (𝕜 := ℝ) (gradient φ (X t i : EucSpace d))
        (Perspective.saField d n β (fun j => (X t j : EucSpace d)) i)) t := by
    rw [gradient, InnerProductSpace.toDual_symm_apply]
    exact (hφ.differentiable (by simp) _).hasFDerivAt.comp_hasDerivAt t
      ((Perspective.SA_iff d n β X).mp hX t i)
  have hsum := (HasDerivAt.fun_sum (fun i (_ : i ∈ Finset.univ) => htoken i)).const_mul
    ((n : ℝ)⁻¹)
  simpa only [Perspective.integral_empiricalProb, smul_eq_mul, fullVF_pureAttentionParams,
    Perspective.vectorField_empiricalProb_eq_saField] using hsum

/-- The ODE and regularity hypotheses hold for a nonzero stationary unit
token and the nonzero constant test function. -/
example : (0 : ℕ) < 1 ∧
    Perspective.SA 1 1 1 (fun _ _ => basePoint 0) ∧
    ContDiff ℝ 1 (fun _ : EucSpace 1 => (1 : ℝ)) :=
  ⟨one_pos, Perspective.SA_const_consensus 1 1 one_pos 1 (basePoint 0), contDiff_const⟩

/-- The actual empirical test-function flux is continuous in time.
Source: arXiv:2411.04551v3, §1, the time integral in `eq: cauchy.pb`,
specialized to the pure-attention parameters of §5. -/
theorem continuous_empirical_flux (hn : 0 < n) (β : ℝ)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun t => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d (pureAttentionParams d β) (Perspective.empiricalProb hn (X t)) t
        (x : EucSpace d))
      ∂(Perspective.empiricalProb hn (X t) : Measure (SSphere d))) := by
  have hc : Continuous (fun t i => (X t i : EucSpace d)) := by
    apply continuous_pi_iff.mpr
    intro i
    exact continuous_iff_continuousAt.mpr fun t =>
      ((Perspective.SA_iff d n β X).mp hX t i).continuousAt
  simp only [Perspective.integral_empiricalProb, smul_eq_mul, fullVF_pureAttentionParams,
    Perspective.vectorField_empiricalProb_eq_saField, gradient,
    InnerProductSpace.toDual_symm_apply]
  apply Continuous.const_mul
  apply continuous_finsetSum
  intro i _
  exact (hφ.continuous_fderiv_apply (by simp)).comp
    (((continuous_apply i).comp hc).prodMk
      ((continuous_apply i).comp ((Perspective.contDiff_saField β).continuous.comp hc)))

example : (0 : ℕ) < 1 ∧
    Perspective.SA 1 1 0 (fun _ _ => basePoint 0) ∧
    ContDiff ℝ 1 (fun _ : EucSpace 1 => (1 : ℝ)) :=
  ⟨one_pos, Perspective.SA_const_consensus 1 1 one_pos 0 (basePoint 0), contDiff_const⟩

/-- **The actual empirical attention trajectory solves the integrated
continuity equation.** Its time flux is integrable by continuity.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §5,
`prop:generic-discrete`. -/
theorem cauchyPB_empirical_of_SA (hn : 0 < n) (β : ℝ)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X) :
    cauchyPB d (pureAttentionParams d β) (fun t => Perspective.empiricalProb hn (X t)) := by
  intro φ hφ t
  have hi := (continuous_empirical_flux hn β X hX φ hφ).intervalIntegrable (μ := volume) 0 t
  refine ⟨hi, ?_⟩
  have heq := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_empirical_test hn β X hX φ hφ s) hi
  linarith only [heq]

example (β : ℝ) (x : SSphere 1) :
    (0 : ℕ) < 1 ∧ Perspective.SA 1 1 β (fun _ _ => x) :=
  ⟨one_pos, Perspective.SA_const_consensus 1 1 one_pos β x⟩

/-- **There is an actual pure-attention probability-measure solution from
every nonempty empirical input.** This supplies the solution quantified
over by the cluster-map hypothesis of arXiv:2411.04551v3, §5,
`prop:generic-discrete`, using the particle equation of §1. -/
theorem exists_cauchyPB_pureAttention (hn : 0 < n) (β : ℝ) (X₀ : SphereTuple d n) :
    ∃ μ : ℝ → Perspective.ProbSphere d,
      (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d n X₀ ∧
      cauchyPB d (pureAttentionParams d β) μ := by
  refine ⟨fun t => Perspective.empiricalProb hn (Perspective.saFlow β X₀ t), ?_, ?_⟩
  · simp only [Perspective.saFlow_zero, Perspective.coe_empiricalProb]
  · exact cauchyPB_empirical_of_SA hn β _ (Perspective.saFlow_spec β X₀).2

example : (0 : ℕ) < 1 := one_pos

/-- A stationary unit Dirac trajectory is an actual integrated solution,
so the construction includes the one-token case used in examples.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §5,
`prop:generic-discrete`; arXiv:2312.10794v5, §2, `SA`. -/
theorem cauchyPB_pureAttention_dirac (β : ℝ) (x : SSphere d) :
    cauchyPB d (pureAttentionParams d β) (fun _ => Perspective.diracProb d x) := by
  simpa only [Perspective.empiricalProb_one] using
    cauchyPB_empirical_of_SA (n := 1) one_pos β (fun _ _ => x)
      (Perspective.SA_const_consensus d 1 one_pos β x)

end Transformer.Interpolation
