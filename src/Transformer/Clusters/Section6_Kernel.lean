/-
# The emergence of clusters in self-attention dynamics — the attention kernel

`e:vectorfield` and `lem: vectorfield.properties` of arXiv:2305.05465v6, §6:
the velocity field driving the continuity equation `e:conteq`, and the three
estimates the well-posedness proof rests on.

**What the source says and what is carried here.**

* `e:vectorfield` is the ratio of two integrals against `μ`; the numerator is
  vector-valued, so the quotient is written as the scalar inverse of the
  denominator acting on the numerator.

* `𝒫_c(ℝ^d)`, compactly supported probability measures, is carried as
  `IsProbabilityMeasure μ` together with `μ (B̄(0,R))ᶜ = 0` for the radius `R`
  that each estimate names anyway.  This avoids `Measure.support` and says
  exactly what the proofs use.

* `e:bddinx` and `e:lipinx` are stated in the source as `L^∞(ℝ^d)` bounds,
  that is for every `x ∈ ℝ^d`; `e:lipinmu` only for `‖x‖ ≤ R`, and the
  source's marginal note insists on that restriction.  Both are carried as
  written.

* `∇_x 𝒳[μ]` is the Fréchet derivative, and the bound is on its operator
  norm; existence of the derivative is part of the statement, since
  `e:lipinx` asserts a bound on an object the source does not separately
  construct.  `‖QᵀK‖_op` is the operator norm of `Q* ∘ K`.

* `W_2` is `Transformer.Wasserstein.W2`.

* `e:lipinx` is in `Section6_KernelDeriv`, `e:lipinmu` in
  `Section6_KernelLip`.

Source: arXiv:2305.05465v6, `e:vectorfield`, `lem: vectorfield.properties`,
`e:bddinx`, `e:lipinx`, `e:lipinmu`.
-/

import Transformer.Clusters.Section1_Dynamics
import Transformer.Wasserstein
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.Analysis.InnerProductSpace.Adjoint

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- **`μ` is carried by the ball `B̄(0,R)`**, the form in which compact support
enters `lem: vectorfield.properties`. -/
def IsCarriedBy (μ : Measure (EucSpace d)) (R : ℝ) : Prop :=
  μ (Metric.closedBall 0 R)ᶜ = 0

/-- The Dirac mass at the origin is carried by every ball of positive radius. -/
theorem isCarriedBy_dirac (R : ℝ) (hR : 0 ≤ R) :
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) R := by
  have hmem : (0 : EucSpace d) ∈ Metric.closedBall (0 : EucSpace d) R := by
    simpa using hR
  rw [IsCarriedBy, Measure.dirac_apply' _ (measurableSet_closedBall.compl)]
  simp [hmem]

/-- **Equation (e:vectorfield), the attention kernel.**

  `𝒳[μ](x) = ∫ e^{⟨Qx, Ky⟩} V y dμ(y) / ∫ e^{⟨Qx, Ky⟩} dμ(y)`.

Source: arXiv:2305.05465v6, `e:vectorfield`. -/
noncomputable def attentionKernel (Q K V : ParamMatrix d) (μ : Measure (EucSpace d))
    (x : EucSpace d) : EucSpace d :=
  (∫ y, Real.exp (inner (𝕜 := ℝ) (Q x) (K y)) ∂μ)⁻¹ •
    ∫ y, Real.exp (inner (𝕜 := ℝ) (Q x) (K y)) • V y ∂μ

/-- **The kernel vanishes at a Dirac mass at the origin**: the only value
averaged is `V 0 = 0`.  This is the stationary solution of `e:conteq`. -/
theorem attentionKernel_dirac_zero (Q K V : ParamMatrix d) (x : EucSpace d) :
    attentionKernel Q K V (Measure.dirac (0 : EucSpace d)) x = 0 := by
  simp [attentionKernel]

/-- A measure carried by `B̄(0,R)` puts almost every point in it. -/
theorem ae_mem_closedBall {μ : Measure (EucSpace d)} {R : ℝ} (hsupp : IsCarriedBy μ R) :
    ∀ᵐ y ∂μ, y ∈ Metric.closedBall (0 : EucSpace d) R :=
  ae_iff.2 hsupp

/-- The attention weight `y ↦ e^{⟨Qx, Ky⟩}` is integrable against a finite measure
carried by a ball, on which it is bounded. -/
theorem integrable_attentionWeight (Q K : ParamMatrix d) (R : ℝ)
    (μ : Measure (EucSpace d)) [IsFiniteMeasure μ] (hsupp : IsCarriedBy μ R) (x : EucSpace d) :
    Integrable (fun y => Real.exp (inner (𝕜 := ℝ) (Q x) (K y))) μ := by
  refine Integrable.of_bound (by fun_prop) (Real.exp (‖Q x‖ * (‖K‖ * R))) ?_
  filter_upwards [ae_mem_closedBall hsupp] with y hy
  rw [Real.norm_of_nonneg (Real.exp_pos _).le, Real.exp_le_exp]
  have hy' : ‖y‖ ≤ R := by simpa using hy
  calc inner (𝕜 := ℝ) (Q x) (K y) ≤ ‖Q x‖ * ‖K y‖ := real_inner_le_norm _ _
    _ ≤ ‖Q x‖ * (‖K‖ * R) := mul_le_mul_of_nonneg_left
        ((K.le_opNorm y).trans (mul_le_mul_of_nonneg_left hy' (norm_nonneg _))) (norm_nonneg _)

/-- **Estimate (e:bddinx).**  `‖𝒳[μ]‖_{L^∞} ≤ ‖V‖_op R`.

As in the source: `‖V y‖ ≤ ‖V‖_op R` on the ball, and `𝒳[μ](x)` is an
average of `V y` against the weights `e^{⟨Qx, Ky⟩}`.

Source: arXiv:2305.05465v6, `e:bddinx`. -/
theorem attentionKernel_norm_le (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R)
    (μ : Measure (EucSpace d)) (hμ : IsProbabilityMeasure μ) (hsupp : IsCarriedBy μ R)
    (x : EucSpace d) :
    ‖attentionKernel Q K V μ x‖ ≤ ‖V‖ * R := by
  set w := fun y => Real.exp (inner (𝕜 := ℝ) (Q x) (K y))
  have hw := integrable_attentionWeight Q K R μ hsupp x
  have hN : ‖∫ y, w y • V y ∂μ‖ ≤ ‖V‖ * R * ∫ y, w y ∂μ := by
    rw [← integral_const_mul]
    refine norm_integral_le_of_norm_le (hw.const_mul _) ?_
    filter_upwards [ae_mem_closedBall hsupp] with y hy
    have hy' : ‖y‖ ≤ R := by simpa using hy
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
    exact mul_le_mul_of_nonneg_right ((V.le_opNorm y).trans
      (mul_le_mul_of_nonneg_left hy' (norm_nonneg _))) (Real.exp_pos _).le
  have hVR : 0 ≤ ‖V‖ * R := mul_nonneg (norm_nonneg _) hR.le
  rw [attentionKernel, norm_smul, Real.norm_eq_abs]
  rcases (integral_nonneg fun y => (Real.exp_pos _).le : 0 ≤ ∫ y, w y ∂μ).eq_or_lt with h | h
  · rw [← h]; simpa using hVR
  · rw [abs_of_pos (inv_pos.2 h), inv_mul_le_iff₀ h, mul_comm (∫ y, w y ∂μ)]
    exact hN

/-- The hypotheses of `e:bddinx` are satisfiable: the Dirac mass at
the origin is a probability measure carried by the unit ball. -/
example : (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one⟩

end Clusters
end Transformer
