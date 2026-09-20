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

/-- **Estimate (e:bddinx).**  `‖𝒳[μ]‖_{L^∞} ≤ ‖V‖_op R`.

Not proved here.

Source: arXiv:2305.05465v6, `e:bddinx`. -/
theorem attentionKernel_norm_le (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R)
    (μ : Measure (EucSpace d)) (hμ : IsProbabilityMeasure μ) (hsupp : IsCarriedBy μ R)
    (x : EucSpace d) :
    ‖attentionKernel Q K V μ x‖ ≤ ‖V‖ * R := by
  sorry

/-- **Estimate (e:lipinx).**  `𝒳[μ]` is differentiable in `x` with
`‖∇_x 𝒳[μ]‖_{L^∞} ≤ 2 ‖QᵀK‖_op ‖V‖_op R²`.

Not proved here.

Source: arXiv:2305.05465v6, `e:lipinx`. -/
theorem attentionKernel_hasFDerivAt (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R)
    (μ : Measure (EucSpace d)) (hμ : IsProbabilityMeasure μ) (hsupp : IsCarriedBy μ R)
    (x : EucSpace d) :
    ∃ D : EucSpace d →L[ℝ] EucSpace d,
      HasFDerivAt (attentionKernel Q K V μ) D x ∧
        ‖D‖ ≤ 2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R ^ 2 := by
  sorry

/-- **Estimate (e:lipinmu).**  On `B(0,R)` the kernel is Lipschitz in the
measure for `W_2`, with a constant depending only on `R`.

Not proved here.

Source: arXiv:2305.05465v6, `e:lipinmu`. -/
theorem attentionKernel_lipschitz_in_measure (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R) :
    ∃ C : ℝ, 0 < C ∧ ∀ μ ν : Measure (EucSpace d), IsProbabilityMeasure μ →
      IsProbabilityMeasure ν → IsCarriedBy μ R → IsCarriedBy ν R →
        ∀ x ∈ Metric.closedBall (0 : EucSpace d) R,
          ‖attentionKernel Q K V μ x - attentionKernel Q K V ν x‖
            ≤ C * Wasserstein.W2 μ ν := by
  sorry

/-- The hypotheses of the three estimates are satisfiable: the Dirac mass at
the origin is a probability measure carried by the unit ball. -/
example : (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one⟩

end Clusters
end Transformer
