/-
# The emergence of clusters in self-attention dynamics — the continuity
  equation

`c:wellposedtransformers` of arXiv:2305.05465v6, §6: global well-posedness of
`e:conteq`, the mean-field form of the dynamics, with a `W_2` stability
estimate.

**What the source says and what is carried here.**

* `e:conteq` and its solutions (`d:solconti`) are `IsContEqSolution`, in
  `Section6_WeakSolution`, where the way they are carried is explained.

* `c:wellposedtransformers` is split in two: existence and uniqueness, and the
  stability estimate `e:w2estimate`, which quantifies over pairs of solutions.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`,
`c:wellposedtransformers`, `e:w2estimate`.
-/

import Transformer.Clusters.Section6_WeakSolution

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- **Proposition (c:wellposedtransformers), existence and uniqueness.**  Every
compactly supported probability measure is the initial datum of exactly one
solution of `e:conteq`.

Not proved here.

Source: arXiv:2305.05465v6, `c:wellposedtransformers`. -/
theorem wellposed_contEq (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d))
    (hμ₀ : IsProbabilityMeasure μ₀) (R : ℝ) (hR : 0 < R) (hsupp : IsCarriedBy μ₀ R) :
    ∃! μ : ℝ → Measure (EucSpace d), IsContEqSolution Q K V μ₀ μ := by
  sorry

/-- The hypotheses of `wellposed_contEq` are satisfiable, and the solution it
asserts exists in that instance: the Dirac mass at the origin. -/
example (Q K V : ParamMatrix d) :
    IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧ (0 : ℝ) < 1 ∧
      IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 ∧
      IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
        (fun _ => Measure.dirac (0 : EucSpace d)) :=
  ⟨inferInstance, one_pos, isCarriedBy_dirac 1 zero_le_one, isContEqSolution_dirac Q K V⟩

/-- **Estimate (e:w2estimate).**  Solutions of `e:conteq` starting in a fixed
ball depend on their initial datum with an exponential `W_2` modulus on every
bounded time interval.

Not proved here.

Source: arXiv:2305.05465v6, `c:wellposedtransformers`, `e:w2estimate`. -/
theorem contEq_w2_stability (Q K V : ParamMatrix d) (R T : ℝ) (hR : 0 < R) (hT : 0 < T) :
    ∃ C : ℝ, 0 < C ∧ ∀ μ₀ ν₀ : Measure (EucSpace d), IsProbabilityMeasure μ₀ →
      IsProbabilityMeasure ν₀ → IsCarriedBy μ₀ R → IsCarriedBy ν₀ R →
        ∀ μ ν : ℝ → Measure (EucSpace d), IsContEqSolution Q K V μ₀ μ →
          IsContEqSolution Q K V ν₀ ν → ∀ t ∈ Set.Icc (0 : ℝ) T,
            Wasserstein.W2 (μ t) (ν t) ≤ Real.exp (C * t) * Wasserstein.W2 μ₀ ν₀ := by
  sorry

/-- The hypotheses of `contEq_w2_stability` are satisfiable: the Dirac mass at
the origin, against itself. -/
example (Q K V : ParamMatrix d) :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
      IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 ∧
      IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
        (fun _ => Measure.dirac (0 : EucSpace d)) :=
  ⟨one_pos, one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one,
    isContEqSolution_dirac Q K V⟩

end Clusters
end Transformer
