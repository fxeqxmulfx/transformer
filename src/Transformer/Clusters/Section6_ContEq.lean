/-
# The emergence of clusters in self-attention dynamics — the continuity
  equation

`e:conteq`, `d:solconti` and `c:wellposedtransformers` of arXiv:2305.05465v6,
§6: the mean-field form of the dynamics, its notion of solution, and global
well-posedness with a `W_2` stability estimate.

**What the source says and what is carried here.**

* `d:solconti` asks for `μ ∈ C⁰_comp(ℝ, 𝒫_c(ℝ^d))`, absolute continuity of
  `t ↦ ∫ g dμ_t` for every `g ∈ C_c^∞`, and the weak identity for almost
  every `t`.  It is carried as: each `μ_t` a probability measure; supports
  equi-contained in a ball on every compact time interval (the definition of
  `C⁰_comp`); `t ↦ ∫ g dμ_t` continuous; and the weak identity for **every**
  `t`.  Demanding the identity everywhere rather than almost everywhere is
  not a strengthening here: both sides are continuous in `t`, so an identity
  valid almost everywhere is valid everywhere, and absolute continuity then
  comes from the right-hand side being an integral in `t`.

* The source's weak identity writes `𝒳[μ_t]` inside the time integral, where
  `s` is the integration variable; that is a typographical slip — the
  vector field must be evaluated at the running time. It is carried as
  `𝒳[μ_s]`, as the proof and every derivation of `e:conteq` require.

* `∫_0^t` is the interval integral, which also gives the statement its meaning
  for `t < 0`; the source's solutions are defined on all of `ℝ`.

* `c:wellposedtransformers` is split in two: existence and uniqueness, and the
  stability estimate `e:w2estimate`, which quantifies over pairs of solutions.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`,
`c:wellposedtransformers`, `e:w2estimate`.
-/

import Transformer.Clusters.Section6_Kernel
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- **Definition (d:solconti), a solution of `e:conteq`.**  A weakly
continuous curve of compactly supported probability measures, equi-supported
on compact time intervals, starting at `μ₀` and satisfying the weak
formulation of `∂_t μ + div(𝒳[μ]μ) = 0`.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`. -/
def IsContEqSolution (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d))
    (μ : ℝ → Measure (EucSpace d)) : Prop :=
  (∀ t : ℝ, IsProbabilityMeasure (μ t)) ∧
    (∀ a b : ℝ, ∃ R : ℝ, 0 < R ∧ ∀ t ∈ Set.Icc a b, IsCarriedBy (μ t) R) ∧
      (∀ g : EucSpace d → ℝ, ContDiff ℝ ⊤ g → HasCompactSupport g →
        Continuous fun t => ∫ x, g x ∂(μ t)) ∧
      μ 0 = μ₀ ∧
      ∀ g : EucSpace d → ℝ, ContDiff ℝ ⊤ g → HasCompactSupport g → ∀ t : ℝ,
        ∫ x, g x ∂(μ t) = (∫ x, g x ∂μ₀) +
          ∫ s in (0 : ℝ)..t,
            ∫ x, inner (𝕜 := ℝ) (gradient g x) (attentionKernel Q K V (μ s) x) ∂(μ s)

/-- **The Dirac mass at the origin is a stationary solution.**  The attention
kernel vanishes there — the only value averaged is `V 0 = 0` — so nothing
moves.  This witnesses `d:solconti`.

Source: arXiv:2305.05465v6, `e:vectorfield`, `d:solconti`. -/
theorem isContEqSolution_dirac (Q K V : ParamMatrix d) :
    IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
      (fun _ => Measure.dirac (0 : EucSpace d)) := by
  refine ⟨fun _ => inferInstance, fun a b => ⟨1, one_pos, fun t _ => isCarriedBy_dirac 1 zero_le_one⟩,
    fun g _ _ => ?_, rfl, fun g _ _ t => ?_⟩
  · simpa using continuous_const
  · simp [attentionKernel_dirac_zero]

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
