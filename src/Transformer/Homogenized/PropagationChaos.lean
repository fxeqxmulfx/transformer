/-
# Homogenized Transformers — propagation of chaos

Formalization of `prop: poc` of arXiv:2604.01978v1, *Homogenized
Transformers*, §4: the `n`-token system driven by the common noise stays
within `(C_T/n) Σ_i 𝔼‖x_i^n(0) - x̄_i(0)‖²` of `n` conditionally i.i.d. copies
of the mean-field limit, uniformly on `[0,T]`.

`CoupledSystem.lean` holds the two systems under one noise; this file holds the
estimate between them, and three readings of the source it makes.

**`𝔼 sup_t`.**  The source's left-hand side is the expectation of a supremum
over an uncountable time set, which is not a measurable function of `ω` for a
process given without path regularity — and the integral of a non-measurable
function is `0` in Mathlib, which would make the inequality trivially true.
The two suprema are therefore carried as explicit random variables `S` and `R`
pinned by `IsLUB`, and their integrability is a hypothesis: what the source
asserts is a bound on `∫ S` for *every* version of `sup_t`.

**`C_T`.**  The source writes "for every `T > 0` there exists `C_T > 0`" after
fixing the system, which as stated would let `C_T` depend on `n` and say
nothing about the limit.  It is quantified here before `n` and before the
system, and after the model data `(d, β, ρ*)` and `T`: that is the content of
the Carmona–Delarue estimate the source cites, whose constant depends on `T`
and on the Lipschitz constant of the coefficient only.

**`n ≥ 1`** is written as `n + 1`; the empirical measure of no tokens is the
zero measure, and `W_2` against it is not the source's object.

Source: arXiv:2604.01978v1, `prop: poc`.
-/

import Transformer.Homogenized.CoupledSystem
import Transformer.Homogenized.RandomChain
import Transformer.Wasserstein

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- **Proposition (prop: poc).**  Under `ass:high_order_short`, for every
`T > 0` there is `C_T > 0` such that, for every `n` and every coupled system of
`IsPoCSystem`,

  `max_i 𝔼 sup_{[0,T]} ‖x_i^n - x̄_i‖² + 𝔼 sup_{[0,T]} W₂²(μ^n, μ̄^n)
      ≤ (C_T/n) Σ_i 𝔼‖x_i^n(0) - x̄_i(0)‖²`.

The source adds that, after choosing an optimal coupling of the initial data,
the right-hand side is `C_T 𝔼 W₂²(μ^n(0), μ̄^n(0))`; that is this bound
composed with the existence of an optimal coupling, which `Wasserstein.lean`
does not have.

See the module docstrings for the three readings this statement makes of the
source: one noise for `2n` tokens, the suprema as `IsLUB` variables, and `C_T`
quantified before `n`.

Not proved here.

Source: arXiv:2604.01978v1, `prop: poc`. -/
theorem propagation_of_chaos {d : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ) (T : ℝ) (hT : 0 < T) :
    ∃ CT : ℝ, 0 < CT ∧
      ∀ (n : ℕ) (Ω : Type) (mΩ : MeasurableSpace Ω) (P : Measure Ω),
        IsProbabilityMeasure P →
      ∀ (ℱ 𝒢 : Filtration ℝ mΩ) (μ₀ : Measure (EucSpace d))
        (X Xbar : ℝ → Ω → (Idx (n + 1) → EucSpace d))
        (μ : ℝ → Ω → Measure (EucSpace d)),
        IsPoCSystem β T ρ P ℱ 𝒢 μ₀ X Xbar μ →
      ∀ (i : Idx (n + 1)) (S R : Ω → ℝ),
        (∀ ω, IsLUB ((fun t => ‖X t ω i - Xbar t ω i‖ ^ 2) '' Set.Icc 0 T) (S ω)) →
        (∀ ω, IsLUB ((fun t =>
          Wasserstein.W2 (empMeasure (X t ω)) (empMeasure (Xbar t ω)) ^ 2) ''
            Set.Icc 0 T) (R ω)) →
        Integrable S P → Integrable R P →
        (∫ ω, S ω ∂P) + ∫ ω, R ω ∂P ≤
          CT / ((n : ℝ) + 1) * ∑ j : Idx (n + 1), ∫ ω, ‖X 0 ω j - Xbar 0 ω j‖ ^ 2 ∂P := by
  sorry

/-- The hypotheses of `propagation_of_chaos` are satisfiable, and the system it
quantifies over is not empty: `ρ* = δ_0` satisfies `ass:high_order_short`,
`T = 1` is positive, and `isPoCSystem_dirac_zero` exhibits a coupled system at
that law. -/
example (d : ℕ) :
    HasHighOrderLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧ (0 : ℝ) < 1 :=
  ⟨hasHighOrderLaw_dirac_zero (d + 1), one_pos⟩

end Homogenized
end Transformer
