/-
# Homogenized Transformers — well-posedness of the mean-field limit

Formalization of `prop:mckean_vlasov_common_noise` and `thm:PoC_wellposedness`
of arXiv:2604.01978v1, *Homogenized Transformers*, §4: the conditional
McKean–Vlasov SDE with common noise has a unique solution, and the conditional
law of that solution is the unique weak solution of the nonlinear
Fokker–Planck SPDE `eq: the.spde`.  Existence is stated here; the rest is false
for the formal solution concepts.

**What the formal solution concepts cannot carry.**  The source fixes a
filtered probability space carrying a cylindrical Wiener process `W` on
`L²(ρ*)`, an `ℱ_0`-measurable `x_0` independent of `W`, and the noise
filtration `𝒢 = (ℱ_t^W)`.  None of the three can be written without `W`, and
this has two consequences.

*Existence is stated on some filtered space, not on a given one.*  The space is
what makes the source's statement true: on a space too small to carry `W` — a
one-point space, say — there is no solution unless the fields vanish.  Since
"carries `W`" cannot be a hypothesis, the space is existentially quantified.

*Uniqueness, `prop: prop.3` and `thm:PoC_wellposedness` are false.*  The
formal `𝒢` is any filtration below `ℱ`; one that learns `x_0` after time `0`
makes `μ(t) = Law(x(t) | 𝒢_t)` jump.  Under the trivial head, two solutions
with one initial law then differ in law, the conditional law of a solution need
not solve the SPDE, and it need not have the law of the weak solution at rest:
`not_mckeanVlasovSolution_unique` and
`not_isWeakSpdeSolution_of_isMcKeanVlasovSolution`, in
`Section4_WellPosedRefutation`, and `not_poc_wellposedness`, in
`Section4_PocRefutation`.  The source's pathwise uniqueness, for solutions
driven by one `W`, is not refuted by them, and is not stated here.
-/

import Transformer.Homogenized.McKeanVlasov
import Transformer.Homogenized.RandomChain

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- Two processes, on possibly different probability spaces, have the same
finite-dimensional distributions on `[0,T]`: every bounded measurable
functional of finitely many time slices has the same expectation under both.

This is what "unique in law" means for a process; for a measure-valued process
it is also what "unique weak solution" can mean once the driving noise is not
part of the solution concept. -/
def SameFdd {E : Type*} [MeasurableSpace E] (T : ℝ)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    {Ω' : Type*} [MeasurableSpace Ω'] (P' : Measure Ω')
    (Z : ℝ → Ω → E) (Z' : ℝ → Ω' → E) : Prop :=
  ∀ (k : ℕ) (r : Fin k → ℝ), (∀ j, r j ∈ Set.Icc (0 : ℝ) T) →
    ∀ F : (Fin k → E) → ℝ, Measurable F → (∃ C : ℝ, ∀ z, |F z| ≤ C) →
      ∫ ω, F (fun j => Z (r j) ω) ∂P = ∫ ω, F (fun j => Z' (r j) ω) ∂P'

/-- The McKean–Vlasov SDE `eq:non_linear_SDE_common` has a solution with
initial law `μ_0`, on some filtered probability space.

See the module docstring for why the space is quantified here and given in the
source. -/
def HasMcKeanVlasovSolution {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    (μ₀ : Measure (EucSpace d)) : Prop :=
  ∃ (Ω : Type) (mΩ : MeasurableSpace Ω) (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ mΩ)
    (x₀ : Ω → EucSpace d) (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)),
    IsProbabilityMeasure P ∧ IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ

/-- **Proposition (prop:mckean_vlasov_common_noise), existence.**  Under
`ass:high_order_short`, for every `μ_0 ∈ 𝒫(𝕊^{d-1})` the conditional
McKean–Vlasov SDE `eq:non_linear_SDE_common` has a solution on `[0,T]`.

The source's proof is a fixed-point argument on the given filtered space,
Lipschitz continuity of `G_μ(x,·)` in `(x,μ)` coming from `prop:satisfying_MF`.

Not proved here.  See the module docstring for the existential over the space.

Source: arXiv:2604.01978v1, `prop:mckean_vlasov_common_noise`. -/
theorem exists_mckeanVlasovSolution {d : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ) (T : ℝ) (hT : 0 < T)
    (μ₀ : Measure (EucSpace d)) (hμ₀ : IsProbabilityMeasure μ₀)
    (hsupp : μ₀ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0) :
    HasMcKeanVlasovSolution β T ρ μ₀ := by
  sorry

/-- The hypotheses of `exists_mckeanVlasovSolution` are satisfiable: the
degenerate weight law `ρ* = δ_0` satisfies `ass:high_order_short`, `T = 1` is
positive, and `μ_0 = δ_{basePoint d}` is a probability measure carried by the
sphere. -/
example (d : ℕ) :
    HasHighOrderLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧ (0 : ℝ) < 1 ∧
      IsProbabilityMeasure
        (Measure.dirac ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))) ∧
      (Measure.dirac ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1)))
        {y : EucSpace (d + 1) | ‖y‖ = 1}ᶜ = 0 := by
  refine ⟨hasHighOrderLaw_dirac_zero (d + 1), one_pos, inferInstance, ?_⟩
  rw [Measure.dirac_apply]
  simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
