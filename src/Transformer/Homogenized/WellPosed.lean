/-
# Homogenized Transformers — well-posedness of the mean-field limit

Formalization of `prop: prop.3`, `prop:mckean_vlasov_common_noise` and
`thm:PoC_wellposedness` of arXiv:2604.01978v1, *Homogenized Transformers*, §4:
the conditional McKean–Vlasov SDE with common noise has a solution, its law is
unique, and the conditional law of that solution is the unique weak solution of
the nonlinear Fokker–Planck SPDE `eq: the.spde`.

**Two deviations from the source, both forced by the missing noise.**

*Existence is stated on some filtered space, not on a given one.*  The source
fixes a filtered probability space carrying a cylindrical Wiener process `W` on
`L²(ρ*)` and an `ℱ_0`-measurable `x_0` independent of `W`, and produces a
solution there.  The space is what makes the statement true: on a space too
small to carry `W` — a one-point space, say — there is no solution unless the
fields vanish.  Since `W` cannot be written, "carries `W`" cannot be a
hypothesis, and existence is stated with the space existentially quantified.

*Uniqueness is uniqueness in law.*  The source's `prop:mckean_vlasov_common_noise`
gives pathwise uniqueness: two solutions on one space driven by *the same* `W`
are indistinguishable.  The martingale-problem formulation of
`McKeanVlasov.lean` does not name the driving noise, so two solutions on one
space may be driven by different Wiener processes, and pathwise uniqueness is
false for it.  What survives, and what the source's result implies, is equality
of the finite-dimensional distributions — `SameFdd`.
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

/-- **Proposition (prop: prop.3).**  If `(x, μ)` solves the McKean–Vlasov SDE
`eq:non_linear_SDE_common` in the sense of `def:nonlinear_SDE_common`, then `μ`
is a weak solution of the SPDE `eq: the.spde` in the sense of `def:weak_spde`.

The source proves it by Itô's formula followed by conditioning on the
common-noise filtration.

Not proved here.

Source: arXiv:2604.01978v1, `prop: prop.3`. -/
theorem isWeakSpdeSolution_of_isMcKeanVlasovSolution {d : ℕ} (β T : ℝ)
    (ρ : Measure (HeadParam d)) {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω)
    (ℱ 𝒢 : Filtration ℝ m) (x₀ : Ω → EucSpace d) (μ₀ : Measure (EucSpace d))
    (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d))
    (h : IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ) :
    IsWeakSpdeSolution β T ρ P 𝒢 μ := by
  sorry

/-- The hypothesis of `isWeakSpdeSolution_of_isMcKeanVlasovSolution` is
satisfiable: the trivial head `ρ* = δ_0`, a token at rest on `basePoint d` and
the constant filtration on the one-point space. -/
example (d : ℕ) (β T : ℝ) :
    IsMcKeanVlasovSolution β T (Measure.dirac (0 : HeadParam (d + 1)))
      (Measure.dirac ()) (Filtration.const ℝ (inferInstanceAs (MeasurableSpace Unit)) le_rfl)
      (Filtration.const ℝ (inferInstanceAs (MeasurableSpace Unit)) le_rfl)
      (fun _ => ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1)))
      (Measure.dirac ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1)))
      (fun _ _ => ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1)))
      (fun _ _ => Measure.dirac ((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))) :=
  isMcKeanVlasovSolution_dirac_zero β T _ _ _ (by simp [basePoint, PiLp.norm_single])

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

/-- **Proposition (prop:mckean_vlasov_common_noise), uniqueness.**  Under
`ass:high_order_short`, two solutions of `eq:non_linear_SDE_common` with the
same initial law have the same law.

The source states pathwise uniqueness on a fixed space; see the module
docstring for why that is not what the martingale-problem formulation can
carry, and why this is.

Not proved here.

Source: arXiv:2604.01978v1, `prop:mckean_vlasov_common_noise`. -/
theorem mckeanVlasovSolution_unique {d : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ) (T : ℝ) (hT : 0 < T)
    (μ₀ : Measure (EucSpace d))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (hP : IsProbabilityMeasure P)
    (ℱ 𝒢 : Filtration ℝ m) (x₀ : Ω → EucSpace d)
    (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d))
    (h : IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ)
    {Ω' : Type*} {m' : MeasurableSpace Ω'} (P' : Measure Ω') (hP' : IsProbabilityMeasure P')
    (ℱ' 𝒢' : Filtration ℝ m') (x₀' : Ω' → EucSpace d)
    (x' : ℝ → Ω' → EucSpace d) (μ' : ℝ → Ω' → Measure (EucSpace d))
    (h' : IsMcKeanVlasovSolution β T ρ P' ℱ' 𝒢' x₀' μ₀ x' μ') :
    SameFdd T P P' (fun t ω => (x t ω, μ t ω)) (fun t ω => (x' t ω, μ' t ω)) := by
  sorry

/-- **Theorem (thm:PoC_wellposedness).**  Under `ass:high_order_short`, for
every `μ_0 ∈ 𝒫(𝕊^{d-1})` the McKean–Vlasov SDE `eq:non_linear_SDE_common` has
a solution `(x, μ)`, unique in law; and `μ` is the unique weak solution of
`eq: the.spde` in the sense of `def:weak_spde`.

That `μ` *is* a weak solution is `isWeakSpdeSolution_of_isMcKeanVlasovSolution`;
what is stated here is the uniqueness half — every weak solution starting at
`μ_0` has the law of the conditional law of a McKean–Vlasov solution.  The
source obtains it from the superposition principle for conditional
McKean–Vlasov equations, which lifts a weak solution of the SPDE to a solution
of the SDE.

Not proved here.

Source: arXiv:2604.01978v1, `thm:PoC_wellposedness`. -/
theorem poc_wellposedness {d : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ) (T : ℝ) (hT : 0 < T)
    (μ₀ : Measure (EucSpace d)) (hμ₀ : IsProbabilityMeasure μ₀)
    (hsupp : μ₀ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0) :
    HasMcKeanVlasovSolution β T ρ μ₀ ∧
      ∀ (Ω : Type) (mΩ : MeasurableSpace Ω) (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ mΩ)
        (x₀ : Ω → EucSpace d) (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)),
        IsProbabilityMeasure P → IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ →
      ∀ (Ω' : Type) (mΩ' : MeasurableSpace Ω') (P' : Measure Ω') (𝒢' : Filtration ℝ mΩ')
        (μ' : ℝ → Ω' → Measure (EucSpace d)),
        IsProbabilityMeasure P' → IsWeakSpdeSolution β T ρ P' 𝒢' μ' →
        (∀ ω, μ' 0 ω = μ₀) → SameFdd T P P' μ μ' := by
  sorry

/-- The hypotheses shared by `exists_mckeanVlasovSolution`,
`mckeanVlasovSolution_unique` and `poc_wellposedness` are satisfiable: the
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
