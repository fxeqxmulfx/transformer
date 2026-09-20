/-
# Homogenized Transformers — the weak approximation error

Formalization of `thm:weak_error_clean` and `cor:weak_error_centered` of
arXiv:2604.01978v1, *Homogenized Transformers*.

The discrete chain `eq:update_tokens`, interpolated in time as
`X^η(t) = X^{⌊t/η⌋}`, is approximated in the weak sense by the solution of the
Itô system `eq: first.sde` on `(𝕊^{d-1})^n`, uniformly on `[0, t_L]` with
`t_L = ηL`, at rate `O(η)`.  The proof of the source shows more: the chain is
approximated by the *modified* equation `eq:SDE_ito_clean` at the improved rate
`η max(η, α)`, and `thm:weak_error_clean` follows because the two limiting
equations differ by a drift of size `O(η)`.  Both are stated.

`C ≥ 1` depends on `‖φ‖_{C⁴}` but not on `η, α, L`: the quantifiers are
arranged so that this is what the statement says — `φ` is fixed before `C` is
produced, and `η`, `L`, the variance proxy and both probability spaces come
after it.

Source: arXiv:2604.01978v1, §2.3.2, `thm:weak_error_clean`.
-/

import Transformer.Homogenized.RandomChain
import Transformer.Homogenized.Generator

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Theorem (thm:weak_error_clean).**  Under `ass:high_order_short`, for any
`φ ∈ C⁴((𝕊^{d-1})^n)` there is `C ≥ 1`, depending on `‖φ‖_{C⁴}` but not on
`η, α, L`, with

  `sup_{t ∈ [0,t_L]} |𝔼φ(X(t)) - 𝔼φ(X^η(t))| ≤ C e^{C t_L} η (t_L + 1) max(1,α)`,

where `X` solves `eq: first.sde`, `X^η` interpolates the chain
`eq:update_tokens`, `t_L = ηL` and `α = η ς²/H`.

**What the source says and what is carried here.**  `X` and `X^η` start from
the same deterministic configuration; the manuscript's `X(0) = X^0` is written
out as the hypothesis `X 0 ω = x₀`, because without it the two expectations at
`t = 0` need not agree and the bound is false.  The solution of
`eq: first.sde` is taken in the sense of `IsFirstSde`, i.e. through the
generator: see `Transformer.Homogenized.Generator`.

Not proved here.

Source: arXiv:2604.01978v1, `thm:weak_error_clean`. -/
theorem weak_error_clean {d n H : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsFirstSde β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ t ∈ Set.Icc (0 : ℝ) (η * L),
        |(∫ ω', φ (X t ω') ∂P') - ∫ ω, φ (interpChain η (Xd ω) t) ∂P| ≤
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max 1 (alphaOf η s H) := by
  sorry

/-- The hypotheses of `weak_error_clean` are satisfiable: the degenerate weight
law `ρ* = δ_0` satisfies `ass:high_order_short` by
`hasHighOrderLaw_dirac_zero`, and a constant `φ` is `C⁴`. -/
example (d n : ℕ) :
    HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨hasHighOrderLaw_dirac_zero d, contDiff_const⟩

/-- **Theorem (thm:weak_error_clean, the improved rate).**  The same statement
against the *modified* equation `eq:SDE_ito_clean`, whose drift carries the
extra `-(η/2) ∇_b b`, holds at the rate `C e^{C t_L} η (t_L+1) max(η,α)`.  The
source states it as the content of the proof of `thm:weak_error_clean`, from
which `thm:weak_error_clean` follows because the two limiting drifts differ
uniformly by `O(η)`.

Not proved here.

Source: arXiv:2604.01978v1, §2.3.2, the paragraph after
`thm:weak_error_clean`. -/
theorem weak_error_modified {d n H : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ t ∈ Set.Icc (0 : ℝ) (η * L),
        |(∫ ω', φ (X t ω') ∂P') - ∫ ω, φ (interpChain η (Xd ω) t) ∂P| ≤
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H) := by
  sorry

/-- The hypotheses of `weak_error_modified` are satisfiable, by the same
witnesses as `weak_error_clean`. -/
example (d n : ℕ) :
    HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨hasHighOrderLaw_dirac_zero d, contDiff_const⟩

end Homogenized
end Transformer
