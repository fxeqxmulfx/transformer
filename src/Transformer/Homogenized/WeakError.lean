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
equations differ by a drift of size `O(η)`.  Both are stated — the improved
one at the grid times where it holds, see `weak_error_modified`.

`lem:stability_generator`, the one-step estimate the telescoping of that proof
runs on, is stated here too: it is the same comparison at a single step, and
it is where the improved rate comes from.

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

The number of heads is positive.  The source writes `Σ_{h=1}^H`, so `H ≥ 1` is
implicit there, and the hypothesis is needed: at `H = 0` the update of
`eq:update_tokens` is the identity and the chain never moves, while the
limiting dynamics has drift `b` and does, so the bound fails for every `φ` that
separates `X(t_L)` from `X⁰`.

Not proved here.

Source: arXiv:2604.01978v1, `thm:weak_error_clean`. -/
theorem weak_error_clean {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
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
    0 < 1 ∧ HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, hasHighOrderLaw_dirac_zero d, contDiff_const⟩

/-- **Theorem (thm:weak_error_clean, the improved rate).**  The same statement
against the *modified* equation `eq:SDE_ito_clean`, whose drift carries the
extra `-(η/2) ∇_b b`, holds at the rate `C e^{C t_L} η (t_L+1) max(η,α)`.  The
source states it as the content of the proof of `thm:weak_error_clean`, from
which `thm:weak_error_clean` follows because the two limiting drifts differ
uniformly by `O(η)`.

**What the source says and what is changed here.**  The source states the
improved rate for `sup_{t ∈ [0,t_L]}`, carrying over the display of
`thm:weak_error_clean`.  That is false for the piecewise-constant
interpolation `X^η(t) = X^{⌊t/η⌋}`: at `t = η/2` the chain is still at `X⁰`,
while `𝔼φ(X(t))` has already moved by `(η/2)(𝖫φ)(X⁰) + O(η²)`.  Take a
deterministic weight law — `ass:high_order_short` holds with `σ_V = σ_A = 0`
and `𝔼V ≠ 0` — so that `α = 0` and `b ≢ 0`, and `φ(X) = ⟨e, x_1⟩`; then at
`t_L = 1` the left-hand side is `Θ(η)` and the right-hand side is
`C e^C η (t_L+1) max(η,0) = Θ(η²)`.

That `Θ(η)` is the interpolation error, and it is what the `max(1,α)` of
`thm:weak_error_clean` leaves room for.  The improved rate is a statement
about the grid, as it is in the stochastic-modified-equation literature the
proof follows, so it is stated at the grid times `t = ℓη`, `ℓ ≤ L`, where
`interpChain_natCast_mul` identifies `X^η(ℓη)` with `X^ℓ`.

Not proved here.

Source: arXiv:2604.01978v1, §2.3.2, the paragraph after
`thm:weak_error_clean`. -/
theorem weak_error_modified {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
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
      ∀ l : ℕ, l ≤ L →
        |(∫ ω', φ (X ((l : ℝ) * η) ω') ∂P') -
            ∫ ω, φ (interpChain η (Xd ω) ((l : ℝ) * η)) ∂P| ≤
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H) := by
  sorry

/-- The hypotheses of `weak_error_modified` are satisfiable, by the same
witnesses as `weak_error_clean`. -/
example (d n : ℕ) :
    0 < 1 ∧ HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, hasHighOrderLaw_dirac_zero d, contDiff_const⟩

/-! ### The one-step estimate -/

/-- **Lemma (lem:stability_generator).**  For `φ ∈ C⁴((𝕊^{d-1})^n)` there is `C > 0`,
uniform in the starting configuration, with

  `max_x |𝔼φ(X(η; x)) - 𝔼φ(X¹(x))| ≤ C η² max(η, α)`,

where `X(·; x)` starts from `x` and `X¹(x)` is one step of the chain
`eq:update_tokens` from `x`.  This is the estimate the telescoping argument of
`thm:weak_error_clean` sums over the `L` layers.

**What the source says and what is carried here.**  The source writes
`X(η; x)` for "the solution of the SDE starting from `x`".  The SDE in question
is the *modified* one: the proof names its drift `V₀(X) = b(X) - (η/2) ∇_{b(X)}
b(X)` three lines in, and it is the matching of that drift's generator against
the chain's that produces the `max(η, α)` rather than `max(1, α)`.  So the
hypothesis carried here is `IsModifiedSde`, as in `weak_error_modified`.

The `max_x` is the universal quantifier over `x` standing inside the `∃ C`:
`C` is produced before the configuration, which is what makes the telescoping
legitimate.  The same is true of `η`, of the variance proxy and of both
probability spaces; `C` may depend on `‖φ‖_{C⁴}` and on the model
`(d, n, H, β, ρ*)`, and those are fixed first.

`X¹(x)` is the chain of `eq:update_tokens` started at `x` and read at index `1`,
so the initial condition the source leaves implicit in the notation is the
hypothesis `IsRandomChain η β ρ P Θ Xd x`; the continuous process is asked for
the same one.  The number of heads is positive, as in `weak_error_clean`.

Not proved here.

Source: arXiv:2604.01978v1, `lem:stability_generator`. -/
theorem stability_generator {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ x : Idx n → EucSpace d, (∀ i, ‖x i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x) →
        |(∫ ω', φ (X η ω') ∂P') - ∫ ω, φ (Xd ω 1) ∂P| ≤
          C * η ^ 2 * max η (alphaOf η s H) := by
  sorry

/-- The hypotheses of `stability_generator` are satisfiable, by the same
witnesses as `weak_error_clean`. -/
example (d n : ℕ) :
    0 < 1 ∧ HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, hasHighOrderLaw_dirac_zero d, contDiff_const⟩

end Homogenized
end Transformer
