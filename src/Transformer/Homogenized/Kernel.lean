/-
# Homogenized Transformers — regularity of the drift and the noise kernel

Formalization of `lem:Kernel_regularity` of arXiv:2604.01978v1, *Homogenized
Transformers*: the regularity that `ass:high_order_short` buys, and which the
proof of `thm:weak_error_clean` runs on.

**What the source says and what is carried here.**

* The `C^k` norms of the source are carried by explicit dominating functions:
  "`‖‖G(·,θ)‖_{C⁵}‖_{L²(ρ*)} ≤ C₂ς`" is written as "there is `N` dominating
  every derivative of order `≤ 5` of `G(·,θ)` at every configuration, with
  `‖N‖_{L²(ρ*)} ≤ C₂ς`".  That is the same statement — the `C⁵` norm is the
  least such `N` — and it avoids the junk value a supremum takes when the
  family is unbounded, which would make the bound vacuously true.

* The two claims "there exist `C₁, C₂ > 0`" and "we can choose
  `C₁ = O(1 + σ_A‖𝔼V‖_op β/σ_V)`, `C₂ = O(1 + β¹⁰σ_A¹⁰d⁵)`" are read as one: a
  universal `K`, produced before `d, n, β, σ_V, σ_A` are chosen, with `C₁` and
  `C₂` those explicit expressions.  The operator norm of `𝔼V` is carried as
  any bound `opV` for it, which is all the estimate uses.

* `ς` is the variance proxy of `eq: defining.alpha`, i.e. the per-token
  supremum `IsVarianceProxy`; the source writes `‖G(X,·)‖_{L²(ρ*)}` for the
  whole configuration in `lem:Kernel_regularity` and per token in
  `eq: defining.alpha`.  The per-token reading is the one `α` is defined from,
  and it is the one used here.

Source: arXiv:2604.01978v1, `lem:Kernel_regularity`, `eq: defining.alpha`.
-/

import Transformer.Homogenized.RandomChain
import Transformer.Homogenized.Generator

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Lemma (lem:Kernel_regularity).**  Under `ass:high_order_short`:

1. `b ∈ C⁴((𝕊^{d-1})^n; T(𝕊^{d-1})^n)`;
2. the variance proxy `ς` of `eq: defining.alpha` is finite, and
   `max_X ‖G(X,·)‖_{L³(ρ*)} ≤ C₁ ς`;
3. for `ρ*`-a.e. `θ`, `G(·,θ) ∈ C⁵`, and
   `‖‖G(·,θ)‖_{C⁵}‖_{L²(ρ*)} + ‖‖∇_G G(·,θ)‖_{C⁵}‖_{L²(ρ*)} ≤ C₂ ς`,

with `C₁ = O(1 + σ_A ‖𝔼V‖_op β / σ_V)` and `C₂ = O(1 + β¹⁰ σ_A¹⁰ d⁵)`, all
independent of `α, η, L`.

Not proved here.

Source: arXiv:2604.01978v1, `lem:Kernel_regularity`. -/
theorem kernel_regularity :
    ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ) (β : ℝ) (C σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
        (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
        (mV mA : Matrix (Fin d) (Fin d) ℝ) (opV : ℝ),
      IsHighOrderLaw d C σV σA ρ P Vr Wr Wr' mV mA →
      (∀ y : EucSpace d, ‖Matrix.toEuclideanLin mV y‖ ≤ opV * ‖y‖) →
      ContDiff ℝ 4 (fun x : Idx n → EucSpace d => bField β ρ x) ∧
      ∃ s : ℝ, IsVarianceProxy d n β ρ s ∧
        (∀ x : Idx n → EucSpace d, (∀ i, ‖x i‖ = 1) → ∀ i : Idx n,
          (∫ θ, ‖Gfield β ρ x θ i‖ ^ 3 ∂ρ) ^ ((1 : ℝ) / 3) ≤
            K * (1 + (σA : ℝ) * opV * β / (σV : ℝ)) * s) ∧
        (∀ᵐ θ ∂ρ, ContDiff ℝ 5 (fun x : Idx n → EucSpace d => Gfield β ρ x θ)) ∧
        ∃ N₁ N₂ : HeadParam d → ℝ,
          (∀ θ : HeadParam d, ∀ k ≤ 5, ∀ x : Idx n → EucSpace d,
            ‖iteratedFDeriv ℝ k (fun y : Idx n → EucSpace d => Gfield β ρ y θ) x‖ ≤ N₁ θ) ∧
          (∀ θ : HeadParam d, ∀ k ≤ 5, ∀ x : Idx n → EucSpace d,
            ‖iteratedFDeriv ℝ k
              (covDeriv (fun y : Idx n → EucSpace d => Gfield β ρ y θ)) x‖ ≤ N₂ θ) ∧
          Real.sqrt (∫ θ, N₁ θ ^ 2 ∂ρ) + Real.sqrt (∫ θ, N₂ θ ^ 2 ∂ρ) ≤
            K * (1 + β ^ 10 * (σA : ℝ) ^ 10 * (d : ℝ) ^ 5) * s := by
  sorry

/-- The hypotheses of `kernel_regularity` are satisfiable: the degenerate law
`ρ* = δ_0` of `isHighOrderLaw_dirac_zero`, whose mean value matrix is `𝔼V = 0`,
for which `opV = 0` is an operator bound. -/
example (d : ℕ) :
    IsHighOrderLaw d 1 0 0 (Measure.dirac (0 : HeadParam d)) (Measure.dirac ())
        (0 : Unit → Matrix (Fin d) (Fin d) ℝ) 0 0 0 0 ∧
      ∀ y : EucSpace d,
        ‖Matrix.toEuclideanLin (0 : Matrix (Fin d) (Fin d) ℝ) y‖ ≤ 0 * ‖y‖ :=
  ⟨isHighOrderLaw_dirac_zero d, fun y => by simp⟩

end Homogenized
end Transformer
