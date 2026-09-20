/-
# Attention's forward pass and Frank-Wolfe — finite `β`

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, up to the Gumbel trick.

At finite `β` the hardmax step is replaced by a softmax one, `eq: softmax.ODE`,
which collapses everything to a single point (`prop: origin`); the Gumbel trick
rewrites the same step as a Markov chain `eq: softmax.process`, whose jump
probabilities are the attention scores.
-/

import Transformer.FrankWolfe.Section1_Models
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- **Proposition (prop: origin).**

Let `β > 0`.  There is `γ_* ∈ (0,1)` such that for every `γ ∈ (0, γ_*)` and
every initial configuration, particles following `eq: softmax.ODE` — that is
`x_i^{t+1} = softmaxStep β γ (x^t) i`, the model `(SA_β)` with `V^t = γ I_d`
and `B^t ≡ I_d` — all converge to one point `x_*` of `conv{x_i^0}`.

So the hardmax dynamics, whose particles stop at the vertices, is only an
approximation of `(SA_β)` on finite time horizons.

Not proved here; the source defers to Geshkovski-Karagodin-Polyanskiy-Rigollet,
`Proposition 2.1`.

Source: arXiv:2508.09628v1, §5, `prop: origin`. -/
theorem softmax_collapse (β : ℝ) (hβ : 0 < β) :
    ∃ γstar ∈ Set.Ioo (0 : ℝ) 1, ∀ γ ∈ Set.Ioo (0 : ℝ) γstar,
      ∀ (X₀ : Idx n → EucSpace d) (x : ℕ → Idx n → EucSpace d),
        x 0 = X₀ → (∀ (t : ℕ) (i : Idx n), x (t + 1) i = softmaxStep β γ (x t) i) →
        ∃ xstar ∈ configHull X₀, ∀ i : Idx n,
          Filter.Tendsto (fun t => x t i) Filter.atTop (nhds xstar) := by
  sorry

/-- The hypothesis of `softmax_collapse` is satisfiable: `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- The attention score particle `i` gives particle `j` at inverse temperature
`β`, for the key-query matrix `B = I_d`:
`e^{β⟨x_i, x_j⟩} / Σ_k e^{β⟨x_i, x_k⟩}`.

Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def attWeight (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (X i) (X j)) /
    ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X k))

/-- **The self-attention process `eq: softmax.process` (SA_ℙ).**

`P` is the law, on path space `(ℝ^d)^{n × ℕ}`, of the Markov chain that starts
at `X₀` and, at every step and for every particle `i`, replaces `x_i^t` by
`(1-γ) x_i^t + γ x_j^t` with probability `attWeight β (x^t) i j`.

The Markov property and the transition law are recorded as one condition: for
every event `S` depending only on the trajectory up to time `t`,

  `ℙ[S ∩ {x_i^{t+1} = (1-γ)x_i^t + γ x_j^t}] = ∫_S attWeight β (x^t) i j dℙ`.

Taking `S` to be the whole space gives the displayed transition probability of
the source; taking `S` in the past says that the jump probability depends on
the past only through `x^t`, which is the source's "clearly this process is a
Markov chain".

This is a predicate of `β`, `γ`, `X₀` and `P`, not a claim: it says which
measures on path space are the process, and it is `P` that theorems quantify
over.

Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def IsSAProcess (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (P : Measure (ℕ → Idx n → EucSpace d)) : Prop :=
  IsProbabilityMeasure P ∧
  (∀ᵐ x ∂P, x 0 = X₀) ∧
  ∀ (t : ℕ) (i j : Idx n) (S : Set (ℕ → Idx n → EucSpace d)), MeasurableSet S →
    (∀ x y : ℕ → Idx n → EucSpace d, (∀ s ≤ t, x s = y s) → (x ∈ S ↔ y ∈ S)) →
    (P (S ∩ {x | x (t + 1) i = (1 - γ) • x t i + γ • x t j})).toReal
      = ∫ x in S, attWeight β (x t) i j ∂P

end FrankWolfe
end Transformer
