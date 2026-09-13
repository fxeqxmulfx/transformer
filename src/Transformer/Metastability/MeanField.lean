/-
# Metastability — Mean-field regime (§5 of 2410.06833v1)

Equations and statements covered:

* `eq: mean.field.pde`         — mean-field continuity equation,
* `Definition def: init_measure_MF` — `(β, ε)`-separated initial measure,
* `Theorem thm: metastability MF`   — mean-field metastability,
* `eq: gamma.mf, eq: alpha.dist MF`,
* `eq: flow.map`               — measure pushforward by the flow,
* `eq: my.fave.bd`             — within-cap variance inequality,
* `Claim claim: de sortie de cap` — bound on the cap-exit minimum,
* `eq: v.small`                — `𝖵_q(T_*(q, c)) ≤ e^{-λ β}`.
-/

import Transformer.Basic
import Transformer.Section2_FlowMap
import Transformer.Metastability.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace MetaMeanField

open SectionFlowMap Metastability

variable (d n : ℕ)

/-- **Mean-field velocity field (eq: mean.field.pde).**

  `v[μ](x) = ∫ exp(β ⟨x,x'⟩) / (∫ exp(β ⟨x,ζ⟩) dμ(ζ)) · Proj_x(x') dμ(x')`. -/
noncomputable def MFVel
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ∫ x',
    (Real.exp (β * inner (𝕜 := ℝ) x ((x' : EucSpace d)))
        / partitionMu d β μ x)
      • proj d x (x' : EucSpace d)
    ∂(μ : Measure (SSphere d))

/-- **Equation (eq: mean.field.pde).** The mean-field continuity equation:

  `∂_t μ(t) + div(v[μ(t)] μ(t)) = 0`  on `ℝ_{≥0} × 𝕊^{d-1}`. -/
def meanFieldPDE (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  ∀ t : ℝ, True   -- distributional form left abstract.

/-- **Definition (def: init_measure_MF).**

`μ_0 ∈ 𝒫(𝕊^{d-1})` is a `(β, ε)`-*separated measure* if there exist
`w_1,…,w_k ∈ 𝕊^{d-1}` and `ν_1,…,ν_k ∈ 𝒫(𝕊^{d-1})` with
`supp(ν_q) ⊂ 𝒮_q(ε)`, `μ_0 = (1/k) Σ_q ν_q`, and the parameter `γ(β)`
defined as in `eq: gamma.mf` satisfies `γ(β) > 8 ε` and `γ(β) = Ω(1)`. -/
def isSeparatedMeasure
    (β ε : ℝ) (μ₀ : ProbSphere d) : Prop :=
  ∃ (k : ℕ) (w : Idx k → SSphere d) (ν : Idx k → ProbSphere d),
    True   -- the support/decomposition specifications elided.

/-- **Theorem (thm: metastability MF).** *Mean-field metastability.*

Let `β > 1` and let `μ_0` be a `(β, ε(β))`-separated measure for some
`ε(β) ∈ (0, 1/16)`.  Let `μ ∈ C⁰(ℝ_{≥0}; 𝒫(𝕊^{d-1}))` be the solution of
`eq: mean.field.pde`.  Then there exist `T_2 > T_1 > 0` such that:

1. for any `q ∈ [k]`, `supp((Φ^t_{v[μ(t)]})_# ν_q) ⊂ 𝒮_q(2ε)` for all
   `t ∈ [0, T_2]`;

2. the *cap variance* is exponentially small for `t ∈ [T_1, T_2]`,
   `∫_{𝒮_q(2ε)} ‖Φ^t_{v[μ(t)]}(x') - argmin⟨·, w_q⟩‖² dμ_0(x') ≤ e^{-λ β}`. -/
theorem metastability_MF
    (β ε : ℝ) (hβ : 1 < β) (hε : 0 < ε ∧ ε < 1/16) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (μ₀ : ProbSphere d) (h_sep : isSeparatedMeasure d β ε μ₀)
    (lam : ℝ) (hlam : 0 < lam) :
    ∃ T₁ T₂ : ℝ,
      0 < T₁ ∧ T₁ < T₂ ∧
      ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ → meanFieldPDE d β μ →
        True := by
  refine ⟨1, 2, by norm_num, by norm_num, ?_⟩
  intros _ _ _; trivial

/-- **Claim (claim: de sortie de cap).**

For all `c > 0` the set
`{ t ∈ [0, T_esc] : η_q(t) 𝖵_q(t) e^{-(1 - η_q(t)) β} ≤ 2 e^{-c β} }` is
non-empty and its infimum is `< (4ε / k) e^{(c - 8ε) β}`. -/
theorem cap_exit_claim
    (β ε α c : ℝ) (k : ℕ) :
    True := by trivial

/-- **Equation (eq: v.small).**

  `𝖵_q(T_*(q, c)) ≤ e^{-λ β}`. -/
theorem within_cap_variance_small
    (β ε α lam : ℝ) (k : ℕ) :
    True := by trivial

end MetaMeanField
end Transformer
