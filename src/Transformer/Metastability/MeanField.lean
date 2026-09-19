/-
# Metastability — Mean-field regime (§5 of 2410.06833v1)

Equations and statements covered:

* `eq: mean.field.pde`              — mean-field continuity equation,
* `Definition def: init_measure_MF` — `(β, ε)`-separated initial measure,
* `Theorem thm: metastability MF`   — mean-field metastability,
* `eq: flow.map`                    — measure pushforward by the flow,
* `Claim claim: de sortie de cap`   — bound on the cap-exit minimum,
* `eq: v.small`                     — `𝖵_q(T_*(q, c)) ≤ e^{-λ β}`.

Three things the paper's statements name are not defined in this development
and are therefore carried as parameters: the characteristic flow
`Φ^t_{v[μ(t)]}` of `eq: flow.map` (no ODE solver is built here), and the two
scalar observables `η_q(t)` and `𝖵_q(t)` of §5.  The qualitative side
conditions "`γ(β) = Ω(1)`" and "`ε(β) ∈ (0, 1/16)` depends on `β`" are
asymptotic in `β` and are not formalized; the quantitative parts are.

The centre of the cap variance in `thm: metastability MF` is an `argmin` in the
paper; here it is an existentially quantified point of the cap, which is
weaker.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Metastability.Basic
import Mathlib.MeasureTheory.Measure.Support

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

open Perspective Metastability

variable (d : ℕ)

/-- **Mean-field velocity field (eq: mean.field.pde).**

  `v[μ](x) = ∫ exp(β ⟨x,x'⟩) / (∫ exp(β ⟨x,ζ⟩) dμ(ζ)) · Proj_x(x') dμ(x')`. -/
noncomputable def MFVel
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ∫ x',
    (Real.exp (β * inner (𝕜 := ℝ) x ((x' : EucSpace d)))
        / partitionMu d β μ x)
      • proj d x (x' : EucSpace d)
    ∂(μ : Measure (SSphere d))

/-- **Equation (eq: mean.field.pde).** The mean-field continuity equation

  `∂_t μ(t) + div(v[μ(t)] μ(t)) = 0`  on `ℝ_{≥0} × 𝕊^{d-1}`,

in the distributional form of `Perspective.continuityEquation`: for every `C¹`
test function `φ` on the ambient space,

  `d/dt ∫ φ dμ(t) = ∫ ⟨∇φ(x), v[μ(t)](x)⟩ dμ(t)(x)`.

The ambient gradient is the right pairing because `MFVel` integrates `proj`,
so the normal part of `∇φ` is annihilated. -/
def meanFieldPDE (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 1 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
          (MFVel d β (μ t) (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t

/-- **Definition (def: init_measure_MF).**

`μ_0 ∈ 𝒫(𝕊^{d-1})` is a `(β, ε)`-*separated measure* if there are
`w_1,…,w_k ∈ 𝕊^{d-1}` and `ν_1,…,ν_k ∈ 𝒫(𝕊^{d-1})` with
`supp(ν_q) ⊂ 𝒮_q(ε)`, `μ_0 = (1/k) Σ_q ν_q`, and `γ(β) > 8 ε` for the `γ` of
`eq: gamma.mf` — which is `Metastability.γβ` with the number of caps `k` in
place of the number of particles.

The paper's further requirement `γ(β) = Ω(1)` is asymptotic in `β` and is not
formalized.  Source: arXiv:2410.06833v1, §5. -/
def isSeparatedMeasure
    (β ε : ℝ) (μ₀ : ProbSphere d) : Prop :=
  ∃ (k : ℕ) (w : Idx k → SSphere d) (ν : Idx k → ProbSphere d),
    0 < k ∧
    (∀ q : Idx k, (ν q : Measure (SSphere d)).support ⊆ sphericalCap d (w q) ε) ∧
    (μ₀ : Measure (SSphere d))
      = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere d)) ∧
    8 * ε < γβ k β (αDist d k w ε) ε

/-- **Theorem (thm: metastability MF).** *Mean-field metastability.*

Let `β > 1`, let `ε ∈ (0, 1/16)`, and let `μ_0 = (1/k) Σ_q ν_q` be a
`(β, ε)`-separated measure with caps `𝒮_q(ε)` around `w_1,…,w_k`.  Let `μ`
solve `eq: mean.field.pde` from `μ_0`, and let `Φ` be the characteristic flow
of `v[μ(t)]` (`eq: flow.map`).  Then there are `T_2 > T_1 > 0` such that:

1. `supp((Φ^t)_# ν_q) ⊂ 𝒮_q(2ε)` for every `q` and every `t ∈ [0, T_2]`;

2. the cap variance is exponentially small on `[T_1, T_2]`: some point `z` of
   `𝒮_q(2ε)` has
   `∫_{𝒮_q(2ε)} ‖Φ^t(x') - z‖² dμ_0(x') ≤ e^{-λ β}`.

Source: arXiv:2410.06833v1, §5. -/
def MetastabilityMF
    (β ε lam : ℝ) (μ₀ : ProbSphere d) (k : ℕ) (w : Idx k → SSphere d)
    (ν : Idx k → ProbSphere d) (Φ : ℝ → SSphere d → SSphere d) : Prop :=
  1 < β → 0 < ε → ε < 1 / 16 → 2 ≤ d → 0 < lam → 0 < k →
  (∀ q : Idx k, (ν q : Measure (SSphere d)).support ⊆ sphericalCap d (w q) ε) →
  (μ₀ : Measure (SSphere d))
      = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere d)) →
  8 * ε < γβ k β (αDist d k w ε) ε →
  ∀ μ : ℝ → ProbSphere d, μ 0 = μ₀ → meanFieldPDE d β μ →
    (∀ t : ℝ, Measurable (Φ t)) → (∀ x : SSphere d, Φ 0 x = x) →
    (∀ x : SSphere d, ∀ t : ℝ,
      HasDerivAt (fun s => (Φ s x : EucSpace d))
        (MFVel d β (μ t) ((Φ t x : EucSpace d))) t) →
      ∃ T₁ T₂ : ℝ, 0 < T₁ ∧ T₁ < T₂ ∧
        (∀ q : Idx k, ∀ t ∈ Set.Icc (0 : ℝ) T₂,
          (Measure.map (Φ t) (ν q : Measure (SSphere d))).support
            ⊆ sphericalCap d (w q) (2 * ε)) ∧
        ∀ q : Idx k, ∀ t ∈ Set.Icc T₁ T₂, ∃ z ∈ sphericalCap d (w q) (2 * ε),
          ∫ x in sphericalCap d (w q) (2 * ε),
              ‖(Φ t x : EucSpace d) - (z : EucSpace d)‖ ^ 2
            ∂(μ₀ : Measure (SSphere d))
              ≤ Real.exp (-lam * β)

/-- The times at which the cap `q` may be left, as in
`claim: de sortie de cap`:

  `{ t ∈ [0, T_esc] : η_q(t) 𝖵_q(t) e^{-(1 - η_q(t)) β} ≤ 2 e^{-c β} }`.

`η_q` and `𝖵_q` — the cap mass and the within-cap variance — are parameters:
neither is defined in this development. -/
def capExitSet (β c Tesc : ℝ) (η V : ℝ → ℝ) : Set ℝ :=
  { t | t ∈ Set.Icc (0 : ℝ) Tesc ∧
      η t * V t * Real.exp (-(1 - η t) * β) ≤ 2 * Real.exp (-c * β) }

/-- **Claim (claim: de sortie de cap).**

For every `c > 0` the set `capExitSet` is non-empty and its infimum `T_*(q, c)`
is smaller than `(4 ε / k) e^{(c - 8 ε) β}`.

Not proved here.

Source: arXiv:2410.06833v1, §5. -/
theorem cap_exit (β ε c Tesc : ℝ) (k : ℕ) (η V : ℝ → ℝ)
    (hc : 0 < c) (hε : 0 < ε) (hk : 0 < k) :
    (capExitSet β c Tesc η V).Nonempty ∧
      sInf (capExitSet β c Tesc η V)
        < (4 * ε / (k : ℝ)) * Real.exp ((c - 8 * ε) * β) := by
  sorry

/-- The hypotheses of `cap_exit` are satisfiable: `c = ε = 1`, one cap. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 < 1 := ⟨one_pos, one_pos, one_pos⟩

/-- **Equation (eq: v.small).**  At the cap-exit time the within-cap variance
is exponentially small:

  `𝖵_q(T_*(q, c)) ≤ e^{-λ β}`.

Not proved here; the exit time it is read at is the one `cap_exit` produces,
which is not proved either.

Source: arXiv:2410.06833v1, §5. -/
theorem variance_small (β c lam Tesc : ℝ) (η V : ℝ → ℝ)
    (hc : 0 < c) (hlam : 0 < lam)
    (hne : (capExitSet β c Tesc η V).Nonempty) :
    V (sInf (capExitSet β c Tesc η V)) ≤ Real.exp (-lam * β) := by
  sorry

/-- The hypotheses of `variance_small` are satisfiable: at zero cap mass the
exit condition `0 ≤ 2 e^{-cβ}` holds at every time, so `t = 0` is in the exit
set. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (capExitSet 1 1 1 (fun _ => 0) (fun _ => 0)).Nonempty := by
  refine ⟨one_pos, one_pos, ⟨0, ⟨le_rfl, zero_le_one⟩, ?_⟩⟩
  have h : (0 : ℝ) ≤ 2 * Real.exp (-1 * 1) := by positivity
  simpa using h

end Metastability
end Transformer
