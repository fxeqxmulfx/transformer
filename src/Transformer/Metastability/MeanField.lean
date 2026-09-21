/-
# Metastability — Mean-field regime (§5 of 2410.06833v1), definitions

The objects §5 is written in:

* `eq: mean.field.pde`              — mean-field continuity equation,
* `Definition def: init_measure_MF` — `(β, ε)`-separated initial measure,
* the characteristic flow `Φ^t_{v[μ(t)]}` of `eq: flow.map`, as the
  predicate `IsMFFlow` it satisfies (no ODE solver is built here, so the flow
  is a parameter pinned down by its equation),
* the observables of the proof of `thm: metastability MF`: the transported
  cap `𝓑_q(t)`, its lowest coordinate `η_q(t)`, a minimiser `x(t)`, the cap
  variance `𝖵_q(t)`, the escape window `[0, T_esc]`, and the set whose
  infimum is `T_*(q, c)`.

The statements are `Metastability.MeanFieldMetastability` (the theorem) and
`Metastability.MeanFieldCapExit` (`claim: de sortie de cap`, `eq: v.small`);
`Metastability.MeanFieldStatic` is the static solution that witnesses their
hypotheses.
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

/-- **The characteristic flow (eq: flow.map).**  `Φ` is the flow of the
velocity field `v[μ(t)]`: `Φ^0 = id`, and every trajectory `t ↦ Φ^t(x)`
solves `ẋ(t) = v[μ(t)](x(t))`.  Measurability of each `Φ^t` is what the
push-forward `(Φ^t)_# μ_0` needs.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`,
`eq: flow.map`. -/
def IsMFFlow (β : ℝ) (μ : ℝ → ProbSphere d) (Φ : ℝ → SSphere d → SSphere d) : Prop :=
  (∀ t : ℝ, Measurable (Φ t)) ∧ (∀ x : SSphere d, Φ 0 x = x) ∧
    ∀ x : SSphere d, ∀ t : ℝ,
      HasDerivAt (fun s => (Φ s x : EucSpace d)) (MFVel d β (μ t) ((Φ t x : EucSpace d))) t

/-- The transported cap `𝓑_q(t) = Φ^t(𝒮_q(ε))`.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, step 1. -/
def capFlowImage (Φ : ℝ → SSphere d → SSphere d) (w : SSphere d) (ε t : ℝ) :
    Set (SSphere d) :=
  Φ t '' sphericalCap d w ε

/-- `η_q(t) = min_{x ∈ 𝓑_q(t)} ⟨x, w_q⟩`, the lowest coordinate of the
transported cap along its centre.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, step 1. -/
noncomputable def capMin (Φ : ℝ → SSphere d → SSphere d) (w : SSphere d) (ε t : ℝ) : ℝ :=
  sInf ((fun x : SSphere d => inner (𝕜 := ℝ) ((x : EucSpace d)) ((w : EucSpace d)))
    '' capFlowImage d Φ w ε t)

/-- `x(t) ∈ argmin_{x ∈ 𝓑_q(t)} ⟨x, w_q⟩` at every time: a point of the
transported cap at which `η_q(t)` is attained.  The minimiser need not be
unique; every statement below holds for each such selection.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, step 1, and
the `argmin` in the statement of `thm: metastability MF`. -/
def IsCapArgmin (Φ : ℝ → SSphere d → SSphere d) (w : SSphere d) (ε : ℝ)
    (x : ℝ → SSphere d) : Prop :=
  ∀ t : ℝ, x t ∈ capFlowImage d Φ w ε t ∧
    ∀ y ∈ capFlowImage d Φ w ε t,
      inner (𝕜 := ℝ) ((x t : EucSpace d)) ((w : EucSpace d))
        ≤ inner (𝕜 := ℝ) ((y : EucSpace d)) ((w : EucSpace d))

/-- At a minimiser, `η_q(t) = ⟨x(t), w_q⟩`. -/
theorem capMin_eq_of_isCapArgmin (Φ : ℝ → SSphere d → SSphere d) (w : SSphere d) (ε : ℝ)
    (x : ℝ → SSphere d) (hx : IsCapArgmin d Φ w ε x) (t : ℝ) :
    capMin d Φ w ε t = inner (𝕜 := ℝ) ((x t : EucSpace d)) ((w : EucSpace d)) := by
  refine IsLeast.csInf_eq ⟨⟨x t, (hx t).1, rfl⟩, ?_⟩
  rintro _ ⟨y, hy, rfl⟩
  exact (hx t).2 y hy

/-- `𝖵_q(t) = (1/2) ∫_{𝒮_q(2ε)} ‖Φ^t(x') - x(t)‖² dμ_0(x')`, the variance of
the cap around the minimiser `x(t)`.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, step 2. -/
noncomputable def capVariance (μ₀ : ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (w : SSphere d) (ε : ℝ) (x : ℝ → SSphere d) (t : ℝ) : ℝ :=
  (1 / 2) * ∫ x' in sphericalCap d w (2 * ε),
    ‖(Φ t x' : EucSpace d) - (x t : EucSpace d)‖ ^ 2 ∂(μ₀ : Measure (SSphere d))

/-- The escape window `[0, T_esc]`, where
`T_esc = min_q inf { t ≥ 0 : 𝓑_q(t) ⊄ 𝒮_q(2ε) }`: the times up to which no
transported cap has left its double.

**What the source says and what is changed here.**  The source writes the
window through the infimum `T_esc`, which may be `+∞`.  It is written here as
the set of times `t ≥ 0` before which every `𝓑_q(s)`, `s ≤ t`, stays in
`𝒮_q(2ε)`.  The two agree when the escape set is open — as it is for a
continuous flow, the caps being closed — and without that the set below may
drop the endpoint `T_esc`; it never adds a time.

Source: arXiv:2410.06833v1, §5, proof of `thm: metastability MF`, step 1. -/
def escapeWindow (k : ℕ) (w : Idx k → SSphere d) (Φ : ℝ → SSphere d → SSphere d) (ε : ℝ) :
    Set ℝ :=
  { t | 0 ≤ t ∧ ∀ s ∈ Set.Icc (0 : ℝ) t, ∀ q : Idx k,
      capFlowImage d Φ (w q) ε s ⊆ sphericalCap d (w q) (2 * ε) }

/-- The times in the escape window at which the cap `q` may be left, whose
infimum is `T_*(q, c)`:

  `{ t ∈ [0, T_esc] : η_q(t) 𝖵_q(t) e^{-(1 - η_q(t)) β} ≤ 2 e^{-c β} }`.

Source: arXiv:2410.06833v1, §5, definition of `T_*(q, c)` and
`claim: de sortie de cap`. -/
def capExitSet (β c ε : ℝ) (k : ℕ) (w : Idx k → SSphere d) (μ₀ : ProbSphere d)
    (Φ : ℝ → SSphere d → SSphere d) (q : Idx k) (x : ℝ → SSphere d) : Set ℝ :=
  { t | t ∈ escapeWindow d k w Φ ε ∧
      capMin d Φ (w q) ε t * capVariance d μ₀ Φ (w q) ε x t
          * Real.exp (-(1 - capMin d Φ (w q) ε t) * β)
        ≤ 2 * Real.exp (-c * β) }

end Metastability
end Transformer
