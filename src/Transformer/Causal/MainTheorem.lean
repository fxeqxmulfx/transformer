/-
# Causal attention — Main theorem for `V = I_d` (§4 of 2411.04990v2)

* `Theorem thm1`        — almost-everywhere convergence to a single cluster
                          for `V = I_d` and *arbitrary* `Q, K`,
* `Conjecture thm1.5`   — analogue for `V` diagonalizable with `λ_max > 0`
                          of multiplicity 1,
* `Conjecture thm2`     — analogue for `V` with `λ_max > 0` of multiplicity
                          `≥ 2`.

All three statements quantify over "almost every initial configuration", which
is relative to a reference measure on `(𝕊^{d-1})^n`; the paper's measure is the
uniform one, and this development does not construct it.  So each is a
`Prop`-valued definition carrying that measure as a parameter, and none of them
is proved here — `thm1.5` and `thm2` are conjectures even in the paper.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.Analysis.InnerProductSpace.Projection.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

open Causal Perspective

variable (d n : ℕ)

/-- **Theorem (thm1).** *Single-cluster convergence with `V = I_d`.*

For arbitrary `Q, K`, `β ≥ 0` and `V = I_d`, for almost any
`(x_1(0),…,x_n(0)) ∈ (𝕊^{d-1})^n` the CSA dynamics satisfy

  `∀ k ∈ [n],  lim_{t → ∞} x_k(t) = x_1(0)`.

The leader `x_1` is a fixed point of `eq: csa`: its own row of the causal mask
sees only itself, so the whole cloud is dragged onto where it started.
Source: arXiv:2411.04990v2, §4. -/
def SingleCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n) : Prop :=
  0 ≤ β →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
    Causal.CSA d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) X →
      ∀ k : Idx n,
        Filter.Tendsto (fun t : ℝ => X t k) Filter.atTop (nhds (X₀ ⟨0, hn⟩))

/-- **Conjecture (thm1.5).**  *Two-cluster convergence (`λ_max > 0`, mult 1).*

Let `λ > 0` be the largest real eigenvalue of `V` — every real eigenvalue is
`≤ λ` — and let it be simple, its eigenspace being the line spanned by
`ξ ∈ 𝕊^{d-1}`.  Then for arbitrary `Q, K` and almost any initialization the
CSA dynamics satisfy

  `∀ k ∈ [n], lim_{t → ∞} x_k(t) ∈ {ξ, -ξ}`.

The two limits are genuinely both possible: `V` fixes the line `ℝ ξ`, not a
ray, so a token starting in the half-space `⟨x, ξ⟩ < 0` is pulled to `-ξ`.
Source: arXiv:2411.04990v2, §4. -/
def TwoCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (ξ : SSphere d) : Prop :=
  0 ≤ β → 0 < lam →
  V (ξ : EucSpace d) = lam • (ξ : EucSpace d) →
  (∀ (v : EucSpace d) (c : ℝ), v ≠ 0 → V v = c • v → c ≤ lam) →
  (∀ v : EucSpace d, V v = lam • v → ∃ c : ℝ, v = c • (ξ : EucSpace d)) →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
    ∀ k : Idx n,
      Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (ξ : EucSpace d)) ∨
        Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (-(ξ : EucSpace d)))

/-- **Conjecture (thm2).**  *Single-cluster convergence with `λ_max > 0`,
`dim L ≥ 2`.*

Let `λ > 0`, let `L` be an eigenspace of `V` for `λ` with `dim L ≥ 2`, let `V`
preserve `L^⊥`, and let `⟨V z, z⟩ < λ ‖z‖²` for every `z ∈ L^⊥ \ {0}` — so `λ`
really is the top eigenvalue and everything transverse to `L` decays relative
to it.  Then almost every initialization yields convergence of every token to
the normalized `L`-component of `x_1(0)`, which is defined as soon as that
component is nonzero.

Source: arXiv:2411.04990v2, §4. -/
def SubspaceCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (L : Submodule ℝ (EucSpace d)) (hn : 1 ≤ n) : Prop :=
  0 ≤ β → 0 < lam →
  (∀ v ∈ L, V v = lam • v) →
  2 ≤ Module.finrank ℝ L →
  (∀ z ∈ Lᗮ, V z ∈ Lᗮ) →
  (∀ z ∈ Lᗮ, z ≠ 0 → inner (𝕜 := ℝ) (V z) z < lam * ‖z‖ ^ 2) →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
    L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d) ≠ 0 →
      ∀ k : Idx n,
        Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (‖L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)‖⁻¹ •
            L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)))

end Causal
end Transformer
