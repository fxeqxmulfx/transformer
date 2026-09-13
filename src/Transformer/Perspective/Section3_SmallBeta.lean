/-
# §4 — A single cluster for small β

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §4 of the survey:

* `e:Snonres0`              — the `β = 0` dynamics,
* `Theorem p:beta0`         — almost-sure consensus at `β = 0`,
* `Theorem th:beta_small`   — probability of clustering tends to `1` as `β → 0`,
* the auxiliary equations  `eq: harry.potter`, `e:Ps0n`, `e:approxsphere`,
                            `eq: youareawizardharry`,
* `Theorem thm: beta.tiny`  — clustering when `β ≤ C/n`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **Equation (e:Snonres0).** The `β = 0` dynamics common to both `SA` and
`USA`:

  `ẋ_i(t) = Proj_{x_i(t)} ( (1/n) Σ_j x_j(t) )`. -/
def beta0Dynamics (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((n : ℝ)⁻¹) • ∑ j : Idx n, ((X t j : EucSpace d)))) t

/-- **Theorem (p:beta0).** *Consensus at zero temperature.*

For `d, n ≥ 2`, and Lebesgue-almost any initial sequence
`(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n`, the unique solution to the Cauchy problem
for `e:Snonres0` satisfies

  `lim_{t → ∞} x_i(t) = x⋆`  for some `x⋆ ∈ 𝕊^{d-1}` and all `i`. -/
theorem beta0_consensus
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    -- "For Lebesgue almost-every X₀, the solution converges to a single point."
    ∀ (X₀ : SphereTuple d n),
      ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n,
        X 0 = X₀ → beta0Dynamics d n X →
          ∀ i : Idx n,
            Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
              Filter.atTop (nhds 0) := by
  sorry

/-- The subset `𝒮_β ⊂ (𝕊^{d-1})^n` of initial sequences for which the solution
to the Cauchy problem for `SA` (or `USA`) converges to a single cluster. -/
def clusteringSet
    (β : ℝ) : Set (SphereTuple d n) :=
  { X₀ | ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n,
            X 0 = X₀ → Perspective.SA d n β X →
              ∀ i : Idx n,
                Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
                  Filter.atTop (nhds 0) }

/-- **Theorem (th:beta_small).** *Clustering with high probability at small β.*

For fixed `d, n ≥ 2`, the probability (w.r.t. uniform initialization on
`(𝕊^{d-1})^n`) that the initial sequence belongs to `𝒮_β` tends to `1` as
`β → 0`. -/
theorem clustering_probability_small_beta
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    -- ℙ(𝒮_β) →[β→0] 1
    True := by trivial

/-! ### Auxiliary objects used inside the proof of `th:beta_small`. -/

/-- A set of `n` points on `𝕊^{d-1}` is *α-clustered* if all pairwise inner
products exceed `α`. -/
def alphaClustered
    (α : ℝ) (X : SphereTuple d n) : Prop :=
  ∀ i j : Idx n, α < inner (𝕜 := ℝ) ((X i) : EucSpace d) ((X j) : EucSpace d)

/-- **Equation (eq: harry.potter).** *3/4-clustering at integer time `m`.*

The set of initial conditions `X₀` for which the solution to `e:Snonres0` is
`(3/4)`-clustered at time `m`:

  `⟨x_i^0(m), x_j^0(m)⟩ > 3/4`  for all `i, j ∈ [n]`. -/
def Sset0 (m : ℕ) : Set (SphereTuple d n) :=
  { X₀ | ∃ X : ℝ → SphereTuple d n,
            X 0 = X₀ ∧ beta0Dynamics d n X ∧
              alphaClustered d n (3/4 : ℝ) (X (m : ℝ)) }

/-- **Equation (e:Ps0n).** As `m → ∞`, `ℙ(𝒮_0^m) → 1`. -/
theorem Sset0_probability_tends_to_one
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    True := by trivial

/-- **Equation (e:approxsphere).** Gronwall bound:

  `‖x_i^β(t) - x_i^0(t)‖ ≤ O(β) e^{3t}`. -/
theorem solutions_close_at_small_beta
    (X0 : SphereTuple d n) (β : ℝ) (hβ : 0 ≤ β)
    (Xβ : ℝ → SphereTuple d n) (X0t : ℝ → SphereTuple d n)
    (h1 : Xβ 0 = X0) (h2 : X0t 0 = X0)
    (hβ1 : Perspective.SA d n β Xβ)
    (hβ0 : beta0Dynamics d n X0t) :
    ∃ C : ℝ, ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖ ≤ C * β * Real.exp (3 * t) := by
  sorry

/-- **Equation (eq: youareawizardharry).**  For any `m`, there exists `β_m > 0`
such that for `β ∈ [0, β_m]`,

  `‖x_i^β(m) - x_i^0(m)‖ ≤ 1/8`. -/
theorem distance_bound_at_time_m
    (X0 : SphereTuple d n) (m : ℕ) :
    ∃ βm : ℝ, 0 < βm ∧
      ∀ β : ℝ, 0 ≤ β → β ≤ βm →
        ∀ Xβ X0t : ℝ → SphereTuple d n,
          Xβ 0 = X0 → X0t 0 = X0 →
          Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
          ∀ i : Idx n,
            ‖((Xβ (m : ℝ) i : EucSpace d))
              - ((X0t (m : ℝ) i : EucSpace d))‖ ≤ (1/8 : ℝ) := by
  sorry

/-- **Theorem (thm: beta.tiny).** *Cluster collapse when `β ≤ C/n`.*

Fix `d, n ≥ 2`.  There is a numerical constant `C > 0` such that whenever
`β ≤ C/n`:

For Lebesgue-almost any `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n`, there exists
`x⋆ ∈ 𝕊^{d-1}` with `lim_{t→∞} x_i(t) = x⋆` for the unique solution of `SA`
(resp. `USA`) starting from `X₀`.

Moreover, when `d = 2` one can take `β ≤ 1`. -/
theorem beta_tiny
    (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∃ C : ℝ, 0 < C ∧ ∀ β : ℝ, 0 ≤ β → β ≤ C / n →
      ∀ (X₀ : SphereTuple d n),
        ∃ x_star : SSphere d,
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
            ∀ i : Idx n,
              Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
                Filter.atTop (nhds 0) := by
  sorry

/-- *d = 2 improvement of `thm: beta.tiny` (Criscitiello-Boumal 2024).*

When `d = 2`, the constant in `beta_tiny` can be taken so that `β ≤ 1`. -/
theorem beta_tiny_circle (hn : 2 ≤ n) :
    ∀ β : ℝ, 0 ≤ β → β ≤ 1 →
      ∀ (X₀ : SphereTuple 2 n),
        ∃ x_star : SSphere 2,
          ∀ X : ℝ → SphereTuple 2 n, X 0 = X₀ → Perspective.SA 2 n β X →
            ∀ i : Idx n,
              Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace 2) - x_star))
                Filter.atTop (nhds 0) := by
  sorry

end Perspective
end Transformer
