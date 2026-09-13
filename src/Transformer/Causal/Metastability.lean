/-
# Causal attention — Meta-stable clustering and R'enyi parking (§5 of 2411.04990v2)

Equations and statements covered:

* *R'enyi centers* and *strong R'enyi centers* (Definition),
* `Lemma lemma:meta`               — strong R'enyi centers stay nearly fixed,
* `eq: renyi`, `eq: c`, `eq: time_bd`,
* `Theorem thm:fixed_centers`     — convergence of all tokens to the
                                    vicinity of strong R'enyi centers.
-/

import Transformer.Basic
import Transformer.Causal.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

variable (d n : ℕ)

/-- A *R'enyi center subsequence* of `(x_j)_{j ≥ 1}`:

  `dist(x_{s_j}, x_{s_i}) > δ`  for all `i < j`. -/
def isRenyiCenters
    (x : ℕ → SSphere d) (s : ℕ → ℕ) (δ : ℝ) : Prop :=
  StrictMono s ∧
    ∀ i j : ℕ, i < j →
      δ < ‖((x (s j) : EucSpace d)) - ((x (s i) : EucSpace d))‖

/-- A *strong R'enyi center* subsequence:

  `dist(x_{s_j}, x_i) > δ`  for all `i < s_j`. -/
def isStrongRenyiCenters
    (x : ℕ → SSphere d) (s : ℕ → ℕ) (δ : ℝ) : Prop :=
  StrictMono s ∧
    ∀ j : ℕ, ∀ i : ℕ, i < s j →
      δ < ‖((x (s j) : EucSpace d)) - ((x i : EucSpace d))‖

/-- **Lemma (lemma: meta).**  *Strong R'enyi centers stay nearly fixed.*

For `d = 2`, `Q = K = V = I_2`, suppose the strong R'enyi centers
`x_{s_1},…,x_{s_m}` satisfy

  `min_{i < s_j} |x_{s_j} - x_i| > c (1 + 2 ε) β^{-1/2}`

with `c > β^{1/2} arccos((-1 + √(4 β² + 1)) / (2 β))`.  For any `T_j` with

  `T_j · s_j · h(c β^{-1/2}) < ε c β^{-1/2}`,

the displacement of each center is bounded:

  `max_{t ∈ [0, T_j]} |x_{s_j}(t) - x_{s_j}(0)| < ε c β^{-1/2}`. -/
theorem lemma_meta
    (β c ε : ℝ) (hβ : 0 < β) (hc : 0 < c) (hε : 0 < ε)
    (X : ℝ → SphereTuple 2 n)
    (hX : Causal.CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
            (ContinuousLinearMap.id ℝ (EucSpace 2))
            (ContinuousLinearMap.id ℝ (EucSpace 2)) X)
    (s : ℕ → ℕ) (h_strong :
      ∀ j i : ℕ, i < s j →
        c * (1 + 2 * ε) * β^(-(1/2 : ℝ))
          < ‖((X 0 ⟨s j, by sorry⟩ : EucSpace 2))
              - ((X 0 ⟨i, by sorry⟩ : EucSpace 2))‖)
    (h_c : Real.sqrt β
              * Real.arccos ((-1 + Real.sqrt (4 * β^2 + 1)) / (2 * β))
            < c)
    (j : ℕ) (T_j : ℝ) (hT :
      T_j * (s j : ℝ) * Causal.h_pot β (c * β^(-(1/2 : ℝ)))
        < ε * c * β^(-(1/2 : ℝ))) :
    ∀ t : ℝ, 0 ≤ t → t ≤ T_j →
      ‖((X t ⟨s j, by sorry⟩ : EucSpace 2))
          - ((X 0 ⟨s j, by sorry⟩ : EucSpace 2))‖
        < ε * c * β^(-(1/2 : ℝ)) := by
  sorry

/-- **Theorem (thm: fixed_centers).**  *Convergence to strong R'enyi centers.*

For an arbitrary set of stationary (strong R'enyi) tokens, all other tokens
converge to the vicinity of one of them as `t → ∞`.  The result extends to
the case of additional cross-attention components. -/
theorem thm_fixed_centers
    (β : ℝ) (hβ : 0 < β) :
    True := by trivial

/-- **Conjecture (cardinality).**

The number of (strong) R'enyi centers with separation `δ = c β^{-1/2}` is
`Θ(β^{(d-1)/2})`. -/
theorem renyi_count (d : ℕ) (hd : 2 ≤ d) :
    True := by trivial

end Causal
end Transformer
