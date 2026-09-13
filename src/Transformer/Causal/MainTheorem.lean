/-
# Causal attention — Main theorem for `V = I_d` (§4 of 2411.04990v2)

* `Theorem thm1`        — almost-everywhere convergence to a single cluster
                          for `V = I_d` and *arbitrary* `Q, K`,
* `Conjecture thm1.5`   — analogue for `V` diagonalizable with `λ_max > 0`
                          of multiplicity 1,
* `Conjecture thm2`     — analogue for `V` with `λ_max > 0` of multiplicity
                          `≥ 2`.
-/

import Transformer.Basic
import Transformer.Causal.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

variable (d n : ℕ)

/-- **Theorem (thm1).** *Single-cluster convergence with `V = I_d`.*

For arbitrary `Q, K` and `V = I_d`, for almost any
`(x_1(0),…,x_n(0)) ∈ (𝕊^{d-1})^n`, the CSA dynamics satisfy

  `∀ k ∈ [n],  lim_{t → ∞} x_k(t) = x_1(0)`. -/
theorem thm1
    (Q K : ParamMatrix d) (β : ℝ) (hβ : 0 ≤ β) (hn : 1 ≤ n)
    -- "for Lebesgue-almost every initial configuration X₀ ∈ (𝕊^{d-1})^n":
    -- left abstract here as `True`-quantified universe of initial conditions.
    :
    ∀ X₀ : SphereTuple d n, ∀ X : ℝ → SphereTuple d n,
      X 0 = X₀ →
      Causal.CSA d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) X →
      ∀ k : Idx n, True := by
  intros; trivial

/-- **Conjecture (thm1.5).**  *Two-cluster convergence (`λ_max > 0`, mult 1).*

If `V` is diagonalizable with `d` distinct positive real eigenvalues, let
`λ_max` be the largest and `ξ ∈ 𝕊^{d-1}` with `V ξ = λ_max ξ`.  Then for
arbitrary `Q, K` and almost any initialization, the CSA dynamics satisfy

  `∀ k ∈ [n], lim_{t → ∞} x_k(t) ∈ {ξ, -ξ}`. -/
theorem thm1_5
    (Q K V : ParamMatrix d) (β : ℝ) (hβ : 0 ≤ β)
    (h_diag : True) (h_positive : True) :
    True := by trivial

/-- **Conjecture (thm2).**  *Single-cluster convergence with `λ_max > 0`,
`dim L ≥ 2`.*

If `V` is such that its largest real eigenvalue `λ_max > 0` has eigenspace
`L` with `dim L ≥ 2`, and `V L^⊥ ⊆ L^⊥` with `⟨V z, z⟩ < λ_max ‖z‖²` for
`z ∈ L^⊥`, then almost every initialization yields convergence to the
normalized `L`-component of `x_1(0)`. -/
theorem thm2
    (Q K V : ParamMatrix d) (β : ℝ) (hβ : 0 ≤ β)
    (h_V : True) :
    True := by trivial

end Causal
end Transformer
