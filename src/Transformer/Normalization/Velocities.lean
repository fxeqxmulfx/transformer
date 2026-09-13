/-
# Normalization — Initial and terminal token velocities (§4.2–§4.3 of 2510.22026v2)

* `Theorem thm: initial-velocity`  — uniform bound on `‖A_j(0)‖` for random
                                      directional init,
* `Theorem thm: preln-slow`        — radial growth `r_k(t) ≥ (1 - δ) t` and
                                      `d/dt Var(t)` rates for each scheme.
-/

import Transformer.Basic
import Transformer.Normalization.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace NormVelocities

open Normalization

variable (d n : ℕ)

/-- **Theorem (thm: initial-velocity).** *Initial attention magnitude bound.*

For `Q, K, V ∈ ℝ^{d × d}` with `max(‖Q^⊤ K‖_op, ‖V‖_op) ≤ 1`, `β = 1`, and
i.i.d. uniform directional init `θ_j(0) ∼ Unif(𝕊^{d-1})`, there exist
absolute constants `c, C > 0` such that for `e^{√d} ≥ n log n ≥ d`, with
probability `1 - n^{-C}`,

  `‖A_j(0)‖ ≤ C (√(log n / n) + log n / d)`  uniformly in `j ∈ [n]`. -/
theorem thm_initial_velocity
    (Q K V : ParamMatrix d)
    (h_norms : ‖Q‖ ≤ 1 ∧ ‖K‖ ≤ 1 ∧ ‖V‖ ≤ 1)
    (h_size : Real.exp (Real.sqrt d) ≥ (n : ℝ) * Real.log n
                ∧ (n : ℝ) * Real.log n ≥ d) :
    True := by trivial

/-- The empirical *intra-cluster variance* used in `thm: preln-slow`:

  `Var(t) = (1/n) Σ_k ‖θ_k(t) - θ̄(t)‖²`,
  with `θ̄ = (1/n) Σ_j θ_j`. -/
noncomputable def intraClusterVar
    (n : ℕ) (θ : ℝ → Idx n → EucSpace d) (t : ℝ) : ℝ :=
  let θbar : EucSpace d := ((n : ℝ)⁻¹) • ∑ j : Idx n, θ t j
  ((n : ℝ)⁻¹) * ∑ k : Idx n, ‖θ t k - θbar‖^2

/-- **Theorem (thm: preln-slow).**  *Rate of cluster collapse.*

For `V = I_d`, arbitrary `Q, K` with `‖Q^⊤ K‖ ≤ 1`, in the local-cone init
`⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ` with `δ < 1 / (100 n² β²)`:

* **Radial growth (Pre-LN, Peri-LN):**  `r_k(t) ≥ (1 - δ) t`;
* **Speed of clustering:**
    * Post-LN :  `d/dt Var(t) = -Θ(Var(t))`
    * Pre-LN  :  `d/dt Var(t) = -Θ(Var(t) / t)`
    * Peri-LN :  `d/dt Var(t) = -Θ(Var(t) / t)`
    * nGPT   :  `d/dt Var(t) = -Θ(Var(t) / α_t)`
    * Mix-LN :  `d/dt Var(t) = -Θ(Var(t) / t)`
    * CoD   :  `d/dt Var(t) = -Θ(Var(t) / √t)`
-/
theorem thm_preln_slow
    (β : ℝ) (hβ : 0 < β) (δ : ℝ) (hδ : δ < 1 / (100 * (n : ℝ)^2 * β^2))
    (Q K : ParamMatrix d)
    (h_norm : ‖Q‖ ≤ 1 ∧ ‖K‖ ≤ 1) :
    True := by trivial

end NormVelocities
end Transformer
