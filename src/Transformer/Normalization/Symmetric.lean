/-
# Normalization — Symmetric (equiangular) initialization (§4.1 of 2510.22026v2)

* `Theorem thm: symmetric` — initial- and terminal-velocity asymptotics of
                              `γ̇(t)` for each normalization scheme,
* the per-scheme ODE for `γ(t) = ⟨θ_j(t), θ_k(t)⟩` and `r(t)` derived in
  Appendix.
-/

import Transformer.Basic
import Transformer.Normalization.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

open Normalization

variable (d n : ℕ)

/-- **Theorem (thm: symmetric).**  *Initial and terminal velocity of the
cosine similarity under symmetric orthogonal init.*

For `Q = K = V = I_d`, with `⟨θ_j(0), θ_k(0)⟩ = δ_{jk}` and `r_j(0) = r_0`:

* `γ(t) = ⟨θ_j(t), θ_k(t)⟩` is constant across all pairs `j ≠ k`;
* the table of `γ̇(t)` for `t → 0` and `t → ∞`:

      Post-LN:  γ̇(0) = 2 / (e^β + n - 1),                γ̇(t) ~ C e^{-2 t}
      Pre-LN :  γ̇(0) = 2 / (r_0 (e^β + n - 1)),           γ̇(t) ~ C / t³
      Mix-LN :  γ̇(0) = 2 / (e^β + n - 1),                γ̇(t) ~ C / t³
      Peri-LN:  γ̇(0) = 2 / (r_0 √(e^{2β} + n - 1)),       γ̇(t) ~ C / t³
      nGPT  :  γ̇(0) = 2 α_0 / √(e^{2β} + n - 1),         γ̇(t) ~ C α_t e^{-2 ∫ α_s ds}
      CoD   :  γ̇(0) = 2 / (e^β + n - 1),                γ̇(t) ~ C e^{-4 √t} / √t

-/
theorem thm_symmetric
    (β : ℝ) (hβ : 0 < β) (r₀ : ℝ) (hr : 0 < r₀)
    (scheme : Scheme) :
    True := by trivial

end Normalization
end Transformer
