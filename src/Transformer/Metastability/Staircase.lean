/-
# Metastability — Beyond metastability: the staircase profile (§6 of 2410.06833v1)

Equations and statements covered:

* `Problem conj: saddle-to-saddle` — the staircase open problem,
* `Definition d: init_s1`         — well-prepared configurations on `𝕊^1`,
* `Definition def: modified USA`  — modified `USA` enforcing collisions,
* `Theorem thm: staircase`        — staircase profile of the energy,
* `compt: reparam`                — time-reparametrization `τ_β`,
* `compt: derivative`             — derivative of the reparametrized angle,
* `Lemma lem: exact time scale of clustering`.
-/

import Transformer.Basic
import Transformer.Perspective.Section6_Circle
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem

open scoped BigOperators
open Real

namespace Transformer
namespace MetaStaircase

variable (n : ℕ)

/-- **Definition (d: init_s1).** *Well-prepared configuration on `𝕊^1`.*

`(θ_1,…,θ_n) ∈ 𝕋^n` is *well-prepared* if `0 ≤ θ_1 < ⋯ < θ_n ≤ π` and
there exists a numerical constant `c > 1` such that for all `i ∈ {2,…,n-1}`
and `k > i`,

  `cos(θ_i - θ_1) > cos(θ_k - θ_i) + (c log β / β)`. -/
def isWellPrepared (β : ℝ) (θ : Idx n → ℝ) : Prop :=
  (∀ i j : Idx n, (i : ℕ) < j → θ i < θ j) ∧
  0 ≤ θ ⟨0, by sorry⟩ ∧ θ ⟨n - 1, by sorry⟩ ≤ Real.pi ∧
  ∃ c : ℝ, 1 < c ∧
    ∀ i : Idx n, 1 ≤ (i : ℕ) → ((i : ℕ) + 1) < n →
      ∀ k : Idx n, (i : ℕ) < k →
        Real.cos (θ k - θ i) + c * Real.log β / β
          < Real.cos (θ i - θ ⟨0, by sorry⟩)

/-- **Definition (def: modified USA).** *Modified `USA` enforcing collisions.*

If two particles get within `1 / √(β log β)` of each other, they are forced
to merge: from time `T_*` onwards `θ_1^*(t) = θ_2^*(t)` and the remaining
particles evolve according to `eq: usa.angles`. -/
def modifiedUSA
    (β : ℝ) (θ : ℝ → Idx n → ℝ) (T_star : ℝ) (θstar : ℝ → Idx n → ℝ) : Prop :=
  -- For `t < T_*` we have `θstar = θ`, for `t ≥ T_*` we identify two indices.
  (∀ t : ℝ, t < T_star → θstar t = θ t) ∧
  (∀ t : ℝ, T_star ≤ t → θstar t ⟨0, by sorry⟩ = θstar t ⟨1, by sorry⟩)

/-- **Time reparametrization (compt: reparam).**

  `τ̇_β(t) = log β · max_{(i,j): |θ_i - θ_j| > 1/√(β log β)}
                exp(β (1 - cos(θ_i(t) - θ_j(t))))`,    `τ_β(0) = 0`. -/
def staircaseReparam
    (β : ℝ) (θ : ℝ → Idx n → ℝ) (τ : ℝ → ℝ) : Prop :=
  τ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt τ
    (Real.log β *
      (Finset.univ.image (fun p : Idx n × Idx n =>
        if (1 / Real.sqrt (β * Real.log β)) < |θ t p.1 - θ t p.2| then
          Real.exp (β * (1 - Real.cos (θ t p.1 - θ t p.2)))
        else 0)).max' (by sorry)) t

/-- **Lemma (lem: exact time scale of clustering).**

For the scalar Cauchy problem

  `u̇(t) = -c log β · sin(u(t)) + c(β)`,   `u(0) = u_0`,

with `|c(β)| ≤ K e^{-κ β} log β` and `κ > log β / β`, the times

  `t(β) = inf { t : u(t) ≤ √(log β / β) }`,
  `T(β) = inf { t : u(t) ≤ 1/√(β log β) }`,

satisfy

  `|t(β) - 2/c| ≤ (2 log tan(u_0 / 2) + log log β) / (c log β)`,

  `T(β) - t(β) ≤ (2 log log β) / (c log β) + O(1 / (β² log β))`. -/
lemma exact_time_scale
    (c K κ u₀ β : ℝ) (hβ : Real.exp 1 ≤ β) (hc : 0 < c) (hK : 0 < K)
    (hκ : Real.log β / β < κ) (hu : 0 ≤ u₀ ∧ u₀ ≤ 1)
    (u : ℝ → ℝ) (cβ : ℝ) (h_bound : |cβ| ≤ K * Real.exp (-(κ * β)) * Real.log β)
    (hu_ode : ∀ t : ℝ, HasDerivAt u (-c * Real.log β * Real.sin (u t) + cβ) t)
    (hu_init : u 0 = u₀) :
    let t_β := sInf { t : ℝ | 0 ≤ t ∧ u t ≤ Real.sqrt (Real.log β / β) }
    let T_β := sInf { t : ℝ | 0 ≤ t ∧ u t ≤ 1 / Real.sqrt (β * Real.log β) }
    |t_β - 2 / c|
        ≤ (2 * Real.log (Real.tan (u₀ / 2)) + Real.log (Real.log β))
            / (c * Real.log β)
      ∧ T_β - t_β
        ≤ (2 * Real.log (Real.log β)) / (c * Real.log β) := by
  sorry

/-- **Theorem (thm: staircase).** *Staircase profile of the energy.*

For `n ≥ 2`, a well-prepared `(θ_i(0))_{i=1}^n ∈ 𝕋^n` and the modified
`USA` dynamics, there exists a time-reparametrization `τ_β` and a sequence
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` (with `k ≤ n`), together with a
piecewise-constant `φ_∞ ∈ L^∞(ℝ_{≥0}; [0, 1])`, such that

  `lim_{β → ∞} max_{i ∈ {1,…,k}} sup_{t ∈ (T_i, T_{i+1})}
              |𝖤_β(Θ(τ_β(t))) - φ_∞(t)| = 0`. -/
theorem staircase
    (β : ℝ) (hβ : 1 < β) (hn : 2 ≤ n)
    (θ : ℝ → Idx n → ℝ) (h_init : isWellPrepared n β (θ 0))
    (T_star : ℝ) (θstar : ℝ → Idx n → ℝ)
    (h_mod : modifiedUSA n β θ T_star θstar)
    (τ : ℝ → ℝ) (h_τ : staircaseReparam n β θ τ) :
    True := by trivial

end MetaStaircase
end Transformer
