/-
# Metastability — Beyond metastability: the staircase profile (§6 of 2410.06833v1)

Equations and statements covered:

* `Problem conj: saddle-to-saddle` — the staircase open problem,
* `Definition d: init_s1`         — well-prepared configurations on `𝕊^1`,
* `Definition def: modified USA`  — modified `USA` enforcing collisions,
* `compt: reparam`                — time-reparametrization `τ_β`,
* `compt: derivative`             — derivative of the reparametrized angle,
* `Lemma lem: exact time scale of clustering`.

`Theorem thm: staircase` itself is `Metastability.StaircaseProfile`.
-/

import Transformer.Basic
import Transformer.Perspective.Section6_Circle
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (n : ℕ)

open Perspective

/-- **Definition (d: init_s1).** *Well-prepared configuration on `𝕊^1`.*

`(θ_1,…,θ_n) ∈ 𝕋^n` is *well-prepared* if `0 ≤ θ_1 < ⋯ < θ_n ≤ π` and
there exists a numerical constant `c > 1` such that for all `i ∈ {2,…,n-1}`
and `k > i`,

  `cos(θ_i - θ_1) > cos(θ_k - θ_i) + (c log β / β)`.

The configuration is indexed by `Idx (n + 2)`: the survey assumes `n ≥ 2`
throughout §6, and carrying the two extremal indices `0` and `n + 1` in the
type is what lets `θ_1` and `θ_n` be written without a side condition.

Source: arXiv:2410.06833v1, §6, `d: init_s1`. -/
def isWellPrepared (β : ℝ) (θ : Angles (n + 2)) : Prop :=
  (∀ i j : Idx (n + 2), (i : ℕ) < j → θ i < θ j) ∧
  0 ≤ θ 0 ∧ θ (Fin.last (n + 1)) ≤ Real.pi ∧
  ∃ c : ℝ, 1 < c ∧
    ∀ i : Idx (n + 2), 1 ≤ (i : ℕ) → ((i : ℕ) + 1) < n + 2 →
      ∀ k : Idx (n + 2), (i : ℕ) < k →
        Real.cos (θ k - θ i) + c * Real.log β / β < Real.cos (θ i - θ 0)

/-- **Definition (def: modified USA).** *Modified `USA` enforcing collisions.*

`θ` solves `eq: usa.angles`; at the time `T_*` at which its first two particles
come within `1 / √(β log β)` of each other they are forced to merge, and from
`T_*` onwards the surviving particles — the indices `i ≥ 1`, the merged pair
being counted twice through `j = 0` and `j = 1` — again solve `eq: usa.angles`.

Source: arXiv:2410.06833v1, §6, `def: modified USA`. -/
def modifiedUSA
    (β : ℝ) (θ θstar : ℝ → Angles (n + 2)) (T_star : ℝ) : Prop :=
  angularUSA (n + 2) β θ ∧
  (∀ t : ℝ, t < T_star → θstar t = θ t) ∧
  |θ T_star 0 - θ T_star 1| ≤ 1 / Real.sqrt (β * Real.log β) ∧
  (∀ t : ℝ, T_star ≤ t → θstar t 0 = θstar t 1) ∧
  ∀ t : ℝ, T_star ≤ t → ∀ i : Idx (n + 2), 1 ≤ (i : ℕ) →
    HasDerivAt (fun s => θstar s i)
      (-(1 / ((n : ℝ) + 2)) *
        ∑ j : Idx (n + 2),
          Real.exp (β * Real.cos (θstar t i - θstar t j)) *
            Real.sin (θstar t i - θstar t j)) t

/-- **Time reparametrization (compt: reparam).**

  `τ̇_β(t) = log β · max_{(i,j): |θ_i - θ_j| > 1/√(β log β)}
                exp(β (1 - cos(θ_i(t) - θ_j(t))))`,    `τ_β(0) = 0`.

The maximum is carried by a separate function `m`, specified by `IsGreatest`
over the values of the admissible pairs, rather than by a `Finset.max'` whose
nonemptiness proof would have to live inside the statement.

Source: arXiv:2410.06833v1, §6, `compt: reparam`, `compt: derivative`. -/
def staircaseReparam
    (β : ℝ) (θ : ℝ → Angles (n + 2)) (τ m : ℝ → ℝ) : Prop :=
  τ 0 = 0 ∧
  (∀ t : ℝ, IsGreatest
    { r : ℝ | ∃ p : Idx (n + 2) × Idx (n + 2),
        1 / Real.sqrt (β * Real.log β) < |θ t p.1 - θ t p.2| ∧
        r = Real.exp (β * (1 - Real.cos (θ t p.1 - θ t p.2))) } (m t)) ∧
  ∀ t : ℝ, HasDerivAt τ (Real.log β * m t) t

/-- **Lemma (lem: exact time scale of clustering) is false as stated.**

For the scalar Cauchy problem

  `u̇(t) = -c log β · sin(u(t)) + c(β)`,   `u(0) = u_0`,

with `u_0 ∈ [0, 1]`, `β ≥ e`, `c > 0`, `K > 0`, `κ > log β / β` and
`|c(β)| ≤ K e^{-κ β} log β`, the survey claims that the times

  `t(β) = inf { t ≥ 0 : u(t) ≤ √(log β / β) }`,
  `T(β) = inf { t ≥ 0 : u(t) ≤ 1/√(β log β) }`

satisfy

  `|t(β) - 2/c| ≤ (2 log tan(u_0 / 2) + log log β) / (c log β)`,
  `T(β) - t(β) ≤ (2 log log β) / (c log β) + O(1 / (β² log β))`.

**The first inequality is false, at `u_0 = 0`, which the source's `u_0 ∈ [0,1]`
admits.**  The equilibrium `u ≡ 0` solves the Cauchy problem with `c(β) = 0`,
it is already below the threshold at `t = 0`, so `t(β) = 0` and the left-hand
side is `2/c > 0`; the right-hand side is `2 log tan(0) / (c log β)`, which is
`-∞` on paper and `0` in Lean, where `log 0 = 0`.  Either way the inequality
fails, and it fails at `β = e` and at every larger `β`.

Two further things are wrong with it, and neither is repaired by excluding
`u_0 = 0`.  *The centre is wrong.*  Separating variables in
`u̇ = -c log β sin u` gives `log tan(u(t)/2) = log tan(u_0/2) - c t log β`, so
the crossing of `√(log β / β)` happens at `1/(2c) + O(1/log β)`, not at `2/c`;
the survey's own proof writes `v̇ = u̇/(2 sin u)` for `v = log tan(u/2)`, whose
derivative is `u̇ / sin u`, and even its own display then gives `1/c` rather
than the `2/c` it reports.  *And the claim is asymptotic* — the source reads
"as `β → +∞`" — while the inequality is stated at a fixed `β`; a faithful
formalization is a statement about the family `β ↦ t(β)`, not this one.

Source: arXiv:2410.06833v1, §6, `lem: exact time scale of clustering`. -/
theorem not_exact_time_scale :
    ¬ ∀ (c K κ u₀ β : ℝ), Real.exp 1 ≤ β → 0 < c → 0 < K →
        Real.log β / β < κ → 0 ≤ u₀ → u₀ ≤ 1 →
        ∀ (u : ℝ → ℝ) (cβ : ℝ), |cβ| ≤ K * Real.exp (-(κ * β)) * Real.log β →
          (∀ t : ℝ, HasDerivAt u (-c * Real.log β * Real.sin (u t) + cβ) t) →
          u 0 = u₀ →
          |sInf { t : ℝ | 0 ≤ t ∧ u t ≤ Real.sqrt (Real.log β / β) } - 2 / c|
              ≤ (2 * Real.log (Real.tan (u₀ / 2)) + Real.log (Real.log β))
                  / (c * Real.log β)
            ∧ sInf { t : ℝ | 0 ≤ t ∧ u t ≤ 1 / Real.sqrt (β * Real.log β) }
                - sInf { t : ℝ | 0 ≤ t ∧ u t ≤ Real.sqrt (Real.log β / β) }
              ≤ (2 * Real.log (Real.log β)) / (c * Real.log β) := by
  intro h
  have hone : Real.log (Real.exp 1) = 1 := Real.log_exp 1
  have hκ : Real.log (Real.exp 1) / Real.exp 1 < 1 := by
    rw [hone, div_lt_one (Real.exp_pos 1)]
    linarith [Real.add_one_le_exp (1 : ℝ)]
  have hbound : |(0 : ℝ)| ≤ 1 * Real.exp (-(1 * Real.exp 1)) * Real.log (Real.exp 1) := by
    rw [abs_zero, hone]
    positivity
  have hode : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ))
      (-1 * Real.log (Real.exp 1) * Real.sin ((fun _ : ℝ => (0 : ℝ)) t) + 0) t :=
    fun t => by simpa using hasDerivAt_const t (0 : ℝ)
  have key := (h 1 1 1 0 (Real.exp 1) le_rfl one_pos one_pos hκ le_rfl zero_le_one
    (fun _ => 0) 0 hbound hode rfl).1
  have hset : { t : ℝ | 0 ≤ t ∧ (fun _ : ℝ => (0 : ℝ)) t
        ≤ Real.sqrt (Real.log (Real.exp 1) / Real.exp 1) } = Set.Ici (0 : ℝ) := by
    ext t
    simp [Real.sqrt_nonneg]
  rw [hset, csInf_Ici, hone, Real.log_one] at key
  norm_num at key

end Metastability
end Transformer
