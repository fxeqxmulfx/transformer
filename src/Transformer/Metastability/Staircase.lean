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

/-- **Lemma (lem: exact time scale of clustering).**

For the scalar Cauchy problem

  `u̇(t) = -c log β · sin(u(t)) + c(β)`,   `u(0) = u_0`,

with `|c(β)| ≤ K e^{-κ β} log β` and `κ > log β / β`, the times

  `t(β) = inf { t : u(t) ≤ √(log β / β) }`,
  `T(β) = inf { t : u(t) ≤ 1/√(β log β) }`,

satisfy

  `|t(β) - 2/c| ≤ (2 log tan(u_0 / 2) + log log β) / (c log β)`,

  `T(β) - t(β) ≤ (2 log log β) / (c log β) + O(1 / (β² log β))`.

The `O(1 / (β² log β))` remainder of the second bound is dropped: what is
stated is the leading term alone.

Source: arXiv:2410.06833v1, §6, `lem: exact time scale of clustering`. -/
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

/-- The hypotheses of `exact_time_scale` are satisfiable: at `u₀ = 0` and
`c(β) = 0` the equilibrium `u ≡ 0` solves the equation, and `β = e`,
`c = K = κ = 1` meet the remaining constraints. -/
example :
    Real.exp 1 ≤ Real.exp 1 ∧ (0 : ℝ) < 1 ∧
      Real.log (Real.exp 1) / Real.exp 1 < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
      |(0 : ℝ)| ≤ 1 * Real.exp (-(1 * Real.exp 1)) * Real.log (Real.exp 1) ∧
      ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ))
        (-1 * Real.log (Real.exp 1) * Real.sin ((fun _ : ℝ => (0 : ℝ)) t) + 0) t := by
  refine ⟨le_rfl, one_pos, ?_, le_rfl, zero_le_one, ?_, fun t => ?_⟩
  · rw [Real.log_exp, div_lt_one (Real.exp_pos 1)]
    linarith [Real.add_one_le_exp (1 : ℝ)]
  · rw [abs_zero, Real.log_exp]
    positivity
  · simpa using hasDerivAt_const t (0 : ℝ)

/-- **Theorem (thm: staircase).** *Staircase profile of the energy.*

For `n ≥ 2`, a well-prepared `(θ_i(0))_{i=1}^n ∈ 𝕋^n` and the modified `USA`
dynamics, there exists a time-reparametrization `τ_β` and a sequence
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` (with `k ≤ n`), together with a
piecewise-constant `φ_∞ ∈ L^∞(ℝ_{≥0}; [0, 1])`, such that

  `lim_{β → ∞} max_{i ∈ {1,…,k}} sup_{t ∈ (T_i, T_{i+1})}
              |𝖤_β(Θ(τ_β(t))) - φ_∞(t)| = 0`.

The limit is in `β`, so the statement is about a whole family of dynamics
indexed by `β`, and the `max`/`sup` is written out as uniform convergence:
for every `ε > 0` there is a `B` past which the error is below `ε` on every
plateau at once.  `φ_∞` is piecewise constant on the plateaux, and takes
values in `[0, 1]`.

A `Prop`-valued definition and not a theorem: this is `conj: saddle-to-saddle`
made precise for the modified dynamics, and none of it is proved here.

Source: arXiv:2410.06833v1, §6, `thm: staircase`. -/
def StaircaseProfile : Prop :=
  ∀ (θ θstar : ℝ → ℝ → Angles (n + 2)) (T_star : ℝ → ℝ) (τ m : ℝ → ℝ → ℝ),
    (∀ β : ℝ, 1 < β → isWellPrepared n β (θ β 0)) →
    (∀ β : ℝ, 1 < β → modifiedUSA n β (θ β) (θstar β) (T_star β)) →
    (∀ β : ℝ, 1 < β → staircaseReparam n β (θ β) (τ β) (m β)) →
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      k ≤ n + 2 ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ i : ℕ, i < k → ∀ s t : ℝ, s ∈ Set.Ioo (T i) (T (i + 1)) →
        t ∈ Set.Ioo (T i) (T (i + 1)) → φ s = φ t) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, i < k → ∀ t : ℝ, t ∈ Set.Ioo (T i) (T (i + 1)) →
          |torusEnergy (n + 2) β (θstar β (τ β t)) - φ t| < ε

end Metastability
end Transformer
