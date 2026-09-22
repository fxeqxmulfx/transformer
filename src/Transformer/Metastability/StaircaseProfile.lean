/-
# Metastability — `thm: staircase`, and the plateau it needs

`Theorem thm: staircase` of §6 of arXiv:2410.06833v1: along the modified
`USA` dynamics, started from a well-prepared configuration and read in the
reparametrized time `τ_β`, the energy converges as `β → ∞` to a
piecewise-constant profile `φ_∞`, uniformly on each plateau.

**What the source says and what is changed here.**

* The energy is the normalized `2β 𝖤_β` (`circleEnergy`): with the printed
  `𝖤_β ≤ 1/(2β)` every profile converges to `0` (`Metastability.EnergyScale`).
* The plateaux are the source's `(T_i, T_{i+1})`, `i ∈ {1,…,k}`, the last one
  `(T_k, +∞)`; `conj: saddle-to-saddle` uses `i ∈ {0,…,k-1}` instead, and the
  theorem is taken as printed.
* The statement asks for `1 ≤ k`: with `k = 0` no plateau is quantified over,
  and the conclusion holds of `T ≡ 0`, `φ ≡ 0` for arbitrary curves and
  time changes — `staircase_profile_vacuous_at_zero`, proved below.
* The initial configuration `θ₀` is one and the same for every `β`, as the
  source's `(θ_i(0))_{i=1}^n` carries no `β`.
-/

import Transformer.Metastability.Staircase

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (n : ℕ)

open Perspective

/-- **Without a plateau the staircase statement is empty.**

Read with `k = 0`, the conclusion of `thm: staircase` quantifies over no
plateau: only `k ≤ n + 2`, `T_0 = 0`, `0 ≤ φ_∞ ≤ 1` and the constancy of
`φ_∞` on `[T_0, +∞)` survive, and `T ≡ 0`, `φ_∞ ≡ 0` satisfy them whatever
the dynamics does — the hypotheses of the theorem are not even needed, and
are not taken.

This is why `staircase_profile` carries `1 ≤ k`.

Source: arXiv:2410.06833v1, §6, `thm: staircase`. -/
theorem staircase_profile_vacuous_at_zero
    (θbar : ℝ → ℝ → Angles (n + 2)) (τ : ℝ → ℝ → ℝ) :
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      k ≤ n + 2 ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ i : ℕ, i ≤ k → ∀ s t : ℝ, T i ≤ s → T i ≤ t →
        (i < k → s < T (i + 1)) → (i < k → t < T (i + 1)) → φ s = φ t) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, 1 ≤ i → i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
          |circleEnergy β (θbar β (τ β t)) - φ t| < ε :=
  ⟨0, fun _ => 0, fun _ => 0, Nat.zero_le _, rfl,
    fun i hi => absurd hi (Nat.not_lt_zero i), fun _ => ⟨le_rfl, zero_le_one⟩,
    fun _ _ _ _ _ _ _ _ => rfl,
    fun _ _ => ⟨0, fun _ _ i h1 h2 => absurd (lt_of_lt_of_le h1 h2) (by omega)⟩⟩

/-- The hypotheses of `staircase_profile_vacuous_at_zero` are satisfiable: the
constant family of configurations and the identity reparametrization. -/
example : ∃ (θbar : ℝ → ℝ → Angles 2) (τ : ℝ → ℝ → ℝ),
    θbar = (fun _ _ _ => 0) ∧ τ = (fun _ t => t) := ⟨_, _, rfl, rfl⟩

/-- **Theorem (thm: staircase).** *Staircase profile of the energy.*

For `n ≥ 2`, a well-prepared `(θ_i(0))_{i=1}^n ∈ 𝕋^n`, the modified `USA`
dynamics `θ̄` and the reparametrization `τ_β` of `compt: reparam`, there are
times `0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` (with `1 ≤ k ≤ n`) and a
`φ_∞ ∈ L^∞(ℝ_{≥0}; [0, 1])`, constant on each `[T_i, T_{i+1})`, such that

  `lim_{β → ∞} max_{i ∈ {1,…,k}} sup_{t ∈ (T_i, T_{i+1})}
              |2β 𝖤_β(θ̄(τ_β(t))) - φ_∞(t)| = 0`.

The limit is in `β`, so the statement is about a whole family of dynamics
indexed by `β`, and the `max`/`sup` is written out as uniform convergence:
for every `ε > 0` there is a `B` past which the error is below `ε` on every
plateau at once.  See the module header for what differs from the source.

The three dynamical hypotheses stay inside the statement rather than becoming
binders: no solution of `modifiedUSA` started from a well-prepared
configuration is constructed in this development — that is the Cauchy
problem the source's §6 solves — so there is no witness to exhibit alongside
the theorem, and claiming one would be claiming the construction.

Source: arXiv:2410.06833v1, §6, `thm: staircase`. -/
theorem staircase_profile :
    ∀ (θ₀ : Angles (n + 2)) (θ θbar : ℝ → ℝ → Angles (n + 2)) (T_star : ℝ → ℝ)
      (τ m : ℝ → ℝ → ℝ),
    (∀ β : ℝ, 1 < β → isWellPrepared n β θ₀) →
    (∀ β : ℝ, 1 < β → θ β 0 = θ₀) →
    (∀ β : ℝ, 1 < β → modifiedUSA n β (θ β) (θbar β) (T_star β)) →
    (∀ β : ℝ, 1 < β → staircaseReparam n β (θbar β) (τ β) (m β)) →
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      1 ≤ k ∧ k ≤ n + 2 ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ i : ℕ, i ≤ k → ∀ s t : ℝ, T i ≤ s → T i ≤ t →
        (i < k → s < T (i + 1)) → (i < k → t < T (i + 1)) → φ s = φ t) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, 1 ≤ i → i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
          |circleEnergy β (θbar β (τ β t)) - φ t| < ε := by
  sorry

end Metastability
end Transformer
