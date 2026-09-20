/-
# Metastability — `thm: staircase`, and the plateau it needs

`Theorem thm: staircase` of §6 of arXiv:2410.06833v1: along the modified
`USA` dynamics, started from a well-prepared configuration and read in the
reparametrized time `τ_β`, the energy converges as `β → ∞` to a
piecewise-constant profile `φ_∞`, uniformly on each plateau.

**What the source says and what is changed here.**  The statement asks for
`1 ≤ k`: at least one plateau.  The source has it — its sequence of times is
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` and the staircase it describes has
at least one step — but the condition is easy to lose in the transcription,
and losing it empties the theorem.  With `k = 0` no interval `(T_i, T_{i+1})`
is quantified over at all, every clause of the conclusion holds of `k = 0`,
`T ≡ 0`, `φ ≡ 0`, and the whole statement is provable in three lines for
*arbitrary* curves and reparametrizations, hypotheses and all.  That is
`staircase_profile_vacuous_at_zero` below, proved, so that the reason for the
extra hypothesis is recorded rather than asserted.
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
interval: the ordering of the times, the constancy of `φ_∞` on the plateaux
and the uniform convergence of the energy are all vacuous, and only
`k ≤ n + 2`, `T_0 = 0` and `0 ≤ φ_∞ ≤ 1` survive.  Those three are satisfied
by `T ≡ 0` and `φ_∞ ≡ 0` whatever the dynamics does — the hypotheses of the
theorem are not even needed, and are not taken.

This is why `staircase_profile` carries `1 ≤ k`.

Source: arXiv:2410.06833v1, §6, `thm: staircase`. -/
theorem staircase_profile_vacuous_at_zero
    (θstar : ℝ → ℝ → Angles (n + 2)) (τ : ℝ → ℝ → ℝ) :
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      k ≤ n + 2 ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ i : ℕ, i < k → ∀ s t : ℝ, s ∈ Set.Ioo (T i) (T (i + 1)) →
        t ∈ Set.Ioo (T i) (T (i + 1)) → φ s = φ t) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, i < k → ∀ t : ℝ, t ∈ Set.Ioo (T i) (T (i + 1)) →
          |torusEnergy (n + 2) β (θstar β (τ β t)) - φ t| < ε :=
  ⟨0, fun _ => 0, fun _ => 0, Nat.zero_le _, rfl,
    fun i hi => absurd hi (Nat.not_lt_zero i), fun _ => ⟨le_rfl, zero_le_one⟩,
    fun i hi => absurd hi (Nat.not_lt_zero i),
    fun _ _ => ⟨0, fun _ _ i hi => absurd hi (Nat.not_lt_zero i)⟩⟩

/-- The hypotheses of `staircase_profile_vacuous_at_zero` are satisfiable: the
constant family of configurations and the identity reparametrization. -/
example : ∃ (θstar : ℝ → ℝ → Angles 2) (τ : ℝ → ℝ → ℝ),
    θstar = (fun _ _ _ => 0) ∧ τ = (fun _ t => t) := ⟨_, _, rfl, rfl⟩

/-- **Theorem (thm: staircase).** *Staircase profile of the energy.*

For `n ≥ 2`, a well-prepared `(θ_i(0))_{i=1}^n ∈ 𝕋^n` and the modified `USA`
dynamics, there exists a time-reparametrization `τ_β` and a sequence
`0 = T_0 < T_1 < ⋯ < T_k < T_{k+1} = +∞` (with `1 ≤ k ≤ n`), together with a
piecewise-constant `φ_∞ ∈ L^∞(ℝ_{≥0}; [0, 1])`, such that

  `lim_{β → ∞} max_{i ∈ {1,…,k}} sup_{t ∈ (T_i, T_{i+1})}
              |𝖤_β(Θ(τ_β(t))) - φ_∞(t)| = 0`.

The limit is in `β`, so the statement is about a whole family of dynamics
indexed by `β`, and the `max`/`sup` is written out as uniform convergence:
for every `ε > 0` there is a `B` past which the error is below `ε` on every
plateau at once.  `φ_∞` is piecewise constant on the plateaux, and takes
values in `[0, 1]`.

`1 ≤ k` is the plateau the source's `T_1 < ⋯ < T_k` carries and without which
the statement is `staircase_profile_vacuous_at_zero`, i.e. nothing.

This is `conj: saddle-to-saddle` made precise for the modified dynamics, and
none of it is proved here.

The three hypotheses stay inside the statement rather than becoming binders:
no solution of `modifiedUSA` started from a well-prepared configuration is
constructed in this development — that is the Cauchy problem the survey's §6
solves — so there is no witness to exhibit alongside the theorem, and claiming
one would be claiming the construction.

Source: arXiv:2410.06833v1, §6, `thm: staircase`. -/
theorem staircase_profile :
    ∀ (θ θstar : ℝ → ℝ → Angles (n + 2)) (T_star : ℝ → ℝ) (τ m : ℝ → ℝ → ℝ),
    (∀ β : ℝ, 1 < β → isWellPrepared n β (θ β 0)) →
    (∀ β : ℝ, 1 < β → modifiedUSA n β (θ β) (θstar β) (T_star β)) →
    (∀ β : ℝ, 1 < β → staircaseReparam n β (θ β) (τ β) (m β)) →
    ∃ (k : ℕ) (T : ℕ → ℝ) (φ : ℝ → ℝ),
      1 ≤ k ∧ k ≤ n + 2 ∧ T 0 = 0 ∧
      (∀ i : ℕ, i < k → T i < T (i + 1)) ∧
      (∀ t : ℝ, φ t ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ i : ℕ, i < k → ∀ s t : ℝ, s ∈ Set.Ioo (T i) (T (i + 1)) →
        t ∈ Set.Ioo (T i) (T (i + 1)) → φ s = φ t) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, i < k → ∀ t : ℝ, t ∈ Set.Ioo (T i) (T (i + 1)) →
          |torusEnergy (n + 2) β (θstar β (τ β t)) - φ t| < ε := by
  sorry

end Metastability
end Transformer
