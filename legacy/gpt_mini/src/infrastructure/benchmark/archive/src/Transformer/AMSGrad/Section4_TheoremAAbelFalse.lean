import Transformer.AMSGrad.Section4_TheoremAAbel

/-
# AMSGrad — the printed Abel bound fails even with the initial maximum

The scalar estimate needed by the printed proof of Theorem A still fails when
`β = β₁ = β_{1,1}`, as required by that theorem.  The counterexample uses
`a_t = t`, `β = 1/2`, `β_t = 1/2` at `t = 1` and at even times, and `0` at
other odd times.  Set `u_t = 1` at even times and `0` at odd times.  At `T = 16`
the two sides of the asserted estimate are `143/2` and `137/2`.

This refutes a scalar proof step, not Theorem A itself: the values `u_t` are
not shown to arise from an AMSGrad run.  In particular, the theorem
`Transformer.AMSGrad.theorem_A` is now proved by a different argument.

Source: arXiv:1904.03590v4, §3, the discussion of the telescoping step in
the proof of Theorem A.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- The scalar estimate with Theorem A's printed constants is false even when
the schedule starts at its stated maximum, `β₁ = β_{1,1}`.  Unlike the simpler
counterexample `not_abel_printed`, this witness has `a_t = t`, so its weights
increase strictly.  This is a counterexample to the proof step, not to the
regret theorem.  Source: arXiv:1904.03590v4, §3, Theorem A proof audit. -/
theorem not_abel_printed_initial_max :
    ¬ ∀ (a b u : ℕ → ℝ) (β E : ℝ), β < 1 → β = b 1 →
      (∀ t, 1 ≤ t → 0 ≤ a t) →
      (∀ t, 2 ≤ t → a (t - 1) ≤ a t) →
      (∀ t, 1 ≤ t → 0 ≤ b t ∧ b t ≤ β) →
      (∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ E) → ∀ T : ℕ, 1 ≤ T →
      ∑ t ∈ Icc 1 T, a t / (2 * (1 - b t)) * (u t - u (t + 1))
          + ∑ t ∈ Icc 2 T, b t * a (t - 1) / (2 * (1 - β)) * u t ≤
        E * a T / (1 - β) + E / (2 * (1 - β)) * ∑ t ∈ Icc 1 T, b t * a t := by
  intro h
  let b : ℕ → ℝ := fun t => if t = 1 ∨ t % 2 = 0 then 1 / 2 else 0
  let u : ℕ → ℝ := fun t => if t % 2 = 0 then 1 else 0
  have h' := h (fun t => (t : ℝ)) b u (1 / 2) 1
    (by norm_num) (by norm_num [b])
    (fun t _ => by positivity)
    (fun t _ => by exact_mod_cast Nat.sub_le t 1)
    (fun t _ => by dsimp [b]; split_ifs <;> norm_num)
    (fun t _ => by dsimp [u]; split_ifs <;> norm_num)
    16 (by norm_num)
  norm_num [b, u, Finset.sum_Icc_succ_top] at h'

end AMSGrad
end Transformer
