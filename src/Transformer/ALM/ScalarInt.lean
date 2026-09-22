/-
# The construction as implemented: scalar integer keys

The `LookUp` primitive compiles to a head with *distinct integer scalar* keys.
In that situation at most **two** keys sit at any given distance `t` from the
query (namely `q - t` and `q + t`), so `softmax_winner_lengthfree` applies
with `M = 2` and yields a bound free of the token count `n`.

Contrast `softmax_winner_ge`, whose bound `1 - (n-1)e^{-β}` is vacuous once
`n > 1 + e^{β}`.
-/

import Transformer.ALM.Defs
import Transformer.ALM.Lattice

open scoped BigOperators
open Real

namespace Transformer
namespace ALM

/-- **The bound actually needed for the VM.**  For distinct integer scalar
keys the winner's softmax weight satisfies

  `w i₀ ≥ 1 / (1 + 2 e^{-β}/(1 - e^{-β}))`,

with **no dependence on the number of tokens `n`**. -/
theorem softmax_winner_scalar_int {n : ℕ} (β : ℝ) (hβ : 0 < β)
    (K : Fin n → ℤ) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β))))
      ≤ Real.exp (β * sScore (K i₀) (K i₀)) / ∑ j, Real.exp (β * sScore (K i₀) (K j)) := by
  set g : Fin n → ℕ := fun j => (K j - K i₀).natAbs with hg
  have hcast : ∀ j, ((g j : ℝ)) = |(K j : ℝ) - (K i₀ : ℝ)| := by
    intro j
    rw [hg]
    push_cast [Nat.cast_natAbs]
    ring_nf
  -- margins dominate the integer distance
  have hgap : ∀ j, j ≠ i₀ → sScore (K i₀) (K j) + (g j : ℝ) ≤ sScore (K i₀) (K i₀) := by
    intro j hj
    have hne : (K j : ℝ) - (K i₀ : ℝ) ≠ 0 := by
      have : K j ≠ K i₀ := fun h => hj (hinj h)
      simpa [sub_eq_zero] using fun h => this (by exact_mod_cast h)
    have h1 : (1 : ℝ) ≤ |(K j : ℝ) - (K i₀ : ℝ)| := by
      rw [← hcast j]
      have : 1 ≤ g j := by
        rw [hg]
        refine Int.natAbs_pos.mpr ?_
        intro h
        exact hne (by push_cast [sub_eq_zero.mp (by exact h : K j - K i₀ = 0)]; ring)
      exact_mod_cast this
    have hdiff : sScore (K i₀) (K i₀) - sScore (K i₀) (K j)
        = ((K j : ℝ) - (K i₀ : ℝ)) ^ 2 := by
      simp only [sScore]; ring
    have hsq : |(K j : ℝ) - (K i₀ : ℝ)| ^ 2 = ((K j : ℝ) - (K i₀ : ℝ)) ^ 2 := sq_abs _
    rw [hcast j]
    nlinarith [abs_nonneg ((K j : ℝ) - (K i₀ : ℝ))]
  have hg1 : ∀ j, j ≠ i₀ → 1 ≤ g j := by
    intro j hj
    rw [hg]
    refine Int.natAbs_pos.mpr ?_
    intro h
    exact hj (hinj (by omega))
  -- at most two integer keys lie at any given distance from the query
  have hM : ∀ t : ℕ, ((Finset.univ.erase i₀).filter (fun j => g j = t)).card ≤ 2 := by
    intro t
    have hcard : ((Finset.univ.erase i₀).filter (fun j => g j = t)).card
        ≤ ({K i₀ - (t : ℤ), K i₀ + (t : ℤ)} : Finset ℤ).card := by
      refine Finset.card_le_card_of_injOn K ?_ ?_
      · intro j hj
        have ht : (K j - K i₀).natAbs = t := (Finset.mem_filter.mp hj).2
        have hcases : K j - K i₀ = (t : ℤ) ∨ K j - K i₀ = -(t : ℤ) := by
          rcases Int.natAbs_eq (K j - K i₀) with h | h
          · left; rw [h, ht]
          · right; rw [h, ht]
        rcases hcases with h | h
        · have heq : K j = K i₀ + (t : ℤ) := by omega
          rw [heq]
          exact Finset.mem_insert_of_mem (Finset.mem_singleton.mpr rfl)
        · have heq : K j = K i₀ - (t : ℤ) := by omega
          rw [heq]
          exact Finset.mem_insert_self _ _
      · intro a _ b _ h; exact hinj h
    refine le_trans hcard ?_
    exact le_trans (Finset.card_insert_le _ _) (by simp)
  have := softmax_winner_lengthfree β hβ (fun j => sScore (K i₀) (K j)) i₀ g 2 hgap hg1 hM
  simpa using this

/-- The hypotheses are satisfiable: the keys `0, 1, 2`, which are distinct,
at `β = 1`. -/
example : let K : Fin 3 → ℤ := fun j => (j.val : ℤ)
    1 / (1 + 2 * (Real.exp (-1) / (1 - Real.exp (-1))))
      ≤ Real.exp (1 * sScore (K 0) (K 0)) / ∑ j, Real.exp (1 * sScore (K 0) (K j)) := by
  intro K
  have hK : Function.Injective K := fun a b h => Fin.ext (by simpa [K] using h)
  exact softmax_winner_scalar_int 1 one_pos K hK 0

end ALM
end Transformer
