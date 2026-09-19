/-
# Sums over `B` independent uniform seeds

Support for arXiv:2601.22813v2, Appendix A: the plot there averages `B`
independent quantized gradients, so every expectation it takes is a sum over
`Fin B → Fin N`, the tuples of `B` seeds drawn out of `N`.

Two sums carry the whole argument.  Writing `c` for a centred quantity —
`∑ i, c i = 0`, which is what unbiasedness says — the sum of `∑ b, c (ω b)`
over all tuples vanishes (`sum_pi_sum_eq_zero`), and the sum of its square is
`B · N^B · ∑ i, c i ^ 2` (`sum_pi_sum_sq`): the cross terms of the square
cancel because the seeds are independent, and only the `B` diagonal terms
survive.  That is the `1/B` of Appendix A, before any of it is divided by
`N^B`.

Both are proved by induction on `B`, splitting a tuple of length `B + 1` into
its first seed and the rest (`sum_pi_succ`).
-/

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

namespace Transformer
namespace Quartet

variable {N : ℕ}

/-- A sum over tuples of `B + 1` seeds splits into the first seed and the
remaining `B` (Appendix A: the draws are independent). -/
theorem sum_pi_succ (B : ℕ) (F : (Fin (B + 1) → Fin N) → ℝ) :
    ∑ ω : Fin (B + 1) → Fin N, F ω =
      ∑ i : Fin N, ∑ ω' : Fin B → Fin N, F (Fin.cons i ω') := by
  rw [← Fintype.sum_prod_type fun p : Fin N × (Fin B → Fin N) => F (Fin.cons p.1 p.2)]
  exact (Fintype.sum_equiv (Fin.consEquiv fun _ : Fin (B + 1) => Fin N)
    (fun p => F (Fin.cons p.1 p.2)) F fun _ => rfl).symm

/-- The number of tuples of `B` seeds out of `N`. -/
theorem card_pi (B : ℕ) : (Fintype.card (Fin B → Fin N) : ℝ) = (N : ℝ) ^ B := by
  simp

/-- **A centred quantity averages to zero over the seeds** (Appendix A, "`Ĝ`
is unbiased, i.e. `E_ω Ĝ(ω) = G`"): summing `∑ b, c (ω b)` over every tuple of
`B` seeds gives `0`. -/
theorem sum_pi_sum_eq_zero {c : Fin N → ℝ} (hc : ∑ i, c i = 0) (B : ℕ) :
    ∑ ω : Fin B → Fin N, ∑ b, c (ω b) = 0 := by
  induction B with
  | zero => simp
  | succ B ih =>
    rw [sum_pi_succ]
    have hterm : ∀ i : Fin N,
        (∑ ω' : Fin B → Fin N, ∑ b : Fin (B + 1), c ((Fin.cons i ω' : Fin (B + 1) → Fin N) b))
          = (N : ℝ) ^ B * c i + ∑ ω' : Fin B → Fin N, ∑ b : Fin B, c (ω' b) := by
      intro i
      simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
      rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, card_pi]
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib, ← Finset.mul_sum, hc,
      ih, Finset.sum_const]
    simp

/-- The hypothesis of `sum_pi_sum_eq_zero` and `sum_pi_sum_sq` is satisfiable,
and is exactly unbiasedness: a quantizer that overshoots on one seed and
undershoots by as much on the other is centred. -/
example : ∑ i : Fin 2, (if i = 0 then (1 : ℝ) else -1) = 0 := by
  simp [Fin.sum_univ_two]

/-- **The square of a centred average keeps only its diagonal** (Appendix A,
the `~1/B` decay): summing `(∑ b, c (ω b))^2` over every tuple of `B` seeds
gives `B · N^B · ∑ i, c i ^ 2`, once multiplied by `N` to clear the one seed
the diagonal term integrates out.  The `B` cross terms of the square vanish by
`sum_pi_sum_eq_zero`. -/
theorem sum_pi_sum_sq {c : Fin N → ℝ} (hc : ∑ i, c i = 0) (B : ℕ) :
    (N : ℝ) * ∑ ω : Fin B → Fin N, (∑ b, c (ω b)) ^ 2 =
      B * (N : ℝ) ^ B * ∑ i, c i ^ 2 := by
  induction B with
  | zero => simp
  | succ B ih =>
    rw [sum_pi_succ]
    have hterm : ∀ i : Fin N,
        (∑ ω' : Fin B → Fin N, (∑ b : Fin (B + 1), c ((Fin.cons i ω' : Fin (B + 1) → Fin N) b)) ^ 2)
          = (N : ℝ) ^ B * c i ^ 2
            + 2 * c i * (∑ ω' : Fin B → Fin N, ∑ b : Fin B, c (ω' b))
            + ∑ ω' : Fin B → Fin N, (∑ b : Fin B, c (ω' b)) ^ 2 := by
      intro i
      simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
      have hexp : ∀ ω' : Fin B → Fin N,
          (c i + ∑ b : Fin B, c (ω' b)) ^ 2 =
            c i ^ 2 + 2 * c i * (∑ b : Fin B, c (ω' b)) + (∑ b : Fin B, c (ω' b)) ^ 2 :=
        fun _ => by ring
      rw [Finset.sum_congr rfl fun ω' _ => hexp ω', Finset.sum_add_distrib,
        Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, card_pi,
        ← Finset.mul_sum]
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
      Finset.sum_add_distrib, sum_pi_sum_eq_zero hc B, ← Finset.mul_sum, ← Finset.sum_mul]
    simp only [mul_zero, add_zero, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    push_cast
    linear_combination (N : ℝ) * ih

end Quartet
end Transformer
