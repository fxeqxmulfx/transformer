/-
# The simplex relaxation of rowwise softmax

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.1,
`sec:convex_attention`.  This proves the displayed inclusion of softmax
attention weights in the simplex set, using the row-stochastic convention
required by rowwise softmax.
-/

import Transformer.Convexifying.Section3_Models

open scoped BigOperators

namespace Transformer.Convexifying

/-- Rowwise softmax of a square score matrix. -/
noncomputable def rowSoftmax {n : ℕ} (U : Mat n n) : Mat n n :=
  fun r k => Real.exp (U r k) / ∑ j, Real.exp (U r j)

/-- The rowwise softmax denominator is positive when there is a token.
Source: arXiv:2211.11052v1, §3.1. -/
theorem softmax_denominator_pos {n : ℕ} (hn : 0 < n)
    (U : Mat n n) (r : Fin n) :
    0 < ∑ j, Real.exp (U r j) := by
  have hnonempty : (Finset.univ : Finset (Fin n)).Nonempty :=
    ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  exact Finset.sum_pos (fun j _ => Real.exp_pos _) hnonempty

/-- Every row of softmax is a probability distribution.
Source: arXiv:2211.11052v1, §3.1, displayed definition of `Δ`. -/
theorem rowSoftmax_stochastic {n : ℕ} (hn : 0 < n) (U : Mat n n) :
    IsRowStochastic (rowSoftmax U) := by
  intro r
  constructor
  · intro k
    exact div_nonneg (Real.exp_pos _).le (softmax_denominator_pos hn U r).le
  · simp only [rowSoftmax]
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt (softmax_denominator_pos hn U r))

/-- The paper's displayed relaxation: for every score matrix and token
matrix, a row-stochastic attention matrix gives exactly the same output as
rowwise softmax.  The witness is the softmax matrix itself.
Source: arXiv:2211.11052v1, §3.1, display before `eq:simplex_regression`. -/
theorem softmax_has_simplex_representative {n d : ℕ} (hn : 0 < n)
    (U : Mat n n) (X : Fin n → Vec d) :
    ∃ W : Mat n n, IsRowStochastic W ∧
      ∀ r q, (∑ k, rowSoftmax U r k * X k q) =
        ∑ k, W r k * X k q := by
  exact ⟨rowSoftmax U, rowSoftmax_stochastic hn U, fun _ _ => rfl⟩

/-- The nonempty-token hypothesis above is satisfiable. -/
example : 0 < (1 : ℕ) := by norm_num

end Transformer.Convexifying
