module

public import Transformer.MHC.Section4_Stability

/-!
# Finite Sinkhorn normalization

arXiv:2512.24880v2, §4.2: exponentiate the logits, then alternate column
and row normalization. Twenty iterations are an implementation choice;
exact double stochasticity in §4.1 is an ideal constraint. This module
proves the finite algorithm's positivity and final row normalization,
and exhibits a positive input for which one cycle is not doubly stochastic.
The paper's limiting convergence claim is outside this module's scope.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.MHC

variable {ι : Type*} [Fintype ι]

/-- Normalize every row, arXiv:2512.24880v2, §4.2, operator `T_r`.
No implementation epsilon is added to the exact mathematical sum. -/
noncomputable def rowNormalize (M : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of (fun i j => M i j / ∑ k, M i k)

/-- Normalize every column, arXiv:2512.24880v2, §4.2, operator `T_c`. -/
noncomputable def columnNormalize (M : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of (fun i j => M i j / ∑ k, M k j)

/-- One Sinkhorn cycle in the order printed in arXiv:2512.24880v2,
§4.2: `M^(t) = T_r (T_c (M^(t-1)))`. -/
noncomputable def sinkhornCycle (M : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  rowNormalize (columnNormalize M)

/-- A finite number of cycles starting from a positive matrix.
Source: arXiv:2512.24880v2, §4.2, normalization recurrence. -/
noncomputable def sinkhornIterate (M : Matrix ι ι ℝ) : ℕ → Matrix ι ι ℝ
  | 0 => M
  | t + 1 => sinkhornCycle (sinkhornIterate M t)

/-- The paper's logit parameterization before finite normalization.
Source: arXiv:2512.24880v2, §4.2, `M^(0) = exp(H_res)`. -/
noncomputable def finiteSinkhorn (logits : Matrix ι ι ℝ) (t : ℕ) : Matrix ι ι ℝ :=
  sinkhornIterate (Matrix.of (fun i j => Real.exp (logits i j))) t

variable [Nonempty ι]

/-- Positive rows have positive normalizers. Auxiliary fact for the
normalization recurrence in arXiv:2512.24880v2, §4.2. -/
theorem row_sum_pos (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (i : ι) :
    0 < ∑ j, M i j := by
  exact Finset.sum_pos (fun j _ => hM i j) Finset.univ_nonempty

/-- A strictly positive four-stream matrix realizes the hypothesis. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- Row normalization preserves positivity.
Source: arXiv:2512.24880v2, §4.2, operator `T_r`. -/
theorem rowNormalize_pos (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) :
    ∀ i j, 0 < rowNormalize M i j := by
  intro i j
  exact div_pos (hM i j) (row_sum_pos M hM i)

/-- A strictly positive matrix satisfies the row-normalization hypothesis. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- Column normalization preserves positivity.
Source: arXiv:2512.24880v2, §4.2, operator `T_c`. -/
theorem columnNormalize_pos (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) :
    ∀ i j, 0 < columnNormalize M i j := by
  intro i j
  exact div_pos (hM i j)
    (Finset.sum_pos (fun k _ => hM k j) Finset.univ_nonempty)

/-- A strictly positive matrix satisfies the column-normalization hypothesis. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- The final row sums are exactly one in real arithmetic.
Source: arXiv:2512.24880v2, §4.2, row normalization. -/
theorem rowNormalize_sum (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (i : ι) :
    ∑ j, rowNormalize M i j = 1 := by
  change (∑ j, M i j / ∑ k, M i k) = 1
  rw [← Finset.sum_div, div_self (row_sum_pos M hM i).ne']

/-- The exact-normalization hypothesis holds in a nonempty four-stream case. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- Column normalization alone produces unit column sums. A subsequent
row normalization can change these sums, as the counterexample below
shows. Source: arXiv:2512.24880v2, §4.2, operator `T_c`. -/
theorem columnNormalize_sum (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (j : ι) :
    ∑ i, columnNormalize M i j = 1 := by
  have hs : 0 < ∑ i, M i j :=
    Finset.sum_pos (fun i _ => hM i j) Finset.univ_nonempty
  change (∑ i, M i j / ∑ k, M k j) = 1
  rw [← Finset.sum_div, div_self hs.ne']

/-- The column-sum theorem's positivity hypothesis is satisfiable. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- Every finite iterate stays positive.
Source: arXiv:2512.24880v2, §4.2, the finite normalization recurrence. -/
theorem sinkhornIterate_pos (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (t : ℕ) :
    ∀ i j, 0 < sinkhornIterate M t i j := by
  induction t with
  | zero => exact hM
  | succ t ih => exact rowNormalize_pos _ (columnNormalize_pos _ ih)

/-- Positive initialization is satisfiable for the iterative theorem. -/
example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  norm_num

/-- A nonzero finite iteration count guarantees unit row sums, without
asserting unit column sums. Source: arXiv:2512.24880v2, §4.2. -/
theorem finiteSinkhorn_row_sum (logits : Matrix ι ι ℝ) (t : ℕ) (i : ι) :
    ∑ j, finiteSinkhorn logits (t + 1) i j = 1 := by
  apply rowNormalize_sum
  apply columnNormalize_pos
  exact sinkhornIterate_pos _ (fun i j => Real.exp_pos (logits i j)) t

/-- The paper's practical count twenty therefore gives exact row sums in
the ideal real algorithm. Epsilon additions and rounding are not modeled.
Source: arXiv:2512.24880v2, §4.2, `t_max = 20`. -/
theorem twenty_iterations_row_sum (logits : Matrix ι ι ℝ) (i : ι) :
    ∑ j, finiteSinkhorn logits 20 i j = 1 := by
  exact finiteSinkhorn_row_sum logits 19 i

/-- A strictly positive finite-step counterexample: normalizing columns
and then rows need not leave columns normalized. This distinguishes the
finite algorithm from the ideal manifold in arXiv:2512.24880v2, §4.1. -/
theorem one_cycle_not_doubly_stochastic :
    ∃ M : Matrix (Fin 2) (Fin 2) ℝ,
      (∀ i j, 0 < M i j) ∧ sinkhornCycle M ∉ doublyStochastic ℝ (Fin 2) := by
  let M : Matrix (Fin 2) (Fin 2) ℝ :=
    Matrix.of (fun i j => if i = 0 ∧ j = 1 then 2 else 1)
  refine ⟨M, ?_, ?_⟩
  · intro i j
    dsimp [M]
    split_ifs <;> norm_num
  · intro h
    have hc := sum_col_of_mem_doublyStochastic h 0
    norm_num [sinkhornCycle, rowNormalize, columnNormalize, M, Fin.sum_univ_two] at hc

end Transformer.MHC
