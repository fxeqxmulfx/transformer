module

public import Transformer.MHC.Section4_Sinkhorn

/-!
# Denominators of the finite Sinkhorn cycles

arXiv:2512.24880v2, §4.2: `M⁽ᵗ⁾ = T_r (T_c (M⁽ᵗ⁻¹⁾))` from `M⁽⁰⁾ = exp(H̃)`, with `T_r`, `T_c` the
row and column normalizations (`rowNormalize`, `columnNormalize`, `finiteSinkhorn`). The paper
bounds none of the divisions; these bounds follow from the operators alone.

* The first cycle can divide by anything: the column sums of `exp(H̃)` and then the row sums of
  the column-normalized matrix are arbitrarily small or large, so an implementation keeps that
  cycle in log space.
* After a column normalization the entries are at most `1`, so every row sum is at most `n`
  and row-normalizing leaves each column sum in `[1/n, n]` (`rowNormalize_column_sum_bounds`),
  and symmetrically for the columns (`columnNormalize_row_sum_bounds`).
* So in every cycle after the first both divisions are by a number in `[1/n, n]`
  (`finiteSinkhorn_denominator_bounds`): for the four streams of the main model, `[1/4, 4]`.
  Nothing here says the cycles converge or that a cycle is doubly stochastic
  (`one_cycle_not_doubly_stochastic`).

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/proofs/MHCNormalization.lean`,
where they are stated for four streams as `MainModelMHC.row_normalize_column_bounds`,
`column_normalize_row_bounds`, `cycle_column_bounds`, `later_denominators`. Exact real
arithmetic: rounding and underflow of the native kernel are not modeled.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.MHC

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- Row-normalizing a positive matrix whose columns sum to `1` leaves every column sum in
`[1/n, n]`, `n` the number of streams: an entry is at most `1`, so a row sum is at most `n`, and
a normalized entry is at least `1/n` of the entry and at most `1`. -/
theorem rowNormalize_column_sum_bounds (M : Matrix ι ι ℝ) (hpos : ∀ i j, 0 < M i j)
    (hcols : ∀ j, ∑ i, M i j = 1) (j : ι) :
    1 / (Fintype.card ι : ℝ) ≤ ∑ i, rowNormalize M i j ∧
      ∑ i, rowNormalize M i j ≤ Fintype.card ι := by
  have entry_le (i k : ι) : M i k ≤ 1 :=
    calc M i k ≤ ∑ a, M a k :=
          Finset.single_le_sum (fun a _ => (hpos a k).le) (Finset.mem_univ i)
      _ = 1 := hcols k
  have row_le (i : ι) : ∑ k, M i k ≤ Fintype.card ι :=
    calc ∑ k, M i k ≤ ∑ _k : ι, (1 : ℝ) := Finset.sum_le_sum fun k _ => entry_le i k
      _ = Fintype.card ι := by simp
  constructor
  · calc 1 / (Fintype.card ι : ℝ) = (∑ i, M i j) / Fintype.card ι := by rw [hcols j]
      _ = ∑ i, M i j / Fintype.card ι := Finset.sum_div _ _ _
      _ ≤ ∑ i, rowNormalize M i j :=
        Finset.sum_le_sum fun i _ =>
          div_le_div_of_nonneg_left (hpos i j).le (row_sum_pos M hpos i) (row_le i)
  · calc ∑ i, rowNormalize M i j ≤ ∑ _i : ι, (1 : ℝ) :=
          Finset.sum_le_sum fun i _ =>
            calc rowNormalize M i j ≤ ∑ k, rowNormalize M i k :=
                  Finset.single_le_sum (fun k _ => (rowNormalize_pos M hpos i k).le)
                    (Finset.mem_univ j)
              _ = 1 := rowNormalize_sum M hpos i
      _ = Fintype.card ι := by simp

/-- Four streams, columns summing to `1` and every entry `1/4`: the row-normalized matrix has
column sums in `[1/4, 4]`. -/
example :
    1 / (Fintype.card (Fin 4) : ℝ) ≤
        ∑ i, rowNormalize (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ)) i 0 ∧
      ∑ i, rowNormalize (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ)) i 0 ≤
        Fintype.card (Fin 4) :=
  rowNormalize_column_sum_bounds (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ))
    (fun _ _ => by norm_num) (fun _ => by simp) 0

/-- The transposed bound: column-normalizing a positive matrix whose rows sum to `1` leaves
every row sum in `[1/n, n]`. -/
theorem columnNormalize_row_sum_bounds (M : Matrix ι ι ℝ) (hpos : ∀ i j, 0 < M i j)
    (hrows : ∀ i, ∑ j, M i j = 1) (i : ι) :
    1 / (Fintype.card ι : ℝ) ≤ ∑ j, columnNormalize M i j ∧
      ∑ j, columnNormalize M i j ≤ Fintype.card ι := by
  simpa only [rowNormalize, columnNormalize, Matrix.of_apply, Matrix.transpose_apply] using
    rowNormalize_column_sum_bounds M.transpose (fun i j => hpos j i) hrows i

/-- Four streams, rows summing to `1` and every entry `1/4`. -/
example :
    1 / (Fintype.card (Fin 4) : ℝ) ≤
        ∑ j, columnNormalize (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ)) 0 j ∧
      ∑ j, columnNormalize (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ)) 0 j ≤
        Fintype.card (Fin 4) :=
  columnNormalize_row_sum_bounds (Matrix.of fun _ _ : Fin 4 => (1 / 4 : ℝ))
    (fun _ _ => by norm_num) (fun _ => by simp) 0

/-- Any completed positive cycle `T_r ∘ T_c` has column sums in `[1/n, n]`, which are the
denominators of the next cycle's `T_c`. The cycle need not be doubly stochastic. -/
theorem sinkhornCycle_column_sum_bounds (M : Matrix ι ι ℝ) (hpos : ∀ i j, 0 < M i j) (j : ι) :
    1 / (Fintype.card ι : ℝ) ≤ ∑ i, sinkhornCycle M i j ∧
      ∑ i, sinkhornCycle M i j ≤ Fintype.card ι :=
  rowNormalize_column_sum_bounds (columnNormalize M) (columnNormalize_pos M hpos)
    (columnNormalize_sum M hpos) j

/-- A positive `4 × 4` matrix, here all ones. -/
example :
    1 / (Fintype.card (Fin 4) : ℝ) ≤
        ∑ i, sinkhornCycle (Matrix.of fun _ _ : Fin 4 => (1 : ℝ)) i 0 ∧
      ∑ i, sinkhornCycle (Matrix.of fun _ _ : Fin 4 => (1 : ℝ)) i 0 ≤ Fintype.card (Fin 4) :=
  sinkhornCycle_column_sum_bounds (Matrix.of fun _ _ : Fin 4 => (1 : ℝ))
    (fun _ _ => by norm_num) 0

/-- **Every cycle after the first divides by numbers in `[1/n, n]`.** After `t + 1` cycles of the
finite algorithm from any logits, the column sums, which the next `T_c` divides by, and the row
sums of the next column normalization, which the next `T_r` divides by, are in `[1/n, n]`. -/
theorem finiteSinkhorn_denominator_bounds (logits : Matrix ι ι ℝ) (t : ℕ) :
    (∀ j, 1 / (Fintype.card ι : ℝ) ≤ ∑ i, finiteSinkhorn logits (t + 1) i j ∧
      ∑ i, finiteSinkhorn logits (t + 1) i j ≤ Fintype.card ι) ∧
    (∀ i, 1 / (Fintype.card ι : ℝ) ≤ ∑ j, columnNormalize (finiteSinkhorn logits (t + 1)) i j ∧
      ∑ j, columnNormalize (finiteSinkhorn logits (t + 1)) i j ≤ Fintype.card ι) := by
  have pos (n : ℕ) : ∀ i j, 0 < finiteSinkhorn logits n i j :=
    sinkhornIterate_pos _ (fun i j => Real.exp_pos (logits i j)) n
  exact ⟨fun j => sinkhornCycle_column_sum_bounds (finiteSinkhorn logits t) (pos t) j,
    fun i => columnNormalize_row_sum_bounds _ (pos (t + 1)) (finiteSinkhorn_row_sum logits t) i⟩

/-- The practical twenty cycles from zero logits: the denominators of cycle `20` lie in
`[1/4, 4]`. -/
example : ∀ j : Fin 4, 1 / (Fintype.card (Fin 4) : ℝ) ≤
    ∑ i, finiteSinkhorn (0 : Matrix (Fin 4) (Fin 4) ℝ) (19 + 1) i j :=
  fun j => ((finiteSinkhorn_denominator_bounds (0 : Matrix (Fin 4) (Fin 4) ℝ) 19).1 j).1

end Transformer.MHC
