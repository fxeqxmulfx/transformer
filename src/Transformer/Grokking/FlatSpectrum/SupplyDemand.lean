module

public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Tactic.FinCases
public import Mathlib.Tactic.Linarith
public import Transformer.Grokking.FlatSpectrum.Transport

/-!
# Hall's weighted condition under the moves of the supply–demand proof

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem F, item 3: the
supply–demand theorem (Gale, 1957). Hall's weighted condition asks `∑_R α ≤ ∑_{N(R)} β` for
every set of rows `R`. The proof of the theorem moves mass along one cell `(i, j)` of `Ω` and
then shrinks `Ω`; each move keeps the condition.

* Shipping `t` along `(i, j)`, from `α_i` and from `β_j`, keeps it when `t` is at most the slack
  `∑_{N(R)} β − ∑_R α` of every `R` that avoids `i` but meets `j` (`sum_le_sum_nbhd_ship`): a set
  `R ∋ i` loses `t` on both sides.
* Once row `i` or column `j` is empty, the cell `(i, j)` can go (`sum_le_sum_nbhd_diff_of_row`,
  `sum_le_sum_nbhd_diff_of_col`).
* A tight set `R₀`, with `∑_{N(R₀)} β ≤ ∑_{R₀} α`, splits the problem. The rows `R₀` keep the
  condition on their own cells (`sum_le_sum_nbhd_inside`). The other rows keep it on the cells
  outside the columns `N(R₀)`, whose capacities drop to `0` (`sum_le_sum_nbhd_outside`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι κ : Type*} [Fintype κ]

/-- **Shipping along a cell keeps Hall's weighted condition**: moving `t` from `α_i` and `β_j`,
for a cell `(i, j)` of `Ω`, keeps `∑_R α ≤ ∑_{N(R)} β` if `t` is at most the slack of every set
of rows that avoids `i` but meets `j`. -/
lemma sum_le_sum_nbhd_ship [DecidableEq ι] [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ}
    {β : κ → ℝ} (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l) {i : ι} {j : κ}
    (hij : (i, j) ∈ Ω) {t : ℝ}
    (ht : ∀ R : Finset ι, i ∉ R → j ∈ nbhd Ω R → t ≤ ∑ l ∈ nbhd Ω R, β l - ∑ k ∈ R, α k)
    (R : Finset ι) :
    ∑ k ∈ R, (α k - if k = i then t else 0) ≤ ∑ l ∈ nbhd Ω R, (β l - if l = j then t else 0) := by
  rw [sum_sub_distrib, sum_sub_distrib, sum_ite_eq', sum_ite_eq']
  by_cases hi : i ∈ R
  · rw [ite_eq_left hi, ite_eq_left (mem_nbhd.mpr ⟨i, hi, hij⟩)]
    linarith [hall R]
  · rw [ite_eq_right hi]
    by_cases hj : j ∈ nbhd Ω R
    · rw [ite_eq_left hj]
      linarith [ht R hi hj]
    · rw [ite_eq_right hj]
      linarith [hall R]

/-- The diagonal pattern meets in the rows `R` exactly the columns `R`. -/
lemma nbhd_diag {n : Type*} [Fintype n] (R : Finset n) :
    nbhd {ij : n × n | ij.2 = ij.1} R = R := by
  ext j
  simp [mem_nbhd]

example := sum_le_sum_nbhd_ship (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1}) (α := ![2, 1])
  (β := ![2, 1]) (fun R => by rw [nbhd_diag]) (i := 0) (j := 0) rfl (t := 0)
  (fun R _ _ => by rw [nbhd_diag, sub_self]) {0}

/-- Once row `i` is empty, deleting the cell `(i, j)` keeps Hall's weighted condition. -/
lemma sum_le_sum_nbhd_diff_of_row [DecidableEq ι] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    (hβ : ∀ l, 0 ≤ β l) (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l) {i : ι}
    (hi : α i = 0) (j : κ) (R : Finset ι) :
    ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd (Ω \ {(i, j)}) R, β l := by
  rw [← sum_erase R hi]
  refine (hall _).trans (sum_le_sum_of_subset_of_nonneg (fun l hl => ?_) fun l _ _ => hβ l)
  obtain ⟨k, hk, hkl⟩ := mem_nbhd.mp hl
  exact mem_nbhd.mpr ⟨k, mem_of_mem_erase hk, Set.mem_sdiff_of_mem hkl
    fun h => ne_of_mem_erase hk (congrArg Prod.fst (Set.mem_singleton_iff.mp h))⟩

example := sum_le_sum_nbhd_diff_of_row (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1})
  (α := ![0, 1]) (β := ![1, 1]) (by simp [Fin.forall_fin_two])
  (fun R => by
    rw [nbhd_diag]
    exact sum_le_sum fun k _ => by fin_cases k <;> simp)
  (i := 0) rfl 0 {0, 1}

/-- Once column `j` is empty, deleting the cell `(i, j)` keeps Hall's weighted condition. -/
lemma sum_le_sum_nbhd_diff_of_col [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    (hβ : ∀ l, 0 ≤ β l) (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l) {j : κ}
    (hj : β j = 0) (i : ι) (R : Finset ι) :
    ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd (Ω \ {(i, j)}) R, β l := by
  refine (hall R).trans ?_
  rw [← sum_erase _ hj]
  refine sum_le_sum_of_subset_of_nonneg (fun l hl => ?_) fun l _ _ => hβ l
  obtain ⟨k, hk, hkl⟩ := mem_nbhd.mp (mem_of_mem_erase hl)
  exact mem_nbhd.mpr ⟨k, hk, Set.mem_sdiff_of_mem hkl
    fun h => ne_of_mem_erase hl (congrArg Prod.snd (Set.mem_singleton_iff.mp h))⟩

example := sum_le_sum_nbhd_diff_of_col (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1})
  (α := ![0, 1]) (β := ![0, 1]) (by simp [Fin.forall_fin_two]) (fun R => by rw [nbhd_diag])
  (j := 0) rfl 0 {0, 1}

/-- The rows `R₀` keep Hall's weighted condition on their own cells. -/
lemma sum_le_sum_nbhd_inside [DecidableEq ι] {Ω : Set (ι × κ)} {α : ι → ℝ} {β : κ → ℝ}
    (hβ : ∀ l, 0 ≤ β l) (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l)
    (R₀ R : Finset ι) :
    ∑ k ∈ R, (if k ∈ R₀ then α k else 0) ≤ ∑ l ∈ nbhd {kl ∈ Ω | kl.1 ∈ R₀} R, β l := by
  rw [sum_ite_mem]
  refine (hall _).trans (sum_le_sum_of_subset_of_nonneg (fun l hl => ?_) fun l _ _ => hβ l)
  obtain ⟨k, hk, hkl⟩ := mem_nbhd.mp hl
  exact mem_nbhd.mpr ⟨k, mem_of_mem_inter_left hk, Set.mem_sep hkl (mem_of_mem_inter_right hk)⟩

example := sum_le_sum_nbhd_inside (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1}) (α := ![2, 1])
  (β := ![2, 1]) (by simp [Fin.forall_fin_two]) (fun R => by rw [nbhd_diag]) {0} {0, 1}

/-- **Splitting at a tight set**: if the rows `R₀` fill the columns `N(R₀)` they meet, the other
rows keep Hall's weighted condition on the cells outside `N(R₀)`, with no capacity left in
`N(R₀)`. -/
lemma sum_le_sum_nbhd_outside [DecidableEq ι] [DecidableEq κ] {Ω : Set (ι × κ)} {α : ι → ℝ}
    {β : κ → ℝ} (hβ : ∀ l, 0 ≤ β l)
    (hall : ∀ R : Finset ι, ∑ k ∈ R, α k ≤ ∑ l ∈ nbhd Ω R, β l) {R₀ : Finset ι}
    (htight : ∑ l ∈ nbhd Ω R₀, β l ≤ ∑ k ∈ R₀, α k) (R : Finset ι) :
    ∑ k ∈ R, (if k ∈ R₀ then 0 else α k) ≤
      ∑ l ∈ nbhd {kl ∈ Ω | kl.2 ∉ nbhd Ω R₀} R, (if l ∈ nbhd Ω R₀ then 0 else β l) := by
  have hL : ∑ k ∈ R, (if k ∈ R₀ then 0 else α k) = ∑ k ∈ R \ R₀, α k := by
    rw [sdiff_eq_filter, sum_filter]
    simp only [ite_not]
  have h := hall (R ∪ R₀)
  rw [nbhd_union, ← sum_sdiff (subset_union_right : R₀ ⊆ R ∪ R₀), union_sdiff_right,
    ← sum_sdiff (subset_union_right : nbhd Ω R₀ ⊆ nbhd Ω R ∪ nbhd Ω R₀), union_sdiff_right] at h
  rw [hL]
  calc ∑ k ∈ R \ R₀, α k ≤ ∑ l ∈ nbhd Ω R \ nbhd Ω R₀, β l := by linarith
    _ = ∑ l ∈ nbhd Ω R \ nbhd Ω R₀, (if l ∈ nbhd Ω R₀ then 0 else β l) :=
        sum_congr rfl fun l hl => (ite_eq_right (mem_sdiff.mp hl).2).symm
    _ ≤ _ := by
        refine sum_le_sum_of_subset_of_nonneg (fun l hl => ?_) fun l _ _ => ?_
        · obtain ⟨hl1, hl2⟩ := mem_sdiff.mp hl
          obtain ⟨k, hk, hkl⟩ := mem_nbhd.mp hl1
          exact mem_nbhd.mpr ⟨k, hk, Set.mem_sep hkl hl2⟩
        · split_ifs
          exacts [le_rfl, hβ l]

example := sum_le_sum_nbhd_outside (Ω := {ij : Fin 2 × Fin 2 | ij.2 = ij.1}) (α := ![2, 1])
  (β := ![2, 1]) (by simp [Fin.forall_fin_two]) (fun R => by rw [nbhd_diag]) (R₀ := {0})
  (by rw [nbhd_diag]) {0, 1}

end Transformer.Grokking.FlatSpectrum
