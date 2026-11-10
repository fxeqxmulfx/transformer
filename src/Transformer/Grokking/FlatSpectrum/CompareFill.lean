module

public import Transformer.Grokking.FlatSpectrum.Compare

/-!
# One unseen cell at the boundary of the comparison kernel

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW5.
Let the cell `(i, i)` of the comparison kernel `C` be the only unseen one, `0 < i < n − 1`.
Putting `−1` there makes row `i` equal to row `i + 1` and column `i` equal to column `i − 1`, so
by `isMinOn_of_row_eq_of_col_eq` the flipped kernel is a minimum of the nuclear norm among the
matrices equal to `C` at the other cells (`isMinOn_compareKernel_flip_diag`). `C` itself is not
one (`not_isMinOn_compareKernel_diag`), so no minimum holds the true value `1` in that cell
(`ne_of_isMinOn_compareKernel_diag`). The same holds for a cell `(i + 1, i)` below the diagonal
with `1` in place of the true `−1` (`isMinOn_compareKernel_flip_succ`,
`ne_of_isMinOn_compareKernel_succ`); there the flip merges rows `i`, `i + 1` and columns `i`,
`i + 1`. So a learner that minimizes the nuclear norm answers a comparison of two neighbours
it has not seen with the opposite of the truth.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {n : ℕ}

/-- The entries of a matrix with one cell changed. -/
lemma add_smul_single_one_apply {m n : Type*} [DecidableEq m] [DecidableEq n] (K : Matrix m n ℂ)
    (i : m) (j : n) (t : ℂ) (r : m) (c : n) :
    (K + t • single i j 1 : Matrix m n ℂ) r c = K r c + if r = i ∧ c = j then t else 0 := by
  rw [Matrix.add_apply, Matrix.smul_apply, single_apply, smul_eq_mul]
  by_cases h : r = i ∧ c = j
  · simp [h.1, h.2]
  · have h' : ¬(i = r ∧ j = c) := fun h' => h ⟨h'.1.symm, h'.2.symm⟩
    simp [h, h']

example := add_smul_single_one_apply (1 : Matrix (Fin 2) (Fin 2) ℂ) 0 0 2 0 0

/-- A matrix equal to `K` off a cell is equal off that cell to `K` with the cell changed. -/
lemma mem_add_smul_single_of_mem {m n : Type*} [DecidableEq m] [DecidableEq n]
    {K X : Matrix m n ℂ} {i : m} {j : n} (t : ℂ)
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2) :
    ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)),
      X kl.1 kl.2 = (K + t • single i j 1 : Matrix m n ℂ) kl.1 kl.2 := fun kl hkl => by
  rw [hX kl hkl, add_smul_single_apply_of_mem t kl hkl]

example := mem_add_smul_single_of_mem (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (X := 1) (i := 0)
  (j := 0) 2 fun _ _ => rfl

/-- **The flipped diagonal cell is a minimum**, `review/flat_spectrum.md`, item CW5: with the
cell `(i, i)` unseen, `0 < i < n − 1`, the comparison kernel with `−1` in that cell has the
least nuclear norm among the matrices equal to the kernel at the other cells. -/
theorem isMinOn_compareKernel_flip_diag {i₀ i i₁ : Fin n} (h₀ : (i₀ : ℕ) + 1 = i)
    (h₁ : (i : ℕ) + 1 = i₁) :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2}
      (compareKernel n + (-2 : ℂ) • single i i 1) := by
  have hne : i ≠ i₁ := fun h => by rw [← h] at h₁; omega
  have hne' : i ≠ i₀ := fun h => by rw [← h] at h₀; omega
  have hmin := isMinOn_of_row_eq_of_col_eq (K := compareKernel n + (-2 : ℂ) • single i i 1)
    (i := i) (i' := i₁) (j := i) (j' := i₀) hne hne'
    (fun c => by
      rw [add_smul_single_one_apply, add_smul_single_one_apply, compareKernel_succ_row h₁ c]
      by_cases hc : c = i
      · simp [hc, hne.symm]
      · simp [hc])
    (fun r => by
      rw [add_smul_single_one_apply, add_smul_single_one_apply, compareKernel_succ_col h₀ r]
      by_cases hr : r = i
      · simp [hr, hne'.symm]
      · simp [hr])
  rw [isMinOn_iff] at hmin ⊢
  exact fun X hX => hmin X (mem_add_smul_single_of_mem _ hX)

/-- **The flipped cell below the diagonal is a minimum**, `review/flat_spectrum.md`, item CW5:
with the cell `(i + 1, i)` unseen, the comparison kernel with `1` in that cell has the least
nuclear norm among the matrices equal to the kernel at the other cells. -/
theorem isMinOn_compareKernel_flip_succ {i i₁ : Fin n} (h₁ : (i : ℕ) + 1 = i₁) :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i₁, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2}
      (compareKernel n + (2 : ℂ) • single i₁ i 1) := by
  have hne : i ≠ i₁ := fun h => by rw [← h] at h₁; omega
  have hmin := isMinOn_of_row_eq_of_col_eq (K := compareKernel n + (2 : ℂ) • single i₁ i 1)
    (i := i₁) (i' := i) (j := i) (j' := i₁) hne.symm hne
    (fun c => by
      rw [add_smul_single_one_apply, add_smul_single_one_apply, compareKernel_succ_row h₁ c]
      by_cases hc : c = i
      · simp [hc, hne]
      · simp [hc])
    (fun r => by
      rw [add_smul_single_one_apply, add_smul_single_one_apply, compareKernel_succ_col h₁ r]
      by_cases hr : r = i₁
      · simp [hr, hne.symm]
      · simp [hr])
  rw [isMinOn_iff] at hmin ⊢
  exact fun X hX => hmin X (mem_add_smul_single_of_mem _ hX)

/-- **No minimum keeps the true diagonal cell**, `review/flat_spectrum.md`, item CW5. -/
theorem ne_of_isMinOn_compareKernel_diag {i : Fin n} {X : Matrix (Fin n) (Fin n) ℂ}
    (hX : ∀ kl ∈ ({(i, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2}
      X) : X i i ≠ 1 := by
  intro h
  have hXC : X = compareKernel n := by
    rw [eq_add_single_of_mem hX, h]
    simp [compareKernel]
  rw [hXC] at hmin
  exact not_isMinOn_compareKernel_diag (i := i) (by simp) hmin

/-- **No minimum keeps the true cell below the diagonal**, `review/flat_spectrum.md`,
item CW5. -/
theorem ne_of_isMinOn_compareKernel_succ {i i₁ : Fin n} (h₁ : (i : ℕ) + 1 = i₁)
    {X : Matrix (Fin n) (Fin n) ℂ}
    (hX : ∀ kl ∈ ({(i₁, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i₁, i)}ᶜ : Set (Fin n × Fin n)), X kl.1 kl.2 = compareKernel n kl.1 kl.2}
      X) : X i₁ i ≠ -1 := by
  intro h
  have hlt : ¬i₁ ≤ i := by rw [Fin.le_def]; omega
  have hXC : X = compareKernel n := by
    rw [eq_add_single_of_mem hX, h]
    simp [compareKernel, hlt]
  rw [hXC] at hmin
  exact not_isMinOn_compareKernel_succ h₁ (by simp) hmin

/-- Three values: the unseen cell `(1, 1)` is filled with `−1` by a minimum, and by no minimum
with the true `1`. -/
example : IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 1)}ᶜ : Set (Fin 3 × Fin 3)), X kl.1 kl.2 = compareKernel 3 kl.1 kl.2}
    (compareKernel 3 + (-2 : ℂ) • single 1 1 1) :=
  isMinOn_compareKernel_flip_diag (i₀ := 0) (i₁ := 2) rfl rfl

example : IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 0)}ᶜ : Set (Fin 2 × Fin 2)), X kl.1 kl.2 = compareKernel 2 kl.1 kl.2}
    (compareKernel 2 + (2 : ℂ) • single 1 0 1) :=
  isMinOn_compareKernel_flip_succ (i := 0) rfl

example : (compareKernel 3 + (-2 : ℂ) • single 1 1 1 : Matrix (Fin 3) (Fin 3) ℂ) 1 1 ≠ 1 :=
  ne_of_isMinOn_compareKernel_diag (add_smul_single_apply_of_mem _)
    (isMinOn_compareKernel_flip_diag (i₀ := 0) (i₁ := 2) rfl rfl)

example : (compareKernel 2 + (2 : ℂ) • single 1 0 1 : Matrix (Fin 2) (Fin 2) ℂ) 1 0 ≠ -1 :=
  ne_of_isMinOn_compareKernel_succ (i := 0) rfl (add_smul_single_apply_of_mem _)
    (isMinOn_compareKernel_flip_succ (i := 0) rfl)

end Transformer.Grokking.FlatSpectrum
