module

public import Transformer.Grokking.FlatSpectrum.OneCellMin

/-!
# One unseen cell: flat kernels of either shape

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC5.

OC5 is stated for orthogonal rows, `K Kᴴ = c 1`, with `g` the rest of the column of the cell.
The adjoint exchanges rows and columns and keeps the nuclear norm, so it carries the fills of
`K` without `(i, j)` to the fills of `Kᴴ` without `(j, i)`, and minima to minima
(`conjTranspose_mem`, `isMinOn_conjTranspose`). So for orthogonal columns, `Kᴴ K = c 1`, the
same holds with `g' = ∑_{q ≠ j} |K_{iq}|²`, the rest of the row of the cell
(`exists_isMinOn_compl_singleton_of_conjTranspose_mul`,
`exists_neg_mul_of_isMinOn_compl_singleton_of_conjTranspose_mul`).

For a flat kernel of either shape: if the unseen cell is not zero and neither its row nor its
column is zero elsewhere, a minimum of M from the other cells exists, and every minimum fills
the cell with the opposite phase (`IsFlat.exists_isMinOn_cell`,
`IsFlat.exists_neg_mul_of_isMinOn_cell`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- The adjoint of a fill of `K` without `(i, j)` is a fill of `Kᴴ` without `(j, i)`. -/
lemma conjTranspose_mem {K Y : Matrix m n ℂ} {i : m} {j : n}
    (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2) :
    ∀ kl ∈ ({(j, i)}ᶜ : Set (n × m)), Yᴴ kl.1 kl.2 = Kᴴ kl.1 kl.2 := by
  rintro ⟨q, p⟩ h
  have h1 := hY (p, q) fun h' => h (by
    have h'' : (p, q) = (i, j) := h'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h''
    exact Set.mem_singleton _)
  simp only [conjTranspose_apply]
  exact congrArg star h1

/-- Minima of M from all cells but one correspond under the adjoint. -/
lemma isMinOn_conjTranspose {K X : Matrix m n ℂ} {i : m} {j : n}
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X) :
    IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(j, i)}ᶜ : Set (n × m)), Y kl.1 kl.2 = Kᴴ kl.1 kl.2} Xᴴ := by
  rw [isMinOn_iff] at hmin ⊢
  intro Y hY
  rw [nuclearNorm_conjTranspose, ← nuclearNorm_conjTranspose Y]
  refine hmin Yᴴ ?_
  have h := conjTranspose_mem hY
  rwa [conjTranspose_conjTranspose] at h

/-- **Orthogonal columns: a minimum of M from all cells but one exists**
(`review/flat_spectrum.md`, OC5). -/
theorem exists_isMinOn_compl_singleton_of_conjTranspose_mul {K : Matrix m n ℂ} {c : ℝ}
    (hc : 0 < c) (hK : Kᴴ * K = (c : ℂ) • 1) {i : m} {j : n} (hg : 0 < restSq Kᴴ j i) :
    ∃ X : Matrix m n ℂ, (∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2) ∧
      IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X := by
  have hK' : Kᴴ * Kᴴᴴ = (c : ℂ) • 1 := by rwa [conjTranspose_conjTranspose]
  obtain ⟨X, hX, hmin⟩ := exists_isMinOn_compl_singleton hc hK' hg
  have h1 := conjTranspose_mem hX
  have h2 := isMinOn_conjTranspose hmin
  rw [conjTranspose_conjTranspose] at h1 h2
  exact ⟨Xᴴ, h1, h2⟩

/-- **Orthogonal columns: the unseen cell is filled with the opposite phase**
(`review/flat_spectrum.md`, OC5): for `Kᴴ K = c 1`, a cell `κ = K_{ij} ≠ 0` and
`g' = ∑_{q ≠ j} |K_{iq}|² > 0`, every minimum of the nuclear norm among the matrices equal to
`K` at the other cells has `X_{ij} = −θ κ` with `0 < θ`, `g' ≤ θ (2c − g')` and
`θ (c − g') ≤ g'`. -/
theorem exists_neg_mul_of_isMinOn_compl_singleton_of_conjTranspose_mul {K X : Matrix m n ℂ}
    {c : ℝ} (hc : 0 < c) (hK : Kᴴ * K = (c : ℂ) • 1) {i : m} {j : n} (hκ : K i j ≠ 0)
    (hg : 0 < restSq Kᴴ j i)
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X) :
    ∃ θ : ℝ, 0 < θ ∧ restSq Kᴴ j i ≤ θ * (2 * c - restSq Kᴴ j i) ∧
      θ * (c - restSq Kᴴ j i) ≤ restSq Kᴴ j i ∧ X i j = -(θ : ℂ) * K i j := by
  have hK' : Kᴴ * Kᴴᴴ = (c : ℂ) • 1 := by rwa [conjTranspose_conjTranspose]
  obtain ⟨θ, h1, h2, h2', h3⟩ := exists_neg_mul_of_isMinOn_compl_singleton hc hK' (i := j)
    (j := i) (by simpa using hκ) hg (conjTranspose_mem hX) (isMinOn_conjTranspose hmin)
  refine ⟨θ, h1, h2, h2', ?_⟩
  have h4 := congrArg star h3
  simpa using h4

/-- **A flat kernel of either shape: a minimum of M from all cells but one exists**
(`review/flat_spectrum.md`, OC5). -/
theorem IsFlat.exists_isMinOn_cell {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    {i : m} {j : n} (hcol : 0 < restSq K i j) (hrow : 0 < restSq Kᴴ j i) :
    ∃ X : Matrix m n ℂ, (∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2) ∧
      IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X :=
  h.elim (fun h => exists_isMinOn_compl_singleton_of_conjTranspose_mul hc h hrow)
    fun h => exists_isMinOn_compl_singleton hc h hcol

/-- **One unseen cell of a flat kernel of either shape is filled with the opposite phase**
(`review/flat_spectrum.md`, OC5): if the cell `κ = K_{ij}` is not zero and neither its column
nor its row is zero elsewhere, every minimum of the nuclear norm among the matrices equal to
`K` at the other cells has `X_{ij} = −θ κ` with `θ > 0`. -/
theorem IsFlat.exists_neg_mul_of_isMinOn_cell {K X : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : IsFlat K c) {i : m} {j : n} (hκ : K i j ≠ 0) (hcol : 0 < restSq K i j)
    (hrow : 0 < restSq Kᴴ j i)
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X) :
    ∃ θ : ℝ, 0 < θ ∧ X i j = -(θ : ℂ) * K i j := by
  obtain h | h := h
  · obtain ⟨θ, h1, -, -, h3⟩ :=
      exists_neg_mul_of_isMinOn_compl_singleton_of_conjTranspose_mul hc h hκ hrow hX hmin
    exact ⟨θ, h1, h3⟩
  · obtain ⟨θ, h1, -, -, h3⟩ := exists_neg_mul_of_isMinOn_compl_singleton hc h hκ hcol hX hmin
    exact ⟨θ, h1, h3⟩

/-- The Hadamard matrix of size two is its own adjoint. -/
lemma hadamard_conjTranspose : (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ = !![1, 1; 1, -1] := by
  ext p q
  fin_cases p <;> fin_cases q <;> simp

/-- Its columns are orthogonal of squared length two as well. -/
lemma hadamard_conjTranspose_mul : (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ *
    (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) = ((2 : ℝ) : ℂ) • 1 := by
  have h := hadamard_mul_conjTranspose
  rw [hadamard_conjTranspose] at h ⊢
  exact h

/-- Its row `0` outside the column `0` has squared length one. -/
lemma restSq_hadamard_conjTranspose :
    restSq (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ 0 0 = 1 := by
  rw [hadamard_conjTranspose, restSq_hadamard]

example := exists_neg_mul_of_isMinOn_compl_singleton_of_conjTranspose_mul two_pos
  hadamard_conjTranspose_mul (i := 0) (j := 0) (by simp)
  (by rw [restSq_hadamard_conjTranspose]; exact one_pos) hadamard_fill_mem isMinOn_hadamard_fill

example := exists_isMinOn_compl_singleton_of_conjTranspose_mul two_pos
  hadamard_conjTranspose_mul (i := 0) (j := 0)
  (by rw [restSq_hadamard_conjTranspose]; exact one_pos)

example := IsFlat.exists_neg_mul_of_isMinOn_cell two_pos (Or.inl hadamard_conjTranspose_mul)
  (i := 0) (j := 0) (by simp) (by rw [restSq_hadamard]; exact one_pos)
  (by rw [restSq_hadamard_conjTranspose]; exact one_pos) hadamard_fill_mem isMinOn_hadamard_fill

example := IsFlat.exists_isMinOn_cell two_pos (Or.inr hadamard_mul_conjTranspose)
  (i := 0) (j := 0) (by rw [restSq_hadamard]; exact one_pos)
  (by rw [restSq_hadamard_conjTranspose]; exact one_pos)

example := conjTranspose_mem hadamard_fill_mem

example := isMinOn_conjTranspose isMinOn_hadamard_fill

end Transformer.Grokking.FlatSpectrum
