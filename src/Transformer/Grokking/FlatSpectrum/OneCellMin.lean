module

public import Transformer.Grokking.FlatSpectrum.OneCellPhase
public import Transformer.Grokking.FlatSpectrum.RankOneFill

/-!
# One unseen cell of a flat kernel is filled with the opposite phase

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC5.

Let `K Kᴴ = c 1` with `c > 0`, and let `(i, j)` be the only unseen cell, with `κ = K_{ij} ≠ 0`
and `g = ∑_{p ≠ i} |K_{pj}|² > 0`. A matrix equal to `K` at every other cell is
`K + (x − κ) e_i e_jᵀ` for its value `x` at the cell, and conversely (`eq_add_single_of_mem`
and `add_smul_single_apply_of_mem` of `RankOneFill`). Its nuclear norm is `√c (m − 2) + N(x)`
with `N` of OC4 (`nuclearNorm_eq_cellNorm`). So

- a minimum of M from the other cells exists (`exists_isMinOn_compl_singleton`);
- every minimum has `X_{ij} = −θ κ` with `g / (2c − g) ≤ θ ≤ g / (c − g)`
  (`exists_neg_mul_of_isMinOn_compl_singleton`).

For the Hadamard matrix of size two and the cell `(0, 0)`, `c = 2`, `κ = 1` and `g = 1`. The
fill `[[−1, 1], [1, −1]]` has norm `2`, and the swap of the two coordinates is a contraction
that pairs every fill to `2`, so this fill is the minimum (`isMinOn_hadamard_fill`):
`θ = 1 = g / (c − g)`, and the upper bound on `θ` is attained.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **The norm of a matrix equal to `K` off one cell** (`review/flat_spectrum.md`, OC5):
`√c (m − 2) + N(Y_{ij})`. -/
lemma nuclearNorm_eq_cellNorm {K Y : Matrix m n ℂ} {c : ℝ} (hc : 0 ≤ c)
    (hK : K * Kᴴ = (c : ℂ) • 1) {i : m} {j : n} (hg : 0 < restSq K i j)
    (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2) :
    nuclearNorm Y = √c * (Fintype.card m - 2) + cellNorm c (restSq K i j) (K i j) (Y i j) := by
  have h := nuclearNorm_add_smul_single hc hK hg (Y i j - K i j)
  rw [← eq_add_single_of_mem hY] at h
  rw [h, cellNorm, add_sub_cancel]

/-- **A minimum of M from all cells but one exists** (`review/flat_spectrum.md`, OC5). -/
theorem exists_isMinOn_compl_singleton {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K * Kᴴ = (c : ℂ) • 1) {i : m} {j : n} (hg : 0 < restSq K i j) :
    ∃ X : Matrix m n ℂ, (∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2) ∧
      IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X := by
  have hcg := norm_sq_add_restSq_le hc hK i j
  obtain ⟨x, hx⟩ := exists_forall_cellNorm_le (c := c) (g := restSq K i j) (κ := K i j)
    (by linarith [restSq_nonneg K i j])
  have hmem := add_smul_single_apply_of_mem (K := K) (i := i) (j := j) (x - K i j)
  refine ⟨_, hmem, isMinOn_iff.2 fun Y hY => ?_⟩
  rw [nuclearNorm_eq_cellNorm hc.le hK hg hmem, nuclearNorm_eq_cellNorm hc.le hK hg hY]
  have h1 : (K + (x - K i j) • single i j (1 : ℂ)) i j = x := by simp
  rw [h1]
  exact add_le_add le_rfl (hx _)

/-- **One unseen cell of a flat kernel is filled with the opposite phase**
(`review/flat_spectrum.md`, OC5): for `K Kᴴ = c 1`, a cell `κ = K_{ij} ≠ 0` and
`g = ∑_{p ≠ i} |K_{pj}|² > 0`, every minimum of the nuclear norm among the matrices equal to `K`
at the other cells has `X_{ij} = −θ κ` with `0 < θ`, `g ≤ θ (2c − g)` and `θ (c − g) ≤ g`. -/
theorem exists_neg_mul_of_isMinOn_compl_singleton {K X : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K * Kᴴ = (c : ℂ) • 1) {i : m} {j : n} (hκ : K i j ≠ 0) (hg : 0 < restSq K i j)
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), Y kl.1 kl.2 = K kl.1 kl.2} X) :
    ∃ θ : ℝ, 0 < θ ∧ restSq K i j ≤ θ * (2 * c - restSq K i j) ∧
      θ * (c - restSq K i j) ≤ restSq K i j ∧ X i j = -(θ : ℂ) * K i j := by
  refine exists_neg_mul_of_forall_cellNorm_le hg hκ (norm_sq_add_restSq_le hc hK i j) fun y => ?_
  have hmem := add_smul_single_apply_of_mem (K := K) (i := i) (j := j) (y - K i j)
  have h := isMinOn_iff.1 hmin _ hmem
  rw [nuclearNorm_eq_cellNorm hc.le hK hg hX, nuclearNorm_eq_cellNorm hc.le hK hg hmem] at h
  have h1 : (K + (y - K i j) • single i j (1 : ℂ)) i j = y := by simp
  rw [h1] at h
  exact le_of_add_le_add_left h

/-- The swap of two coordinates is a contraction. -/
lemma isContraction_swap : IsContraction (!![0, 1; 1, 0] : Matrix (Fin 2) (Fin 2) ℂ) := by
  have h : (!![0, 1; 1, 0] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ * !![0, 1; 1, 0] = 1 := by
    ext p q
    fin_cases p <;> fin_cases q <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  rw [IsContraction, h, sub_self]
  exact PosSemidef.zero

/-- The fill `[[−1, 1], [1, −1]]` is the Hadamard matrix off the cell `(0, 0)`. -/
lemma hadamard_fill_mem : ∀ kl ∈ ({(0, 0)}ᶜ : Set (Fin 2 × Fin 2)),
    (!![-1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) kl.1 kl.2 =
      (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) kl.1 kl.2 := by
  rintro ⟨p, q⟩ h
  fin_cases p <;> fin_cases q <;> simp at h ⊢

/-- **The Hadamard matrix of size two without the cell `(0, 0)`** (`review/flat_spectrum.md`,
OC5): the fill `[[−1, 1], [1, −1]]` is a minimum of M. -/
theorem isMinOn_hadamard_fill :
    IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(0, 0)}ᶜ : Set (Fin 2 × Fin 2)),
      Y kl.1 kl.2 = (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) kl.1 kl.2}
      (!![-1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) := by
  rw [isMinOn_iff]
  intro Y hY
  have h1 : nuclearNorm (!![-1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) = 2 := by
    rw [nuclearNorm_fin_two]
    simp only [frobeniusSq, Fin.sum_univ_two, Matrix.det_fin_two]
    norm_num
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq zero_le_two]
  have h01 : Y 0 1 = 1 := by simpa using hY (0, 1) (by simp)
  have h10 : Y 1 0 = 1 := by simpa using hY (1, 0) (by simp)
  have h3 : ((!![0, 1; 1, 0] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ * Y).trace = 2 := by
    simp [Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two, h01, h10]
    norm_num
  have h2 := norm_trace_le_nuclearNorm isContraction_swap Y
  rw [h3] at h2
  rw [h1]
  simpa using h2

/-- The fill of the Hadamard matrix is `−θ` times its cell, with `g ≤ θ (2c − g)` and
`θ (c − g) ≤ g`. -/
example := exists_neg_mul_of_isMinOn_compl_singleton two_pos hadamard_mul_conjTranspose
  (i := 0) (j := 0) (by simp) (by rw [restSq_hadamard]; exact one_pos) hadamard_fill_mem
  isMinOn_hadamard_fill

/-- There `θ = 1`, and `θ (c − g) = g`: the upper bound on `θ` is attained. -/
example : (!![-1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) 0 0 =
    -((1 : ℝ) : ℂ) * (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) 0 0 ∧
    (1 : ℝ) * (2 - restSq (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) 0 0) =
      restSq (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) 0 0 := by
  rw [restSq_hadamard]
  norm_num

example := exists_isMinOn_compl_singleton two_pos hadamard_mul_conjTranspose (i := 0) (j := 0)
  (by rw [restSq_hadamard]; exact one_pos)

example := nuclearNorm_eq_cellNorm zero_le_two hadamard_mul_conjTranspose
  (by rw [restSq_hadamard]; exact one_pos) hadamard_fill_mem

end Transformer.Grokking.FlatSpectrum
