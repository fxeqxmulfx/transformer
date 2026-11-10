module

public import Transformer.Grokking.FlatSpectrum.MergedLevels
public import Transformer.Grokking.FlatSpectrum.PolarDiagonal

/-!
# The comparison kernel

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW2.
The comparison kernel on `n` ordered values has `C_ij = 1` for `i ≤ j` and `−1` for `j < i`.
It is `1 + A` with `A` skew: `C + Cᴴ = 2` (`compareKernel_add_conjTranspose`). Its boundary is
the diagonal, `i = j`, and the cells just below it, `i = j + 1`.

* Rows `i` and `i + 1` differ only in column `i` (`compareKernel_succ_row`), and columns `j` and
  `j + 1` only in row `j + 1` (`compareKernel_succ_col`).
* `C` is not a minimum of the nuclear norm among the matrices equal to it on `Ω`, once `Ω`
  misses a cell of the boundary (`not_isMinOn_compareKernel_diag`,
  `not_isMinOn_compareKernel_succ`): the polar factor has `Re W_ii = S_ii > 0` and
  `Re W_{i+1,i} = −S_ii < 0`.
* `CompareFill` shows that, with one cell of the boundary unseen, the kernel with that cell
  flipped is a minimum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

/-- The comparison kernel: `1` where `i ≤ j`, `−1` where `j < i`,
`review/flat_spectrum.md`, item CW2. -/
def compareKernel (n : ℕ) : Matrix (Fin n) (Fin n) ℂ := of fun i j => if i ≤ j then 1 else -1

variable {n : ℕ}

/-- `C + Cᴴ = 2`: the comparison kernel is `1 + A` with `A` skew. -/
theorem compareKernel_add_conjTranspose (n : ℕ) :
    compareKernel n + (compareKernel n)ᴴ = ((2 * 1 : ℝ) : ℂ) • 1 := by
  ext i j
  rw [Matrix.add_apply, conjTranspose_apply, Matrix.smul_apply, one_apply]
  simp only [compareKernel, of_apply]
  rcases lt_trichotomy i j with h | h | h
  · simp [h.le, h.not_ge, h.ne]
  · subst h
    simp
    norm_num
  · simp [h.le, h.not_ge, h.ne']

/-- Rows `i` and `i + 1` differ only in column `i`. -/
lemma compareKernel_succ_row {i i' : Fin n} (h : (i : ℕ) + 1 = i') (c : Fin n) :
    compareKernel n i' c = compareKernel n i c + if c = i then -2 else 0 := by
  simp only [compareKernel, of_apply, Fin.le_def, Fin.ext_iff]
  split_ifs <;> first | omega | norm_num

example (c : Fin 2) := compareKernel_succ_row (n := 2) (i := 0) (i' := 1) rfl c

/-- Columns `j` and `j + 1` differ only in row `j + 1`. -/
lemma compareKernel_succ_col {j j' : Fin n} (h : (j : ℕ) + 1 = j') (r : Fin n) :
    compareKernel n r j' = compareKernel n r j + if r = j' then 2 else 0 := by
  simp only [compareKernel, of_apply, Fin.le_def, Fin.ext_iff]
  split_ifs <;> first | omega | norm_num

example (r : Fin 2) := compareKernel_succ_col (n := 2) (j := 0) (j' := 1) rfl r

/-- A missing diagonal cell of the comparison kernel is not recovered,
`review/flat_spectrum.md`, item CW2. -/
theorem not_isMinOn_compareKernel_diag {Ω : Set (Fin n × Fin n)} {i : Fin n} (hi : (i, i) ∉ Ω) :
    ¬IsMinOn nuclearNorm
      {X | ∀ kl ∈ Ω, X kl.1 kl.2 = compareKernel n kl.1 kl.2} (compareKernel n) :=
  not_isMinOn_of_add_conjTranspose one_pos (compareKernel_add_conjTranspose n) hi

example : ¬IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(0, 0)}ᶜ : Set (Fin 2 × Fin 2)), X kl.1 kl.2 = compareKernel 2 kl.1 kl.2}
    (compareKernel 2) :=
  not_isMinOn_compareKernel_diag (i := 0) (by simp)

/-- Below the diagonal the polar factor is negative: `Re W_{i+1,i} = −S_ii`. -/
theorem re_polarFactor_compareKernel_succ {i i' : Fin n} (h : (i : ℕ) + 1 = i') :
    (polarFactor (compareKernel n) i' i).re < 0 := by
  obtain ⟨s, hs, hsi, hre⟩ :=
    re_polarFactor_apply_self one_pos (compareKernel_add_conjTranspose n) i
  have h1 : polarFactor (compareKernel n) i' i =
      polarFactor (compareKernel n) i i + -2 * gramInvSqrt (compareKernel n) i i := by
    rw [polarFactor_eq_mul_gramInvSqrt, Matrix.mul_apply, Matrix.mul_apply]
    simp only [compareKernel_succ_row h, add_mul, sum_add_distrib, ite_mul, zero_mul,
      sum_ite_eq', mem_univ, ite_true]
  rw [h1, hsi, Complex.add_re, hre, ← Complex.ofReal_ofNat, ← Complex.ofReal_neg,
    ← Complex.ofReal_mul, Complex.ofReal_re]
  linarith

example := re_polarFactor_compareKernel_succ (n := 2) (i := 0) (i' := 1) rfl

/-- A missing cell just below the diagonal of the comparison kernel is not recovered,
`review/flat_spectrum.md`, item CW2. -/
theorem not_isMinOn_compareKernel_succ {Ω : Set (Fin n × Fin n)} {i i' : Fin n}
    (h : (i : ℕ) + 1 = i') (hi : (i', i) ∉ Ω) :
    ¬IsMinOn nuclearNorm
      {X | ∀ kl ∈ Ω, X kl.1 kl.2 = compareKernel n kl.1 kl.2} (compareKernel n) := by
  intro hmin
  have hdet := det_ne_zero_of_add_conjTranspose one_pos (compareKernel_add_conjTranspose n)
  have h0 := (isMinOn_nuclearNorm_iff hdet Ω).mp hmin i' i hi
  have h1 := re_polarFactor_compareKernel_succ h
  rw [h0, Complex.zero_re] at h1
  exact lt_irrefl _ h1

example : ¬IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(1, 0)}ᶜ : Set (Fin 2 × Fin 2)), X kl.1 kl.2 = compareKernel 2 kl.1 kl.2}
    (compareKernel 2) :=
  not_isMinOn_compareKernel_succ (i := 0) (i' := 1) rfl (by simp)

end Transformer.Grokking.FlatSpectrum
