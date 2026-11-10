module

public import Transformer.Grokking.FlatSpectrum.ContractionDual
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.PartialIsometry

/-!
# Flat kernels of any shape

Source: `review/flat_spectrum.md`, *Flat kernels of any shape* (added), item R1. The note calls
a square kernel flat when its singular values are all equal. A kernel `K` with rows `m` and
columns `n` is flat with squared singular value `c` (`IsFlat K c`) when `Kᴴ K = c I` (orthogonal
columns of squared length `c`) or `K Kᴴ = c I` (orthogonal rows). Equal singular values `s` give
`Kᴴ K = s² I` (`isFlat_of_singularValues_eq`), and `Kᴴ K = c I` gives singular values `√c`
(`singularValues_eq_of_conjTranspose_mul_self`), so for square kernels this is the notion of the
note.

For a flat `K` with `c > 0`:

* `(Kᴴ K)² = c Kᴴ K` (`IsFlat.gram_mul_gram`), so `‖K‖_S1 = rank K · √c`,
  `‖K‖_S2² = rank K · c` and `r_eff K = rank K` by the partial-isometry lemmas;
* `rank K = min(|m|, |n|)` (`IsFlat.rank_eq`): `I = (Kᴴ / c) K` or `I = K (Kᴴ / c)`;
* so `r_eff K = min(|m|, |n|)` (`IsFlat.effectiveRank_eq`), the largest value an `m × n` matrix
  can have, as `r_eff ≤ rank`;
* `K / √c` is a contraction (`IsFlat.isContraction`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- A kernel is **flat** with squared singular value `c` when its columns or its rows are
orthogonal of squared length `c`: `Kᴴ K = c I` or `K Kᴴ = c I` (`review/flat_spectrum.md`,
R1). -/
def IsFlat (K : Matrix m n ℂ) (c : ℝ) : Prop :=
  Kᴴ * K = (c : ℂ) • 1 ∨ K * Kᴴ = (c : ℂ) • 1

/-- The adjoint of a flat kernel is flat. -/
theorem IsFlat.conjTranspose {K : Matrix m n ℂ} {c : ℝ} (h : IsFlat K c) : IsFlat Kᴴ c :=
  h.elim (fun h => Or.inr (by rwa [conjTranspose_conjTranspose]))
    fun h => Or.inl (by rwa [conjTranspose_conjTranspose])

/-- The identity is flat, with zero entries off the diagonal. -/
lemma isFlat_one : IsFlat (1 : Matrix n n ℂ) 1 :=
  Or.inl (by simp)

/-- The `a³ + a b` kernel is flat with `c = p`, Proposition 2. -/
lemma isFlat_cubicKernel (p : ℕ) [NeZero p] : IsFlat (cubicKernel p) p :=
  Or.inl (by rw [conjTranspose_mul_cubicKernel, Complex.ofReal_natCast])

/-- The `1 × 2` kernel `(1, 1)` is flat with `c = 2`: one row, two columns, no zero entry. -/
lemma isFlat_row_ones : IsFlat (Matrix.of ![![(1 : ℂ), 1]]) 2 :=
  Or.inr (by
    ext i j
    simp [Matrix.mul_apply, Fin.sum_univ_two, Subsingleton.elim i j]
    norm_num)

example : IsFlat (1 : Matrix (Fin 2) (Fin 2) ℂ)ᴴ 1 := isFlat_one.conjTranspose

omit [Fintype n] [DecidableEq m] in
/-- For `c > 0` and `t = √c`: `Aᴴ A = c I` gives `(A / t)ᴴ (A / t) = I`. -/
lemma conjTranspose_smul_mul_smul_eq_one {A : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : Aᴴ * A = (c : ℂ) • 1) :
    ((((√c : ℝ) : ℂ))⁻¹ • A)ᴴ * ((((√c : ℝ) : ℂ))⁻¹ • A) = 1 := by
  have h0 : ((√c : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.2 (Real.sqrt_ne_zero'.2 hc)
  have hone : star (((√c : ℝ) : ℂ))⁻¹ * (((√c : ℝ) : ℂ))⁻¹ * (c : ℂ) = 1 := by
    nth_rewrite 3 [← Real.mul_self_sqrt hc.le]
    rw [star_inv₀, Complex.star_def, Complex.conj_ofReal, Complex.ofReal_mul]
    field_simp
  rw [conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, h, smul_smul, smul_smul, hone,
    one_smul]

example := conjTranspose_smul_mul_smul_eq_one (A := (1 : Matrix (Fin 2) (Fin 2) ℂ)) one_pos
  (by simp)

/-- Equal singular values `s` give `Kᴴ K = s² I`: such a kernel is flat. -/
theorem isFlat_of_singularValues_eq {K : Matrix m n ℂ} {s : ℝ}
    (h : ∀ i, singularValues K i = s) : IsFlat K (s ^ 2) := by
  left
  have hD := gram_diagonalization K
  have hUU := eigenvectorUnitary_mul_conjTranspose (isHermitian_conjTranspose_mul_self K)
  generalize ((isHermitian_conjTranspose_mul_self K).eigenvectorUnitary : Matrix n n ℂ) = U
    at hD hUU
  simp only [h] at hD
  calc Kᴴ * K = U * Uᴴ * (Kᴴ * K) * (U * Uᴴ) := by rw [hUU, Matrix.one_mul, Matrix.mul_one]
    _ = U * (Uᴴ * (Kᴴ * K) * U) * Uᴴ := by simp only [Matrix.mul_assoc]
    _ = ((s ^ 2 : ℝ) : ℂ) • 1 := by
        rw [hD, ← smul_one_eq_diagonal, Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul, hUU]

example := isFlat_of_singularValues_eq (singularValues_cubicKernel 3)

omit [DecidableEq m] in
/-- `Kᴴ K = c I` gives singular values `√c`. -/
theorem singularValues_eq_of_conjTranspose_mul_self {K : Matrix m n ℂ} {c : ℝ}
    (h : Kᴴ * K = (c : ℂ) • 1) (i : n) : singularValues K i = √c := by
  rw [singularValues, eigenvalues_eq_of_eq_smul_one _ h]

example (i : Fin 2) := singularValues_eq_of_conjTranspose_mul_self
  (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1) (by simp) i

/-- A flat kernel has `(Kᴴ K)² = c Kᴴ K`: it is `√c` times a partial isometry. -/
theorem IsFlat.gram_mul_gram {K : Matrix m n ℂ} {c : ℝ} (h : IsFlat K c) :
    Kᴴ * K * (Kᴴ * K) = (c : ℂ) • (Kᴴ * K) := by
  rcases h with h | h
  · rw [h, Matrix.smul_mul, Matrix.one_mul]
  · rw [Matrix.mul_assoc, ← Matrix.mul_assoc K, h, Matrix.smul_mul, Matrix.one_mul,
      Matrix.mul_smul]

omit [DecidableEq m] in
/-- Orthogonal columns: `Kᴴ K = c I` with `c ≠ 0` gives `rank K = |n|`. -/
lemma rank_eq_card_of_conjTranspose_mul_self {K : Matrix m n ℂ} {c : ℝ} (hc : c ≠ 0)
    (h : Kᴴ * K = (c : ℂ) • 1) : K.rank = Fintype.card n := by
  have h1 : ((c : ℂ)⁻¹ • Kᴴ) * K = 1 := by
    rw [Matrix.smul_mul, h, smul_smul, inv_mul_cancel₀ (Complex.ofReal_ne_zero.2 hc), one_smul]
  refine le_antisymm (rank_le_card_width K) ?_
  calc Fintype.card n = (((c : ℂ)⁻¹ • Kᴴ) * K).rank := by rw [h1, rank_one]
    _ ≤ K.rank := rank_mul_le_right _ _

omit [DecidableEq n] in
/-- Orthogonal rows: `K Kᴴ = c I` with `c ≠ 0` gives `rank K = |m|`. -/
lemma rank_eq_card_of_mul_conjTranspose_self {K : Matrix m n ℂ} {c : ℝ} (hc : c ≠ 0)
    (h : K * Kᴴ = (c : ℂ) • 1) : K.rank = Fintype.card m := by
  have h1 : K * ((c : ℂ)⁻¹ • Kᴴ) = 1 := by
    rw [Matrix.mul_smul, h, smul_smul, inv_mul_cancel₀ (Complex.ofReal_ne_zero.2 hc), one_smul]
  refine le_antisymm (rank_le_card_height K) ?_
  calc Fintype.card m = (K * ((c : ℂ)⁻¹ • Kᴴ)).rank := by rw [h1, rank_one]
    _ ≤ K.rank := rank_mul_le_left _ _

example := rank_eq_card_of_conjTranspose_mul_self (K := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (c := 1) one_ne_zero (by simp)

example := rank_eq_card_of_mul_conjTranspose_self (K := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (c := 1) one_ne_zero (by simp)

/-- **A flat kernel has full rank** `min(|m|, |n|)` (`review/flat_spectrum.md`, R1). -/
theorem IsFlat.rank_eq {K : Matrix m n ℂ} {c : ℝ} (hc : c ≠ 0) (h : IsFlat K c) :
    K.rank = min (Fintype.card m) (Fintype.card n) := by
  rcases h with h | h
  · have h1 := rank_eq_card_of_conjTranspose_mul_self hc h
    rw [h1, min_eq_right (h1 ▸ rank_le_card_height K)]
  · have h1 := rank_eq_card_of_mul_conjTranspose_self hc h
    rw [h1, min_eq_left (h1 ▸ rank_le_card_width K)]

/-- **A flat kernel has the largest effective rank**: `r_eff K = min(|m|, |n|)`
(`review/flat_spectrum.md`, R1). -/
theorem IsFlat.effectiveRank_eq {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c) :
    effectiveRank K = min (Fintype.card m) (Fintype.card n) := by
  rw [effectiveRank_eq_rank hc h.gram_mul_gram, h.rank_eq hc.ne']

/-- `‖K‖_S2² = min(|m|, |n|) · c` for a flat kernel (`review/flat_spectrum.md`, R1). -/
theorem IsFlat.frobeniusSq_eq {K : Matrix m n ℂ} {c : ℝ} (hc : c ≠ 0) (h : IsFlat K c) :
    frobeniusSq K = min (Fintype.card m) (Fintype.card n) * c := by
  rw [frobeniusSq_eq_rank_mul h.gram_mul_gram, h.rank_eq hc]

/-- `‖K‖_S1 = min(|m|, |n|) · √c` for a flat kernel (`review/flat_spectrum.md`, R1). -/
theorem IsFlat.nuclearNorm_eq {K : Matrix m n ℂ} {c : ℝ} (hc : c ≠ 0) (h : IsFlat K c) :
    nuclearNorm K = min (Fintype.card m) (Fintype.card n) * √c := by
  rw [nuclearNorm_eq_rank_mul_sqrt h.gram_mul_gram, h.rank_eq hc]

/-- A flat kernel divided by its singular value `√c` is a contraction. -/
theorem IsFlat.isContraction {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c) :
    IsContraction ((((√c : ℝ) : ℂ))⁻¹ • K) := by
  rcases h with h | h
  · rw [IsContraction, conjTranspose_smul_mul_smul_eq_one hc h, sub_self]
    exact PosSemidef.zero
  · have hZ : IsContraction ((((√c : ℝ) : ℂ))⁻¹ • Kᴴ) := by
      rw [IsContraction, conjTranspose_smul_mul_smul_eq_one hc
        (by rwa [conjTranspose_conjTranspose]), sub_self]
      exact PosSemidef.zero
    have h1 := hZ.conjTranspose
    rwa [conjTranspose_smul, conjTranspose_conjTranspose, star_inv₀, Complex.star_def,
      Complex.conj_ofReal] at h1

/-- The `a³ + a b` kernel for `p = 3`: rank, effective rank and norms. -/
example := (isFlat_cubicKernel 3).gram_mul_gram
example := (isFlat_cubicKernel 3).rank_eq (by norm_num)
example := (isFlat_cubicKernel 3).effectiveRank_eq (by norm_num)
example := (isFlat_cubicKernel 3).frobeniusSq_eq (by norm_num)
example := (isFlat_cubicKernel 3).nuclearNorm_eq (by norm_num)
example := (isFlat_cubicKernel 3).isContraction (by norm_num)

end Transformer.Grokking.FlatSpectrum
