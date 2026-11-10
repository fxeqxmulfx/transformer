module

public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Transformer.Grokking.FlatSpectrum.GramBlock

/-!
# A flat kernel with one changed cell: its Gram matrix

Source: `review/flat_spectrum.md`, *One unseen cell* (added), items OC1 and OC2.

OC1. The nuclear norm of the adjoint is that of the matrix (`nuclearNorm_conjTranspose`), by
the duality with contractions. And a Gram matrix `Xᴴ X = c 1 + U M Uᴴ`, with two orthogonal
columns `U` of squared lengths `d`, has
`‖X‖_S1 = √c (n − 2) + √(tr (B diag d) + 2 √(|det B| d₀ d₁))` with `B = M + c diag(1/d)`
(`nuclearNorm_of_gram_eq_smul_one_add`).

OC2. Let `K Kᴴ = c 1` and change the cell `(i, j)` by `t`: `X = K + t e_i e_jᵀ`. Then

    X Xᴴ = c 1 + U [[t κ̄ + t̄ κ + |t|², t], [t̄, 0]] Uᴴ,       κ = K_{ij},

for the frame `U = [e_i, k_j − κ e_i]` of the unit column of `i` and the rest of the column
`j` of `K` (`cellFrame`, `mul_conjTranspose_add_smul_single`). Its columns are orthogonal of
squared lengths `1` and `g = ∑_{p ≠ i} |K_{pj}|²` (`restSq`, `cellFrame_gram`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The adjoint does not have a larger nuclear norm: pair `X` with the adjoint of the
contraction that attains `‖Xᴴ‖_S1`. -/
lemma nuclearNorm_conjTranspose_le (X : Matrix m n ℂ) : nuclearNorm Xᴴ ≤ nuclearNorm X := by
  obtain ⟨Z, hZ, htr⟩ := exists_isContraction_trace_eq Xᴴ
  have h := norm_trace_le_nuclearNorm hZ.conjTranspose X
  have h2 : (Zᴴᴴ * X).trace = star (Zᴴ * Xᴴ).trace := by
    rw [← Matrix.trace_conjTranspose, conjTranspose_mul, conjTranspose_conjTranspose,
      conjTranspose_conjTranspose, Matrix.trace_mul_comm]
  rw [h2, htr, norm_star, Complex.norm_real, Real.norm_of_nonneg (nuclearNorm_nonneg _)] at h
  exact h

/-- **The nuclear norm of the adjoint** (`review/flat_spectrum.md`, OC1):
`‖Xᴴ‖_S1 = ‖X‖_S1`. -/
theorem nuclearNorm_conjTranspose (X : Matrix m n ℂ) : nuclearNorm Xᴴ = nuclearNorm X :=
  le_antisymm (nuclearNorm_conjTranspose_le X)
    (by simpa using nuclearNorm_conjTranspose_le Xᴴ)

example : nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ)ᴴ =
    nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_conjTranspose 1

omit [DecidableEq m] in
/-- **A Gram matrix that is a multiple of the identity plus a block on two orthogonal columns**
(`review/flat_spectrum.md`, OC1): if `Xᴴ X = c 1 + U M Uᴴ` with `Uᴴ U = diag d`, `d > 0`, then
`‖X‖_S1 = √c (n − 2) + √(tr (B diag d) + 2 √(|det B| d₀ d₁))` for `B = M + c diag(1/d)`. -/
theorem nuclearNorm_of_gram_eq_smul_one_add {X : Matrix m n ℂ} {U : Matrix n (Fin 2) ℂ}
    {M : Matrix (Fin 2) (Fin 2) ℂ} {c : ℝ} {d : Fin 2 → ℝ} (hc : 0 ≤ c) (hd : ∀ i, 0 < d i)
    (hU : Uᴴ * U = diagonal fun i => (d i : ℂ))
    (hX : Xᴴ * X = (c : ℂ) • 1 + U * M * Uᴴ) :
    nuclearNorm X = √c * (Fintype.card n - 2) +
      √(((M + (c : ℂ) • diagonal fun i => ((d i : ℂ))⁻¹) *
          diagonal fun i => (d i : ℂ)).trace.re +
        2 * √(‖(M + (c : ℂ) • diagonal fun i => ((d i : ℂ))⁻¹).det‖ * (d 0 * d 1))) := by
  obtain ⟨D', hD'⟩ : ∃ D' : Matrix (Fin 2) (Fin 2) ℂ, D' = diagonal fun i => ((d i : ℂ))⁻¹ :=
    ⟨_, rfl⟩
  have hDD : D' * (diagonal fun i => (d i : ℂ)) = 1 := by
    rw [hD', diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext i
    exact inv_mul_cancel₀ (Complex.ofReal_ne_zero.2 (hd i).ne')
  have hDD2 : (diagonal fun i => (d i : ℂ)) * D' = 1 := by
    rw [hD', diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext i
    exact mul_inv_cancel₀ (Complex.ofReal_ne_zero.2 (hd i).ne')
  have hD'h : D'ᴴ = D' := by
    rw [hD', diagonal_conjTranspose]
    congr 1
    funext i
    simp
  obtain ⟨P, hP⟩ : ∃ P : Matrix n n ℂ, P = U * D' * Uᴴ := ⟨_, rfl⟩
  have hPU : P * U = U := by
    rw [hP, Matrix.mul_assoc, hU, Matrix.mul_assoc, hDD, Matrix.mul_one]
  have hPh : Pᴴ = P := by
    rw [hP, conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose, hD'h,
      Matrix.mul_assoc]
  have hPP : P * P = P :=
    calc P * P = P * (U * D' * Uᴴ) := by rw [← hP]
      _ = P * U * D' * Uᴴ := by simp only [Matrix.mul_assoc]
      _ = P := by rw [hPU, ← hP]
  have hEU : (1 - P) * U = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hPU, sub_self]
  have hEh : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hPh]
  have hEE : (1 - P) * (1 - P) = 1 - P := by
    rw [Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul, Matrix.one_mul, hPP, sub_self, sub_zero]
  have htr : (1 - P).trace.re = Fintype.card n - 2 := by
    rw [Matrix.trace_sub, Matrix.trace_one, hP, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hU,
      hDD2, Matrix.trace_one]
    simp
  have hX' : Xᴴ * X = ((√c ^ 2 : ℝ) : ℂ) • (1 - P) + U * (M + (c : ℂ) • D') * Uᴴ := by
    rw [Real.sq_sqrt hc, hX, smul_sub, Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul,
      Matrix.smul_mul, ← hP]
    abel
  rw [nuclearNorm_of_gram_eq_proj_add_diagonal (Real.sqrt_nonneg c) hd hEh hEE hU hEU hX', htr,
    hD']

/-- The identity, with `U = 1`, `M = 0`, `c = 1` and `d = (1, 1)`. -/
example := nuclearNorm_of_gram_eq_smul_one_add (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (U := 1)
  (M := 0) (c := 1) (d := fun _ => 1) zero_le_one (fun _ => one_pos) (by simp) (by simp)

/-- An entry of `K (t e_i e_jᵀ)ᴴ`. -/
lemma mul_conjTranspose_smul_single_apply {m : Type*} [DecidableEq m] (K : Matrix m n ℂ)
    (t : ℂ) (i : m) (j : n) (p q : m) :
    (K * (t • single i j (1 : ℂ))ᴴ) p q = if i = q then K p j * star t else 0 := by
  simp [Matrix.mul_apply, single_apply, ite_and]

/-- An entry of `(t e_i e_jᵀ) Kᴴ`. -/
lemma smul_single_mul_conjTranspose_apply {m : Type*} [DecidableEq m] (K : Matrix m n ℂ)
    (t : ℂ) (i : m) (j : n) (p q : m) :
    ((t • single i j (1 : ℂ)) * Kᴴ) p q = if i = p then t * star (K q j) else 0 := by
  simp [Matrix.mul_apply, single_apply, ite_and]

/-- **The frame of a cell** (`review/flat_spectrum.md`, OC2): the unit column of the row `i`
and the column `j` of `K` outside the row `i`. -/
def cellFrame (K : Matrix m n ℂ) (i : m) (j : n) : Matrix m (Fin 2) ℂ :=
  of fun p α => if i = p then ![1, 0] α else ![0, K p j] α

/-- The squared length `g = ∑_{p ≠ i} |K_{pj}|²` of the column `j` of `K` outside the row `i`
(`review/flat_spectrum.md`, OC2). -/
noncomputable def restSq (K : Matrix m n ℂ) (i : m) (j : n) : ℝ :=
  ∑ p, if i = p then 0 else ‖K p j‖ ^ 2

omit [Fintype n] [DecidableEq n] in
/-- The columns of the frame are orthogonal of squared lengths `1` and `g`
(`review/flat_spectrum.md`, OC2). -/
lemma cellFrame_gram (K : Matrix m n ℂ) (i : m) (j : n) :
    (cellFrame K i j)ᴴ * cellFrame K i j =
      diagonal fun α => ((![1, restSq K i j] α : ℝ) : ℂ) := by
  ext α β
  fin_cases α <;> fin_cases β
  · simp [Matrix.mul_apply, cellFrame]
  · simp only [Matrix.mul_apply, conjTranspose_apply, cellFrame, of_apply]
    refine (Finset.sum_eq_zero fun p _ => ?_).trans (by simp)
    by_cases hp : i = p <;> simp [hp]
  · simp [Matrix.mul_apply, cellFrame]
  · simp only [Matrix.mul_apply, conjTranspose_apply, cellFrame, of_apply]
    refine Eq.trans (Finset.sum_congr rfl fun p _ =>
      (?_ : _ = ((if i = p then 0 else ‖K p j‖ ^ 2 : ℝ) : ℂ))) (by simp [restSq])
    by_cases hp : i = p
    · simp [hp]
    · simpa [hp] using Complex.conj_mul' (K p j)

omit [Fintype m] in
/-- **The Gram matrix of a flat kernel with one changed cell** (`review/flat_spectrum.md`,
OC2): `X Xᴴ = c 1 + U [[t κ̄ + t̄ κ + |t|², t], [t̄, 0]] Uᴴ` for `X = K + t e_i e_jᵀ`. -/
lemma mul_conjTranspose_add_smul_single {K : Matrix m n ℂ} {c : ℝ}
    (hK : K * Kᴴ = (c : ℂ) • 1) (i : m) (j : n) (t : ℂ) :
    (K + t • single i j (1 : ℂ)) * (K + t • single i j (1 : ℂ))ᴴ = (c : ℂ) • 1 +
      cellFrame K i j * !![t * star (K i j) + star t * K i j + t * star t, t; star t, 0] *
        (cellFrame K i j)ᴴ := by
  ext p q
  have h0 : (K * Kᴴ) p q = if p = q then (c : ℂ) else 0 := by
    rw [hK]
    simp [Matrix.one_apply]
  rw [conjTranspose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add]
  simp only [Matrix.add_apply, h0, mul_conjTranspose_smul_single_apply,
    smul_single_mul_conjTranspose_apply]
  simp only [Matrix.smul_apply, Matrix.one_apply, Matrix.mul_apply, conjTranspose_apply,
    cellFrame, of_apply, Fin.sum_univ_two, smul_eq_mul, single_apply]
  by_cases hp : i = p
  · subst hp
    by_cases hq : i = q
    · subst hq
      simp
      ring
    · simp [hq]
  · by_cases hq : i = q
    · subst hq
      have hp' : ¬ p = i := fun h => hp h.symm
      simp [hp, hp']
    · simp [hp, hq]

/-- The identity of size two with the cell `(0, 1)` changed. -/
example (t : ℂ) := mul_conjTranspose_add_smul_single (K := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (c := 1) (by simp) 0 1 t

end Transformer.Grokking.FlatSpectrum
