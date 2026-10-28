module

public import Transformer.Grokking.FlatSpectrum.Factorization

/-!
# Matrices with `(X* X)² = c X* X`

Used for the real readouts in `review/flat_spectrum.md`, Torch check. For `c > 0` such an
`X` is `√c` times a partial isometry. If `A = X* X` has `A² = c A`, then `D² = c D` for the
diagonal `D = U* A U`, so every eigenvalue of `A` is `0` or `c`. So every singular value of `X`
is `0` or `√c`, `rank X · c = tr A = ‖X‖_S2²`, `‖X‖_S1 = rank X · √c`, and `r_eff X = rank X`
for `c > 0`. The identity holds for `X = Q K` when `K K* = c I` and `Q* Q` is idempotent, and then
`tr A = c tr (Q* Q)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- A Hermitian `A` with `A A = c A` has every eigenvalue `0` or `c`. -/
lemma eigenvalues_eq_zero_or_of_mul_self {A : Matrix n n ℂ} (hA : A.IsHermitian) {c : ℝ}
    (h : A * A = (c : ℂ) • A) (i : n) : hA.eigenvalues i = 0 ∨ hA.eigenvalues i = c := by
  have hD := eigenvectorUnitary_conj hA
  have hUU := eigenvectorUnitary_mul_conjTranspose hA
  generalize (hA.eigenvectorUnitary : Matrix n n ℂ) = U at hD hUU
  have h1 : (diagonal fun i => ((hA.eigenvalues i : ℝ) : ℂ)) *
      (diagonal fun i => ((hA.eigenvalues i : ℝ) : ℂ)) =
      (c : ℂ) • diagonal fun i => ((hA.eigenvalues i : ℝ) : ℂ) := by
    rw [← hD]
    calc Uᴴ * A * U * (Uᴴ * A * U) = Uᴴ * (A * (U * Uᴴ) * A) * U := by
          simp only [Matrix.mul_assoc]
      _ = (c : ℂ) • (Uᴴ * A * U) := by
          rw [hUU, Matrix.mul_one, h, Matrix.mul_smul, Matrix.smul_mul]
  have h2 := congrFun (congrFun h1 i) i
  rw [diagonal_mul_diagonal, diagonal_apply_eq, Matrix.smul_apply, diagonal_apply_eq,
    smul_eq_mul] at h2
  have h3 : hA.eigenvalues i * hA.eigenvalues i = c * hA.eigenvalues i := by exact_mod_cast h2
  exact (mul_eq_mul_right_iff.mp h3).symm

example (i : Fin 2) := eigenvalues_eq_zero_or_of_mul_self (isHermitian_one (n := Fin 2) (α := ℂ))
  (c := 1) (by simp) i

/-- A Hermitian `A` with `A A = c A` has `rank A · c = tr A`. -/
lemma rank_mul_eq_trace_of_mul_self {A : Matrix n n ℂ} (hA : A.IsHermitian) {c : ℝ}
    (h : A * A = (c : ℂ) • A) : (A.rank : ℂ) * c = A.trace := by
  have hev (i : n) : (if hA.eigenvalues i ≠ 0 then (c : ℂ) else 0) =
      ((hA.eigenvalues i : ℝ) : ℂ) := by
    split_ifs with h0
    · rw [(eigenvalues_eq_zero_or_of_mul_self hA h i).resolve_left h0]
    · rw [not_not.mp h0, Complex.ofReal_zero]
  calc (A.rank : ℂ) * c = ∑ i, if hA.eigenvalues i ≠ 0 then (c : ℂ) else 0 := by
        rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul,
          hA.rank_eq_card_non_zero_eigs, Fintype.card_subtype]
    _ = ∑ i, ((hA.eigenvalues i : ℝ) : ℂ) := Finset.sum_congr rfl fun i _ => hev i
    _ = A.trace := hA.trace_eq_sum_eigenvalues.symm

example := rank_mul_eq_trace_of_mul_self (isHermitian_one (n := Fin 2) (α := ℂ)) (c := 1)
  (by simp)

/-- If `(X* X)² = c X* X`, every singular value of `X` is `0` or `√c`. -/
theorem singularValues_eq_zero_or {X : Matrix m n ℂ} {c : ℝ}
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) (i : n) :
    singularValues X i = 0 ∨ singularValues X i = √c := by
  rcases eigenvalues_eq_zero_or_of_mul_self (isHermitian_conjTranspose_mul_self X) h i with
    h0 | h0
  · left
    rw [singularValues, h0, Real.sqrt_zero]
  · right
    rw [singularValues, h0]

example (i : Fin 2) := singularValues_eq_zero_or (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1)
  (by simp) i

/-- `rank X · c = tr (X* X)` if `(X* X)² = c X* X`. -/
lemma rank_mul_eq_trace {X : Matrix m n ℂ} {c : ℝ}
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) : (X.rank : ℂ) * c = (Xᴴ * X).trace := by
  rw [← rank_conjTranspose_mul_self]
  exact rank_mul_eq_trace_of_mul_self (isHermitian_conjTranspose_mul_self X) h

example := rank_mul_eq_trace (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1) (by simp)

/-- `‖X‖_S1 = rank X · √c` if `(X* X)² = c X* X`. -/
theorem nuclearNorm_eq_rank_mul_sqrt {X : Matrix m n ℂ} {c : ℝ}
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) : nuclearNorm X = X.rank * √c := by
  have hA := isHermitian_conjTranspose_mul_self X
  have hsv (i : n) : singularValues X i = if hA.eigenvalues i ≠ 0 then √c else 0 := by
    split_ifs with h0
    · rw [singularValues, (eigenvalues_eq_zero_or_of_mul_self hA h i).resolve_left h0]
    · rw [singularValues, not_not.mp h0, Real.sqrt_zero]
  rw [nuclearNorm, Finset.sum_congr rfl fun i _ => hsv i, Finset.sum_ite, Finset.sum_const_zero,
    add_zero, Finset.sum_const, nsmul_eq_mul, ← rank_conjTranspose_mul_self,
    hA.rank_eq_card_non_zero_eigs, Fintype.card_subtype]

example := nuclearNorm_eq_rank_mul_sqrt (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1)
  (by simp)

/-- `‖X‖_S2² = rank X · c` if `(X* X)² = c X* X`. -/
theorem frobeniusSq_eq_rank_mul {X : Matrix m n ℂ} {c : ℝ}
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) : frobeniusSq X = X.rank * c := by
  rw [frobeniusSq_eq_re_trace, ← rank_mul_eq_trace h]
  simp

example := frobeniusSq_eq_rank_mul (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (c := 1) (by simp)

/-- `r_eff X = rank X` if `(X* X)² = c X* X` with `c > 0`. -/
theorem effectiveRank_eq_rank {X : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : Xᴴ * X * (Xᴴ * X) = (c : ℂ) • (Xᴴ * X)) : effectiveRank X = X.rank := by
  rw [effectiveRank, nuclearNorm_eq_rank_mul_sqrt h, frobeniusSq_eq_rank_mul h,
    Real.sqrt_mul (Nat.cast_nonneg _), mul_div_mul_right _ _ (Real.sqrt_pos.mpr hc).ne', div_pow,
    Real.sq_sqrt (Nat.cast_nonneg _), sq, mul_self_div_self]

example := effectiveRank_eq_rank (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) one_pos (by simp)

omit [Fintype m] [DecidableEq n] in
/-- `(Q K)* (Q K) = K* (Q* Q) K`. -/
lemma conjTranspose_mul_mul_self (Q K : Matrix n n ℂ) :
    (Q * K)ᴴ * (Q * K) = Kᴴ * (Qᴴ * Q) * K := by
  rw [conjTranspose_mul]
  simp only [Matrix.mul_assoc]

/-- If `K K* = c I` and `Q* Q` is idempotent, `A = (Q K)* (Q K)` has `A² = c A`. -/
theorem mul_self_conjTranspose_mul_self {Q K : Matrix n n ℂ} {c : ℂ} (hK : K * Kᴴ = c • 1)
    (hQ : Qᴴ * Q * (Qᴴ * Q) = Qᴴ * Q) :
    (Q * K)ᴴ * (Q * K) * ((Q * K)ᴴ * (Q * K)) = c • ((Q * K)ᴴ * (Q * K)) := by
  rw [conjTranspose_mul_mul_self]
  calc Kᴴ * (Qᴴ * Q) * K * (Kᴴ * (Qᴴ * Q) * K) = Kᴴ * (Qᴴ * Q) * (K * Kᴴ) * (Qᴴ * Q) * K := by
        simp only [Matrix.mul_assoc]
    _ = c • (Kᴴ * (Qᴴ * Q * (Qᴴ * Q)) * K) := by
        rw [hK]
        simp only [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, Matrix.mul_assoc]
    _ = c • (Kᴴ * (Qᴴ * Q) * K) := by rw [hQ]

example := mul_self_conjTranspose_mul_self (Q := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (K := 1)
  (c := 1) (by simp) (by simp)

/-- If `K K* = c I`, then `tr ((Q K)* (Q K)) = c tr (Q* Q)`. -/
lemma trace_conjTranspose_mul_self_of_mul {Q K : Matrix n n ℂ} {c : ℂ} (hK : K * Kᴴ = c • 1) :
    ((Q * K)ᴴ * (Q * K)).trace = c * (Qᴴ * Q).trace := by
  rw [conjTranspose_mul_mul_self, trace_mul_comm, ← Matrix.mul_assoc, hK, Matrix.smul_mul,
    Matrix.one_mul, trace_smul, smul_eq_mul]

example := trace_conjTranspose_mul_self_of_mul (Q := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (K := 1)
  (c := 1) (by simp)

end Transformer.Grokking.FlatSpectrum
