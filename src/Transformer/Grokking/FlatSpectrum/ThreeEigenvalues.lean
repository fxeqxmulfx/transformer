module

public import Transformer.Grokking.FlatSpectrum.Basic

/-!
# Hermitian matrices with eigenvalues `0`, `α`, `β`

Used for the kernel `[χ(a³ + a b)]` of the `b/a²` remark in `review/flat_spectrum.md`,
Theorem B. If a Hermitian `A` has `A³ = (α + β) A² − α β A`, the diagonal `D = U* A U` has the
same identity, so every eigenvalue is `0`, `α` or `β`. Then every sum `∑ g(λ)` with `g 0 = 0`
is `m_α g(α) + m_β g(β)`, `m_α` and `m_β` the multiplicities. The traces `tr A = ∑ λ` and
`tr A² = ∑ λ²` are two such sums, and `rank A = m_α + m_β` for nonzero `α` and `β`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A Hermitian `A` with `A³ = (α + β) A² − α β A` has every eigenvalue `0`, `α` or `β`. -/
lemma eigenvalues_mem_of_cubic {A : Matrix n n ℂ} (hA : A.IsHermitian) {α β : ℝ}
    (h : A * A * A = ((α + β : ℝ) : ℂ) • (A * A) - ((α * β : ℝ) : ℂ) • A) (i : n) :
    hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β := by
  have hD := eigenvectorUnitary_conj hA
  have hUU := eigenvectorUnitary_mul_conjTranspose hA
  generalize (hA.eigenvectorUnitary : Matrix n n ℂ) = U at hD hUU
  set D : Matrix n n ℂ := diagonal fun i => ((hA.eigenvalues i : ℝ) : ℂ) with hDdef
  have h2 : Uᴴ * (A * A) * U = D * D := by
    rw [← hD]
    calc Uᴴ * (A * A) * U = Uᴴ * (A * (U * Uᴴ) * A) * U := by rw [hUU, Matrix.mul_one]
      _ = Uᴴ * A * U * (Uᴴ * A * U) := by simp only [Matrix.mul_assoc]
  have h3 : Uᴴ * (A * A * A) * U = D * D * D := by
    rw [← h2, ← hD]
    calc Uᴴ * (A * A * A) * U = Uᴴ * (A * A * (U * Uᴴ) * A) * U := by rw [hUU, Matrix.mul_one]
      _ = Uᴴ * (A * A) * U * (Uᴴ * A * U) := by simp only [Matrix.mul_assoc]
  have h1 : D * D * D = ((α + β : ℝ) : ℂ) • (D * D) - ((α * β : ℝ) : ℂ) • D := by
    rw [← h3, h, ← h2, ← hD]
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul]
  have he := congrFun (congrFun h1 i) i
  simp only [hDdef, diagonal_mul_diagonal, diagonal_apply_eq, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul] at he
  have he' : hA.eigenvalues i * hA.eigenvalues i * hA.eigenvalues i =
      (α + β) * (hA.eigenvalues i * hA.eigenvalues i) - α * β * hA.eigenvalues i := by
    exact_mod_cast he
  have h0 : hA.eigenvalues i * ((hA.eigenvalues i - α) * (hA.eigenvalues i - β)) = 0 := by
    linear_combination he'
  rcases mul_eq_zero.mp h0 with h0 | h0
  · exact Or.inl h0
  · rcases mul_eq_zero.mp h0 with h0 | h0
    · exact Or.inr (Or.inl (sub_eq_zero.mp h0))
    · exact Or.inr (Or.inr (sub_eq_zero.mp h0))

example (i : Fin 2) := eigenvalues_mem_of_cubic (isHermitian_zero (n := Fin 2) (α := ℂ))
  (α := 1) (β := 2) (by simp) i

/-- `tr A² = ∑ λ²` for a Hermitian `A`. -/
lemma trace_mul_self_eq_sum_sq {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    (A * A).trace = ∑ i, ((hA.eigenvalues i ^ 2 : ℝ) : ℂ) := by
  have hD := eigenvectorUnitary_conj hA
  have hUU := eigenvectorUnitary_mul_conjTranspose hA
  generalize (hA.eigenvectorUnitary : Matrix n n ℂ) = U at hD hUU
  have h2 : Uᴴ * (A * A) * U = Uᴴ * A * U * (Uᴴ * A * U) := by
    calc Uᴴ * (A * A) * U = Uᴴ * (A * (U * Uᴴ) * A) * U := by rw [hUU, Matrix.mul_one]
      _ = Uᴴ * A * U * (Uᴴ * A * U) := by simp only [Matrix.mul_assoc]
  have htr : (A * A).trace = (Uᴴ * (A * A) * U).trace := by
    rw [Matrix.trace_mul_comm (Uᴴ * (A * A)) U, ← Matrix.mul_assoc U Uᴴ (A * A), hUU,
      Matrix.one_mul]
  rw [htr, h2, hD, diagonal_mul_diagonal, trace_diagonal]
  push_cast
  simp only [sq]

example := trace_mul_self_eq_sum_sq (isHermitian_zero (n := Fin 2) (α := ℂ))

variable {A : Matrix n n ℂ} (hA : A.IsHermitian) {α β : ℝ}

/-- If every eigenvalue is `0`, `α` or `β`, with `α ≠ β`, then
`∑ g(λ) = m_α g(α) + m_β g(β)` for every `g` with `g 0 = 0`. -/
lemma sum_comp_eigenvalues_of_mem (hαβ : α ≠ β)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β)
    (g : ℝ → ℝ) (hg : g 0 = 0) :
    ∑ i, g (hA.eigenvalues i) =
      #{i | hA.eigenvalues i = α} * g α + #{i | hA.eigenvalues i = β} * g β := by
  have hpt (i : n) : g (hA.eigenvalues i) =
      (if hA.eigenvalues i = α then g α else 0) + (if hA.eigenvalues i = β then g β else 0) := by
    rcases hmem i with h | h | h
    · rw [h, hg]
      split_ifs <;> subst_vars <;> simp [hg]
    · rw [h, ite_eq_left rfl, ite_eq_right hαβ, add_zero]
    · rw [h, ite_eq_left rfl, ite_eq_right (Ne.symm hαβ), zero_add]
  rw [Finset.sum_congr rfl fun i _ => hpt i, Finset.sum_add_distrib, ← Finset.sum_filter,
    ← Finset.sum_filter, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul]

example := sum_comp_eigenvalues_of_mem (isHermitian_zero (n := Fin 2) (α := ℂ)) (α := 1)
  (β := 2) (by norm_num) (eigenvalues_mem_of_cubic _ (by simp)) id rfl

/-- `tr A = m_α α + m_β β`. -/
lemma re_trace_eq_of_mem (hαβ : α ≠ β)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β) :
    A.trace.re = #{i | hA.eigenvalues i = α} * α + #{i | hA.eigenvalues i = β} * β := by
  rw [hA.trace_eq_sum_eigenvalues, Complex.re_sum]
  simpa using sum_comp_eigenvalues_of_mem hA hαβ hmem id rfl

example := re_trace_eq_of_mem (isHermitian_zero (n := Fin 2) (α := ℂ)) (α := 1) (β := 2)
  (by norm_num) (eigenvalues_mem_of_cubic _ (by simp))

/-- `tr A² = m_α α² + m_β β²`. -/
lemma re_trace_mul_self_eq_of_mem (hαβ : α ≠ β)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β) :
    (A * A).trace.re =
      #{i | hA.eigenvalues i = α} * α ^ 2 + #{i | hA.eigenvalues i = β} * β ^ 2 := by
  rw [trace_mul_self_eq_sum_sq hA,
    ← sum_comp_eigenvalues_of_mem hA hαβ hmem (· ^ 2) (by norm_num)]
  simp only [Complex.re_sum, Complex.ofReal_re]

example := re_trace_mul_self_eq_of_mem (isHermitian_zero (n := Fin 2) (α := ℂ)) (α := 1)
  (β := 2) (by norm_num) (eigenvalues_mem_of_cubic _ (by simp))

/-- `rank A = m_α + m_β` for distinct nonzero `α` and `β`. -/
lemma rank_eq_of_mem (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α ≠ β)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β) :
    (A.rank : ℝ) = #{i | hA.eigenvalues i = α} + #{i | hA.eigenvalues i = β} := by
  have h := sum_comp_eigenvalues_of_mem hA hαβ hmem (fun x => if x = 0 then 0 else 1)
    (by simp)
  simp only [hα, hβ, ite_false, mul_one] at h
  rw [hA.rank_eq_card_non_zero_eigs, Fintype.card_subtype, ← h]
  simp only [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul,
    mul_one, ne_eq]

example := rank_eq_of_mem (isHermitian_zero (n := Fin 2) (α := ℂ)) (α := 1) (β := 2)
  one_ne_zero two_ne_zero (by norm_num) (eigenvalues_mem_of_cubic _ (by simp))

/-- For square `X`, `X* X` and `X X*` have the same eigenvalues: their characteristic
polynomials agree. -/
lemma eigenvalues_conjTranspose_mul_self_eq (X : Matrix n n ℂ) :
    (isHermitian_conjTranspose_mul_self X).eigenvalues =
      (isHermitian_mul_conjTranspose_self X).eigenvalues :=
  ((isHermitian_conjTranspose_mul_self X).eigenvalues_eq_eigenvalues_iff
    (isHermitian_mul_conjTranspose_self X)).mpr (charpoly_mul_comm _ _)

/-- The singular values of a square `X` are the square roots of the eigenvalues of `X X*`. -/
lemma singularValues_eq_sqrt_eigenvalues_mul_conjTranspose (X : Matrix n n ℂ) (i : n) :
    singularValues X i = √((isHermitian_mul_conjTranspose_self X).eigenvalues i) := by
  rw [singularValues, eigenvalues_conjTranspose_mul_self_eq]

end Transformer.Grokking.FlatSpectrum
