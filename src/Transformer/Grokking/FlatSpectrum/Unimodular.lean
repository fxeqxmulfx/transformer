module

public import Transformer.Grokking.FlatSpectrum.Kernels
public import Mathlib.Algebra.Order.Chebyshev

/-!
# Proposition 2: unimodular bounds and relabeling

Source: `review/flat_spectrum.md`, Proposition 2. For entries of modulus one,
`∑ σ² = ‖K‖_S2² = N²`, so `N ≤ ‖K‖_S1 ≤ N^{3/2}`; by `Kernels.lean` addition attains the
minimum and `x³ + x y` the maximum. Relabeling the vocabulary, with the readout
carried along, permutes rows and columns of the task kernel and keeps the multiset of
singular values.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- `∑ σ_i² = ‖X‖_S2²`. -/
theorem sum_singularValues_sq (X : Matrix m n ℂ) :
    ∑ i, singularValues X i ^ 2 = frobeniusSq X := by
  have h := (isHermitian_conjTranspose_mul_self X).trace_eq_sum_eigenvalues
  rw [trace_conjTranspose_mul_self] at h
  have h' : ((∑ i, (isHermitian_conjTranspose_mul_self X).eigenvalues i : ℝ) : ℂ) =
      (frobeniusSq X : ℂ) := by
    rw [h]
    push_cast
    rfl
  simp only [singularValues, Real.sq_sqrt (eigenvalues_gram_nonneg X _)]
  exact_mod_cast h'

/-- `x^{3/2} = x √x` for `x ≥ 0`. -/
lemma rpow_three_halves {x : ℝ} (hx : 0 ≤ x) : x ^ (3 / 2 : ℝ) = x * √x := by
  rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add' hx (by norm_num), Real.rpow_one,
    Real.sqrt_eq_rpow]

example := rpow_three_halves (x := 4) (by norm_num)

/-- Unimodular kernels have `N ≤ ‖K‖_S1 ≤ N^{3/2}`, `review/flat_spectrum.md`,
Proposition 2. -/
theorem card_le_nuclearNorm_le_of_unimodular {K : Matrix n n ℂ} (hK : ∀ i j, ‖K i j‖ = 1) :
    (Fintype.card n : ℝ) ≤ nuclearNorm K ∧
      nuclearNorm K ≤ (Fintype.card n : ℝ) ^ (3 / 2 : ℝ) := by
  set N : ℝ := (Fintype.card n : ℝ)
  have hS : ∑ i, singularValues K i ^ 2 = N ^ 2 := by
    rw [sum_singularValues_sq]
    simp [frobeniusSq, hK, N, sq]
  have hnn : ∀ i ∈ (Finset.univ : Finset n), 0 ≤ singularValues K i :=
    fun i _ => Real.sqrt_nonneg _
  have hT : 0 ≤ nuclearNorm K := Finset.sum_nonneg hnn
  have hN : 0 ≤ N := Nat.cast_nonneg _
  have h1 : N ^ 2 ≤ nuclearNorm K ^ 2 := by
    have := Finset.sum_sq_le_sq_sum_of_nonneg hnn
    rw [hS] at this
    exact this
  have h2 : nuclearNorm K ^ 2 ≤ N * N ^ 2 := by
    have := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := singularValues K)
    rw [hS, Finset.card_univ] at this
    exact this
  refine ⟨(pow_le_pow_iff_left₀ hN hT two_ne_zero).mp h1, ?_⟩
  rw [rpow_three_halves hN]
  refine (pow_le_pow_iff_left₀ hT (mul_nonneg hN (Real.sqrt_nonneg _)) two_ne_zero).mp ?_
  rw [mul_pow, Real.sq_sqrt hN]
  linarith

example := card_le_nuclearNorm_le_of_unimodular (K := cubicKernel 2)
  (by simp [cubicKernel, taskKernel])

/-- Permuting rows and columns keeps the multiset of singular values. -/
theorem singularValues_submatrix (K : Matrix m n ℂ) (σ : m ≃ m) (τ : n ≃ n) :
    Multiset.map (singularValues (K.submatrix σ τ)) Finset.univ.val =
      Multiset.map (singularValues K) Finset.univ.val := by
  have hG : (K.submatrix σ τ)ᴴ * K.submatrix σ τ = reindex τ.symm τ.symm (Kᴴ * K) := by
    rw [reindex_apply, Equiv.symm_symm, submatrix_mul _ _ _ _ _ σ.bijective,
      conjTranspose_submatrix]
  have hc : ((K.submatrix σ τ)ᴴ * K.submatrix σ τ).charpoly = (Kᴴ * K).charpoly := by
    rw [hG, charpoly_reindex]
  have h := (isHermitian_conjTranspose_mul_self (K.submatrix σ τ)).roots_charpoly_eq_eigenvalues
  rw [hc, (isHermitian_conjTranspose_mul_self K).roots_charpoly_eq_eigenvalues] at h
  have h3 := congrArg (Multiset.map fun z : ℂ => √(RCLike.re z)) h
  simp only [Multiset.map_map, Function.comp_def, RCLike.ofReal_re] at h3
  exact h3.symm

/-- Permuting rows and columns keeps the nuclear norm. -/
theorem nuclearNorm_submatrix (K : Matrix m n ℂ) (σ : m ≃ m) (τ : n ≃ n) :
    nuclearNorm (K.submatrix σ τ) = nuclearNorm K := by
  rw [nuclearNorm, nuclearNorm, Finset.sum_eq_multiset_sum, Finset.sum_eq_multiset_sum,
    singularValues_submatrix]

/-- Relabeling the vocabulary — the inputs by `σ` and `τ`, the answers by `π`, with the
readout carried along — keeps the singular values of the task kernel,
`review/flat_spectrum.md`, Proposition 2. -/
theorem singularValues_taskKernel_relabel {G : Type*} [Fintype G] [DecidableEq G]
    (f : G → G → G) (φ : G → ℂ) (σ τ π : G ≃ G) :
    Multiset.map (singularValues (taskKernel (fun a b => π (f (σ a) (τ b))) (φ ∘ π.symm)))
        Finset.univ.val =
      Multiset.map (singularValues (taskKernel f φ)) Finset.univ.val := by
  have : taskKernel (fun a b => π (f (σ a) (τ b))) (φ ∘ π.symm) =
      (taskKernel f φ).submatrix σ τ := by
    ext a b
    simp [taskKernel]
  rw [this, singularValues_submatrix]

/-- Among `p × p` kernels with entries of modulus one, addition has the least nuclear norm,
`p`, and `x³ + x y` the greatest, `p^{3/2}`, `review/flat_spectrum.md`, Proposition 2. -/
theorem nuclearNorm_unimodular_extremes (p : ℕ) [NeZero p] :
    IsLeast (nuclearNorm '' {K : Matrix (ZMod p) (ZMod p) ℂ | ∀ i j, ‖K i j‖ = 1})
        (nuclearNorm (addKernel p)) ∧
      IsGreatest (nuclearNorm '' {K : Matrix (ZMod p) (ZMod p) ℂ | ∀ i j, ‖K i j‖ = 1})
        (nuclearNorm (cubicKernel p)) := by
  have hcard : (Fintype.card (ZMod p) : ℝ) = p := by rw [ZMod.card]
  refine ⟨⟨⟨addKernel p, by simp [addKernel, taskKernel], rfl⟩, ?_⟩,
    ⟨⟨cubicKernel p, by simp [cubicKernel, taskKernel], rfl⟩, ?_⟩⟩
  · rintro _ ⟨K, hK, rfl⟩
    rw [nuclearNorm_addKernel, ← hcard]
    exact (card_le_nuclearNorm_le_of_unimodular hK).1
  · rintro _ ⟨K, hK, rfl⟩
    rw [nuclearNorm_cubicKernel, ← hcard]
    exact (card_le_nuclearNorm_le_of_unimodular hK).2

example := nuclearNorm_unimodular_extremes 97

end Transformer.Grokking.FlatSpectrum
