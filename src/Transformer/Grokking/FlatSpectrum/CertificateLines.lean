module

public import Transformer.Grokking.FlatSpectrum.CertificateOrbit

/-!
# A row or column orthogonal to the others has to be seen in full

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A column of `K`
orthogonal to all the others is an eigenvector of `Kᴴ K`, so a certificate `Y` equals `K / c` on it
and has no zero where `K` has none (`not_isMinOn_of_orthogonal_column`). For a row `k` orthogonal to
the others the eigenvector is `kᴴ`, so `⟨Y row, k⟩ = ‖k‖`, and a zero of `Y` at a cell with `k ≠ 0`
breaks Cauchy–Schwarz (`not_isMinOn_of_orthogonal_row`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **A column orthogonal to all the others has to be seen in full.** -/
theorem not_isMinOn_of_orthogonal_column {K : Matrix m n ℂ} {Ω : Set (m × n)} {i₀ : m} {j₀ : n}
    (hort : ∀ j, j ≠ j₀ → ∑ i, star (K i j) * K i j₀ = 0) (hK : K i₀ j₀ ≠ 0)
    (hΩ : (i₀, j₀) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  intro hmin
  obtain ⟨Y, hYΩ, hY, htr⟩ := exists_isContraction_of_isMinOn hmin
  have hnn : 0 ≤ ∑ i, ‖K i j₀‖ ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hx : (Kᴴ * K) *ᵥ Pi.single j₀ 1 =
      (((√(∑ i, ‖K i j₀‖ ^ 2)) ^ 2 : ℝ) : ℂ) • Pi.single j₀ (1 : ℂ) := by
    ext j
    rw [mulVec_single_one, Real.sq_sqrt hnn]
    by_cases hj : j = j₀
    · subst hj
      simp [conjTranspose_mul_self_apply_self]
    · have h := hort j hj
      rw [col_apply, Matrix.mul_apply]
      simp only [conjTranspose_apply]
      rw [h]
      simp [hj]
  have h := congrFun (smul_mulVec_of_certificate hY htr (Real.sqrt_nonneg _) hx) i₀
  simp only [mulVec_single_one, Pi.smul_apply, col_apply, smul_eq_mul, hYΩ i₀ j₀ hΩ, mul_zero] at h
  exact hK h.symm

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_orthogonal_column (i₀ := 0) (j₀ := 0)
    (fun j hj => by fin_cases j <;> simp_all [Fin.sum_univ_two]) (by simp) (by simp)

/-- **A row orthogonal to all the others has to be seen in full.** -/
theorem not_isMinOn_of_orthogonal_row {K : Matrix m n ℂ} {Ω : Set (m × n)} {i₀ : m} {j₀ : n}
    (hort : ∀ i, i ≠ i₀ → ∑ j, K i j * star (K i₀ j) = 0) (hK : K i₀ j₀ ≠ 0)
    (hΩ : (i₀, j₀) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  intro hmin
  obtain ⟨Y, hYΩ, hY, htr⟩ := exists_isContraction_of_isMinOn hmin
  set S : ℝ := ∑ j, ‖K i₀ j‖ ^ 2 with hS
  have hS0 : 0 ≤ S := sum_nonneg fun _ _ => sq_nonneg _
  have hj₀ : 0 < ‖K i₀ j₀‖ ^ 2 := pow_pos (norm_pos_iff.2 hK) 2
  have hSpos : 0 < S := lt_of_lt_of_le hj₀
    (single_le_sum (f := fun j => ‖K i₀ j‖ ^ 2) (fun _ _ => sq_nonneg _) (mem_univ j₀))
  set x : n → ℂ := fun j => star (K i₀ j) with hx
  have hKx : K *ᵥ x = Pi.single i₀ (S : ℂ) := by
    ext i
    by_cases hi : i = i₀
    · subst hi
      rw [Pi.single_eq_same, hS]
      simp only [mulVec, dotProduct, hx, Complex.star_def, Complex.mul_conj,
        Complex.normSq_eq_norm_sq]
      push_cast
      rfl
    · rw [Pi.single_eq_of_ne hi]
      exact hort i hi
  have hxe : (Kᴴ * K) *ᵥ x = (((√S) ^ 2 : ℝ) : ℂ) • x := by
    rw [← mulVec_mulVec, hKx, Real.sq_sqrt hS0]
    ext j
    simp only [mulVec, dotProduct, Pi.single_apply, mul_ite, mul_zero, Finset.sum_ite_eq',
      mem_univ, ite_true, conjTranspose_apply, Pi.smul_apply, smul_eq_mul, hx]
    exact mul_comm _ _
  have h := congrFun (smul_mulVec_of_certificate hY htr (Real.sqrt_nonneg S) hxe) i₀
  rw [hKx, Pi.smul_apply, Pi.single_eq_same, smul_eq_mul] at h
  have hsq : 0 < √S := Real.sqrt_pos.2 hSpos
  have hval : (Y *ᵥ x) i₀ = (√S : ℂ) := by
    have h2 : ((√S : ℝ) : ℂ) * ((√S : ℝ) : ℂ) = (S : ℂ) := by
      rw [← Complex.ofReal_mul, Real.mul_self_sqrt hS0]
    exact mul_left_cancel₀ (by exact_mod_cast hsq.ne') (h.trans h2.symm)
  have hcs := norm_sum_star_mul_le (univ.erase j₀) (fun j => K i₀ j) (fun j => Y i₀ j)
  have hsum : ∑ j ∈ univ.erase j₀, star (K i₀ j) * Y i₀ j = (√S : ℂ) := by
    rw [← hval, Finset.sum_erase _ (by simp [hYΩ i₀ j₀ hΩ])]
    simp only [mulVec, dotProduct, hx]
    exact sum_congr rfl fun j _ => mul_comm _ _
  rw [hsum, Complex.norm_real, Real.norm_of_nonneg hsq.le] at hcs
  have hY1 : ∑ j ∈ univ.erase j₀, ‖Y i₀ j‖ ^ 2 ≤ 1 := by
    have h1 := hY.conjTranspose.sum_norm_sq_le (U := 1) (by simp) i₀
    simp only [Matrix.mul_one, conjTranspose_apply, norm_star] at h1
    exact (sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => sq_nonneg _).trans h1
  have hK1 : ∑ j ∈ univ.erase j₀, ‖K i₀ j‖ ^ 2 = S - ‖K i₀ j₀‖ ^ 2 := by
    rw [hS, ← Finset.add_sum_erase _ _ (mem_univ j₀)]
    ring
  have hle : √S ≤ √(S - ‖K i₀ j₀‖ ^ 2) := by
    calc √S ≤ √(∑ j ∈ univ.erase j₀, ‖K i₀ j‖ ^ 2) * √(∑ j ∈ univ.erase j₀, ‖Y i₀ j‖ ^ 2) := hcs
      _ ≤ √(S - ‖K i₀ j₀‖ ^ 2) * 1 := by
        rw [hK1]
        exact mul_le_mul_of_nonneg_left (Real.sqrt_le_one.2 hY1) (Real.sqrt_nonneg _)
      _ = _ := mul_one _
  have hlt : √(S - ‖K i₀ j₀‖ ^ 2) < √S :=
    Real.sqrt_lt_sqrt (by rw [← hK1]; exact sum_nonneg fun _ _ => sq_nonneg _) (by linarith)
  linarith

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_orthogonal_row (i₀ := 0) (j₀ := 0)
    (fun i hi => by fin_cases i <;> simp_all [Fin.sum_univ_two]) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
