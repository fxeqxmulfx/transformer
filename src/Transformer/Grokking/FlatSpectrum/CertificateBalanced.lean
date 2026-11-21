module

public import Transformer.Grokking.FlatSpectrum.CertificateTests

/-!
# A criterion for a table whose rows sum to zero

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. If the rows of `K` sum
to zero, then `K` is not invertible and the columns of the identity are not in its row space. When
`K + J` is invertible, with `J` the matrix of ones, that row space is all of `1^⊥`, so the
differences `e_s − e_{j₀}` are test vectors for `not_isMinOn_of_tests`. Each has only two nonzero
entries, and a symmetry has to hide both of them at once.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **A table with zero row sums and `K + J` invertible.** If for every row `i` and column `s` a
symmetry hides the cells of the columns `s` and `j₀` in that row, then M does not return `K`. -/
theorem not_isMinOn_of_balanced [Nonempty n] {K : Matrix n n ℂ} (hK0 : K ≠ 0)
    (hrow : ∀ i, ∑ j, K i j = 0) (hdet : (K + Matrix.of fun _ _ => (1 : ℂ)).det ≠ 0)
    {Ω : Set (n × n)} (j₀ : n) {ι : Type*} (σ τ : ι → Equiv.Perm n) (α β : ι → n → ℂ)
    (hα : ∀ t i, ‖α t i‖ = 1) (hβ : ∀ t j, ‖β t j‖ = 1)
    (hsym : ∀ t i j, K (σ t i) (τ t j) = α t i * K i j * β t j)
    (hcover : ∀ i s, ∃ t, (σ t i, τ t s) ∉ Ω ∧ (σ t i, τ t j₀) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  set J : Matrix n n ℂ := Matrix.of fun _ _ => 1 with hJ
  set B : Matrix n n ℂ :=
    Matrix.of fun j s => (if j = s then 1 else 0) - (if j = j₀ then 1 else 0) with hB
  have hdetH : IsUnit (Kᴴ + J).det := by
    have h : Kᴴ + J = (K + J)ᴴ := by
      ext i j
      simp [hJ]
    rw [h, det_conjTranspose]
    exact (star_ne_zero.2 hdet).isUnit
  set C : Matrix n n ℂ := (Kᴴ + J)⁻¹ * B with hCdef
  have hCB : (Kᴴ + J) * C = B := by
    rw [hCdef, ← Matrix.mul_assoc, mul_nonsing_inv _ hdetH, Matrix.one_mul]
  have hcard : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  have hC0 (s : n) : ∑ k, C k s = 0 := by
    have h1 : ∑ i, ((Kᴴ + J) * C) i s = (Fintype.card n : ℂ) * ∑ k, C k s := by
      simp only [Matrix.mul_apply]
      rw [Finset.sum_comm, Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [← Finset.sum_mul]
      congr 1
      have h : ∑ i, (starRingEnd ℂ) (K k i) = 0 := by rw [← map_sum, hrow k, map_zero]
      simp [hJ, Finset.sum_add_distrib, h]
    have h2 : ∑ i, B i s = 0 := by
      simp [hB, Finset.sum_sub_distrib]
    rw [hCB, h2] at h1
    exact (mul_eq_zero.1 h1.symm).resolve_left hcard
  have hC : Kᴴ * C = B := by
    have hJC : J * C = 0 := by
      ext i s
      simp [hJ, Matrix.mul_apply, hC0 s]
    rw [← hCB, Matrix.add_mul, hJC, add_zero]
  have hD : B * Kᴴ = Kᴴ := by
    ext j c
    have h : ∑ s, (starRingEnd ℂ) (K c s) = 0 := by rw [← map_sum, hrow c, map_zero]
    simp [hB, Matrix.mul_apply, sub_mul, Finset.sum_sub_distrib, h]
  refine not_isMinOn_of_tests hK0 B C Kᴴ hC hD σ τ α β hα hβ hsym fun i s => ?_
  obtain ⟨t, h1, h2⟩ := hcover i s
  refine ⟨t, fun j hj => ?_⟩
  by_cases hjs : j = s
  · exact hjs ▸ h1
  · by_cases hj0 : j = j₀
    · exact hj0 ▸ h2
    · exact (hj (by simp [hB, hjs, hj0])).elim

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)),
      X ij.1 ij.2 = (!![1, -1; -1, 1] : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2}
    (!![1, -1; -1, 1] : Matrix (Fin 2) (Fin 2) ℂ) := by
  refine not_isMinOn_of_balanced (fun h => one_ne_zero (α := ℂ) ?_) ?_ ?_ 0
    (fun _ : Unit => Equiv.refl _) (fun _ => Equiv.refl _) (fun _ _ => 1) (fun _ _ => 1)
    (by simp) (by simp) ?_ (fun _ _ => ⟨(), Set.notMem_empty _, Set.notMem_empty _⟩)
  · simpa using congrFun (congrFun h 0) 0
  · intro i
    fin_cases i <;> simp [Fin.sum_univ_two]
  · have h : ((!![1, -1; -1, 1] : Matrix (Fin 2) (Fin 2) ℂ) + Matrix.of fun _ _ => (1 : ℂ)) =
        !![2, 0; 0, 2] := by
      ext i j
      fin_cases i <;> fin_cases j <;> norm_num
    rw [h, det_fin_two_of]
    norm_num
  · intro t i j
    simp

end Transformer.Grokking.FlatSpectrum
