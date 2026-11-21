module

public import Transformer.Grokking.FlatSpectrum.CertificateTests

/-!
# A criterion for a table whose columns fall into classes

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. If `K i j` depends on
the column `j` only through its class `cl j`, then `K = L P`, with `P` the membership matrix of the
classes. When `L` has an invertible minor on all classes, the indicator of each class lies in the
row space of `K`, so these indicators are test vectors for `not_isMinOn_of_tests`. A symmetry then
has to hide one whole class in a row, and no more than that.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **A table with classes of columns.** Let `K i j = L i (cl j)` with an invertible minor of `L`
on all classes. If for every row and class a symmetry sends that class in that row outside `Ω`, then
M does not return `K`. -/
theorem not_isMinOn_of_classes {K : Matrix m n ℂ} (hK0 : K ≠ 0) {Ω : Set (m × n)} {r ι : Type*}
    [Fintype r] [DecidableEq r] (cl : n → r) (L : Matrix m r ℂ) (hKL : ∀ i j, K i j = L i (cl j))
    (ρ : r → m) (hdet : (L.submatrix ρ id).det ≠ 0)
    (σ : ι → Equiv.Perm m) (τ : ι → Equiv.Perm n) (α : ι → m → ℂ) (β : ι → n → ℂ)
    (hα : ∀ t i, ‖α t i‖ = 1) (hβ : ∀ t j, ‖β t j‖ = 1)
    (hsym : ∀ t i j, K (σ t i) (τ t j) = α t i * K i j * β t j)
    (hcover : ∀ i c, ∃ t, ∀ j, cl j = c → (σ t i, τ t j) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  set A : Matrix r r ℂ := L.submatrix ρ id with hAdef
  have hA : IsUnit A.det := isUnit_iff_ne_zero.2 hdet
  set S : Matrix r m ℂ := Matrix.of fun c i => if i = ρ c then 1 else 0 with hS
  have hSL : S * L = A := by
    ext c c'
    simp [hS, hAdef, Matrix.mul_apply]
  set E : Matrix r m ℂ := A⁻¹ * S with hE
  have hEL : E * L = 1 := by
    rw [hE, Matrix.mul_assoc, hSL, nonsing_inv_mul _ hA]
  set B : Matrix n r ℂ := Matrix.of fun j c => if cl j = c then 1 else 0 with hB
  have hC : Kᴴ * Eᴴ = B := by
    ext j c
    have h : (E * K) c j = (1 : Matrix r r ℂ) c (cl j) := by
      rw [← hEL]
      simp only [Matrix.mul_apply, hKL]
    rw [← conjTranspose_mul, conjTranspose_apply, h]
    by_cases hc : cl j = c
    · subst hc
      simp [hB]
    · have hc' : c ≠ cl j := fun h' => hc h'.symm
      simp [hB, hc, one_apply_ne hc']
  have hD : B * Lᴴ = Kᴴ := by
    ext j i
    simp [hB, Matrix.mul_apply, hKL]
  refine not_isMinOn_of_tests hK0 B Eᴴ Lᴴ hC hD σ τ α β hα hβ hsym fun i c => ?_
  obtain ⟨t, ht⟩ := hcover i c
  refine ⟨t, fun j hj => ht j ?_⟩
  by_contra hne
  exact hj (by simp [hB, hne])

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_classes one_ne_zero id (1 : Matrix (Fin 2) (Fin 2) ℂ) (fun _ _ => rfl) id
    (by simp) (fun _ : Unit => Equiv.refl _) (fun _ => Equiv.refl _) (fun _ _ => 1) (fun _ _ => 1)
    (by simp) (by simp) (fun _ i j => show (1 : Matrix (Fin 2) (Fin 2) ℂ) i j =
      1 * (1 : Matrix (Fin 2) (Fin 2) ℂ) i j * 1 by ring)
    (fun _ _ => ⟨(), fun _ _ => Set.notMem_empty _⟩)

end Transformer.Grokking.FlatSpectrum
