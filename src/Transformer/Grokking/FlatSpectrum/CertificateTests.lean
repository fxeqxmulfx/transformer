module

public import Transformer.Grokking.FlatSpectrum.CertificateOrbit

/-!
# A criterion through symmetries for "M does not return `K`"

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A minimum of M has a
certificate `Y` vanishing off the seen cells (item 5.1), and all certificates agree on the row space
of `K` (`mul_conjTranspose_eq_of_certificates`).

A symmetry of `K`, a permutation of rows and columns up to phases, carries `Y` to another
certificate, so `Y b` does not change along the orbit of a test vector `b` in the row space. If
every pair of a row and a test vector meets an orbit where the vector is unseen, then `Y b = 0` for
all of them, so `Y Kᴴ = 0` and `‖K‖_S1 = 0` (`not_isMinOn_of_tests`). For an invertible `K` the test
vectors are the columns of the identity (`not_isMinOn_of_orbits`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] in
/-- Phases and permutations of rows and columns keep a contraction. -/
lemma IsContraction.phased {Y : Matrix m n ℂ} (hY : IsContraction Y) (σ : Equiv.Perm m)
    (τ : Equiv.Perm n) {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1) (hb : ∀ j, ‖b j‖ = 1) :
    IsContraction (Matrix.of fun i j => a i * Y (σ i) (τ j) * b j) := by
  rw [isContraction_iff_sum_norm_sq_le] at hY ⊢
  intro v
  set v' : n → ℂ := fun j => b (τ.symm j) * v (τ.symm j) with hv'
  have hmul (i : m) : (Matrix.of (fun i j => a i * Y (σ i) (τ j) * b j) *ᵥ v) i =
      a i * (Y *ᵥ v') (σ i) := by
    simp only [mulVec, dotProduct, of_apply, Finset.mul_sum]
    rw [← Equiv.sum_comp τ (fun j => a i * (Y (σ i) j * v' j))]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [hv', Equiv.symm_apply_apply]
    ring
  calc ∑ i, ‖(Matrix.of (fun i j => a i * Y (σ i) (τ j) * b j) *ᵥ v) i‖ ^ 2
      = ∑ i, ‖(Y *ᵥ v') (σ i)‖ ^ 2 := by simp only [hmul, norm_mul, ha, one_mul]
    _ = ∑ i, ‖(Y *ᵥ v') i‖ ^ 2 := Equiv.sum_comp σ (fun i => ‖(Y *ᵥ v') i‖ ^ 2)
    _ ≤ ∑ j, ‖v' j‖ ^ 2 := hY v'
    _ = ∑ j, ‖v j‖ ^ 2 := by
      rw [← Equiv.sum_comp τ (fun j => ‖v' j‖ ^ 2)]
      simp only [hv', Equiv.symm_apply_apply, norm_mul, hb, one_mul]

example := (isContraction_one (n := Fin 2)).phased 1 1 (a := fun _ => 1) (b := fun _ => 1)
  (by simp) (by simp)

omit [DecidableEq m] [DecidableEq n] in
/-- A symmetry of `K` carries a certificate to a certificate. -/
lemma trace_phased {K Y : Matrix m n ℂ} {σ : Equiv.Perm m} {τ : Equiv.Perm n} {α : m → ℂ}
    {β : n → ℂ} (hsym : ∀ i j, K (σ i) (τ j) = α i * K i j * β j) :
    ((Matrix.of fun i j => star (α i) * Y (σ i) (τ j) * star (β j))ᴴ * K).trace =
      (Yᴴ * K).trace := by
  rw [trace_conjTranspose_mul, trace_conjTranspose_mul,
    ← Equiv.sum_comp σ (fun i => ∑ j, star (Y i j) * K i j)]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Equiv.sum_comp τ (fun j => star (Y (σ i) j) * K (σ i) j)]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [of_apply, star_mul', star_star, hsym]
  ring

example (K Y : Matrix (Fin 2) (Fin 2) ℂ) := trace_phased (K := K) (Y := Y) (σ := 1) (τ := 1)
  (α := fun _ => 1) (β := fun _ => 1) (by simp)

/-- **A criterion through symmetries and test vectors.** Let the columns of `B` span the row space
of `K` (`Kᴴ C = B`, `B D = Kᴴ`), and let every row `i` and test vector `s` have a symmetry `t` that
sends the support of `s` in the row `i` outside `Ω`. Then M does not return `K`. -/
theorem not_isMinOn_of_tests {K : Matrix m n ℂ} (hK0 : K ≠ 0) {Ω : Set (m × n)} {r ι : Type*}
    [Fintype r] (B : Matrix n r ℂ) (C : Matrix m r ℂ) (D : Matrix r m ℂ)
    (hC : Kᴴ * C = B) (hD : B * D = Kᴴ)
    (σ : ι → Equiv.Perm m) (τ : ι → Equiv.Perm n) (α : ι → m → ℂ) (β : ι → n → ℂ)
    (hα : ∀ t i, ‖α t i‖ = 1) (hβ : ∀ t j, ‖β t j‖ = 1)
    (hsym : ∀ t i j, K (σ t i) (τ t j) = α t i * K i j * β t j)
    (hcover : ∀ i s, ∃ t, ∀ j, B j s ≠ 0 → (σ t i, τ t j) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  intro hmin
  obtain ⟨Y, hYΩ, hY, htr⟩ := exists_isContraction_of_isMinOn hmin
  have hYB : Y * B = 0 := by
    ext i s
    obtain ⟨t, ht⟩ := hcover i s
    set Z : Matrix m n ℂ :=
      Matrix.of fun i j => star (α t i) * Y (σ t i) (τ t j) * star (β t j) with hZdef
    have hZ : IsContraction Z :=
      hY.phased (σ t) (τ t) (fun i => by simp [hα]) (fun j => by simp [hβ])
    have hZtr : (Zᴴ * K).trace = nuclearNorm K := by rw [hZdef, trace_phased (hsym t), htr]
    have hEq : Y * B = Z * B := by
      rw [← hC, ← Matrix.mul_assoc, ← Matrix.mul_assoc,
        mul_conjTranspose_eq_of_certificates hY htr hZ hZtr]
    rw [hEq, Matrix.mul_apply, Matrix.zero_apply]
    refine Finset.sum_eq_zero fun j _ => ?_
    by_cases hj : B j s = 0
    · rw [hj, mul_zero]
    · simp [hZdef, hYΩ _ _ (ht j hj)]
  have h0 : Y * Kᴴ = 0 := by rw [← hD, ← Matrix.mul_assoc, hYB, Matrix.zero_mul]
  have htr0 : (nuclearNorm K : ℂ) = 0 := by
    rw [← htr, show Yᴴ * K = (Kᴴ * Y)ᴴ by rw [conjTranspose_mul, conjTranspose_conjTranspose],
      trace_conjTranspose, Matrix.trace_mul_comm, h0, trace_zero, star_zero]
  exact hK0 (eq_zero_of_nuclearNorm_eq_zero (by exact_mod_cast htr0))

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_tests (r := Fin 2) one_ne_zero (1 : Matrix (Fin 2) (Fin 2) ℂ) 1 1 (by simp)
    (by simp) (fun _ : Unit => Equiv.refl _) (fun _ => Equiv.refl _) (fun _ _ => 1) (fun _ _ => 1)
    (by simp) (by simp) (fun _ i j => show (1 : Matrix (Fin 2) (Fin 2) ℂ) i j =
      1 * (1 : Matrix (Fin 2) (Fin 2) ℂ) i j * 1 by ring)
    (fun _ _ => ⟨(), fun _ _ => Set.notMem_empty _⟩)

/-- **An invertible table with an unseen cell on every orbit is not returned.** -/
theorem not_isMinOn_of_orbits [Nonempty n] {K : Matrix n n ℂ} (hK : K.det ≠ 0) {Ω : Set (n × n)}
    {ι : Type*} (σ τ : ι → Equiv.Perm n) (α β : ι → n → ℂ)
    (hα : ∀ t i, ‖α t i‖ = 1) (hβ : ∀ t j, ‖β t j‖ = 1)
    (hsym : ∀ t i j, K (σ t i) (τ t j) = α t i * K i j * β t j)
    (hcover : ∀ i j, ∃ t, (σ t i, τ t j) ∉ Ω) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  have hK0 : K ≠ 0 := fun h => hK (by rw [h, det_zero])
  have hdet : IsUnit (Kᴴ).det := by
    rw [det_conjTranspose]
    exact (star_ne_zero.2 hK).isUnit
  refine not_isMinOn_of_tests hK0 (1 : Matrix n n ℂ) (Kᴴ)⁻¹ Kᴴ (mul_nonsing_inv _ hdet)
    (Matrix.one_mul _) σ τ α β hα hβ hsym fun i s => ?_
  obtain ⟨t, ht⟩ := hcover i s
  refine ⟨t, fun j hj => ?_⟩
  have hjs : j = s := by
    by_contra hne
    exact hj (one_apply_ne hne)
  exact hjs ▸ ht

example : ¬ IsMinOn nuclearNorm {X : Matrix (Fin 2) (Fin 2) ℂ |
    ∀ ij ∈ (∅ : Set (Fin 2 × Fin 2)), X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2} 1 :=
  not_isMinOn_of_orbits (by simp) (fun _ : Unit => Equiv.refl _) (fun _ => Equiv.refl _)
    (fun _ _ => 1) (fun _ _ => 1) (by simp) (by simp) (fun _ i j => show
      (1 : Matrix (Fin 2) (Fin 2) ℂ) i j = 1 * (1 : Matrix (Fin 2) (Fin 2) ℂ) i j * 1 by ring)
    (fun _ _ => ⟨(), Set.notMem_empty _⟩)

end Transformer.Grokking.FlatSpectrum
