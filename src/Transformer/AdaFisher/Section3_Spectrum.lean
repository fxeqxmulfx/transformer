/-
# AdaFisher: inverse and spectral bounds

arXiv:2405.16397v3, Proposition 3.2, Appendix A.2, Part 2.
-/

import Transformer.AdaFisher.Section3_Efficient

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {a b d : ℕ}

/-- The ordinary matrix inverse is the entrywise reciprocal when all
diagonal entries are nonzero, Proposition 3.2, the closed-form inverse. -/
theorem diagonal_inverse {ι : Type*} [Fintype ι] [DecidableEq ι]
    (f : ι → ℝ) (hf : ∀ i, f i ≠ 0) :
    (Matrix.diagonal f)⁻¹ = Matrix.diagonal (fun i => (f i)⁻¹) := by
  apply Matrix.inv_eq_left_inv
  ext i j
  by_cases hij : i = j
  · subst j
    simp [hf i]
  · simp [hij]

example : ∀ i : Fin 1, (fun _ => (2 : ℝ)) i ≠ 0 := by norm_num

/-- All real eigenvalues of the efficient Fisher block have the bounds
printed in Proposition 3.2, Appendix A.2. -/
theorem fisherMatrix_eigenvalue_bounds (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ)
    (hh : ∀ i, 0 ≤ h i ∧ h i ≤ 1) (hs : ∀ j, 0 ≤ s j ∧ s j ≤ 1)
    (z : ℝ) (v : Fin a × Fin b → ℝ) (hv : v ≠ 0)
    (heig : (fisherMatrix δ h s).mulVec v = z • v) : δ ≤ z ∧ z ≤ 1 + δ := by
  have hne : ∃ p, v p ≠ 0 := by
    by_contra hn
    push Not at hn
    apply hv
    funext p
    exact hn p
  obtain ⟨p, hp⟩ := hne
  have heq : fisherDiagonal δ h s p = z := by
    apply mul_right_cancel₀ hp
    simpa [fisherMatrix, Matrix.mulVec_diagonal] using congrFun heig p
  rw [← heq]
  exact fisherDiagonal_bounds δ h s hh hs p

example :
    let h : Fin 1 → ℝ := fun _ => 0
    let s : Fin 1 → ℝ := fun _ => 0
    let v : Fin 1 × Fin 1 → ℝ := fun _ => 1
    (∀ i, 0 ≤ h i ∧ h i ≤ 1) ∧ (∀ j, 0 ≤ s j ∧ s j ≤ 1) ∧
      v ≠ 0 ∧ (fisherMatrix 1 h s).mulVec v = (1 : ℝ) • v := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · intro hv
    have hi := congrFun hv (0, 0)
    norm_num at hi
  · funext p
    simp [fisherMatrix, fisherDiagonal, Matrix.mulVec_diagonal]

/-- Spectral reciprocal bounds for the inverse preconditioner,
Proposition 3.2, Appendix A.2. The denominator can be as small as damping;
an upper bound on a matrix norm cannot replace this lower entry bound. -/
theorem fisherDiagonal_inverse_bounds (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ)
    (hδ : 0 < δ) (hh : ∀ i, 0 ≤ h i ∧ h i ≤ 1)
    (hs : ∀ j, 0 ≤ s j ∧ s j ≤ 1) (p : Fin a × Fin b) :
    (1 + δ)⁻¹ ≤ (fisherDiagonal δ h s p)⁻¹ ∧ (fisherDiagonal δ h s p)⁻¹ ≤ δ⁻¹ := by
  obtain ⟨hl, hu⟩ := fisherDiagonal_bounds δ h s hh hs p
  have hf : 0 < fisherDiagonal δ h s p := lt_of_lt_of_le hδ hl
  constructor
  · exact inv_anti₀ hf hu
  · exact inv_anti₀ hδ hl

example : (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 1, 0 ≤ (fun _ => (0 : ℝ)) i ∧ (fun _ => (0 : ℝ)) i ≤ 1) ∧
    (∀ j : Fin 2, 0 ≤ (fun _ => (1 : ℝ)) j ∧ (fun _ => (1 : ℝ)) j ≤ 1) := by norm_num

end Transformer.AdaFisher
