/-
# AdaFisher: Gershgorin circle theorem

arXiv:2405.16397v3, Appendix A.1, Theorem A.1 and §3.1.
The localization theorem holds for every complex matrix; being inside its
discs does not by itself establish diagonal dominance of a Fisher factor.
-/

import Transformer.AdaFisher.Section2_Fisher
import Mathlib.Analysis.Complex.Basic
import Mathlib.Data.Finset.Max

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {n : ℕ}

/-- Off-diagonal row radius, Appendix A.1, Theorem A.1. -/
def gershgorinRadius (A : Matrix (Fin n) (Fin n) ℂ) (i : Fin n) : ℝ :=
  ∑ j ∈ Finset.univ.erase i, ‖A i j‖

/-- Every complex eigenvalue lies in some Gershgorin disc, Appendix A.1,
Theorem A.1. An eigenvalue is expressed by its nonzero eigenvector. -/
theorem gershgorin (A : Matrix (Fin n) (Fin n) ℂ) (z : ℂ) (v : Fin n → ℂ)
    (hv : v ≠ 0) (heig : A.mulVec v = z • v) :
    ∃ i, ‖z - A i i‖ ≤ gershgorinRadius A i := by
  have hne : ∃ k, v k ≠ 0 := by
    by_contra h
    push Not at h
    apply hv
    funext k
    exact h k
  obtain ⟨k, hk⟩ := hne
  obtain ⟨i, hi, hmax⟩ := Finset.exists_max_image Finset.univ (fun j => ‖v j‖)
    ⟨k, Finset.mem_univ k⟩
  have hpos : 0 < ‖v i‖ := lt_of_lt_of_le (norm_pos_iff.mpr hk)
    (hmax k (Finset.mem_univ k))
  have hrow : ∑ j, A i j * v j = z * v i := by
    simpa [Matrix.mulVec, dotProduct, Pi.smul_apply, smul_eq_mul] using congrFun heig i
  have hsplit := Finset.sum_erase_add Finset.univ (fun j => A i j * v j) (Finset.mem_univ i)
  have hsum : (z - A i i) * v i = ∑ j ∈ Finset.univ.erase i, A i j * v j := by
    calc
      _ = z * v i - A i i * v i := by ring
      _ = (∑ j, A i j * v j) - A i i * v i := by rw [hrow]
      _ = _ := by rw [← hsplit]; ring
  have hbound : ‖z - A i i‖ * ‖v i‖ ≤ gershgorinRadius A i * ‖v i‖ := by
    calc
      _ = ‖(z - A i i) * v i‖ := (norm_mul _ _).symm
      _ = ‖∑ j ∈ Finset.univ.erase i, A i j * v j‖ := by rw [hsum]
      _ ≤ ∑ j ∈ Finset.univ.erase i, ‖A i j * v j‖ := norm_sum_le _ _
      _ = ∑ j ∈ Finset.univ.erase i, ‖A i j‖ * ‖v j‖ := by simp only [norm_mul]
      _ ≤ ∑ j ∈ Finset.univ.erase i, ‖A i j‖ * ‖v i‖ := by
        apply Finset.sum_le_sum
        intro j hj
        exact mul_le_mul_of_nonneg_left (hmax j (Finset.mem_univ j)) (norm_nonneg _)
      _ = _ := by simp only [gershgorinRadius, Finset.sum_mul]
  exact ⟨i, (mul_le_mul_iff_left₀ hpos).mp (by simpa [mul_comm] using hbound)⟩

example : (fun _ : Fin 1 => (1 : ℂ)) ≠ 0 ∧
    (0 : Matrix (Fin 1) (Fin 1) ℂ).mulVec (fun _ => 1) =
      (0 : ℂ) • (fun _ => 1) := by
  constructor
  · intro h
    have hi := congrFun h 0
    norm_num at hi
  · simp

/-- Gershgorin localization does not imply diagonal dominance, §3.1 and
Appendix A.1. The all-ones Gram matrix of three deterministic features
is positive semidefinite, but each off-diagonal radius is two and its
diagonal is only one. The source's empirical observation is not a
consequence of the localization theorem. -/
theorem gershgorin_not_diagonal_dominance :
    let A : Matrix (Fin 3) (Fin 3) ℂ := fun _ _ => 1
    A.mulVec (fun _ => 1) = (3 : ℂ) • (fun _ => 1) ∧
    ‖A 0 0‖ < gershgorinRadius A 0 := by
  constructor
  · funext i
    norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  · norm_num [gershgorinRadius, Finset.sum_erase_eq_sub, Fin.sum_univ_succ]

end Transformer.AdaFisher
