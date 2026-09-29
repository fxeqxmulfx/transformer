/-
# The paper's Q_SR remains unbiased after a fixed RHT

arXiv:2601.22813v2, §3.1 gives a concrete FP4-SR quantizer; §3.2 gives the
RHT. Their composition is exactly unbiased under the group normal-range
condition needed by the paper's FP8 round-to-nearest scale formula.
-/

import Transformer.Quartet.Section3_SRRotation

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- **The paper's concrete `Q_SR` is unbiased after a fixed RHT** (§3.1–§3.2).
The group's normal-range condition, omitted in the prose of §3.1, prevents
its FP8 scale from underflowing and is exactly the condition already needed
by `integral_qSRAt`. With it, the paper's `c = 6` scales satisfy the generic
non-clipping theorem. -/
theorem integral_rhtInv_qSRAt {c : ℝ} (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (x : Fin (2 ^ k) → Fin 16 → ℝ) (hc : 0 < c) (hc' : c ≤ 6)
    (hy : 0 < absMax (rht k ε x))
    (hgroups : ∀ i, absMax (rht k ε x) ≤
      2 ^ (14 : ℕ) * groupAbsMax (rht k ε x) i)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k,
      rhtInv k ε (fun i' j' => qSRAt c (rht k ε x) i' j' (u (i', j'))) i j =
        x i j := by
  have ht : 0 < tensorScaleSR c (rht k ε x) := by
    unfold tensorScaleSR
    exact div_pos hy (by positivity)
  have hg : ∀ i, 0 < groupScaleSR c (rht k ε x) i :=
    fun i => groupScaleSR_pos hc hy i (hgroups i)
  have hclip : ∀ i j,
      |rht k ε x i j /
        (groupScaleSR c (rht k ε x) i * tensorScaleSR c (rht k ε x))| ≤ 6 :=
    fun i j => abs_div_groupScaleSR_le hc hc' hy i (hgroups i) j
  simpa only [qSRScaledAt, qSRAt] using
    (integral_rhtInv_qSRScaledAt ε x (groupScaleSR c (rht k ε x))
      (tensorScaleSR c (rht k ε x)) ht hg hclip i j)

/-- A nonzero rotated tensor with one FP8 group meets the concrete `Q_SR`
range hypothesis at `c = 6`: take the all-ones input and the all-positive RHT
seed (§3.1–§3.2). -/
example : 0 < (6 : ℝ) ∧ (6 : ℝ) ≤ 6 ∧
    0 < absMax (rht 0 (fun _ _ => false) (fun _ _ => (1 : ℝ))) ∧
    ∀ i : Fin (2 ^ 0),
      absMax (rht 0 (fun _ _ => false) (fun _ _ => (1 : ℝ))) ≤
        2 ^ (14 : ℕ) * groupAbsMax (rht 0 (fun _ _ => false) (fun _ _ => (1 : ℝ))) i := by
  let y : Fin (2 ^ 0) → Fin 16 → ℝ := rht 0 (fun _ _ => false) (fun _ _ => (1 : ℝ))
  have hcoord : y 0 0 = 4 := by
    change rht 0 (fun _ _ => false) (fun _ _ => (1 : ℝ)) 0 0 = 4
    simp [rht, hadamard_zero_row]
    have h : Real.sqrt ((2 : ℝ) ^ 4) = 4 := by
      rw [show (2 : ℝ) ^ 4 = 4 ^ 2 by norm_num, Real.sqrt_sq_eq_abs]
      norm_num
    rw [h]
    norm_num
  have hmax : 0 < absMax y := by
    have hbound := (abs_le_groupAbsMax y 0 0).trans (groupAbsMax_le_absMax y 0)
    rw [hcoord] at hbound
    norm_num at hbound
    linarith
  refine ⟨by norm_num, le_rfl, hmax, ?_⟩
  intro i
  have hi : i = 0 := Fin.eq_zero i
  subst i
  have hle : absMax y ≤ groupAbsMax y 0 := by
    unfold absMax
    exact Finset.sup'_le _ _ fun p _ => by
      have hp : p.1 = 0 := Fin.eq_zero p.1
      simpa [hp] using abs_le_groupAbsMax y 0 p.2
  have hnonneg := groupAbsMax_nonneg y 0
  norm_num
  nlinarith

end Quartet
end Transformer
