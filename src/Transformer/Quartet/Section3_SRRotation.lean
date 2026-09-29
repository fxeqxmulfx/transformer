/-
# Exact unbiasedness from element-wise FP4 stochastic rounding after RHT

arXiv:2601.22813v2, §3.1 defines `Q_SR` and states that its dequantized
entries are unbiased. Section 3.2 defines the randomized Hadamard transform.
Applying SR after a fixed rotation, then undoing that rotation, yields exact
unbiasedness. Fixed scales must be positive and prevent FP4 clipping; a
conservative scale construction below satisfies this for every real input.
-/

import Transformer.Quartet.Section3_SRCoinCube
import Transformer.Quartet.Hadamard

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- Every entry has its own SR coin; integrating out the whole cube preserves
the one-entry expectation of §3.1. -/
theorem integral_qSRScaledAt_cube (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (ht : 0 < t) (hg : ∀ i, 0 < g i)
    (hclip : ∀ i j, |y i j / (g i * t)| ≤ 6)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k, qSRScaledAt g t y i j (u (i, j)) = y i j := by
  unfold srCoinCube
  rw [integral_srCoinCube_eval]
  exact integral_qSRScaledAt g t y ht hg hclip i j

/-- The one-entry observables are integrable on the full cube of SR coins. -/
theorem integrable_qSRScaledAt_cube (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    Integrable (fun u : (Fin (2 ^ k) × Fin 16) → ℝ =>
      qSRScaledAt g t y i j (u (i, j))) (volume.restrict (srCoinCube k)) := by
  simpa [srCoinCube] using
    (integrable_srCoinCube_eval (integrable_qSRScaledAt g t y i j) (i, j))

/-- For each fixed RHT seed, independent FP4 stochastic rounding followed by
the inverse rotation is exactly unbiased. This follows from the §3.1 `Q_SR`
identity and the §3.2 RHT inverse. The conditions on fixed scales say that no
group underflows to zero and no normalized entry clips outside E2M1. -/
theorem integral_rhtInv_qSRScaledAt (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (x : Fin (2 ^ k) → Fin 16 → ℝ) (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (ht : 0 < t) (hg : ∀ i, 0 < g i)
    (hclip : ∀ i j, |rht k ε x i j / (g i * t)| ≤ 6)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k,
      rhtInv k ε (fun i' j' => qSRScaledAt g t (rht k ε x) i' j' (u (i', j'))) i j =
        x i j := by
  have hentry (i' : Fin (2 ^ k)) (j' : Fin 16) :
      Integrable (fun u : (Fin (2 ^ k) × Fin 16) → ℝ =>
        hadamard k i i' j j' * qSRScaledAt g t (rht k ε x) i' j' (u (i', j')))
        (volume.restrict (srCoinCube k)) :=
    (integrable_qSRScaledAt_cube g t (rht k ε x) i' j').const_mul _
  have houter (i' : Fin (2 ^ k)) :
      Integrable (fun u : (Fin (2 ^ k) × Fin 16) → ℝ =>
        ∑ j' : Fin 16,
          hadamard k i i' j j' * qSRScaledAt g t (rht k ε x) i' j' (u (i', j')))
        (volume.restrict (srCoinCube k)) :=
    integrable_finsetSum _ fun j' _ => hentry i' j'
  calc
    ∫ u in srCoinCube k,
        rhtInv k ε (fun i' j' => qSRScaledAt g t (rht k ε x) i' j' (u (i', j'))) i j
        = rhtInv k ε (rht k ε x) i j := by
          unfold rhtInv
          rw [integral_const_mul, integral_finsetSum _ fun i' _ => houter i']
          congr 1
          refine Finset.sum_congr rfl fun i' _ => ?_
          rw [integral_finsetSum _ fun j' _ => hentry i' j']
          refine Finset.sum_congr rfl fun j' _ => ?_
          rw [integral_const_mul, integral_qSRScaledAt_cube g t (rht k ε x) ht hg hclip i' j']
    _ = x i j := rhtInv_rht ε x i j

/-- A conservative tensor scale that uses the E4M3 group scale `1` for every
group. It covers every real input, including zero, and keeps the FP4 argument
inside `[-6, 6]` (§3.1). It is an existence construction, not an MSE-optimal
replacement for the paper's per-group scales. -/
noncomputable def safeTensorScale (y : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  if absMax y = 0 then 1 else absMax y / 6

/-- The constant group scale of the conservative construction is an E4M3
number (§3.1). -/
theorem one_mem_fp8 : (1 : ℝ) ∈ fp8 := by
  refine ⟨by norm_num, 8, -3, by norm_num, by norm_num, ?_⟩
  norm_num

/-- The conservative tensor scale never vanishes, even on the zero tensor. -/
theorem safeTensorScale_pos (y : Fin (2 ^ k) → Fin 16 → ℝ) :
    0 < safeTensorScale y := by
  have hnonneg : 0 ≤ absMax y :=
    (abs_nonneg (y 0 0)).trans
      ((abs_le_groupAbsMax y 0 0).trans (groupAbsMax_le_absMax y 0))
  unfold safeTensorScale
  split_ifs with h
  · norm_num
  · have hpos : 0 < absMax y := lt_of_le_of_ne hnonneg (Ne.symm h)
    exact div_pos hpos (by norm_num)

/-- The conservative scale prevents FP4 clipping for every entry. -/
theorem safeTensorScale_nonclipping (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    |y i j / ((1 : ℝ) * safeTensorScale y)| ≤ 6 := by
  have hbound : |y i j| ≤ absMax y :=
    (abs_le_groupAbsMax y i j).trans (groupAbsMax_le_absMax y i)
  by_cases hzero : absMax y = 0
  · have hyzero : y i j = 0 := abs_eq_zero.mp (le_antisymm (hbound.trans_eq hzero)
      (abs_nonneg _))
    simp [safeTensorScale, hzero, hyzero]
  · have hscale : safeTensorScale y = absMax y / 6 := by
      simp [safeTensorScale, hzero]
    have htpos : 0 < absMax y / 6 := by rw [← hscale]; exact safeTensorScale_pos y
    rw [one_mul, hscale, abs_div, abs_of_pos htpos]
    apply (div_le_iff₀ htpos).2
    nlinarith [hbound]

/-- **A universally defined, exactly unbiased FP4-SR/RHT quantizer.** For
every input and every *fixed* RHT seed, use the positive non-clipping scale
above and one SR coin per FP4 entry; the expected inverse-rotated output is
the original input. This is a conservative instance of the SR variant in
§3.1, composed with the RHT of §3.2. The tensor scale is an exact real number,
as in the existing NVFP4 model; finite FP32 rounding, MSE, and throughput are
outside this statement. -/
theorem integral_rhtInv_qSRSafe (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k,
      rhtInv k ε (fun i' j' =>
        qSRScaledAt (fun _ => 1) (safeTensorScale (rht k ε x)) (rht k ε x)
          i' j' (u (i', j'))) i j = x i j := by
  apply integral_rhtInv_qSRScaledAt ε x (fun _ => 1) (safeTensorScale (rht k ε x))
    (safeTensorScale_pos _) (by intro i'; norm_num)
    (by intro i' j'; exact safeTensorScale_nonclipping _ i' j')

/-- The scale hypotheses used by the full-vector result have a concrete
witness, even for the zero input. -/
example : ∃ (g : Fin (2 ^ 0) → ℝ) (t : ℝ), 0 < t ∧
    (∀ i, 0 < g i) ∧
    (∀ i j, |rht 0 (fun _ _ => false) (fun _ _ => (0 : ℝ)) i j / (g i * t)| ≤ 6) := by
  refine ⟨fun _ => 1, safeTensorScale (rht 0 (fun _ _ => false) (fun _ _ => 0)),
    safeTensorScale_pos _, ?_, ?_⟩
  · intro i; norm_num
  · intro i j; exact safeTensorScale_nonclipping _ i j
end Quartet
end Transformer
