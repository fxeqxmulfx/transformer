/-
# An everywhere non-clipping E4M3 scale policy for FP4 stochastic rounding

arXiv:2601.22813v2, §3.1 uses an FP32 tensor scale and E4M3 group scales,
then rounds each FP4 value stochastically. Its E4M3 round-to-nearest scale
can underflow or make an FP4 argument exceed `6` in a very small group.

This alternative keeps the same two-level layout but rounds each positive
group scale *up* to E4M3. A zero group receives the representable scale `1`.
It is an explicit corrected construction: no input-dependent normal-range
hypothesis is needed for exact unbiasedness after any fixed RHT. As in the
rest of the formalization, the FP32 tensor scale is modeled by an exact real.
-/

import Transformer.Quartet.Section3_SRRotation

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- The tensor scale `max|y| / (6 · 448)` when `y ≠ 0`, with a positive
fallback for `y = 0` (§3.1, adjusted for group scales rounded upward). -/
noncomputable def ceilTensorScale (y : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  safeTensorScale y / 448

/-- The E4M3 group scale is the least representable value above the ideal
`max_g|y|/(6 · tensorScale)`; a zero group uses `1` (§3.1, corrected scale
selection). -/
noncomputable def ceilGroupScale (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) : ℝ :=
  if groupAbsMax y i = 0 then 1 else
    ceilOn fp8 (groupAbsMax y i / (ceilTensorScale y * 6))

/-- The adjusted tensor scale is strictly positive. -/
theorem ceilTensorScale_pos (y : Fin (2 ^ k) → Fin 16 → ℝ) :
    0 < ceilTensorScale y := by
  unfold ceilTensorScale
  exact div_pos (safeTensorScale_pos y) (by norm_num)

/-- For every group, the upward-rounded E4M3 scale is representable,
positive, and large enough that its maximum normalized FP4 entry is at most
`6`. This includes subnormal and zero groups. -/
theorem ceilGroupScale_spec (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) :
    ceilGroupScale y i ∈ fp8 ∧ 0 < ceilGroupScale y i ∧
      groupAbsMax y i ≤ ceilGroupScale y i * ceilTensorScale y * 6 := by
  have ht : 0 < ceilTensorScale y := ceilTensorScale_pos y
  by_cases hzero : groupAbsMax y i = 0
  · simp only [ceilGroupScale, ite_eq_left hzero]
    exact ⟨one_mem_fp8, by norm_num, by rw [hzero]; positivity⟩
  · have hg0 : 0 < groupAbsMax y i :=
      lt_of_le_of_ne (groupAbsMax_nonneg y i) (Ne.symm hzero)
    have hM : 0 < absMax y := lt_of_lt_of_le hg0 (groupAbsMax_le_absMax y i)
    have hts : ceilTensorScale y * 6 * 448 = absMax y := by
      unfold ceilTensorScale safeTensorScale
      simp only [ite_eq_right hM.ne']
      ring
    let v := groupAbsMax y i / (ceilTensorScale y * 6)
    have hv0 : 0 < v := div_pos hg0 (mul_pos ht (by norm_num))
    have hv448 : v ≤ 448 := by
      dsimp [v]
      apply (div_le_iff₀ (mul_pos ht (by norm_num))).2
      nlinarith [groupAbsMax_le_absMax y i]
    have hfp0 : (0 : ℝ) ∈ fp8 := ⟨by norm_num, 0, 0, by norm_num⟩
    have hfp448 : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num⟩
    obtain ⟨-, -, hmem, hceil⟩ := floorOn_mem_le_and_le_ceilOn_mem fp8_finite
      ⟨0, hfp0, hv0.le⟩ ⟨448, hfp448, hv448⟩
    have hbound : groupAbsMax y i ≤ ceilOn fp8 v * (ceilTensorScale y * 6) :=
      (div_le_iff₀ (mul_pos ht (by norm_num))).mp hceil
    simp only [ceilGroupScale, ite_eq_right hzero]
    exact ⟨hmem, hv0.trans_le hceil, by nlinarith [hbound]⟩

/-- Every normalized entry stays in E2M1 range `[-6, 6]`, including when a
group is smaller than the normal E4M3 range (§3.1, corrected scale policy). -/
theorem ceilGroupScale_nonclipping (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    |y i j / (ceilGroupScale y i * ceilTensorScale y)| ≤ 6 := by
  obtain ⟨-, hg, hbound⟩ := ceilGroupScale_spec y i
  have hscale : 0 < ceilGroupScale y i * ceilTensorScale y :=
    mul_pos hg (ceilTensorScale_pos y)
  rw [abs_div, abs_of_pos hscale]
  apply (div_le_iff₀ hscale).2
  nlinarith [abs_le_groupAbsMax y i j]

/-- **The expected quantized vector equals the fixed rotated vector**, one
coordinate at a time, with E4M3 ceilings and one FP4 SR coin per entry
(§3.1). -/
theorem integral_qSRCeilScale (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k,
      qSRScaledAt (ceilGroupScale y) (ceilTensorScale y) y i j (u (i, j)) =
        y i j :=
  integral_qSRScaledAt_cube (ceilGroupScale y) (ceilTensorScale y) y
    (ceilTensorScale_pos y) (fun i' => (ceilGroupScale_spec y i').2.1)
    (fun i' j' => ceilGroupScale_nonclipping y i' j') i j

/-- **Exact unbiasedness with E4M3 group scales for every input and every
fixed RHT seed.** The group scales are rounded upward; FP4 values are rounded
stochastically with independent coins. This is the §3.1 SR route with an
everywhere non-clipping scale choice, rather than the §3.3 MS-EDEN route. -/
theorem integral_rhtInv_qSRCeilScale (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in srCoinCube k,
      rhtInv k ε (fun i' j' =>
        qSRScaledAt (ceilGroupScale (rht k ε x)) (ceilTensorScale (rht k ε x))
          (rht k ε x) i' j' (u (i', j'))) i j = x i j := by
  apply integral_rhtInv_qSRScaledAt ε x (ceilGroupScale (rht k ε x))
    (ceilTensorScale (rht k ε x)) (ceilTensorScale_pos _)
    (fun i' => (ceilGroupScale_spec _ i').2.1)
    (fun i' j' => ceilGroupScale_nonclipping _ i' j')

end Quartet
end Transformer
