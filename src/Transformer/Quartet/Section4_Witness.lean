/-
# The tensor Four Over Six is biased on, and its NVFP4 scales

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), §4.2 and Appendix A.

The paper asserts that Four Over Six "does not constitute an unbiased
estimation" and backs the assertion with a measurement — the plateau of
Figure 5 — not with a construction.  A proof needs one tensor whose
expectation moves, and this module fixes it: a single group of `16` entries,
the first `1` and the rest `0`.

That choice makes every scale explicit.  With one group `max_g|x| = max|x|`,
so the E4M3 argument of §3.1 is exactly `448` whatever the grid maximum `c`
is, `RTN_FP8` leaves it alone, and the two scales multiply to `17/(16 c)`.
The sole non-zero entry is therefore presented to E2M1 as `16 c / 17`: as
`64/17` on the `4.0` branch, whose neighbours are `3` and `4`, and as `96/17`
on the `6.0` branch, whose neighbours are `4` and `6`.  Those four grid
points, and `RTN_FP8(448) = 448`, are what the module computes first.
-/

import Transformer.Quartet.Section4_FourOverSix

namespace Transformer
namespace Quartet

/-- `0` is its own E2M1 neighbour from below, so `SR_FP4` leaves the zero
entries of the witness alone (§3.1). -/
theorem floorOn_fp4_zero : floorOn fp4 0 = 0 := by
  refine floorOn_eq (by norm_num [fp4]) le_rfl ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- And from above (§3.1). -/
theorem ceilOn_fp4_zero : ceilOn fp4 0 = 0 := by
  refine ceilOn_eq (by norm_num [fp4]) le_rfl ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- The E2M1 point below `64/17`, the argument of the `4.0` branch (§4.2). -/
theorem floorOn_fp4_64_17 : floorOn fp4 (64 / 17) = 3 := by
  refine floorOn_eq (by norm_num [fp4]) (by norm_num) ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- The E2M1 point above `64/17` (§4.2). -/
theorem ceilOn_fp4_64_17 : ceilOn fp4 (64 / 17) = 4 := by
  refine ceilOn_eq (by norm_num [fp4]) (by norm_num) ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- The E2M1 point below `96/17`, the argument of the `6.0` branch (§4.2). -/
theorem floorOn_fp4_96_17 : floorOn fp4 (96 / 17) = 4 := by
  refine floorOn_eq (by norm_num [fp4]) (by norm_num) ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- The E2M1 point above `96/17` (§4.2).  The gap is `2` here against `1` on
the other branch, and that asymmetry is the whole source of the bias. -/
theorem ceilOn_fp4_96_17 : ceilOn fp4 (96 / 17) = 6 := by
  refine ceilOn_eq (by norm_num [fp4]) (by norm_num) ?_
  intro y hy hle
  simp only [fp4, Set.mem_insert_iff, Set.mem_singleton_iff] at hy
  rcases hy with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    (revert hle; norm_num)

/-- `448` is the largest element of the E4M3 grid, so `RTN_FP8` fixes it
(§3.1).  This is the group scale of every one-group tensor. -/
theorem rtn_fp8_448 : rtn fp8 448 = 448 := by
  have hmem : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num⟩
  rw [rtn, floorOn_eq hmem le_rfl fun _ _ hle => hle,
    ceilOn_eq hmem le_rfl fun _ _ hle => hle]
  norm_num

/-- **The witness of §4.2**: one group of `16` entries, the first `1` and the
rest `0`.  Only the first is off the E2M1 grid at either scale, so it alone
carries the rounding error Four Over Six compares. -/
def biasWitness : Fin (2 ^ 0) → Fin 16 → ℝ := fun _ j => if j = 0 then 1 else 0

/-- Its largest absolute value is `1` (§3.1, the `max|x|` of the tensor
scale). -/
theorem absMax_biasWitness : absMax biasWitness = 1 := by
  refine le_antisymm (Finset.sup'_le _ _ fun p _ => ?_) ?_
  · show |biasWitness p.1 p.2| ≤ 1
    unfold biasWitness
    split <;> norm_num
  · unfold absMax
    refine le_trans ?_ (Finset.le_sup' (fun p : Fin (2 ^ 0) × Fin 16 => |biasWitness p.1 p.2|)
      (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))
    norm_num [biasWitness]

/-- With a single group, the `max_g|x|` of the group scale is the same `1`
(§3.1). -/
theorem groupAbsMax_biasWitness : groupAbsMax biasWitness 0 = 1 := by
  refine le_antisymm (Finset.sup'_le _ _ fun j _ => ?_) ?_
  · show |biasWitness 0 j| ≤ 1
    unfold biasWitness
    split <;> norm_num
  · unfold groupAbsMax
    refine le_trans ?_ (Finset.le_sup' (fun j : Fin 16 => |biasWitness 0 j|)
      (Finset.mem_univ (0 : Fin 16)))
    norm_num [biasWitness]

/-- Its per-tensor FP32 scale, `max|x| / (c · 16/17 · 448)` (§3.1). -/
theorem tensorScaleSR_biasWitness (c : ℝ) : tensorScaleSR c biasWitness = 17 / (c * 7168) := by
  unfold tensorScaleSR
  rw [absMax_biasWitness]
  ring

/-- Its per-group E4M3 scale is the top of the grid, `448`: with one group the
argument of `RTN_FP8` is exactly `448`, whatever the grid maximum is (§3.1). -/
theorem groupScaleSR_biasWitness (c : ℝ) (hc : c ≠ 0) : groupScaleSR c biasWitness 0 = 448 := by
  unfold groupScaleSR
  rw [groupAbsMax_biasWitness, tensorScaleSR_biasWitness,
    show 17 / (c * 7168) * c * (16 / 17) = 1 / 448 by field_simp; ring]
  norm_num
  exact rtn_fp8_448

/-- The hypothesis of `groupScaleSR_biasWitness` is satisfiable at both grid
maxima Four Over Six chooses between. -/
example : (4 : ℝ) ≠ 0 ∧ (6 : ℝ) ≠ 0 := ⟨by norm_num, by norm_num⟩

end Quartet
end Transformer
