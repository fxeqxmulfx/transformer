/-
# How far `RTN_FP8` can move a group scale

Support for arXiv:2601.22813v2, §3.1 and §3.3.  The non-clipping claims of the
paper — "Given the choice of constants, stochastic rounding `SR_FP4` does not
clip its arguments" (§3.1) and "Setting the clipping factor `s` to `6 × 16/17`
or lower makes the scheme non-clipping" (§3.3) — both rest on one property of
the E4M3 grid, named in §3.1 as "`16/17`, the maximum factor by which
`RTN_FP8` can increase the underlying values".

Rounding to nearest moves a value by at most half the local step, and a
*normal* E4M3 value is at least `8` steps from zero, so `RTN_FP8(v)` is never
below `16/17 · v`.  That is `le_rtn_fp8` below.

The hypothesis `2^{-6} ≤ v` is the normal range, and it cannot be dropped: the
E4M3 subnormals are spaced `2^{-9}` apart all the way down to zero, so
`RTN_FP8(3 · 2^{-10}) = 2^{-9}` moves its argument by a factor `2/3`, far
below `16/17`.  The paper's claims are stated without it; what makes them
harmless in practice is that a group whose largest entry is `2^{14}` times
smaller than the tensor's is already zero in NVFP4.
-/

import Transformer.Quartet.Section3_Grids
import Mathlib.Algebra.Order.Archimedean.Basic
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

namespace Transformer
namespace Quartet

/-- A nonnegative multiple `m · 2^e` of a power of two is an E4M3 number as
soon as `m ≤ 16` and `-9 ≤ e`: at `m = 16` the significand rolls over into the
next binade, `16 · 2^e = 8 · 2^{e+1}`. -/
theorem mem_fp8 {m e : ℤ} (hm0 : 0 ≤ m) (hm : m ≤ 16) (he : -9 ≤ e)
    (h : (m : ℝ) * 2 ^ e ≤ 448) : (m : ℝ) * 2 ^ e ∈ fp8 := by
  have hm0' : (0 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm0
  have habs : |(m : ℝ) * 2 ^ e| = (m : ℝ) * 2 ^ e :=
    abs_of_nonneg (mul_nonneg hm0' (by positivity))
  refine ⟨by rw [habs]; exact h, ?_⟩
  rcases eq_or_lt_of_le hm with hm16 | hm16
  · subst hm16
    refine ⟨8, e + 1, by norm_num, by omega, ?_⟩
    rw [habs, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    push_cast
    ring
  · refine ⟨m.toNat, e, by omega, he, ?_⟩
    rw [habs]
    congr 1
    exact_mod_cast (Int.toNat_of_nonneg hm0).symm

/-- Its hypotheses are satisfiable at the largest E4M3 number, `448 = 14 · 2^5`. -/
example : (0 : ℤ) ≤ 14 ∧ (14 : ℤ) ≤ 16 ∧ (-9 : ℤ) ≤ 5 ∧ ((14 : ℤ) : ℝ) * 2 ^ (5 : ℤ) ≤ 448 := by
  norm_num

/-- **`RTN_FP8` never drops a normal value below `16/17` of itself** (§3.1,
"`16/17`, the maximum factor by which `RTN_FP8` can increase the underlying
values"): rounding to nearest moves `v` by at most half a step, and a normal
E4M3 value is at least `8` steps from zero. -/
theorem le_rtn_fp8 {v : ℝ} (hlo : (2 : ℝ) ^ (-6 : ℤ) ≤ v) (hhi : v ≤ 448) :
    16 / 17 * v ≤ rtn fp8 v := by
  have hv0 : (0 : ℝ) < v := lt_of_lt_of_le (by positivity) hlo
  obtain ⟨L, hL1, hL2⟩ := exists_mem_Ico_zpow hv0 (by norm_num : (1 : ℝ) < 2)
  -- the binade `[2^L, 2^{L+1})` of `v`, and its step `u = 2^{L-3}`
  have hL6 : -6 ≤ L := by
    have : (-6 : ℤ) < L + 1 :=
      (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)).mp (lt_of_le_of_lt hlo hL2)
    omega
  set u : ℝ := (2 : ℝ) ^ (L - 3) with hudef
  have hu0 : (0 : ℝ) < u := by rw [hudef]; positivity
  have h8u : (8 : ℝ) * u = 2 ^ L := by
    rw [hudef, show (8 : ℝ) = 2 ^ (3 : ℤ) by norm_num,
      ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    ring
  have h16u : (16 : ℝ) * u = 2 ^ (L + 1) := by
    rw [hudef, show (16 : ℝ) = 2 ^ (4 : ℤ) by norm_num,
      ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    ring
  -- the grid point `p = m · u` below `v`, with a significand `m ∈ [8, 15]`
  obtain ⟨m, hm⟩ : ∃ m : ℤ, m = ⌊v / u⌋ := ⟨_, rfl⟩
  have hdiv8 : (8 : ℝ) ≤ v / u := (le_div_iff₀ hu0).mpr (by rw [h8u]; exact hL1)
  have hdiv16 : v / u < 16 := (div_lt_iff₀ hu0).mpr (by rw [h16u]; exact hL2)
  have hm8 : (8 : ℝ) ≤ (m : ℝ) := by
    have h : (8 : ℤ) ≤ m := by rw [hm]; exact Int.le_floor.mpr (by exact_mod_cast hdiv8)
    exact_mod_cast h
  have hm15 : (m : ℝ) ≤ 15 := by
    have h : m < 16 := by rw [hm]; exact Int.floor_lt.mpr (by exact_mod_cast hdiv16)
    have h' : m ≤ 15 := by omega
    exact_mod_cast h'
  have hm0 : (0 : ℤ) ≤ m := by
    have h : (0 : ℝ) ≤ (m : ℝ) := le_trans (by norm_num) hm8
    exact_mod_cast h
  have hm16 : m ≤ 16 := by
    have h : (m : ℝ) ≤ 16 := le_trans hm15 (by norm_num)
    exact_mod_cast h
  have hm15z : m ≤ 15 := by exact_mod_cast hm15
  have hpv : (m : ℝ) * u ≤ v := (le_div_iff₀ hu0).mp (by rw [hm]; exact Int.floor_le _)
  have hvpu : v < (m : ℝ) * u + u := by
    have h : v < ((m : ℝ) + 1) * u :=
      (div_lt_iff₀ hu0).mp (by rw [hm]; exact Int.lt_floor_add_one _)
    linarith [h]
  have h8up : 8 * u ≤ (m : ℝ) * u := by nlinarith
  -- the two sides of the grid, read off the supremum and the infimum
  have hbddA : BddAbove {y | y ∈ fp8 ∧ y ≤ v} := ⟨448, fun y hy => (abs_le.mp hy.1.1).2⟩
  have hbddB : BddBelow {y | y ∈ fp8 ∧ v ≤ y} := ⟨-448, fun y hy => (abs_le.mp hy.1.1).1⟩
  have hmem448 : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num, by norm_num, by norm_num⟩
  have hmemp : (m : ℝ) * u ∈ fp8 := by
    rw [hudef]
    exact mem_fp8 hm0 hm16 (by omega) (by rw [← hudef]; linarith)
  have hfl : floorOn fp8 v = sSup {y | y ∈ fp8 ∧ y ≤ v} :=
    ite_eq_left ⟨_, hmemp, hpv⟩
  have hce : ceilOn fp8 v = sInf {y | y ∈ fp8 ∧ v ≤ y} :=
    ite_eq_left ⟨448, hmem448, hhi⟩
  have hpfl : (m : ℝ) * u ≤ floorOn fp8 v := hfl ▸ le_csSup hbddA ⟨hmemp, hpv⟩
  have hvce : v ≤ ceilOn fp8 v := hce ▸ le_csInf ⟨448, hmem448, hhi⟩ fun _ hy => hy.2
  have hceil : ceilOn fp8 v ≤ (m : ℝ) * u + u := by
    rw [hce]
    rcases le_or_gt ((m : ℝ) * u + u) 448 with h | h
    · have hmem : (m : ℝ) * u + u ∈ fp8 := by
        have hrw : (m : ℝ) * u + u = ((m + 1 : ℤ) : ℝ) * 2 ^ (L - 3) := by
          rw [hudef]; push_cast; ring
        rw [hrw]
        exact mem_fp8 (by omega) (by omega) (by omega) (by rw [← hrw]; exact h)
      exact csInf_le hbddB ⟨hmem, hvpu.le⟩
    · exact le_trans (csInf_le hbddB ⟨hmem448, hhi⟩) h.le
  -- half a step below, and never more
  unfold rtn
  rcases le_or_gt v ((m : ℝ) * u + u / 2) with hcase | hcase
  · split
    · linarith
    · linarith
  · split
    · rename_i hnear
      linarith
    · linarith

/-- Its hypotheses are satisfiable at the smallest normal E4M3 value, which is
also the smallest group scale the bound holds for. -/
example : (2 : ℝ) ^ (-6 : ℤ) ≤ 2 ^ (-6 : ℤ) ∧ (2 : ℝ) ^ (-6 : ℤ) ≤ 448 := by
  norm_num

end Quartet
end Transformer
