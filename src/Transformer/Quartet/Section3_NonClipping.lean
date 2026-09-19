/-
# Neither quantizer clips, and `Q_SR` is unbiased

arXiv:2601.22813v2, §3.1 and §3.3.  Two claims of the paper say that the E2M1
rounding is never asked for a value outside `[-6, 6]`: "Given the choice of
constants, stochastic rounding `SR_FP4` does not clip its arguments" (§3.1),
and "Setting the clipping factor `s` to `6 × 16/17` or lower makes the scheme
non-clipping" (§3.3).  Both are the same computation: an entry is at most
`max_g|x|`, the group scale is `RTN_FP8` of `max_g|x|` divided by the rest of
the display, and `RTN_FP8` gives back at least `16/17` of its argument
(`le_rtn_fp8`).  Unbiasedness of `Q_SR` (§3.1) then follows from `integral_sr`,
since the two scales do not depend on the coin.

**The hypothesis the paper does not state.**  `le_rtn_fp8` needs its argument
to be a normal E4M3 value, and here that is `max|x| ≤ 2^14 · max_g|x|`: no
group is more than `2^14` times smaller than the tensor.  Without it the
claims are false — a group scale landing among the E4M3 subnormals can be
rounded down by a factor `2/3`, and the E2M1 argument then leaves `[-6, 6]`.
The hypothesis is harmless for the paper's purpose, since a group that far
below the tensor maximum quantizes to zero anyway, but it is a hypothesis.
-/

import Transformer.Quartet.Fp8Grid
import Transformer.Quartet.Section3_NVFP4

namespace Transformer
namespace Quartet

variable {k : ℕ} {c s : ℝ}

/-- Both quantizers hand `RTN_FP8` an argument `b · max_g|x| / max|x|` with
`256 ≤ b ≤ 448`, and such an argument is a normal E4M3 value exactly when the
group is not vanishingly small against the tensor. -/
theorem normal_groupScale_arg {x : Fin (2 ^ k) → Fin 16 → ℝ} {b : ℝ} (hb : 256 ≤ b)
    (hb' : b ≤ 448) (hx : 0 < absMax x) (i : Fin (2 ^ k))
    (hg : absMax x ≤ 2 ^ (14 : ℕ) * groupAbsMax x i) :
    (2 : ℝ) ^ (-6 : ℤ) ≤ b * groupAbsMax x i / absMax x ∧
      b * groupAbsMax x i / absMax x ≤ 448 := by
  have hgm := groupAbsMax_le_absMax x i
  have hgn := groupAbsMax_nonneg x i
  norm_num at hg
  refine ⟨?_, ?_⟩
  · rw [show ((2 : ℝ) ^ (-6 : ℤ)) = 1 / 64 by norm_num, le_div_iff₀ hx]
    nlinarith [mul_nonneg (sub_nonneg.mpr hb) hgn]
  · rw [div_le_iff₀ hx]
    nlinarith [mul_nonneg (sub_nonneg.mpr hb') hgn]

/-- Its hypotheses are satisfiable: a tensor of ones has `max|x| = max_g|x| = 1`. -/
example : (256 : ℝ) ≤ 448 ∧ (448 : ℝ) ≤ 448 ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ∧
    absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤
      2 ^ (14 : ℕ) * groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
  have h1 : absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤ 1 := by
    unfold absMax
    exact Finset.sup'_le _ _ fun p _ => by norm_num
  have h2 : (1 : ℝ) ≤ groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
    simpa using abs_le_groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 0
  refine ⟨by norm_num, le_rfl, ?_, by norm_num; linarith⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

/-- **`Q_SR` never clips**: with the constants of §3.1 the argument handed to
`SR_FP4` stays in `[-6, 6]`, the range of E2M1 — "Given the choice of
constants, stochastic rounding `SR_FP4` does not clip its arguments".  The
factor `16/17`, "the maximum factor by which `RTN_FP8` can increase the
underlying values", is exactly what buys this. -/
theorem abs_div_groupScaleSR_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hc : 0 < c) (hc' : c ≤ 6)
    (hx : 0 < absMax x) (i : Fin (2 ^ k)) (hg : absMax x ≤ 2 ^ (14 : ℕ) * groupAbsMax x i)
    (j : Fin 16) : |x i j / (groupScaleSR c x i * tensorScaleSR c x)| ≤ 6 := by
  have hT : 0 < tensorScaleSR c x := div_pos hx (by positivity)
  have hTc : 0 < tensorScaleSR c x * c * (16 / 17) := by positivity
  have hTs : tensorScaleSR c x * c * (16 / 17) = absMax x / 448 := by
    unfold tensorScaleSR
    field_simp
  have harg : groupAbsMax x i / (tensorScaleSR c x * c * (16 / 17))
      = 448 * groupAbsMax x i / absMax x := by
    rw [hTs, div_div_eq_mul_div]
    ring
  obtain ⟨hlo, hhi⟩ := normal_groupScale_arg (b := 448) (by norm_num) le_rfl hx i hg
  have hv0 : (0 : ℝ) < 448 * groupAbsMax x i / absMax x := lt_of_lt_of_le (by positivity) hlo
  have hG : 16 / 17 * (448 * groupAbsMax x i / absMax x) ≤ groupScaleSR c x i := by
    unfold groupScaleSR
    rw [harg]
    exact le_rtn_fp8 hlo hhi
  have hGpos : 0 < groupScaleSR c x i := lt_of_lt_of_le (by linarith) hG
  have hxij : |x i j| ≤ 448 * groupAbsMax x i / absMax x * tensorScaleSR c x * c * (16 / 17) := by
    rw [show 448 * groupAbsMax x i / absMax x * tensorScaleSR c x * c * (16 / 17)
        = 448 * groupAbsMax x i / absMax x * (tensorScaleSR c x * c * (16 / 17)) by ring,
      ← harg, div_mul_cancel₀ _ hTc.ne']
    exact abs_le_groupAbsMax x i j
  have hstep : 448 * groupAbsMax x i / absMax x * tensorScaleSR c x * c * (16 / 17)
      ≤ 448 * groupAbsMax x i / absMax x * tensorScaleSR c x * 6 * (16 / 17) := by
    have := mul_le_mul_of_nonneg_left hc' (mul_pos hv0 hT).le
    nlinarith [this]
  have h1 : 16 / 17 * (448 * groupAbsMax x i / absMax x) * tensorScaleSR c x
      ≤ groupScaleSR c x i * tensorScaleSR c x := mul_le_mul_of_nonneg_right hG hT.le
  rw [abs_div, abs_of_pos (mul_pos hGpos hT), div_le_iff₀ (mul_pos hGpos hT)]
  linarith

/-- The hypotheses are satisfiable at the grid maximum `6.0` of §3.1 and a
tensor of ones. -/
example : 0 < (6 : ℝ) ∧ (6 : ℝ) ≤ 6 ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ∧
    absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤
      2 ^ (14 : ℕ) * groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
  have h1 : absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤ 1 := by
    unfold absMax
    exact Finset.sup'_le _ _ fun p _ => by norm_num
  have h2 : (1 : ℝ) ≤ groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
    simpa using abs_le_groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 0
  refine ⟨by norm_num, le_rfl, ?_, by norm_num; linarith⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

/-- **`Q_RTN(·, s)` clips nothing when `s ≤ 6 · 16/17`** (§3.3, "Setting the
clipping factor `s` to `6 × 16/17` or lower makes the scheme non-clipping");
above that value it does clip, which is what the paper's numerically optimal
`s = 6 · 16/17 / 0.93` trades error for. -/
theorem abs_div_groupScaleRTN_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hs : 0 < s)
    (hs' : s ≤ 6 * (16 / 17)) (hx : 0 < absMax x) (i : Fin (2 ^ k))
    (hg : absMax x ≤ 2 ^ (14 : ℕ) * groupAbsMax x i) (j : Fin 16) :
    |x i j / (groupScaleRTN s x i * tensorScaleRTN s x)| ≤ 6 := by
  have hT : 0 < tensorScaleRTN s x := div_pos hx (by positivity)
  have hTs0 : 0 < tensorScaleRTN s x * s := mul_pos hT hs
  have hTs : tensorScaleRTN s x * s = absMax x / 256 := by
    unfold tensorScaleRTN
    field_simp
  have harg : groupAbsMax x i / (tensorScaleRTN s x * s)
      = 256 * groupAbsMax x i / absMax x := by
    rw [hTs, div_div_eq_mul_div]
    ring
  obtain ⟨hlo, hhi⟩ := normal_groupScale_arg (b := 256) le_rfl (by norm_num) hx i hg
  have hv0 : (0 : ℝ) < 256 * groupAbsMax x i / absMax x := lt_of_lt_of_le (by positivity) hlo
  have hG : 16 / 17 * (256 * groupAbsMax x i / absMax x) ≤ groupScaleRTN s x i := by
    unfold groupScaleRTN
    rw [harg]
    exact le_rtn_fp8 hlo hhi
  have hGpos : 0 < groupScaleRTN s x i := lt_of_lt_of_le (by linarith) hG
  have hxij : |x i j| ≤ 256 * groupAbsMax x i / absMax x * tensorScaleRTN s x * s := by
    rw [show 256 * groupAbsMax x i / absMax x * tensorScaleRTN s x * s
        = 256 * groupAbsMax x i / absMax x * (tensorScaleRTN s x * s) by ring,
      ← harg, div_mul_cancel₀ _ hTs0.ne']
    exact abs_le_groupAbsMax x i j
  have hstep : 256 * groupAbsMax x i / absMax x * tensorScaleRTN s x * s
      ≤ 256 * groupAbsMax x i / absMax x * tensorScaleRTN s x * (6 * (16 / 17)) :=
    mul_le_mul_of_nonneg_left hs' (mul_pos hv0 hT).le
  have h1 : 16 / 17 * (256 * groupAbsMax x i / absMax x) * tensorScaleRTN s x
      ≤ groupScaleRTN s x i * tensorScaleRTN s x := mul_le_mul_of_nonneg_right hG hT.le
  rw [abs_div, abs_of_pos (mul_pos hGpos hT), div_le_iff₀ (mul_pos hGpos hT)]
  linarith

/-- The hypotheses are satisfiable at the paper's own non-clipping bound
`s = 6 · 16/17` and a tensor of ones. -/
example : 0 < 6 * (16 / 17 : ℝ) ∧ 6 * (16 / 17 : ℝ) ≤ 6 * (16 / 17) ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ∧
    absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤
      2 ^ (14 : ℕ) * groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
  have h1 : absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) ≤ 1 := by
    unfold absMax
    exact Finset.sup'_le _ _ fun p _ => by norm_num
  have h2 : (1 : ℝ) ≤ groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 := by
    simpa using abs_le_groupAbsMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) 0 0
  refine ⟨by norm_num, le_rfl, ?_, by norm_num; linarith⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

end Quartet
end Transformer
