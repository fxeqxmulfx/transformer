/-
# The two NVFP4 quantizers: `Q_SR` and the clipping `Q_RTN`

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), §3.1 and §3.3.

A tensor here is `2^k` groups of `16` entries — the layout NVFP4 quantizes,
and the one the randomized rotation of §3.2 will need.  Both quantizers of the
paper build the same three levels, and differ in how they round:

* `qSR` (§3.1) is the scheme of every earlier FP4 training method: the per-
  tensor scale is `max|x| / (6 · 16/17 · 448)`, the group scales are rounded to
  nearest in E4M3, and the entries are rounded *stochastically* to E2M1.  The
  factor `16/17` is the most `RTN_FP8` can raise a group scale, which is what
  keeps the E2M1 argument inside `[-6, 6]`.  The grid maximum is a parameter
  `c`, `6.0` in §3.1 and one of `4.0`, `6.0` in §4.2.
* `qRTN s` (§3.3) is what `MS-EDEN` quantizes with: round-to-nearest at both
  levels, a free clipping factor `s` in place of `6 · 16/17`, and group scales
  capped by `256` rather than `448` "for them not to overflow when applying
  EDEN correction".  It is unbiased for no `s` at all — the rotation and the
  rescaling of §3.2 are what will restore unbiasedness.

`qSRAt` and `qRTN` return the *dequantized* value `x^FP4 · x^FP8 · x^FP32`,
since that is what the guarantees of the paper speak about.  That neither of
them clips, and that `Q_SR` is unbiased, is
`Transformer.Quartet.Section3_NonClipping`.
-/

import Transformer.Quartet.Section3_Grids

namespace Transformer
namespace Quartet

variable {k : ℕ} {c s : ℝ}

/-- The largest absolute value of a tensor laid out as `2^k` groups of `16`
entries: the `max_{i=1…d} |x_i|` of the scale displays of §3.1. -/
noncomputable def absMax (x : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  Finset.univ.sup' ⟨(⟨0, Nat.two_pow_pos k⟩, 0), Finset.mem_univ _⟩
    fun p : Fin (2 ^ k) × Fin 16 => |x p.1 p.2|

/-- The largest absolute value inside one group of `16` entries. -/
noncomputable def groupAbsMax (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun j : Fin 16 => |x i j|

/-- A group's largest absolute value is at most the tensor's: the `max_g |x|`
of the group-scale displays never exceeds the `max|x|` of the tensor scale. -/
theorem groupAbsMax_le_absMax (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) :
    groupAbsMax x i ≤ absMax x :=
  Finset.sup'_le _ _ fun j _ =>
    Finset.le_sup' (fun p : Fin (2 ^ k) × Fin 16 => |x p.1 p.2|) (Finset.mem_univ (i, j))

/-- Every entry of a group is bounded by the group's largest absolute value. -/
theorem abs_le_groupAbsMax (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    |x i j| ≤ groupAbsMax x i :=
  Finset.le_sup' (fun j : Fin 16 => |x i j|) (Finset.mem_univ j)

/-- And it is never negative, being a supremum of absolute values. -/
theorem groupAbsMax_nonneg (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) :
    0 ≤ groupAbsMax x i := by
  unfold groupAbsMax
  exact (abs_nonneg (x i 0)).trans
    (Finset.le_sup' (fun j : Fin 16 => |x i j|) (Finset.mem_univ (0 : Fin 16)))

/-- The per-tensor FP32 scale of `Q_SR`, `max|x| / (c · 16/17 · 448)` (§3.1),
where the grid maximum `c` is `6.0`.  The Four Over Six heuristic of §4.2 is
the same display with `c = 4.0`, which is why `c` is a parameter here. -/
noncomputable def tensorScaleSR (c : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  absMax x / (c * (16 / 17) * 448)

/-- Its per-group E4M3 scale, `RTN_FP8(max_g |x| / (x^FP32 · c · 16/17))` (§3.1). -/
noncomputable def groupScaleSR (c : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  rtn fp8 (groupAbsMax x i / (tensorScaleSR c x * c * (16 / 17)))

/-- One dequantized entry of `Q_SR`, with the rounding coin `t` of that entry
(§3.1): the E2M1 value is drawn by stochastic rounding, the two scales are
deterministic. -/
noncomputable def qSRAt (c : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16)
    (t : ℝ) : ℝ :=
  sr fp4 (x i j / (groupScaleSR c x i * tensorScaleSR c x)) t *
    groupScaleSR c x i * tensorScaleSR c x

/-- The per-tensor FP32 scale of the clipping `Q_RTN(x, s)`, `max|x| / (s · 256)`
(§3.3): `256.0` instead of `448.0` leaves room for the EDEN correction. -/
noncomputable def tensorScaleRTN (s : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  absMax x / (s * 256)

/-- Its per-group E4M3 scale, `RTN_FP8(max_g |x| / (x^FP32 · s))` (§3.3). -/
noncomputable def groupScaleRTN (s : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  rtn fp8 (groupAbsMax x i / (tensorScaleRTN s x * s))

/-- One dequantized entry of `Q_RTN(x, s)` (§3.3): round-to-nearest at every
level, clipping the entries whenever `s` exceeds `6 · 16/17`. -/
noncomputable def qRTN (s : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    ℝ :=
  rtn fp4 (x i j / (groupScaleRTN s x i * tensorScaleRTN s x)) *
    groupScaleRTN s x i * tensorScaleRTN s x

/-- **The group scales of `Q_RTN` stay below `256`**, whatever `s` is: the
headroom of §3.3 that lets `MS-EDEN` scale a group up without overflowing
E4M3, whose largest value is `448`. -/
theorem groupScaleRTN_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hs : 0 < s) (hx : 0 < absMax x)
    (i : Fin (2 ^ k)) : groupScaleRTN s x i ≤ 256 := by
  have hscale : tensorScaleRTN s x * s = absMax x / 256 := by
    unfold tensorScaleRTN
    field_simp
  have hpos : (0 : ℝ) < absMax x / 256 := by positivity
  have hv0 : 0 ≤ groupAbsMax x i / (tensorScaleRTN s x * s) := by
    rw [hscale]; exact div_nonneg (groupAbsMax_nonneg x i) hpos.le
  have hv256 : groupAbsMax x i / (tensorScaleRTN s x * s) ≤ 256 := by
    rw [hscale, div_le_iff₀ hpos]
    have := groupAbsMax_le_absMax x i
    linarith
  unfold groupScaleRTN rtn
  split
  · refine csSup_le ⟨0, ⟨by norm_num, 0, 0, by norm_num⟩, hv0⟩ ?_
    rintro y ⟨-, hyv⟩
    exact hyv.trans hv256
  · refine csInf_le ⟨-448, ?_⟩ ⟨⟨by norm_num, 8, 5, by norm_num⟩, hv256⟩
    rintro y ⟨hy, -⟩
    exact (abs_le.mp hy.1).1

/-- Its hypotheses are satisfiable at the paper's own non-clipping bound
`s = 6 · 16/17`. -/
example : 0 < 6 * (16 / 17 : ℝ) ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) := by
  refine ⟨by norm_num, ?_⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

end Quartet
end Transformer
