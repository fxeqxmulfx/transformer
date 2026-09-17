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
  keeps the E2M1 argument inside `[-6, 6]`.
* `qRTN s` (§3.3) is what `MS-EDEN` quantizes with: round-to-nearest at both
  levels, a free clipping factor `s` in place of `6 · 16/17`, and group scales
  capped by `256` rather than `448` "for them not to overflow when applying
  EDEN correction".  It is unbiased for no `s` at all — the rotation and the
  rescaling of §3.2 are what will restore unbiasedness.

`qSRAt` and `qRTN` return the *dequantized* value `x^FP4 · x^FP8 · x^FP32`,
since that is what the guarantees of the paper speak about.
-/

import Transformer.Quartet.Section3_Grids

namespace Transformer
namespace Quartet

variable {k : ℕ} {s : ℝ}

/-- The largest absolute value of a tensor laid out as `2^k` groups of `16`
entries: the `max_{i=1…d} |x_i|` of the scale displays of §3.1. -/
noncomputable def absMax (x : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  Finset.univ.sup' ⟨(⟨0, Nat.two_pow_pos k⟩, 0), Finset.mem_univ _⟩
    fun p : Fin (2 ^ k) × Fin 16 => |x p.1 p.2|

/-- The largest absolute value inside one group of `16` entries. -/
noncomputable def groupAbsMax (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun j : Fin 16 => |x i j|

/-- The per-tensor FP32 scale of `Q_SR`, `max|x| / (6 · 16/17 · 448)` (§3.1). -/
noncomputable def tensorScaleSR (x : Fin (2 ^ k) → Fin 16 → ℝ) : ℝ :=
  absMax x / (6 * (16 / 17) * 448)

/-- Its per-group E4M3 scale, `RTN_FP8(max_g |x| / (x^FP32 · 6 · 16/17))` (§3.1). -/
noncomputable def groupScaleSR (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  rtn fp8 (groupAbsMax x i / (tensorScaleSR x * 6 * (16 / 17)))

/-- One dequantized entry of `Q_SR`, with the rounding coin `t` of that entry
(§3.1): the E2M1 value is drawn by stochastic rounding, the two scales are
deterministic. -/
noncomputable def qSRAt (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) (t : ℝ) :
    ℝ :=
  sr fp4 (x i j / (groupScaleSR x i * tensorScaleSR x)) t * groupScaleSR x i * tensorScaleSR x

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

/-- **`Q_SR` is unbiased**, entry by entry: "`E_ω[x_i^FP4 × x^FP8_{i//16} ×
x^FP32] = x_i`" (§3.1).  This is the property every FP4 training method before
the paper pays for with element-wise stochastic rounding. -/
theorem integral_qSRAt {x : Fin (2 ^ k) → Fin 16 → ℝ} (hx : 0 < absMax x) (i : Fin (2 ^ k))
    (j : Fin 16) : ∫ t in (0 : ℝ)..1, qSRAt x i j t = x i j :=
  sorry

/-- **`Q_SR` never clips**: with the constants of §3.1 the argument handed to
`SR_FP4` stays in `[-6, 6]`, the range of E2M1 — "Given the choice of
constants, stochastic rounding `SR_FP4` does not clip its arguments".  The
factor `16/17`, "the maximum factor by which `RTN_FP8` can increase the
underlying values", is exactly what buys this. -/
theorem abs_div_groupScaleSR_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hx : 0 < absMax x)
    (i : Fin (2 ^ k)) (j : Fin 16) : |x i j / (groupScaleSR x i * tensorScaleSR x)| ≤ 6 :=
  sorry

/-- The hypothesis of the two statements above is satisfiable: a tensor of
ones has largest absolute value `1`. -/
example : 0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) := by
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

/-- **`Q_RTN(·, s)` clips nothing when `s ≤ 6 · 16/17`** (§3.3, "Setting the
clipping factor `s` to `6 × 16/17` or lower makes the scheme non-clipping");
above that value it does clip, which is what the paper's numerically optimal
`s = 6 · 16/17 / 0.93` trades error for. -/
theorem abs_div_groupScaleRTN_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hs : 0 < s)
    (hs' : s ≤ 6 * (16 / 17)) (hx : 0 < absMax x) (i : Fin (2 ^ k)) (j : Fin 16) :
    |x i j / (groupScaleRTN s x i * tensorScaleRTN s x)| ≤ 6 :=
  sorry

/-- **The group scales of `Q_RTN` stay below `256`**, whatever `s` is: the
headroom of §3.3 that lets `MS-EDEN` scale a group up without overflowing
E4M3, whose largest value is `448`. -/
theorem groupScaleRTN_le {x : Fin (2 ^ k) → Fin 16 → ℝ} (hs : 0 < s) (hx : 0 < absMax x)
    (i : Fin (2 ^ k)) : groupScaleRTN s x i ≤ 256 :=
  sorry

/-- The hypotheses of the two statements above are satisfiable at the paper's
own non-clipping bound `s = 6 · 16/17`. -/
example : 0 < 6 * (16 / 17 : ℝ) ∧ 6 * (16 / 17 : ℝ) ≤ 6 * (16 / 17) ∧
    0 < absMax (fun _ _ => (1 : ℝ) : Fin (2 ^ 0) → Fin 16 → ℝ) := by
  refine ⟨by norm_num, le_rfl, ?_⟩
  unfold absMax
  exact lt_of_lt_of_le (by norm_num)
    (Finset.le_sup' _ (Finset.mem_univ ((0 : Fin (2 ^ 0)), (0 : Fin 16))))

end Quartet
end Transformer
