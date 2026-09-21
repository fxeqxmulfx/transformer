/-
# The 4-bit floats: `MXFP4`, `NVFP4`

`ggml/src/ggml-common.h` (`block_mxfp4`, `block_nvfp4`, `kvalues_mxfp4`),
`ggml_e8m0_to_fp32_half` and `ggml_ue4m3_to_fp32` in `ggml/src/ggml-impl.h`,
and `dequantize_row_mxfp4`, `dequantize_row_nvfp4`, `quantize_row_mxfp4_ref`,
`best_index_mxfp4` in `ggml/src/ggml-quants.c`.

Every weight is a 4-bit float E2M1 (sign, two exponent bits, one mantissa bit)
times a block scale.  ggml stores E2M1 doubled, as the integers
`kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, -0, -1, …, -12}`, and halves the scale
to compensate.

| format | bytes | weights | scale | bits/weight |
| --- | ---: | ---: | --- | ---: |
| `MXFP4` | 17 | 32 | E8M0 `e`: `2^{e-127}`, one per 32 | 4.25 |
| `NVFP4` | 36 | 64 | UE4M3, one per 16 | 4.5 |

Unlike the codebooks of `IQ4_NL`, `0` is a level (twice: `+0` and `-0`), so
the formats have a dead zone.  The E8M0 scale is a power of two, so the
quantizer can only pick `d = 2^{⌊log₂ amax⌋ - 3}`, and every weight below
`amax/32` of its block is stored as `0` (`mxfp4_dead_zone`).
-/

import Transformer.GGUF.Legacy
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Int.Log

namespace Transformer
namespace GGUF

/-- `kvalues_mxfp4`: twice the E2M1 values. -/
def kFP4 : Fin 16 → ℤ := ![0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

/-- The levels lie in `[-12, 12]`. -/
theorem kFP4_bounds (k : Fin 16) : -12 ≤ kFP4 k ∧ kFP4 k ≤ 12 := by
  revert k; decide

/-- **`0` is a level**, and every other level is at least `1` in absolute value. -/
theorem kFP4_zero_or (k : Fin 16) : kFP4 k = 0 ∨ 1 ≤ |kFP4 k| := by
  revert k; decide

/-- `ggml_e8m0_to_fp32_half`: `2^{e-128}` for every byte, half the E8M0 value.
The OCP NaN `0xFF` is not special: it decodes to `2^{127}`. -/
noncomputable def e8m0Half (e : ℕ) : ℝ := (2 : ℝ) ^ ((e : ℤ) - 128)

theorem e8m0Half_pos (e : ℕ) : 0 < e8m0Half e := zpow_pos (by norm_num) _

/-- `ggml_ue4m3_to_fp32`: exponent bits `3..6` with bias `7`, mantissa bits
`0..2`, times `1/2`.  Bit `7` is ignored, and `0x7F` alone is sent to `0`. -/
noncomputable def ue4m3 (b : ℕ) : ℝ :=
  if b = 0 ∨ b = 0x7F then 0
  else if b / 8 % 16 = 0 then (b % 8 : ℕ) * (2 : ℝ) ^ (-9 : ℤ) / 2
  else (1 + (b % 8 : ℕ) / 8) * (2 : ℝ) ^ (((b / 8 % 16 : ℕ) : ℤ) - 7) / 2

/-- **The UE4M3 quirks of ggml.**  The largest finite E4M3 number, `0x7E`, is
`448/2`; `0x7F`, E4M3's NaN, is `0`; but `0xFF`, the same pattern with bit `7`
set, is `240`, above every scale the quantizer emits, and `0x80` is `0`. -/
theorem ue4m3_quirks : ue4m3 0x7E = 224 ∧ ue4m3 0x7F = 0 ∧ ue4m3 0xFF = 240 ∧ ue4m3 0x80 = 0 := by
  norm_num [ue4m3]

theorem ue4m3_nonneg (b : ℕ) : 0 ≤ ue4m3 b := by
  unfold ue4m3; split_ifs <;> positivity

/-- `MXFP4`: `e`, `qs[16]`; the nibbles are laid out as in `Q4_0`. -/
noncomputable def MXFP4 (x : Bytes 17) : Fin 32 → ℝ :=
  fun i => e8m0Half (byte x 0) * kFP4 ⟨nib x 1 i, nib_lt x 1 i⟩

/-- The 4-bit field of weight `i` of `NVFP4`: sub-block `s = i / 16` owns the
8 bytes from `4 + 8s`, low nibbles first. -/
def nibNV {n : ℕ} (x : Bytes n) (i : Fin 64) : ℕ :=
  byte x (4 + 8 * (i.val / 16) + i.val % 8) / 16 ^ (i.val % 16 / 8) % 16

theorem nibNV_lt {n : ℕ} (x : Bytes n) (i : Fin 64) : nibNV x i < 16 := by
  unfold nibNV; omega

/-- `NVFP4`: `d[4]` (UE4M3), `qs[32]`. -/
noncomputable def NVFP4 (x : Bytes 36) : Fin 64 → ℝ :=
  fun i => ue4m3 (byte x (i.val / 16)) * kFP4 ⟨nibNV x i, nibNV_lt x i⟩

/-- **`MXFP4` stores a doubled E2M1 level per weight and a power of two per
block of 32:** `y_i = 2^{e-128} · K[q_i]`. -/
theorem MXFP4_levels (x : Bytes 17) :
    ∃ e : ℕ, e < 256 ∧ ∀ i, ∃ k : Fin 16, MXFP4 x i = (2 : ℝ) ^ ((e : ℤ) - 128) * kFP4 k :=
  ⟨byte x 0, byte_lt x 0, fun _ => ⟨_, rfl⟩⟩

/-- **`NVFP4` stores a doubled E2M1 level per weight and a UE4M3 scale per
sub-block of 16:** `y_i = d_{i/16} · K[q_i]`. -/
theorem NVFP4_levels (x : Bytes 36) :
    ∃ d : ℕ → ℝ, (∀ s, 0 ≤ d s ∧ d s ≤ 240) ∧ ∀ i : Fin 64, ∃ k : Fin 16, NVFP4 x i = d (i.val / 16) * kFP4 k := by
  refine ⟨fun s => ue4m3 (byte x s), fun s => ⟨ue4m3_nonneg _, ?_⟩, fun _ => ⟨_, rfl⟩⟩
  show ue4m3 (byte x s) ≤ 240
  have hb := byte_lt x s
  generalize byte x s = b at hb
  unfold ue4m3
  have hm : ((b % 8 : ℕ) : ℝ) ≤ 7 := by exact_mod_cast (by omega : b % 8 ≤ 7)
  have he : b / 8 % 16 ≤ 15 := by omega
  split_ifs with h0 h1
  · norm_num
  · have : (2 : ℝ) ^ (-9 : ℤ) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) (-9 : ℤ)]
  · have h8 : (2 : ℝ) ^ (((b / 8 % 16 : ℕ) : ℤ) - 7) ≤ 2 ^ (8 : ℤ) :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    have hp : (0 : ℝ) ≤ 2 ^ (((b / 8 % 16 : ℕ) : ℤ) - 7) := by positivity
    rw [show (2 : ℝ) ^ (8 : ℤ) = 256 by norm_num] at h8
    have hq : 1 + ((b % 8 : ℕ) : ℝ) / 8 ≤ 15 / 8 := by linarith
    nlinarith [mul_le_mul hq h8 hp (by norm_num)]

/-- `best_index_mxfp4`: the first index of least error `|K[i] d - x|`, scanning
`i = 0, …, 15` and moving only on a strict improvement. -/
noncomputable def bestFP4 (x d : ℝ) : Fin 16 :=
  (List.finRange 16).foldl (fun b i => if |kFP4 i * d - x| < |kFP4 b * d - x| then i else b) 0

theorem foldl_best (f : Fin 16 → ℝ) (l : List (Fin 16)) (b : Fin 16) :
    f (l.foldl (fun b i => if f i < f b then i else b) b) ≤ f b ∧
      ∀ i ∈ l, f (l.foldl (fun b i => if f i < f b then i else b) b) ≤ f i := by
  induction l generalizing b with
  | nil => simp
  | cons j l ih =>
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp]
    obtain ⟨h1, h2⟩ := ih (if f j < f b then j else b)
    by_cases h : f j < f b
    · simp only [h, ↓reduceIte] at h1 h2 ⊢; exact ⟨by linarith, h1, h2⟩
    · simp only [h, ↓reduceIte] at h1 h2 ⊢; exact ⟨h1, by linarith, h2⟩

/-- `best_index_mxfp4` finds a nearest level. -/
theorem bestFP4_le (x d : ℝ) (k : Fin 16) :
    |kFP4 (bestFP4 x d) * d - x| ≤ |kFP4 k * d - x| :=
  (foldl_best (fun i => |kFP4 i * d - x|) _ 0).2 k (List.mem_finRange k)

/-- **Below half the scale, a weight is stored as `0`.** -/
theorem bestFP4_eq_zero {x d : ℝ} (hd : 0 < d) (hx : |x| < d / 2) : kFP4 (bestFP4 x d) = 0 := by
  rcases kFP4_zero_or (bestFP4 x d) with h | h
  · exact h
  exfalso
  have h0 := bestFP4_le x d 0
  have hk : (1 : ℝ) ≤ |(kFP4 (bestFP4 x d) : ℝ)| := by exact_mod_cast h
  simp only [show kFP4 0 = 0 from rfl, Int.cast_zero, zero_mul, zero_sub, abs_neg] at h0
  have := abs_sub_abs_le_abs_sub (kFP4 (bestFP4 x d) * d) x
  rw [abs_mul, abs_of_pos hd] at this
  nlinarith

/-- The scale `quantize_row_mxfp4_ref` picks for a block of largest absolute
value `amax > 0`: `d = 2^{e - 128}` with `e = ⌊log₂ amax⌋ - 2 + 127`.  The C code
computes `⌊log₂ amax⌋` as `floorf(log2f(amax))`; here it is exact. -/
noncomputable def mxfp4Scale (amax : ℝ) : ℝ := (2 : ℝ) ^ (Int.log 2 amax - 3)

/-- The smallest nonzero level `d` exceeds `amax / 16`. -/
theorem amax_lt_mxfp4Scale {amax : ℝ} : amax < 16 * mxfp4Scale amax := by
  have := Int.lt_zpow_succ_log_self (R := ℝ) (by norm_num : 1 < 2) amax
  unfold mxfp4Scale
  have h : (2 : ℝ) ^ (Int.log 2 amax + 1) = 16 * 2 ^ (Int.log 2 amax - 3) := by
    rw [show Int.log 2 amax + 1 = 4 + (Int.log 2 amax - 3) by ring,
      zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  push_cast at this
  linarith

/-- **The dead zone of `MXFP4`.**  In a block of largest absolute value `amax`,
`quantize_row_mxfp4_ref` stores every weight with `|x| ≤ amax/32` as `0`: the
E8M0 scale is a power of two, the smallest nonzero level is above `amax/16`,
and nearest rounding sends everything under half of it to `0`. -/
theorem mxfp4_dead_zone {amax x : ℝ} (hx : |x| ≤ amax / 32) :
    kFP4 (bestFP4 x (mxfp4Scale amax)) = 0 :=
  bestFP4_eq_zero (zpow_pos (by norm_num) _) (by linarith [amax_lt_mxfp4Scale (amax := amax)])

/-- The hypothesis of `mxfp4_dead_zone` is satisfiable: `amax = 1`, `x = 1/32`. -/
example : |(1 / 32 : ℝ)| ≤ 1 / 32 := by norm_num

end GGUF
end Transformer
