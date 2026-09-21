/-
# GGUF tensor storage: bytes, bit fields, and IEEE floats

The tensor types of the GGUF format, read from their reference implementation
in ggml (`ggml/src/ggml-common.h` for the block layouts, `ggml/src/ggml-quants.c`
for `dequantize_row_*`), llama.cpp master at commit `335b21f`.

A block is a fixed number of bytes (`Bytes n`); a format is a decoding of those
bytes to real numbers.  The bit operations of the C code are written with
their arithmetic equivalents on `ℕ`: `b & 0xF` is `b % 16`, `b >> 4` is
`b / 16`, `b & 63` is `b % 64`, and bit `k` of a little-endian field is
`b / 2^k % 2`.

The scales of every block are IEEE binary floats (`ggml_half` is binary16), and
`F16`, `BF16`, `F32` are tensor types of their own.  `ieee E M` decodes a bit
pattern with `E` exponent and `M` mantissa bits exactly as `GGML_FP16_TO_FP32`
and its siblings do, except that the infinities and NaNs have no value in `ℝ`
and decode to `none`.
-/

import Mathlib.Algebra.Order.Round
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Field.Power

namespace Transformer
namespace GGUF

/-- `n` bytes, each a number in `[0, 256)`. -/
abbrev Bytes (n : ℕ) : Type := Fin n → Fin 256

/-- The low nibble `b & 0xF`. -/
def lo (b : ℕ) : ℕ := b % 16

/-- The high nibble `b >> 4`. -/
def hi (b : ℕ) : ℕ := b / 16

/-- The byte at offset `o`, and `0` past the end. -/
def byte {n : ℕ} (x : Bytes n) (o : ℕ) : ℕ := if h : o < n then (x ⟨o, h⟩).val else 0

theorem byte_lt {n : ℕ} (x : Bytes n) (o : ℕ) : byte x o < 256 := by
  unfold byte; split_ifs <;> omega

/-- Bit `k` of the little-endian bit string starting at offset `o`: bit `k % 8`
of byte `o + k / 8`. -/
def bitAt {n : ℕ} (x : Bytes n) (o k : ℕ) : ℕ := byte x (o + k / 8) / 2 ^ (k % 8) % 2

/-- The little-endian `uint16_t` at offset `o`. -/
def u16 {n : ℕ} (x : Bytes n) (o : ℕ) : ℕ := byte x o + 256 * byte x (o + 1)

/-- The little-endian `uint32_t` at offset `o`. -/
def u32 {n : ℕ} (x : Bytes n) (o : ℕ) : ℕ := u16 x o + 65536 * u16 x (o + 2)

theorem lo_lt (b : ℕ) : lo b < 16 := Nat.mod_lt _ (by norm_num)

theorem hi_lt {b : ℕ} (hb : b < 256) : hi b < 16 := by
  unfold hi; omega

theorem bitAt_le {n : ℕ} (x : Bytes n) (o k : ℕ) : bitAt x o k ≤ 1 := by
  unfold bitAt; omega

/-- The exponent bias `2^{E-1} - 1` of an IEEE binary format. -/
def bias (E : ℕ) : ℤ := 2 ^ (E - 1) - 1

/-- **IEEE binary decoding.**  A pattern of `1 + E + M` bits: sign, biased
exponent `e`, mantissa `m`.  `e = 0` is subnormal, `m · 2^{1 - bias - M}`;
`0 < e < 2^E - 1` is normal, `(2^M + m) · 2^{e - bias - M}`; `e = 2^E - 1` is an
infinity or a NaN and has no real value (IEEE 754-2008, §3.4). -/
noncomputable def ieee (E M bits : ℕ) : Option ℝ :=
  let m := bits % 2 ^ M
  let e := bits / 2 ^ M % 2 ^ E
  let sgn : ℝ := if bits / 2 ^ (E + M) % 2 = 1 then -1 else 1
  if e = 2 ^ E - 1 then none
  else if e = 0 then some (sgn * m * (2 : ℝ) ^ (1 - bias E - M))
  else some (sgn * (2 ^ M + m) * (2 : ℝ) ^ ((e : ℤ) - bias E - M))

/-- `ggml_half`, IEEE binary16: `GGML_FP16_TO_FP32`. -/
noncomputable def f16 (bits : ℕ) : Option ℝ := ieee 5 10 bits

/-- `ggml_bf16_t`, bfloat16: the upper half of a binary32, `GGML_BF16_TO_FP32`. -/
noncomputable def bf16 (bits : ℕ) : Option ℝ := ieee 8 7 bits

/-- IEEE binary32, `float`. -/
noncomputable def f32 (bits : ℕ) : Option ℝ := ieee 8 23 bits

/-- The smallest positive number of the format: the subnormal `2^{1 - bias - M}`. -/
noncomputable def minSub (E M : ℕ) : ℝ := (2 : ℝ) ^ (1 - bias E - M)

theorem minSub_pos (E M : ℕ) : 0 < minSub E M := zpow_pos (by norm_num) _

/-- **No number of the format is closer to `0` than the smallest subnormal.** -/
theorem minSub_le_abs {E M bits : ℕ} {v : ℝ} (h : ieee E M bits = some v) (hv : v ≠ 0) :
    minSub E M ≤ |v| := by
  unfold ieee at h
  simp only at h
  have hsgn : |(if bits / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1)| = 1 := by
    split_ifs <;> simp
  generalize (if bits / 2 ^ (E + M) % 2 = 1 then (-1 : ℝ) else 1) = sgn at h hsgn
  split_ifs at h with h1 h2
  · obtain rfl := Option.some.inj h
    have hm : bits % 2 ^ M ≠ 0 := by rintro hm; apply hv; simp [hm]
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _), minSub]
    have : (1 : ℝ) ≤ (bits % 2 ^ M : ℕ) := by exact_mod_cast Nat.one_le_iff_ne_zero.2 hm
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) (1 - bias E - M)]
  · obtain rfl := Option.some.inj h
    rw [abs_mul, abs_mul, hsgn, one_mul, abs_of_nonneg (by positivity),
      abs_of_pos (zpow_pos (by norm_num) _), minSub]
    have he : (1 : ℤ) ≤ (bits / 2 ^ M % 2 ^ E : ℕ) := by exact_mod_cast Nat.one_le_iff_ne_zero.2 h2
    have hz : (2 : ℝ) ^ (1 - bias E - M) ≤ 2 ^ (((bits / 2 ^ M % 2 ^ E : ℕ) : ℤ) - bias E - M) :=
      zpow_le_zpow_right₀ (by norm_num) (by linarith)
    have hM : (1 : ℝ) ≤ 2 ^ M + (bits % 2 ^ M : ℕ) := by
      have := one_le_pow₀ (n := M) (by norm_num : (1 : ℝ) ≤ 2)
      have : (0 : ℝ) ≤ (bits % 2 ^ M : ℕ) := by positivity
      linarith
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) (1 - bias E - M)]

/-- The hypotheses of `minSub_le_abs` are satisfiable: the binary16 pattern
`0x3C00` is `1`. -/
example : f16 0x3C00 = some 1 := by
  norm_num [f16, ieee, bias]

/-- The smallest positive binary16 number is `2^{-24}`. -/
theorem minSub_f16 : minSub 5 10 = 2 ^ (-24 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive bfloat16 number is `2^{-133}`. -/
theorem minSub_bf16 : minSub 8 7 = 2 ^ (-133 : ℤ) := by norm_num [minSub, bias]

/-- The smallest positive binary32 number is `2^{-149}`. -/
theorem minSub_f32 : minSub 8 23 = 2 ^ (-149 : ℤ) := by norm_num [minSub, bias]

/-- The pattern `0x0001` is the smallest positive binary16 number. -/
theorem f16_one : f16 1 = some (minSub 5 10) := by
  norm_num [f16, ieee, minSub, bias]

/-- The largest finite binary16 number, `0x7BFF`, is `65504`. -/
theorem f16_max : f16 0x7BFF = some 65504 := by
  norm_num [f16, ieee, bias]

end GGUF
end Transformer
