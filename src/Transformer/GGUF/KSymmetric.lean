/-
# The symmetric K-quants: `Q3_K`, `Q6_K`

`ggml/src/ggml-common.h` (`block_q3_K`, `block_q6_K`, `QK_K = 256`) and
`dequantize_row_q3_K`, `dequantize_row_q6_K` in `ggml/src/ggml-quants.c`.

A super-block holds 256 weights in 16 sub-blocks of 16, one binary16 `d`, and a
signed integer scale `s` per sub-block.  Weight `i` is

  `y_i = d · s_{i/16} · q_i`,

with no min: the levels are symmetric about `0`, up to one extra negative one.

| format | bytes | `s` | `q` | bits/weight |
| --- | ---: | --- | --- | ---: |
| `Q3_K` | 110 | 6 bits, `-32 ≤ s < 32` | 3 bits, `-4 ≤ q < 4` | 3.4375 |
| `Q6_K` | 210 | `int8_t` | 6 bits, `-32 ≤ q < 32` | 6.5625 |

`Q3_K` packs its sixteen 6-bit scales into 12 bytes; `dequantize_row_q3_K`
unpacks them with the `kmask1`/`kmask2` shuffle of four `uint32_t`s, which
`scaleK3` writes out byte by byte.
-/

import Transformer.GGUF.Legacy

namespace Transformer
namespace GGUF

/-- The 6-bit scale `k < 16` of `Q3_K`, packed in the 12 bytes at offset `o`.
With `g = k / 4`, `b = k mod 4`: the low four bits are a nibble of byte
`4 (g mod 2) + b` (low for `g < 2`, high otherwise), the top two are bits
`2g, 2g + 1` of byte `8 + b`.  This is byte `k` of `aux` after the shuffle in
`dequantize_row_q3_K`. -/
def scaleK3 {n : ℕ} (x : Bytes n) (o k : ℕ) : ℕ :=
  (if k / 4 < 2 then lo (byte x (o + 4 * (k / 4 % 2) + k % 4))
    else hi (byte x (o + 4 * (k / 4 % 2) + k % 4))) +
    16 * (byte x (o + 8 + k % 4) / 4 ^ (k / 4) % 4)

theorem scaleK3_lt {n : ℕ} (x : Bytes n) (o k : ℕ) : scaleK3 x o k < 64 := by
  have h1 : (if k / 4 < 2 then lo (byte x (o + 4 * (k / 4 % 2) + k % 4))
      else hi (byte x (o + 4 * (k / 4 % 2) + k % 4))) < 16 := by
    split_ifs
    · exact lo_lt _
    · exact hi_lt (byte_lt _ _)
  unfold scaleK3; omega

/-- `Q3_K`: `hmask[32]`, `qs[64]`, `scales[12]`, `d`.  The low two bits of
weight `i` are at shift `2 ((i mod 128) / 32)` of byte `32 (i / 128) + i mod 32`
of `qs`; its high bit is bit `i / 32` of `hmask[i mod 32]`, and a clear high bit
subtracts `4`. -/
noncomputable def Q3_K (x : Bytes 110) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 108)).map fun d i =>
    d * (((scaleK3 x 96 (i.val / 16) : ℕ) : ℤ) - 32 : ℤ) *
      (((byte x (32 + 32 * (i.val / 128) + i.val % 32) / 4 ^ (i.val % 128 / 32) % 4 : ℕ) : ℤ) -
        (if byte x (i.val % 32) / 2 ^ (i.val / 32) % 2 = 1 then 0 else 4) : ℤ)

/-- The 6-bit field of weight `i` in `Q6_K`, before the offset `-32`.  With
`c = i / 128`, `k = (i mod 128) / 32`, `l = i mod 32`: a nibble of `ql` byte
`64c + l + 32 (k mod 2)` (low for `k < 2`) and bits `2k, 2k + 1` of `qh` byte
`32c + l`. -/
def fieldQ6 {n : ℕ} (x : Bytes n) (i : Fin 256) : ℕ :=
  (if i.val % 128 / 32 < 2 then lo (byte x (64 * (i.val / 128) + i.val % 32 + 32 * (i.val % 128 / 32 % 2)))
    else hi (byte x (64 * (i.val / 128) + i.val % 32 + 32 * (i.val % 128 / 32 % 2)))) +
    16 * (byte x (128 + 32 * (i.val / 128) + i.val % 32) / 4 ^ (i.val % 128 / 32) % 4)

theorem fieldQ6_lt {n : ℕ} (x : Bytes n) (i : Fin 256) : fieldQ6 x i < 64 := by
  have h1 : (if i.val % 128 / 32 < 2
      then lo (byte x (64 * (i.val / 128) + i.val % 32 + 32 * (i.val % 128 / 32 % 2)))
      else hi (byte x (64 * (i.val / 128) + i.val % 32 + 32 * (i.val % 128 / 32 % 2)))) < 16 := by
    split_ifs
    · exact lo_lt _
    · exact hi_lt (byte_lt _ _)
  unfold fieldQ6; omega

/-- `Q6_K`: `ql[128]`, `qh[64]`, `scales[16]` (`int8_t`), `d`. -/
noncomputable def Q6_K (x : Bytes 210) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 208)).map fun d i =>
    d * (int8 (byte x (192 + i.val / 16)) : ℝ) * (((fieldQ6 x i : ℕ) : ℤ) - 32 : ℤ)

/-- **`Q3_K` stores 8 levels per weight and 64 scales per sub-block of 16:**
`y_i = d s q`, `-4 ≤ q ≤ 3`, `-32 ≤ s ≤ 31`. -/
theorem Q3_K_levels {x : Bytes 110} {y : Fin 256 → ℝ} (h : Q3_K x = some y) :
    ∃ d, f16 (u16 x 108) = some d ∧ ∃ s : ℕ → ℤ, (∀ j, -32 ≤ s j ∧ s j ≤ 31) ∧
      ∀ i : Fin 256, ∃ q : ℤ, -4 ≤ q ∧ q ≤ 3 ∧ y i = d * s (i.val / 16) * q := by
  unfold Q3_K at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨d, hd, fun j => ((scaleK3 x 96 j : ℕ) : ℤ) - 32,
    fun j => by have := scaleK3_lt x 96 j; dsimp only; omega, fun i =>
      ⟨((byte x (32 + 32 * (i.val / 128) + i.val % 32) / 4 ^ (i.val % 128 / 32) % 4 : ℕ) : ℤ) -
        (if byte x (i.val % 32) / 2 ^ (i.val / 32) % 2 = 1 then 0 else 4), ?_, ?_, by push_cast; rfl⟩⟩ <;>
  · have : byte x (32 + 32 * (i.val / 128) + i.val % 32) / 4 ^ (i.val % 128 / 32) % 4 < 4 :=
      Nat.mod_lt _ (by norm_num)
    split_ifs <;> omega

/-- **`Q6_K` stores 64 levels per weight and 256 scales per sub-block of 16:**
`y_i = d s q`, `-32 ≤ q ≤ 31`, `-128 ≤ s ≤ 127`. -/
theorem Q6_K_levels {x : Bytes 210} {y : Fin 256 → ℝ} (h : Q6_K x = some y) :
    ∃ d, f16 (u16 x 208) = some d ∧ ∃ s : ℕ → ℤ, (∀ j, -128 ≤ s j ∧ s j ≤ 127) ∧
      ∀ i : Fin 256, ∃ q : ℤ, -32 ≤ q ∧ q ≤ 31 ∧ y i = d * s (i.val / 16) * q := by
  unfold Q6_K at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun j => int8 (byte x (192 + j)), fun j => int8_bounds (byte_lt _ _),
    fun i => ⟨_, by have := fieldQ6_lt x i; omega, by have := fieldQ6_lt x i; omega, rfl⟩⟩

/-- The hypotheses of the `_levels` theorems are satisfiable: the all-zero
super-blocks decode, their `d` being the binary16 `0`. -/
example : Q3_K (fun _ => 0) = some (fun _ => (0 : ℝ)) ∧ Q6_K (fun _ => 0) = some (fun _ => (0 : ℝ)) := by
  constructor <;> norm_num [Q3_K, Q6_K, f16, ieee, u16, byte]

end GGUF
end Transformer
