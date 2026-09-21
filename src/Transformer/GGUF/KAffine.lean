/-
# The affine K-quants: `Q2_K`, `Q4_K`, `Q5_K`

`ggml/src/ggml-common.h` (`block_q2_K`, `block_q4_K`, `block_q5_K`, `QK_K = 256`)
and `dequantize_row_q2_K`, `dequantize_row_q4_K`, `dequantize_row_q5_K`,
`get_scale_min_k4` in `ggml/src/ggml-quants.c`.

A super-block holds 256 weights, two binary16 numbers `d` and `dmin`, and for
each sub-block an integer scale `s` and an integer min `m`.  Weight `i` of
sub-block `j` is

  `y_i = d · s_j · q_i - dmin · m_j`,

a two-level quantization: the per-sub-block scale and min are themselves
quantized against the super-block's `d` and `dmin`.

| format | bytes | sub-blocks | `s`, `m` | `q` | bits/weight |
| --- | ---: | --- | --- | --- | ---: |
| `Q2_K` | 84 | 16 × 16 | 4 bits each | 2 bits | 2.625 |
| `Q4_K` | 144 | 8 × 32 | 6 bits each | 4 bits | 4.5 |
| `Q5_K` | 176 | 8 × 32 | 6 bits each | 5 bits | 5.5 |

The loops of the C code visit weight `i` with sub-block `i / 16` (`Q2_K`) or
`i / 32` (`Q4_K`, `Q5_K`); the byte offsets below are those loops' pointers
written as functions of `i`.
-/

import Transformer.GGUF.Basic

namespace Transformer
namespace GGUF

/-- `get_scale_min_k4`, the scale: the 6-bit scale of sub-block `j < 8`, packed
in the 12 bytes at offset `o`. -/
def scaleK4 {n : ℕ} (x : Bytes n) (o j : ℕ) : ℕ :=
  if j < 4 then byte x (o + j) % 64 else lo (byte x (o + j + 4)) + 16 * (byte x (o + j - 4) / 64)

/-- `get_scale_min_k4`, the min: the 6-bit min of sub-block `j < 8`. -/
def minK4 {n : ℕ} (x : Bytes n) (o j : ℕ) : ℕ :=
  if j < 4 then byte x (o + j + 4) % 64 else hi (byte x (o + j + 4)) + 16 * (byte x (o + j) / 64)

theorem scaleK4_lt {n : ℕ} (x : Bytes n) (o j : ℕ) : scaleK4 x o j < 64 := by
  have := byte_lt x (o + j - 4)
  unfold scaleK4 lo; split_ifs <;> omega

theorem minK4_lt {n : ℕ} (x : Bytes n) (o j : ℕ) : minK4 x o j < 64 := by
  have := byte_lt x (o + j + 4); have := byte_lt x (o + j)
  unfold minK4 hi; split_ifs <;> omega

/-- The 4-bit field of weight `i` in the `qs` array at offset `o`: weights
`64c + l` and `64c + 32 + l` share byte `32c + l`, low and high nibble. -/
def nibK {n : ℕ} (x : Bytes n) (o : ℕ) (i : Fin 256) : ℕ :=
  if i.val % 64 < 32 then lo (byte x (o + 32 * (i.val / 64) + i.val % 32))
  else hi (byte x (o + 32 * (i.val / 64) + i.val % 32))

theorem nibK_lt {n : ℕ} (x : Bytes n) (o : ℕ) (i : Fin 256) : nibK x o i < 16 := by
  unfold nibK; split_ifs
  · exact lo_lt _
  · exact hi_lt (byte_lt _ _)

/-- `Q2_K`: `scales[16]`, `qs[64]`, `d`, `dmin`.  Scale byte `i / 16` holds the
scale in its low and the min in its high nibble; the 2-bit field of weight `i`
is at shift `2 ((i mod 128) / 32)` of byte `32 (i / 128) + i mod 32`. -/
noncomputable def Q2_K (x : Bytes 84) : Option (Fin 256 → ℝ) :=
  match f16 (u16 x 80), f16 (u16 x 82) with
  | some d, some dmin => some fun i =>
      d * lo (byte x (i.val / 16)) *
          (byte x (16 + 32 * (i.val / 128) + i.val % 32) / 4 ^ (i.val % 128 / 32) % 4 : ℕ) -
        dmin * hi (byte x (i.val / 16))
  | _, _ => none

/-- `Q4_K`: `d`, `dmin`, `scales[12]`, `qs[128]`. -/
noncomputable def Q4_K (x : Bytes 144) : Option (Fin 256 → ℝ) :=
  match f16 (u16 x 0), f16 (u16 x 2) with
  | some d, some dmin => some fun i =>
      d * scaleK4 x 4 (i.val / 32) * nibK x 16 i - dmin * minK4 x 4 (i.val / 32)
  | _, _ => none

/-- `Q5_K`: `d`, `dmin`, `scales[12]`, `qh[32]`, `qs[128]`.  The fifth bit of
weight `i` is bit `i / 32` of `qh[i mod 32]` (`u1`, `u2` in the C code). -/
noncomputable def Q5_K (x : Bytes 176) : Option (Fin 256 → ℝ) :=
  match f16 (u16 x 0), f16 (u16 x 2) with
  | some d, some dmin => some fun i =>
      d * scaleK4 x 4 (i.val / 32) *
          (nibK x 48 i + 16 * (byte x (16 + i.val % 32) / 2 ^ (i.val / 32) % 2) : ℕ) -
        dmin * minK4 x 4 (i.val / 32)
  | _, _ => none

/-- **`Q2_K` stores 4 levels per weight, and 16 scales and 16 mins per
sub-block of 16:** `y_i = d s q - dmin m`, `q < 4`, `s, m < 16`. -/
theorem Q2_K_levels {x : Bytes 84} {y : Fin 256 → ℝ} (h : Q2_K x = some y) :
    ∃ d dmin, f16 (u16 x 80) = some d ∧ f16 (u16 x 82) = some dmin ∧
      ∃ s m : ℕ → ℕ, (∀ j, s j < 16 ∧ m j < 16) ∧
        ∀ i : Fin 256, ∃ q : ℕ, q < 4 ∧ y i = d * s (i.val / 16) * q - dmin * m (i.val / 16) := by
  unfold Q2_K at h
  split at h
  · rename_i d dmin hd hm
    obtain rfl := Option.some.inj h
    refine ⟨d, dmin, hd, hm, fun j => lo (byte x j), fun j => hi (byte x j),
      fun j => ⟨lo_lt _, hi_lt (byte_lt _ _)⟩, fun i => ⟨_, Nat.mod_lt _ (by norm_num), rfl⟩⟩
  · simp at h

/-- **`Q4_K` stores 16 levels per weight, and 64 scales and 64 mins per
sub-block of 32:** `y_i = d s q - dmin m`, `q < 16`, `s, m < 64`. -/
theorem Q4_K_levels {x : Bytes 144} {y : Fin 256 → ℝ} (h : Q4_K x = some y) :
    ∃ d dmin, f16 (u16 x 0) = some d ∧ f16 (u16 x 2) = some dmin ∧
      ∃ s m : ℕ → ℕ, (∀ j, s j < 64 ∧ m j < 64) ∧
        ∀ i : Fin 256, ∃ q : ℕ, q < 16 ∧ y i = d * s (i.val / 32) * q - dmin * m (i.val / 32) := by
  unfold Q4_K at h
  split at h
  · rename_i d dmin hd hm
    obtain rfl := Option.some.inj h
    exact ⟨d, dmin, hd, hm, scaleK4 x 4, minK4 x 4, fun j => ⟨scaleK4_lt x 4 j, minK4_lt x 4 j⟩,
      fun i => ⟨_, nibK_lt x 16 i, rfl⟩⟩
  · simp at h

/-- **`Q5_K` stores 32 levels per weight, and 64 scales and 64 mins per
sub-block of 32:** `y_i = d s q - dmin m`, `q < 32`, `s, m < 64`. -/
theorem Q5_K_levels {x : Bytes 176} {y : Fin 256 → ℝ} (h : Q5_K x = some y) :
    ∃ d dmin, f16 (u16 x 0) = some d ∧ f16 (u16 x 2) = some dmin ∧
      ∃ s m : ℕ → ℕ, (∀ j, s j < 64 ∧ m j < 64) ∧
        ∀ i : Fin 256, ∃ q : ℕ, q < 32 ∧ y i = d * s (i.val / 32) * q - dmin * m (i.val / 32) := by
  unfold Q5_K at h
  split at h
  · rename_i d dmin hd hm
    obtain rfl := Option.some.inj h
    refine ⟨d, dmin, hd, hm, scaleK4 x 4, minK4 x 4, fun j => ⟨scaleK4_lt x 4 j, minK4_lt x 4 j⟩,
      fun i => ⟨_, ?_, rfl⟩⟩
    have := nibK_lt x 48 i
    omega
  · simp at h

/-- The hypotheses of the `_levels` theorems are satisfiable: the all-zero
super-blocks decode, their `d` and `dmin` being the binary16 `0`. -/
example : Q2_K (fun _ => 0) = some (fun _ => (0 : ℝ)) ∧ Q4_K (fun _ => 0) = some (fun _ => (0 : ℝ)) ∧
    Q5_K (fun _ => 0) = some (fun _ => (0 : ℝ)) := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [Q2_K, Q4_K, Q5_K, f16, ieee, u16, byte]

end GGUF
end Transformer
