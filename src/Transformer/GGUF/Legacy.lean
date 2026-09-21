/-
# The 32-weight block formats: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`

`ggml/src/ggml-common.h` (`block_q4_0` … `block_q8_0`, `QK = 32`) and
`dequantize_row_q4_0` … `dequantize_row_q8_0` in `ggml/src/ggml-quants.c`.

Each block holds 32 weights, a binary16 scale `d`, sometimes a binary16 offset
`m`, and one small integer per weight:

| format | bytes | layout | weight | bits/weight |
| --- | ---: | --- | --- | ---: |
| `Q4_0` | 18 | `d`, 16 × nibble pair | `(q - 8) d`, `q < 16` | 4.5 |
| `Q4_1` | 20 | `d`, `m`, 16 × nibble pair | `q d + m`, `q < 16` | 5 |
| `Q5_0` | 22 | `d`, 32 high bits, 16 × nibble pair | `(q - 16) d`, `q < 32` | 5.5 |
| `Q5_1` | 24 | `d`, `m`, 32 high bits, 16 × nibble pair | `q d + m`, `q < 32` | 6 |
| `Q8_0` | 34 | `d`, 32 × `int8` | `q d`, `-128 ≤ q < 128` | 8.5 |

Weight `j < 16` takes the low nibble of byte `j` of the nibble array, weight
`j + 16` its high nibble; in the 5-bit formats weight `i` takes bit `i` of the
little-endian `uint32_t qh` as its fifth bit (`xh_0`, `xh_1` in the C code).
A block whose scale is an infinity or a NaN decodes to `none`.
-/

import Transformer.GGUF.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Tactic.FieldSimp

namespace Transformer
namespace GGUF

/-- The 4-bit field of weight `i` in a nibble array at offset `o`: the low
nibble of byte `i` for `i < 16`, the high nibble of byte `i - 16` otherwise. -/
def nib {n : ℕ} (x : Bytes n) (o : ℕ) (i : Fin 32) : ℕ :=
  if i.val < 16 then lo (byte x (o + i)) else hi (byte x (o + i - 16))

theorem nib_lt {n : ℕ} (x : Bytes n) (o : ℕ) (i : Fin 32) : nib x o i < 16 := by
  unfold nib; split_ifs
  · exact lo_lt _
  · exact hi_lt (byte_lt _ _)

/-- `int8_t` read from a byte, two's complement. -/
def int8 (b : ℕ) : ℤ := if b < 128 then b else (b : ℤ) - 256

theorem int8_bounds {b : ℕ} (hb : b < 256) : -128 ≤ int8 b ∧ int8 b ≤ 127 := by
  unfold int8; split_ifs <;> omega

/-- `Q4_0`: `y_i = (q_i - 8) d`. -/
noncomputable def Q4_0 (x : Bytes 18) : Option (Fin 32 → ℝ) :=
  (f16 (u16 x 0)).map fun d i => ((nib x 2 i : ℤ) - 8 : ℤ) * d

/-- `Q4_1`: `y_i = q_i d + m`. -/
noncomputable def Q4_1 (x : Bytes 20) : Option (Fin 32 → ℝ) :=
  match f16 (u16 x 0), f16 (u16 x 2) with
  | some d, some m => some fun i => (nib x 4 i : ℝ) * d + m
  | _, _ => none

/-- `Q5_0`: `y_i = (q_i - 16) d`, `q_i` a nibble and a fifth bit. -/
noncomputable def Q5_0 (x : Bytes 22) : Option (Fin 32 → ℝ) :=
  (f16 (u16 x 0)).map fun d i => ((nib x 6 i + 16 * bitAt x 2 i : ℕ) - 16 : ℤ) * d

/-- `Q5_1`: `y_i = q_i d + m`, `q_i` a nibble and a fifth bit. -/
noncomputable def Q5_1 (x : Bytes 24) : Option (Fin 32 → ℝ) :=
  match f16 (u16 x 0), f16 (u16 x 2) with
  | some d, some m => some fun i => ((nib x 8 i + 16 * bitAt x 4 i : ℕ) : ℝ) * d + m
  | _, _ => none

/-- `Q8_0`: `y_i = q_i d`, `q_i` an `int8_t`. -/
noncomputable def Q8_0 (x : Bytes 34) : Option (Fin 32 → ℝ) :=
  (f16 (u16 x 0)).map fun d i => (int8 (byte x (2 + i)) : ℝ) * d

/-- **`Q4_0` stores 16 levels:** `(q - 8) d`, `-8 ≤ q - 8 ≤ 7`. -/
theorem Q4_0_levels {x : Bytes 18} {y : Fin 32 → ℝ} (h : Q4_0 x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∀ i, ∃ q : ℤ, -8 ≤ q ∧ q ≤ 7 ∧ y i = q * d := by
  unfold Q4_0 at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun i => ⟨_, by have := nib_lt x 2 i; omega, by have := nib_lt x 2 i; omega, rfl⟩⟩

/-- **`Q4_1` stores 16 levels above an offset:** `q d + m`, `0 ≤ q ≤ 15`. -/
theorem Q4_1_levels {x : Bytes 20} {y : Fin 32 → ℝ} (h : Q4_1 x = some y) :
    ∃ d m, f16 (u16 x 0) = some d ∧ f16 (u16 x 2) = some m ∧
      ∀ i, ∃ q : ℕ, q ≤ 15 ∧ y i = q * d + m := by
  unfold Q4_1 at h
  split at h
  · rename_i d m hd hm
    obtain rfl := Option.some.inj h
    exact ⟨d, m, hd, hm, fun i => ⟨_, by have := nib_lt x 4 i; omega, rfl⟩⟩
  · simp at h

/-- **`Q5_0` stores 32 levels:** `(q - 16) d`, `-16 ≤ q - 16 ≤ 15`. -/
theorem Q5_0_levels {x : Bytes 22} {y : Fin 32 → ℝ} (h : Q5_0 x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∀ i, ∃ q : ℤ, -16 ≤ q ∧ q ≤ 15 ∧ y i = q * d := by
  unfold Q5_0 at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨d, hd, fun i => ⟨_, ?_, ?_, rfl⟩⟩ <;>
    · have := nib_lt x 6 i; have := bitAt_le x 2 i; push_cast; omega

/-- **`Q5_1` stores 32 levels above an offset:** `q d + m`, `0 ≤ q ≤ 31`. -/
theorem Q5_1_levels {x : Bytes 24} {y : Fin 32 → ℝ} (h : Q5_1 x = some y) :
    ∃ d m, f16 (u16 x 0) = some d ∧ f16 (u16 x 2) = some m ∧
      ∀ i, ∃ q : ℕ, q ≤ 31 ∧ y i = q * d + m := by
  unfold Q5_1 at h
  split at h
  · rename_i d m hd hm
    obtain rfl := Option.some.inj h
    exact ⟨d, m, hd, hm, fun i =>
      ⟨_, by have := nib_lt x 8 i; have := bitAt_le x 4 i; omega, rfl⟩⟩
  · simp at h

/-- **`Q8_0` stores 256 levels:** `q d`, `-128 ≤ q ≤ 127`. -/
theorem Q8_0_levels {x : Bytes 34} {y : Fin 32 → ℝ} (h : Q8_0 x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∀ i, ∃ q : ℤ, -128 ≤ q ∧ q ≤ 127 ∧ y i = q * d := by
  unfold Q8_0 at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun i => ⟨_, (int8_bounds (byte_lt x _)).1, (int8_bounds (byte_lt x _)).2, rfl⟩⟩

/-- The hypotheses of the `_levels` theorems are satisfiable: the all-zero
blocks of `Q4_0` and `Q8_0` decode, their scale being the binary16 `0`. -/
example : Q4_0 (fun _ => 0) = some (fun _ => (0 : ℝ)) ∧ Q8_0 (fun _ => 0) = some (fun _ => (0 : ℝ)) := by
  constructor <;> norm_num [Q4_0, Q8_0, f16, ieee, u16, byte, nib, int8, lo, hi]

/-- The `Q8_0` block with binary16 scale pattern `h` and integers `q`, laid out
as `block_q8_0`: `d` little-endian, then `qs[32]` in two's complement. -/
def encodeQ8_0 (h : ℕ) (q : Fin 32 → ℤ) : Bytes 34 := fun j =>
  if j.val < 2 then ⟨h / 256 ^ j.val % 256, Nat.mod_lt _ (by norm_num)⟩
  else ⟨(q ⟨j.val - 2, by omega⟩ % 256).toNat, by omega⟩

/-- **`encodeQ8_0` is decoded by `Q8_0`.**  Every scale pattern and every
integers in `[-128, 127]` are a `Q8_0` block. -/
theorem Q8_0_encode {h : ℕ} (hh : h < 2 ^ 16) {q : Fin 32 → ℤ}
    (hq : ∀ i, -128 ≤ q i ∧ q i ≤ 127) :
    Q8_0 (encodeQ8_0 h q) = (f16 h).map fun d i => (q i : ℝ) * d := by
  have hu : u16 (encodeQ8_0 h q) 0 = h := by
    simp [u16, byte, encodeQ8_0]; omega
  have hb : ∀ i : Fin 32, int8 (byte (encodeQ8_0 h q) (2 + i)) = q i := by
    intro i
    have h1 : 2 + i.val < 34 := by omega
    have h2 := hq i
    simp only [byte, h1, dite_true, encodeQ8_0, show ¬(2 + i.val < 2) by omega, ite_false,
      show (⟨2 + i.val - 2, by omega⟩ : Fin 32) = i from Fin.ext (by simp), int8]
    split_ifs <;> omega
  unfold Q8_0
  rw [hu]
  congr 1
  funext d i
  rw [hb]

/-- **What `Q8_0` keeps of a block.**  Given a representable binary16 scale
`d > 0` with `|x_i| ≤ 127 d` for every weight, rounding `x_i / d` to nearest
(`quantize_row_q8_0_ref`, `roundf(x * id)`) gives a `Q8_0` block within `d/2`
of every weight, and every weight below `d/2` is stored as `0`.  The reference
quantizer takes `d = amax / 127` rounded to binary16, so up to that rounding
the dead zone of `Q8_0` is `amax / 254`: relative to the largest weight of its
block, not absolute. -/
theorem exists_Q8_0 {h : ℕ} (hh : h < 2 ^ 16) {d : ℝ} (hd : f16 h = some d) (hd0 : 0 < d)
    {x : Fin 32 → ℝ} (hx : ∀ i, |x i| ≤ 127 * d) :
    ∃ blk y, Q8_0 blk = some y ∧ ∀ i, |y i - x i| ≤ d / 2 ∧ (|x i| < d / 2 → y i = 0) := by
  have hr : ∀ i, |x i / d - round (x i / d)| ≤ 1 / 2 := fun i => abs_sub_round _
  have hxd : ∀ i, |x i / d| = |x i| / d := fun i => by rw [abs_div, abs_of_pos hd0]
  have hq : ∀ i, -128 ≤ round (x i / d) ∧ round (x i / d) ≤ 127 := by
    intro i
    have h1 : |x i / d| ≤ 127 := by rw [hxd, div_le_iff₀ hd0]; linarith [hx i]
    have h2 := hr i
    rw [abs_le] at h1 h2
    have h3 : ((round (x i / d) : ℤ) : ℝ) < 128 := by linarith
    have h4 : (-129 : ℝ) < ((round (x i / d) : ℤ) : ℝ) := by linarith
    constructor
    · have : (-129 : ℤ) < round (x i / d) := by exact_mod_cast h4
      omega
    · have : round (x i / d) < 128 := by exact_mod_cast h3
      omega
  refine ⟨encodeQ8_0 h fun i => round (x i / d), _, by rw [Q8_0_encode hh hq, hd]; rfl,
    fun i => ⟨?_, fun hsmall => ?_⟩⟩
  · have : (round (x i / d) : ℝ) * d - x i = -(d * (x i / d - round (x i / d))) := by
      field_simp; ring
    rw [this, abs_neg, abs_mul, abs_of_pos hd0]
    nlinarith [hr i]
  · have h1 : |x i / d| < 1 / 2 := by rw [hxd, div_lt_iff₀ hd0]; linarith
    have h2 := hr i
    rw [abs_lt] at h1
    rw [abs_le] at h2
    have h3 : |((round (x i / d) : ℤ) : ℝ)| < 1 := by rw [abs_lt]; constructor <;> linarith
    have h4 : round (x i / d) = 0 := by
      have : |round (x i / d)| < 1 := by exact_mod_cast h3
      rw [abs_lt] at this; omega
    simp [h4]

/-- The hypotheses of `exists_Q8_0` are satisfiable: the scale `1`
(`0x3C00`) and the weights `0`. -/
example : (0x3C00 : ℕ) < 2 ^ 16 ∧ f16 0x3C00 = some 1 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 32, |(fun _ => (0 : ℝ)) i| ≤ 127 * 1 :=
  ⟨by norm_num, by norm_num [f16, ieee, bias], by norm_num, fun _ => by simp⟩

end GGUF
end Transformer
