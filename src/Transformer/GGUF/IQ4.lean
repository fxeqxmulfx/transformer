/-
# The non-linear 4-bit formats: `IQ4_NL`, `IQ4_XS`

`ggml/src/ggml-common.h` (`block_iq4_nl`, `block_iq4_xs`, `kvalues_iq4nl`) and
`dequantize_row_iq4_nl`, `dequantize_row_iq4_xs` in `ggml/src/ggml-quants.c`.

A 4-bit field does not scale an integer here; it indexes the codebook
`kvalues_iq4nl` of sixteen `int8_t` levels, spaced densely near `0` and
sparsely at the ends.

| format | bytes | weights | weight | bits/weight |
| --- | ---: | ---: | --- | ---: |
| `IQ4_NL` | 18 | 32 | `d · K[q]` | 4.5 |
| `IQ4_XS` | 136 | 256 | `d · (s - 32) · K[q]`, `s` 6 bits per 32 | 4.25 |

The codebook has no `0`: it jumps from `-10` to `1`.  So in a block whose
scale is not `0`, no weight is stored as `0` (`IQ4_NL_ne_zero`): unlike the
integer formats, `IQ4_NL` has no dead zone at all.
-/

import Transformer.GGUF.Legacy
import Mathlib.Data.Fin.VecNotation
import Mathlib.Order.Fin.Basic

namespace Transformer
namespace GGUF

/-- `kvalues_iq4nl`. -/
def kIQ4 : Fin 16 → ℤ :=
  ![-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing. -/
theorem kIQ4_strictMono : StrictMono kIQ4 :=
  Fin.strictMono_iff_lt_succ.2 (by decide)

/-- **The codebook has no `0`.** -/
theorem kIQ4_ne_zero (k : Fin 16) : kIQ4 k ≠ 0 := by
  revert k; decide

/-- The codebook lies in `[-127, 113]`. -/
theorem kIQ4_bounds (k : Fin 16) : -127 ≤ kIQ4 k ∧ kIQ4 k ≤ 113 := by
  revert k; decide

/-- `IQ4_NL`: `d`, `qs[16]`; the nibbles are laid out as in `Q4_0`. -/
noncomputable def IQ4_NL (x : Bytes 18) : Option (Fin 32 → ℝ) :=
  (f16 (u16 x 0)).map fun d i => d * kIQ4 ⟨nib x 2 i, nib_lt x 2 i⟩

/-- The 6-bit scale of sub-block `b < 8` of `IQ4_XS`: a nibble of
`scales_l[b / 2]` (low for even `b`) and bits `2b, 2b + 1` of the little-endian
`scales_h`. -/
def scaleIQ4XS {n : ℕ} (x : Bytes n) (b : ℕ) : ℕ :=
  byte x (4 + b / 2) / 16 ^ (b % 2) % 16 + 16 * (u16 x 2 / 4 ^ b % 4)

theorem scaleIQ4XS_lt {n : ℕ} (x : Bytes n) (b : ℕ) : scaleIQ4XS x b < 64 := by
  unfold scaleIQ4XS; omega

/-- The 4-bit field of weight `i` of `IQ4_XS`: sub-block `i / 32` owns the
16 bytes from `8 + 16 (i / 32)`, low nibbles first. -/
def nibXS {n : ℕ} (x : Bytes n) (i : Fin 256) : ℕ :=
  byte x (8 + 16 * (i.val / 32) + i.val % 16) / 16 ^ (i.val % 32 / 16) % 16

theorem nibXS_lt {n : ℕ} (x : Bytes n) (i : Fin 256) : nibXS x i < 16 := by
  unfold nibXS; omega

/-- `IQ4_XS`: `d`, `scales_h`, `scales_l[4]`, `qs[128]`. -/
noncomputable def IQ4_XS (x : Bytes 136) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 0)).map fun d i =>
    d * (((scaleIQ4XS x (i.val / 32) : ℕ) : ℤ) - 32 : ℤ) * kIQ4 ⟨nibXS x i, nibXS_lt x i⟩

/-- **`IQ4_NL` stores a codebook level per weight:** `y_i = d · K[q_i]`. -/
theorem IQ4_NL_levels {x : Bytes 18} {y : Fin 32 → ℝ} (h : IQ4_NL x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∀ i, ∃ k : Fin 16, y i = d * kIQ4 k := by
  unfold IQ4_NL at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun i => ⟨_, rfl⟩⟩

/-- **`IQ4_NL` has no dead zone.**  In a block whose scale is not `0`, every
weight is stored as a nonzero number: `0` is not a level of the format. -/
theorem IQ4_NL_ne_zero {x : Bytes 18} {y : Fin 32 → ℝ} (h : IQ4_NL x = some y)
    (hd : f16 (u16 x 0) ≠ some 0) (i : Fin 32) : y i ≠ 0 := by
  obtain ⟨d, hd', hy⟩ := IQ4_NL_levels h
  obtain ⟨k, hk⟩ := hy i
  have hd0 : d ≠ 0 := by rintro rfl; exact hd hd'
  rw [hk]
  exact mul_ne_zero hd0 (by exact_mod_cast kIQ4_ne_zero k)

/-- **`IQ4_XS` stores a codebook level per weight and 64 scales per sub-block
of 32:** `y_i = d · s · K[q_i]`, `-32 ≤ s ≤ 31`. -/
theorem IQ4_XS_levels {x : Bytes 136} {y : Fin 256 → ℝ} (h : IQ4_XS x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∃ s : ℕ → ℤ, (∀ j, -32 ≤ s j ∧ s j ≤ 31) ∧
      ∀ i : Fin 256, ∃ k : Fin 16, y i = d * s (i.val / 32) * kIQ4 k := by
  unfold IQ4_XS at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun j => ((scaleIQ4XS x j : ℕ) : ℤ) - 32,
    fun j => by have := scaleIQ4XS_lt x j; dsimp only; omega, fun i => ⟨_, rfl⟩⟩

/-- The hypotheses of `IQ4_NL_ne_zero` are satisfiable: the block with scale
`1` (`0x3C00`) and every nibble `8`, which decodes to the level `1`. -/
example : IQ4_NL (fun j => if j.val = 1 then 0x3C else if j.val < 2 then 0 else 0x88) =
      some (fun _ => 1) ∧
    f16 (u16 (fun j : Fin 18 => if j.val = 1 then (0x3C : Fin 256) else if j.val < 2 then 0 else 0x88) 0)
      ≠ some 0 := by
  have hu : u16 (fun j : Fin 18 => if j.val = 1 then (0x3C : Fin 256) else if j.val < 2 then 0 else 0x88)
      0 = 0x3C00 := by decide
  have h1 : f16 0x3C00 = some 1 := by norm_num [f16, ieee, bias]
  refine ⟨?_, by rw [hu, h1]; norm_num⟩
  unfold IQ4_NL
  rw [hu, h1]
  simp only [Option.map_some, Option.some.injEq]
  funext i
  have : nib (fun j : Fin 18 => if j.val = 1 then (0x3C : Fin 256) else if j.val < 2 then 0 else 0x88) 2 i
      = 8 := by revert i; decide
  simp [this, kIQ4]

/-- The hypothesis of `IQ4_XS_levels` is satisfiable: the all-zero super-block
decodes. -/
example : IQ4_XS (fun _ => 0) = some (fun _ => (0 : ℝ)) := by
  norm_num [IQ4_XS, f16, ieee, u16, byte]

end GGUF
end Transformer
