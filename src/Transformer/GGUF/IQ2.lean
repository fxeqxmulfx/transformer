/-
# The 2-bit lattice formats: `IQ2_XXS`, `IQ2_XS`

`ggml/src/ggml-common.h` (`block_iq2_xxs`, `block_iq2_xs`) and
`dequantize_row_iq2_xxs`, `dequantize_row_iq2_xs` in `ggml/src/ggml-quants.c`.

A super-block of 256 weights is 32 groups of 8.  Each group is a point of the
grid `iq2xxs_grid` (256 points) or `iq2xs_grid` (512 points), every coordinate
`8`, `25` or `43`, with 7 sign bits completed to an even number of minus signs
by `ksigns_iq2xs`.  A 4-bit scale `s` per 32 (`IQ2_XXS`) or per 16 (`IQ2_XS`)
weights gives

  `y_i = d · (2s + 1)/8 · g_i · ±1`.

| format | bytes | group code | bits/weight |
| --- | ---: | --- | ---: |
| `IQ2_XXS` | 66 | 8-bit grid index, 7 sign bits | 2.0625 |
| `IQ2_XS` | 74 | 9-bit grid index, 7 sign bits | 2.3125 |

No coordinate of either grid is `0`, so in a block whose `d` is not `0` no
weight is stored as `0` (`IQ2_XXS_ne_zero`, `IQ2_XS_ne_zero`).
-/

import Transformer.GGUF.Basic
import Transformer.GGUF.Grid.IQ2XXS
import Transformer.GGUF.Grid.IQ2XS

namespace Transformer
namespace GGUF

open Grid

/-- `IQ2_XXS`: `d`, `qs[32]` (`uint16_t`).  Sub-block `b = i / 32` is 8 bytes
from `2 + 8b`: four grid indices, then a `uint32_t` whose top 4 bits are the
scale and whose 7-bit fields `l = 0..3` are the sign codes of the groups. -/
noncomputable def IQ2_XXS (x : Bytes 66) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 0)).map fun d i =>
    d * ((2 * (u32 x (6 + 8 * (i.val / 32)) / 2 ^ 28) + 1 : ℕ) / 8 : ℝ) *
      (gridByte iq2xxs (byte x (2 + 8 * (i.val / 32) + i.val % 32 / 8)) (i.val % 8) : ℕ) *
      sgn (ksigns (u32 x (6 + 8 * (i.val / 32)) / 2 ^ (7 * (i.val % 32 / 8)) % 128)) (i.val % 8)

/-- `IQ2_XS`: `d`, `qs[32]` (`uint16_t`), `scales[8]`.  Group `i / 8` is the
`uint16_t` at `2 + 2 (i / 8)`: a 9-bit grid index and a 7-bit sign code; the
scale is a nibble of `scales[i / 32]`, low for the first 16 weights. -/
noncomputable def IQ2_XS (x : Bytes 74) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 0)).map fun d i =>
    d * ((2 * (byte x (66 + i.val / 32) / 16 ^ (i.val % 32 / 16) % 16) + 1 : ℕ) / 8 : ℝ) *
      (gridByte iq2xs (u16 x (2 + 2 * (i.val / 8)) % 512) (i.val % 8) : ℕ) *
      sgn (ksigns (u16 x (2 + 2 * (i.val / 8)) / 512)) (i.val % 8)

theorem u16_lt {n : ℕ} (x : Bytes n) (o : ℕ) : u16 x o < 65536 := by
  have := byte_lt x o; have := byte_lt x (o + 1); unfold u16; omega

theorem u32_lt {n : ℕ} (x : Bytes n) (o : ℕ) : u32 x o < 2 ^ 32 := by
  have := u16_lt x o; have := u16_lt x (o + 2); unfold u32; omega

/-- A weight `d · (2s + 1)/8 · g · σ` with `d ≠ 0`, `g ∈ {8, 25, 43}`, `σ = ±1`
is not `0`. -/
theorem iq2_weight_ne_zero {d : ℝ} (hd : d ≠ 0) (s : ℕ) {g : ℕ} (hg : g ∈ [8, 25, 43]) {σ : ℤ}
    (hσ : σ = 1 ∨ σ = -1) : d * ((2 * s + 1 : ℕ) / 8 : ℝ) * (g : ℝ) * (σ : ℝ) ≠ 0 := by
  have hg0 : (g : ℝ) ≠ 0 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl <;> norm_num
  have hσ0 : (σ : ℝ) ≠ 0 := by rcases hσ with rfl | rfl <;> norm_num
  have hs : ((2 * s + 1 : ℕ) / 8 : ℝ) ≠ 0 := by positivity
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero hd hs) hg0) hσ0

/-- **`IQ2_XXS` stores grid points:** `y_i = d · (2s + 1)/8 · g · σ` with a
4-bit `s` per sub-block of 32, `g ∈ {8, 25, 43}`, `σ = ±1`. -/
theorem IQ2_XXS_levels {x : Bytes 66} {y : Fin 256 → ℝ} (h : IQ2_XXS x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∃ s : ℕ → ℕ, (∀ b, s b < 16) ∧ ∀ i : Fin 256,
      ∃ g : ℕ, g ∈ [8, 25, 43] ∧ ∃ σ : ℤ, (σ = 1 ∨ σ = -1) ∧
        y i = d * ((2 * s (i.val / 32) + 1 : ℕ) / 8 : ℝ) * (g : ℝ) * (σ : ℝ) := by
  unfold IQ2_XXS at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨d, hd, fun b => u32 x (6 + 8 * b) / 2 ^ 28,
    fun b => by have := u32_lt x (6 + 8 * b); dsimp only; omega, fun i => ⟨_, iq2xxs_mem (byte_lt _ _)
      (Nat.mod_lt _ (by norm_num)), _, sgn_eq _ _, rfl⟩⟩

/-- **`IQ2_XS` stores grid points:** `y_i = d · (2s + 1)/8 · g · σ` with a
4-bit `s` per 16 weights, `g ∈ {8, 25, 43}`, `σ = ±1`. -/
theorem IQ2_XS_levels {x : Bytes 74} {y : Fin 256 → ℝ} (h : IQ2_XS x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∃ s : ℕ → ℕ, (∀ b, s b < 16) ∧ ∀ i : Fin 256,
      ∃ g : ℕ, g ∈ [8, 25, 43] ∧ ∃ σ : ℤ, (σ = 1 ∨ σ = -1) ∧
        y i = d * ((2 * s (i.val / 16) + 1 : ℕ) / 8 : ℝ) * (g : ℝ) * (σ : ℝ) := by
  unfold IQ2_XS at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨d, hd, fun b => byte x (66 + b / 2) / 16 ^ (b % 2) % 16, fun b => Nat.mod_lt _ (by norm_num),
    fun i => ⟨gridByte iq2xs (u16 x (2 + 2 * (i.val / 8)) % 512) (i.val % 8),
      iq2xs_mem (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)),
      sgn (ksigns (u16 x (2 + 2 * (i.val / 8)) / 512)) (i.val % 8), sgn_eq _ _, ?_⟩⟩
  have h1 : i.val / 16 / 2 = i.val / 32 := by omega
  have h2 : i.val / 16 % 2 = i.val % 32 / 16 := by omega
  simp only [h1, h2]

/-- **`IQ2_XXS` has no dead zone.** -/
theorem IQ2_XXS_ne_zero {x : Bytes 66} {y : Fin 256 → ℝ} (h : IQ2_XXS x = some y)
    (hd : f16 (u16 x 0) ≠ some 0) (i : Fin 256) : y i ≠ 0 := by
  obtain ⟨d, hd', s, -, hy⟩ := IQ2_XXS_levels h
  obtain ⟨g, hg, σ, hσ, hi⟩ := hy i
  rw [hi]
  exact iq2_weight_ne_zero (by rintro rfl; exact hd hd') _ hg hσ

/-- **`IQ2_XS` has no dead zone.** -/
theorem IQ2_XS_ne_zero {x : Bytes 74} {y : Fin 256 → ℝ} (h : IQ2_XS x = some y)
    (hd : f16 (u16 x 0) ≠ some 0) (i : Fin 256) : y i ≠ 0 := by
  obtain ⟨d, hd', s, -, hy⟩ := IQ2_XS_levels h
  obtain ⟨g, hg, σ, hσ, hi⟩ := hy i
  rw [hi]
  exact iq2_weight_ne_zero (by rintro rfl; exact hd hd') _ hg hσ

/-- The hypotheses of the theorems above are satisfiable: the scale-`1`
(`0x3C00`) super-blocks decode, and `1` is not `0`. -/
example : (∃ y, IQ2_XXS (fun j => if j.val = 1 then 0x3C else 0) = some y) ∧
    (∃ y, IQ2_XS (fun j => if j.val = 1 then 0x3C else 0) = some y) ∧
    f16 (u16 (fun j : Fin 66 => if j.val = 1 then (0x3C : Fin 256) else 0) 0) ≠ some 0 ∧
    f16 (u16 (fun j : Fin 74 => if j.val = 1 then (0x3C : Fin 256) else 0) 0) ≠ some 0 := by
  have h66 : u16 (fun j : Fin 66 => if j.val = 1 then (0x3C : Fin 256) else 0) 0 = 0x3C00 := by
    decide
  have h74 : u16 (fun j : Fin 74 => if j.val = 1 then (0x3C : Fin 256) else 0) 0 = 0x3C00 := by
    decide
  have h1 : f16 0x3C00 = some 1 := by norm_num [f16, ieee, bias]
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ?_, ?_⟩
  · rw [h66, h1]; norm_num
  · rw [h74, h1]; norm_num

end GGUF
end Transformer
