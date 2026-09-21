/-
# The 1.5-bit lattice format `IQ1_S`

`ggml/src/ggml-common.h` (`block_iq1_s`, `IQ1S_DELTA = 0.125`) and
`dequantize_row_iq1_s` in `ggml/src/ggml-quants.c`.

A super-block of 256 weights is 32 groups of 8, each a point of `iq1s_grid`
(2048 points, coordinates in `{-1, 0, 1}`) addressed by 11 bits.  Each
sub-block of 32 has a 3-bit scale `s` and a shift `δ = ±1/8`:

  `y_i = d · (2s + 1) · (g_i + δ)`,

at 50 bytes per 256 weights, 1.5625 bits per weight.  Since `g + δ` is never
`0`, no weight of a block with `d ≠ 0` is stored as `0` (`IQ1_S_ne_zero`): the
grid has a `0` coordinate, but the shift moves it off.
-/

import Transformer.GGUF.IQ2
import Transformer.GGUF.Grid.IQ1S
import Transformer.GGUF.Legacy

namespace Transformer
namespace GGUF

open Grid

/-- `IQ1_S`: `d`, `qs[32]`, `qh[8]` (`uint16_t`).  Group `i / 8` takes the low
8 bits of its index from `qs[i / 8]` and the high 3 from bits `3l .. 3l + 2`
of `qh[i / 32]`, `l = (i mod 32) / 8`; bits 12–14 of `qh[i / 32]` are the scale
and bit 15 the sign of the shift. -/
noncomputable def IQ1_S (x : Bytes 50) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 0)).map fun d i =>
    d * ((2 * (u16 x (34 + 2 * (i.val / 32)) / 2 ^ 12 % 8) + 1 : ℕ) : ℝ) *
      ((int8 (gridByte iq1s (byte x (2 + i.val / 8) +
          256 * (u16 x (34 + 2 * (i.val / 32)) / 8 ^ (i.val % 32 / 8) % 8)) (i.val % 8)) : ℝ) +
        (if u16 x (34 + 2 * (i.val / 32)) / 2 ^ 15 % 2 = 1 then -1 / 8 else 1 / 8))

/-- **`IQ1_S` stores shifted grid points:** `y_i = d · (2s + 1) · (g + δ)` with
a 3-bit `s` and a shift `δ = ±1/8` per sub-block of 32, and `g ∈ {-1, 0, 1}`. -/
theorem IQ1_S_levels {x : Bytes 50} {y : Fin 256 → ℝ} (h : IQ1_S x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∃ (s : ℕ → ℕ) (δ : ℕ → ℝ), (∀ b, s b < 8 ∧ (δ b = 1 / 8 ∨ δ b = -1 / 8)) ∧
      ∀ i : Fin 256, ∃ g : ℤ, (g = -1 ∨ g = 0 ∨ g = 1) ∧
        y i = d * ((2 * s (i.val / 32) + 1 : ℕ) : ℝ) * ((g : ℝ) + δ (i.val / 32)) := by
  unfold IQ1_S at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨d, hd, fun b => u16 x (34 + 2 * b) / 2 ^ 12 % 8,
    fun b => if u16 x (34 + 2 * b) / 2 ^ 15 % 2 = 1 then -1 / 8 else 1 / 8,
    fun b => ⟨Nat.mod_lt _ (by norm_num), by dsimp only; split_ifs <;> simp⟩, fun i => ⟨_, ?_, rfl⟩⟩
  have hk : byte x (2 + i.val / 8) + 256 * (u16 x (34 + 2 * (i.val / 32)) / 8 ^ (i.val % 32 / 8) % 8)
      < 2048 := by
    have := byte_lt x (2 + i.val / 8)
    have : u16 x (34 + 2 * (i.val / 32)) / 8 ^ (i.val % 32 / 8) % 8 < 8 := Nat.mod_lt _ (by norm_num)
    omega
  have hm := iq1s_mem hk (Nat.mod_lt i.val (by norm_num : 0 < 8))
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with hm | hm | hm <;> simp [hm, int8]

/-- **`IQ1_S` has no dead zone.** -/
theorem IQ1_S_ne_zero {x : Bytes 50} {y : Fin 256 → ℝ} (h : IQ1_S x = some y)
    (hd : f16 (u16 x 0) ≠ some 0) (i : Fin 256) : y i ≠ 0 := by
  obtain ⟨d, hd', s, δ, hsδ, hy⟩ := IQ1_S_levels h
  obtain ⟨g, hg, hi⟩ := hy i
  rw [hi]
  have hd0 : d ≠ 0 := by rintro rfl; exact hd hd'
  have hs : ((2 * s (i.val / 32) + 1 : ℕ) : ℝ) ≠ 0 := by positivity
  have hgδ : (g : ℝ) + δ (i.val / 32) ≠ 0 := by
    rcases (hsδ (i.val / 32)).2 with h' | h' <;> rw [h'] <;>
      rcases hg with rfl | rfl | rfl <;> norm_num
  exact mul_ne_zero (mul_ne_zero hd0 hs) hgδ

/-- The hypotheses of `IQ1_S_levels` and `IQ1_S_ne_zero` are satisfiable:
the scale-`1` (`0x3C00`) super-block decodes, and `1` is not `0`. -/
example : (∃ y, IQ1_S (fun j => if j.val = 1 then 0x3C else 0) = some y) ∧
    f16 (u16 (fun j : Fin 50 => if j.val = 1 then (0x3C : Fin 256) else 0) 0) ≠ some 0 := by
  have hu : u16 (fun j : Fin 50 => if j.val = 1 then (0x3C : Fin 256) else 0) 0 = 0x3C00 := by
    decide
  have h1 : f16 0x3C00 = some 1 := by norm_num [f16, ieee, bias]
  exact ⟨⟨_, rfl⟩, by rw [hu, h1]; norm_num⟩

end GGUF
end Transformer
