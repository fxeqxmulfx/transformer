/-
# The 3-bit lattice format `IQ3_XXS`

`ggml/src/ggml-common.h` (`block_iq3_xxs`) and `dequantize_row_iq3_xxs` in
`ggml/src/ggml-quants.c`.

A super-block of 256 weights is 64 groups of 4, each a point of `iq3xxs_grid`
(256 points, every coordinate one of `4, 12, …, 52, 62`), with signs per 8
weights as in `IQ2_XXS`.  A 4-bit scale `s` per 32 weights gives

  `y_i = d · (2s + 1)/4 · g_i · ±1`,

at 98 bytes per 256 weights, 3.0625 bits per weight.  No coordinate is `0`,
so no weight of a block with `d ≠ 0` is stored as `0` (`IQ3_XXS_ne_zero`).
-/

import Transformer.GGUF.IQ2
import Transformer.GGUF.Grid.IQ3XXS

namespace Transformer
namespace GGUF

open Grid

/-- `IQ3_XXS`: `d`, `qs[96]`.  The first 64 bytes of `qs` index the grid, one
per group of 4; the last 32 are a `uint32_t` per sub-block `i / 32`, whose top
4 bits are the scale and whose 7-bit fields `l = 0..3` are the sign codes of its
groups of 8. -/
noncomputable def IQ3_XXS (x : Bytes 98) : Option (Fin 256 → ℝ) :=
  (f16 (u16 x 0)).map fun d i =>
    d * ((2 * (u32 x (66 + 4 * (i.val / 32)) / 2 ^ 28) + 1 : ℕ) / 4 : ℝ) *
      (gridByte iq3xxs (byte x (2 + i.val / 4)) (i.val % 4) : ℕ) *
      sgn (ksigns (u32 x (66 + 4 * (i.val / 32)) / 2 ^ (7 * (i.val % 32 / 8)) % 128)) (i.val % 8)

/-- **`IQ3_XXS` stores grid points:** `y_i = d · (2s + 1)/4 · g · σ` with a
4-bit `s` per sub-block of 32, `g ∈ {4, 12, 20, 28, 36, 44, 52, 62}`, `σ = ±1`. -/
theorem IQ3_XXS_levels {x : Bytes 98} {y : Fin 256 → ℝ} (h : IQ3_XXS x = some y) :
    ∃ d, f16 (u16 x 0) = some d ∧ ∃ s : ℕ → ℕ, (∀ b, s b < 16) ∧ ∀ i : Fin 256,
      ∃ g : ℕ, g ∈ [4, 12, 20, 28, 36, 44, 52, 62] ∧ ∃ σ : ℤ, (σ = 1 ∨ σ = -1) ∧
        y i = d * ((2 * s (i.val / 32) + 1 : ℕ) / 4 : ℝ) * (g : ℝ) * (σ : ℝ) := by
  unfold IQ3_XXS at h
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨d, hd, fun b => u32 x (66 + 4 * b) / 2 ^ 28,
    fun b => by have := u32_lt x (66 + 4 * b); dsimp only; omega, fun i => ⟨_, iq3xxs_mem (byte_lt _ _)
      (Nat.mod_lt _ (by norm_num)), _, sgn_eq _ _, rfl⟩⟩

/-- **`IQ3_XXS` has no dead zone.** -/
theorem IQ3_XXS_ne_zero {x : Bytes 98} {y : Fin 256 → ℝ} (h : IQ3_XXS x = some y)
    (hd : f16 (u16 x 0) ≠ some 0) (i : Fin 256) : y i ≠ 0 := by
  obtain ⟨d, hd', s, -, hy⟩ := IQ3_XXS_levels h
  obtain ⟨g, hg, σ, hσ, hi⟩ := hy i
  rw [hi]
  have hd0 : d ≠ 0 := by rintro rfl; exact hd hd'
  have hg0 : (g : ℝ) ≠ 0 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num
  have hσ0 : (σ : ℝ) ≠ 0 := by rcases hσ with rfl | rfl <;> norm_num
  have hs : ((2 * s (i.val / 32) + 1 : ℕ) / 4 : ℝ) ≠ 0 := by positivity
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero hd0 hs) hg0) hσ0

/-- The hypotheses of `IQ3_XXS_levels` and `IQ3_XXS_ne_zero` are satisfiable:
the scale-`1` (`0x3C00`) super-block decodes, and `1` is not `0`. -/
example : (∃ y, IQ3_XXS (fun j => if j.val = 1 then 0x3C else 0) = some y) ∧
    f16 (u16 (fun j : Fin 98 => if j.val = 1 then (0x3C : Fin 256) else 0) 0) ≠ some 0 := by
  have hu : u16 (fun j : Fin 98 => if j.val = 1 then (0x3C : Fin 256) else 0) 0 = 0x3C00 := by
    decide
  have h1 : f16 0x3C00 = some 1 := by norm_num [f16, ieee, bias]
  exact ⟨⟨_, rfl⟩, by rw [hu, h1]; norm_num⟩

end GGUF
end Transformer
