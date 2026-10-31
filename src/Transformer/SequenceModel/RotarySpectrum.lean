module

public import Transformer.SequenceModel.Rotary

/-!
# Position-dependent rotations need not preserve a flat spectrum

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, S1.
Two orthogonal unit tokens have Gram eigenvalues 1,1. Applying RoPE at
positions 0 and 1 with frequency pi/2 makes them opposite: eigenvalues
2,0. Each token keeps its norm. A common orthogonal change of basis and
different rotations of different rows must therefore be distinguished.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel

open Landscape.Neural
open scoped BigOperators Matrix

/-- The Gram table before the positional rotations, including every channel. -/
def orthogonalGram : Matrix (Fin 2) (Fin 2) ℝ := fun i j =>
  ∑ c : Channels (Fin 1), (if c.2 = i then 1 else 0) * (if c.2 = j then 1 else 0)

/-- The same tokens after a valid two-position RoPE frequency grid. -/
def rotaryGram : Matrix (Fin 2) (Fin 2) ℝ := fun i j =>
  ∑ c : Channels (Fin 1),
    rope (fun _ => Real.pi / 2) i.val (fun c => if c.2 = i then 1 else 0) c *
    rope (fun _ => Real.pi / 2) j.val (fun c => if c.2 = j then 1 else 0) c

/-- S1: the original table is exactly flat, with both eigenvalues equal to one. -/
theorem orthogonalGram_eq : orthogonalGram = !![1, 0; 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [orthogonalGram, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- S1: norm-preserving per-position RoPE changes the full Gram table. -/
theorem rotaryGram_eq : rotaryGram = !![1, -1; -1, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [rotaryGram, rope, rotate, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- S1: both unit vectors remain unit, as read from the Gram diagonal. -/
theorem rotaryGram_diagonal (i : Fin 2) : rotaryGram i i = orthogonalGram i i := by
  rw [rotaryGram_eq, orthogonalGram_eq]
  fin_cases i <;> norm_num

/-- S1: a nonzero eigenvector changes eigenvalue from one to zero. -/
theorem rotaryGram_zero_mode :
    orthogonalGram *ᵥ ![1, 1] = ![1, 1] ∧ rotaryGram *ᵥ ![1, 1] = 0 := by
  rw [orthogonalGram_eq, rotaryGram_eq]
  constructor <;> ext i <;> fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- S1: the independent remaining eigenvector changes eigenvalue from one to two. -/
theorem rotaryGram_double_mode :
    orthogonalGram *ᵥ ![1, -1] = ![1, -1] ∧ rotaryGram *ᵥ ![1, -1] = (2 : ℝ) • ![1, -1] := by
  rw [orthogonalGram_eq, rotaryGram_eq]
  constructor <;> ext i <;> fin_cases i <;> norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- S1: positive common Q/K normalization and attention scaling cannot repair
the lost mode. This covers equal-norm Q/K vectors with positive RMS epsilon. -/
theorem scaled_rotaryGram_modes (c : ℝ) (hc : 0 < c) :
    (c • rotaryGram) *ᵥ ![1, 1] = 0 ∧ (c • rotaryGram) *ᵥ ![1, -1] =
      (2 * c) • ![1, -1] ∧ 0 < 2 * c := by
  rw [rotaryGram_eq]
  refine ⟨?_, ?_, by positivity⟩ <;>
    ext i <;> fin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

example := scaled_rotaryGram_modes 2 (by norm_num)

end Transformer.SequenceModel
