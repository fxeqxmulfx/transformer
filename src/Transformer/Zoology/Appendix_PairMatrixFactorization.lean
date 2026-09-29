/-
# Universal two-coordinate factorization, including singular blocks

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`lmm: kaleido-coyote`. An alternative proof construction factors every
2 × 2 block into four shears and one diagonal map. Pivot preparation
handles zero columns and the zero matrix without a nondegeneracy assumption.
-/

import Transformer.Zoology.Appendix_InPlaceLinear

noncomputable section

namespace Transformer.Zoology

/-- Four scalar entries of a coordinate pair's matrix.
Source: Appendix `def: butterfly`, four diagonal blocks. -/
structure PairMatrix where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Exact action on the upper and lower coordinate of the pair.
Source: Appendix `def: butterfly`, ordinary 2 × 2 matrix multiplication. -/
def PairMatrix.apply (M : PairMatrix) (v : ℝ × ℝ) : ℝ × ℝ :=
  (M.a * v.1 + M.b * v.2, M.c * v.1 + M.d * v.2)

/-- Add t times the lower coordinate to the upper coordinate.
Source: Appendix `prop: butterfly-hyena`, in-place alternative construction. -/
def pairUpperShear (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (v.1 + t * v.2, v.2)

/-- Add t times the upper coordinate to the lower coordinate.
Source: Appendix `prop: butterfly-hyena`, in-place alternative construction. -/
def pairLowerShear (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (v.1, v.2 + t * v.1)

/-- First repair a zero first column by adding the second column.
Source: Appendix `lmm: kaleido-coyote`, construction for arbitrary matrices. -/
def PairMatrix.columnPrep (M : PairMatrix) : ℝ := if M.a = 0 ∧ M.c = 0 then 1 else 0

/-- Repair a zero top entry by adding the lower row.
Source: Appendix `lmm: kaleido-coyote`, construction for arbitrary matrices. -/
def PairMatrix.rowPrep (M : PairMatrix) : ℝ :=
  if M.a + M.columnPrep * M.b = 0 then 1 else 0

/-- Prepared pivot, possibly zero only for the zero matrix.
Source: Appendix `lmm: kaleido-coyote`, singular cases retained. -/
def PairMatrix.pivot (M : PairMatrix) : ℝ :=
  M.a + M.columnPrep * M.b + M.rowPrep * (M.c + M.columnPrep * M.d)

/-- A zero prepared pivot implies every matrix entry is zero.
Source: Appendix `def: butterfly`, no invertibility assumption on its blocks. -/
theorem PairMatrix.zero_of_pivot_zero (M : PairMatrix) (h : M.pivot = 0) :
    M.a = 0 ∧ M.b = 0 ∧ M.c = 0 ∧ M.d = 0 := by
  by_cases hcol : M.a = 0 ∧ M.c = 0
  · by_cases hb : M.b = 0
    · simp [PairMatrix.pivot, PairMatrix.columnPrep, PairMatrix.rowPrep, hcol, hb] at h
      exact ⟨hcol.1, hb, hcol.2, h⟩
    · simp [PairMatrix.pivot, PairMatrix.columnPrep, PairMatrix.rowPrep, hcol, hb] at h
  · by_cases ha : M.a = 0
    · have hc : M.c ≠ 0 := by tauto
      simp [PairMatrix.pivot, PairMatrix.columnPrep, PairMatrix.rowPrep, ha, hc] at h
    · simp [PairMatrix.pivot, PairMatrix.columnPrep, PairMatrix.rowPrep, ha] at h

/-- Four shears and one diagonal map realize every real pair matrix.
Source: Appendix `lmm: kaleido-coyote`, a constructive alternative that
requires no sequence workspace and includes all singular blocks. -/
theorem PairMatrix.factorization (M : PairMatrix) (v : ℝ × ℝ) :
    M.apply v =
      pairUpperShear (-M.rowPrep)
        (pairLowerShear ((M.c + M.columnPrep * M.d) / M.pivot)
          ((fun z : ℝ × ℝ =>
            (M.pivot * z.1,
              (M.d - (M.c + M.columnPrep * M.d) / M.pivot *
                (M.b + M.rowPrep * M.d)) * z.2))
            (pairUpperShear ((M.b + M.rowPrep * M.d) / M.pivot)
              (pairLowerShear (-M.columnPrep) v)))) := by
  by_cases hp : M.pivot = 0
  · rcases M.zero_of_pivot_zero hp with ⟨ha, hb, hc, hd⟩
    simp [PairMatrix.apply, pairUpperShear, pairLowerShear, hp, ha, hb, hc, hd]
  · apply Prod.ext <;>
      simp only [PairMatrix.apply, pairUpperShear, pairLowerShear]
    · field_simp [hp]
      simp only [PairMatrix.pivot]
      ring
    · field_simp [hp]
      simp only [PairMatrix.pivot]
      ring

/-- The zero-pivot hypothesis is satisfiable by an allowed singular block.
Source: Appendix `def: butterfly`, all four diagonals may be zero. -/
example : ({a := 0, b := 0, c := 0, d := 0} : PairMatrix).pivot = 0 := by
  norm_num [PairMatrix.pivot, PairMatrix.columnPrep, PairMatrix.rowPrep]

end Transformer.Zoology
