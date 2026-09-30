/-
# DASH — counted matrix products for the Newton solvers

arXiv:2602.02016v2, §3.2–3.3. A state counter increments only when an
actual matrix-matrix product is evaluated. Scalar arithmetic is excluded.
-/

import Transformer.DASH.Section3_NewtonModels

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Evaluate one matrix product and increment the operation counter.
Source: arXiv:2602.02016v2, §3.2–3.3, the matmul cost model. -/
def countedMatmul (A B : Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (Matrix (Fin n) (Fin n) ℝ) := fun count => (A * B, count + 1)

/-- CN for `p=2`, with an explicit product for the correction square.
Source: arXiv:2602.02016v2, §3.2, `equation:CN-X-M`. -/
def countedCnTwo (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) := do
  let C := cnCorrection 2 state.2
  let X ← countedMatmul state.1 C
  let C₂ ← countedMatmul C C
  let M ← countedMatmul C₂ state.2
  pure (X, M)

/-- CN for `p=4` reuses the square to compute the fourth power.
Source: arXiv:2602.02016v2, §3.2, the two products for `C^4`. -/
def countedCnFour (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) := do
  let C := cnCorrection 4 state.2
  let X ← countedMatmul state.1 C
  let C₂ ← countedMatmul C C
  let C₄ ← countedMatmul C₂ C₂
  let M ← countedMatmul C₄ state.2
  pure (X, M)

/-- The unoptimized NDB step, including the product used for its correction.
Source: arXiv:2602.02016v2, §3.3, `equation:NDB-E`–`equation:NDB-Y-Z`. -/
def countedNdbStep (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) := do
  let ZY ← countedMatmul state.2 state.1
  let E := (1 / 2 : ℝ) • ((3 : ℝ) • 1 - ZY)
  let Y ← countedMatmul state.1 E
  let Z ← countedMatmul E state.2
  pure (Y, Z)

/-- The first NDB step evaluates only `A E`; the two identity products are
replaced by closed forms. Source: arXiv:2602.02016v2, §3.3, first-step optimization. -/
def countedNdbFirst (A : Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) := do
  let E := (3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A
  let Y ← countedMatmul A E
  pure (Y, E)

/-- A square-root CN step returns the source's state using three products.
Source: arXiv:2602.02016v2, §3.2, overhead per iteration. -/
theorem countedCnTwo_correct
    (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) (count : ℕ) :
    (countedCnTwo state).run count = (cnStep 2 state, count + 3) := by
  change ((state.1 * cnCorrection 2 state.2,
      (cnCorrection 2 state.2 * cnCorrection 2 state.2) * state.2), count + 3) = _
  rw [cnStep, pow_two]

/-- A fourth-root CN step returns the source's state using four products.
Source: arXiv:2602.02016v2, §3.2, overhead per iteration. -/
theorem countedCnFour_correct
    (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) (count : ℕ) :
    (countedCnFour state).run count = (cnStep 4 state, count + 4) := by
  have hpow (C : Matrix (Fin n) (Fin n) ℝ) : C ^ 4 = (C * C) * (C * C) := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, pow_two, pow_two]
  change ((state.1 * cnCorrection 4 state.2,
      ((cnCorrection 4 state.2 * cnCorrection 4 state.2) *
        (cnCorrection 4 state.2 * cnCorrection 4 state.2)) * state.2), count + 4) = _
  rw [cnStep, hpow]

/-- The general NDB step evaluates exactly three products.
Source: arXiv:2602.02016v2, §3.3, overhead per iteration. -/
theorem countedNdbStep_correct
    (state : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) (count : ℕ) :
    (countedNdbStep state).run count = (ndbStep state, count + 3) := by
  rfl

/-- First-step optimization preserves the iterate while saving two products.
Source: arXiv:2602.02016v2, §3.3, the redundant identity multiplications. -/
theorem countedNdbFirst_correct (A : Matrix (Fin n) (Fin n) ℝ) (count : ℕ) :
    (countedNdbFirst A).run count = (ndbIterate A 1, count + 1) := by
  rw [ndb_first_step]
  rfl

/-- Counted NDB with the optimized first step followed by the unchanged
recurrence. Source: arXiv:2602.02016v2, §3.3, first-step optimization. -/
def countedNdbIterate (A : Matrix (Fin n) (Fin n) ℝ) :
    ℕ → StateM ℕ (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ)
  | 0 => pure (A, 1)
  | 1 => countedNdbFirst A
  | k + 2 => do
      let state ← countedNdbIterate A (k + 1)
      countedNdbStep state

/-- All optimized NDB iterates are unchanged and their total cost is
`max(3k-2,0)`. Source: arXiv:2602.02016v2, §3.3, saving two first-step products. -/
theorem countedNdbIterate_correct (A : Matrix (Fin n) (Fin n) ℝ) (k count : ℕ) :
    (countedNdbIterate A k).run count = (ndbIterate A k, count + (3 * k - 2)) := by
  induction k generalizing count with
  | zero => rfl
  | succ k ih =>
    cases k with
    | zero => exact countedNdbFirst_correct A count
    | succ k =>
      change ((countedNdbIterate A (k + 1) >>= countedNdbStep).run count) = _
      rw [StateT.run_bind, ih]
      change (countedNdbStep (ndbIterate A (k + 1))).run (count + (3 * (k + 1) - 2)) = _
      rw [countedNdbStep_correct]
      change (ndbIterate A (k + 2), count + (3 * (k + 1) - 2) + 3) = _
      congr 1

end Transformer.DASH
