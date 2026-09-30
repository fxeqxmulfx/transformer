/-
# DASH — counts of batched products in the Newton solvers

arXiv:2602.02016v2, §3.2–3.3 and §4. The counter counts calls to
the batched multiplication primitive, independently of the number of
matrices in that call. It makes no assertion about hardware runtime.
-/

import Transformer.DASH.Section4_BatchedNewton

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {m n p : ℕ}

/-- Evaluate one batched product and increment the call counter once.
Source: arXiv:2602.02016v2, §4, the `bmm` implementation of the solvers. -/
def countedBatchMatmul (A : ι → Matrix (Fin m) (Fin n) ℝ)
    (B : ι → Matrix (Fin n) (Fin p) ℝ) :
    StateM ℕ (ι → Matrix (Fin m) (Fin p) ℝ) :=
  fun count => (batchMul A B, count + 1)

/-- Batched square-root CN, computing each correction square by `bmm`.
Source: arXiv:2602.02016v2, §3.2 and §4, three products per iteration. -/
def countedBatchCnTwo
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) :
    StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) := do
  let C := fun j => cnCorrection 2 (state.2 j)
  let X ← countedBatchMatmul state.1 C
  let C₂ ← countedBatchMatmul C C
  let M ← countedBatchMatmul C₂ state.2
  pure (X, M)

/-- Batched fourth-root CN reuses its correction squares.
Source: arXiv:2602.02016v2, §3.2 and §4, four products per iteration. -/
def countedBatchCnFour
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) :
    StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) := do
  let C := fun j => cnCorrection 4 (state.2 j)
  let X ← countedBatchMatmul state.1 C
  let C₂ ← countedBatchMatmul C C
  let C₄ ← countedBatchMatmul C₂ C₂
  let M ← countedBatchMatmul C₄ state.2
  pure (X, M)

/-- The batched NDB recurrence with its three counted `bmm` calls.
Source: arXiv:2602.02016v2, §3.3 and §4, the correction and two state updates. -/
def countedBatchNdbStep
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) :
    StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) := do
  let ZY ← countedBatchMatmul state.2 state.1
  let E := fun j => (1 / 2 : ℝ) • ((3 : ℝ) • 1 - ZY j)
  let Y ← countedBatchMatmul state.1 E
  let Z ← countedBatchMatmul E state.2
  pure (Y, Z)

/-- Optimized first batched NDB step omits both identity products.
Source: arXiv:2602.02016v2, §3.3 and §4, the first-step optimization. -/
def countedBatchNdbFirst (A : ι → Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) := do
  let E := fun j => (3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A j
  let Y ← countedBatchMatmul A E
  pure (Y, E)

/-- A counted square-root CN call returns the complete batched state and
uses three `bmm` calls. Source: arXiv:2602.02016v2, §3.2 and §4. -/
theorem countedBatchCnTwo_correct
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
    (count : ℕ) :
    (countedBatchCnTwo state).run count = (batchCnStep 2 state, count + 3) := by
  change ((fun j => state.1 j * cnCorrection 2 (state.2 j),
      fun j => (cnCorrection 2 (state.2 j) * cnCorrection 2 (state.2 j)) * state.2 j),
      count + 3) = _
  simp only [batchCnStep, pow_two]
  rfl

/-- A fourth-root CN call returns the same batch using four `bmm` calls.
Source: arXiv:2602.02016v2, §3.2 and §4, reused correction powers. -/
theorem countedBatchCnFour_correct
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
    (count : ℕ) :
    (countedBatchCnFour state).run count = (batchCnStep 4 state, count + 4) := by
  have hpow (C : Matrix (Fin n) (Fin n) ℝ) : C ^ 4 = (C * C) * (C * C) := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, pow_two, pow_two]
  change ((fun j => state.1 j * cnCorrection 4 (state.2 j),
      fun j => ((cnCorrection 4 (state.2 j) * cnCorrection 4 (state.2 j)) *
        (cnCorrection 4 (state.2 j) * cnCorrection 4 (state.2 j))) * state.2 j),
      count + 4) = _
  simp only [batchCnStep, hpow]
  rfl

/-- A general NDB iteration uses three `bmm` calls for the entire batch.
Source: arXiv:2602.02016v2, §3.3 and §4, batched NDB updates. -/
theorem countedBatchNdbStep_correct
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
    (count : ℕ) :
    (countedBatchNdbStep state).run count = (batchNdbStep state, count + 3) := by
  rfl

/-- The first batched NDB iteration uses one product and preserves all
outputs. Source: arXiv:2602.02016v2, §3.3 and §4, identity-product elimination. -/
theorem countedBatchNdbFirst_correct (A : ι → Matrix (Fin n) (Fin n) ℝ) (count : ℕ) :
    (countedBatchNdbFirst A).run count = (batchNdbIterate A 1, count + 1) := by
  rw [← batchNdbFirst_eq]
  rfl

/-- Counted NDB uses its optimized first call and the general recurrence
thereafter. Source: arXiv:2602.02016v2, §3.3 and §4. -/
def countedBatchNdbIterate (A : ι → Matrix (Fin n) (Fin n) ℝ) :
    ℕ → StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
  | 0 => pure (A, fun _ => 1)
  | 1 => countedBatchNdbFirst A
  | k + 2 => do
      let state ← countedBatchNdbIterate A (k + 1)
      countedBatchNdbStep state

/-- `k` optimized NDB iterations on any batch use `max(3k-2,0)` calls
and equal the original recurrence at every block. The count is independent
of batch cardinality. Source: arXiv:2602.02016v2, §3.3 and §4. -/
theorem countedBatchNdbIterate_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (k count : ℕ) :
    (countedBatchNdbIterate A k).run count =
      (batchNdbIterate A k, count + (3 * k - 2)) := by
  induction k generalizing count with
  | zero => rfl
  | succ k ih =>
    cases k with
    | zero => exact countedBatchNdbFirst_correct A count
    | succ k =>
      change ((countedBatchNdbIterate A (k + 1) >>= countedBatchNdbStep).run count) = _
      rw [StateT.run_bind, ih]
      change (countedBatchNdbStep (batchNdbIterate A (k + 1))).run
        (count + (3 * (k + 1) - 2)) = _
      rw [countedBatchNdbStep_correct]
      change (batchNdbIterate A (k + 2), count + (3 * (k + 1) - 2) + 3) = _
      congr 1

end Transformer.DASH
