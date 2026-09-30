/-
# DASH — Newton recurrences using batched matrix multiplication

arXiv:2602.02016v2, §4. The batching axis is explicit in each product
of the CN and NDB recurrences. The complete batched state agrees
with the single-matrix solver on every component and iteration.
-/

import Transformer.DASH.Section4_Blocking
import Transformer.DASH.Section3_NewtonModels

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type*} {n : ℕ}

/-- CN step with the source's `bmm` primitive on the batching axis.
Source: arXiv:2602.02016v2, §3.2 and §4, the stacked CN implementation. -/
def batchCnStep (p : ℕ)
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) :
    (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ) :=
  let C := fun j => cnCorrection p (state.2 j)
  (batchMul state.1 C, batchMul (fun j => C j ^ p) state.2)

/-- The complete batched CN recurrence, with independently scaled inputs
and a common iteration count. Source: arXiv:2602.02016v2, §3.2 and §4. -/
def batchCnIterate (p : ℕ) (A : ι → Matrix (Fin n) (Fin n) ℝ) (c : ι → ℝ) :
    ℕ → (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)
  | 0 => (fun j => (c j)⁻¹ • 1, fun j => (c j ^ p)⁻¹ • A j)
  | k + 1 => batchCnStep p (batchCnIterate p A c k)

/-- Batched NDB evaluates `ZY`, `YE` and `EZ` by batched products,
keeping the transpose-free multiplication order of the source.
Source: arXiv:2602.02016v2, §3.3 and §4, the stacked NDB implementation. -/
def batchNdbStep
    (state : (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)) :
    (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ) :=
  let ZY := batchMul state.2 state.1
  let E := fun j => (1 / 2 : ℝ) • ((3 : ℝ) • 1 - ZY j)
  (batchMul state.1 E, batchMul E state.2)

/-- The actual batched NDB state from `Y₀=A,Z₀=I`.
Source: arXiv:2602.02016v2, §3.3 and §4, all inverse-root blocks in one solve. -/
def batchNdbIterate (A : ι → Matrix (Fin n) (Fin n) ℝ) :
    ℕ → (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ)
  | 0 => (A, fun _ => 1)
  | k + 1 => batchNdbStep (batchNdbIterate A k)

/-- Every batched CN output is exactly its unbatched counterpart; the
proof follows the actual batched recurrence rather than defining the
batch result as a map of already completed single-matrix solves.
Source: arXiv:2602.02016v2, §4, unchanged per-block inverse-root computation. -/
theorem batchCnIterate_eq (p : ℕ) (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (c : ι → ℝ) (k : ℕ) :
    batchCnIterate p A c k =
      (fun j => (cnIterate p (A j) (c j) k).1, fun j => (cnIterate p (A j) (c j) k).2) := by
  induction k with
  | zero => rfl
  | succ k ih => rw [batchCnIterate, ih]; rfl

/-- Every batched NDB output equals the corresponding single-matrix
iteration, including both root buffers.
Source: arXiv:2602.02016v2, §4, the same inverse-root procedure on stacked blocks. -/
theorem batchNdbIterate_eq (A : ι → Matrix (Fin n) (Fin n) ℝ) (k : ℕ) :
    batchNdbIterate A k =
      (fun j => (ndbIterate (A j) k).1, fun j => (ndbIterate (A j) k).2) := by
  induction k with
  | zero => rfl
  | succ k ih => rw [batchNdbIterate, ih]; rfl

/-- Optimized first batched NDB step, with just the product `AE`.
Source: arXiv:2602.02016v2, §3.3 and §4, eliminating two identity products. -/
def batchNdbFirst (A : ι → Matrix (Fin n) (Fin n) ℝ) :
    (ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ) :=
  let E := fun j => (3 / 2 : ℝ) • 1 - (1 / 2 : ℝ) • A j
  (batchMul A E, E)

/-- The optimized batched initialization preserves every first-step
output, so it is valid for the entire stacked bucket.
Source: arXiv:2602.02016v2, §3.3 and §4, first-step optimization and batching. -/
theorem batchNdbFirst_eq (A : ι → Matrix (Fin n) (Fin n) ℝ) :
    batchNdbFirst A = batchNdbIterate A 1 := by
  rw [batchNdbIterate_eq]
  apply Prod.ext
  · funext j
    exact (congrArg Prod.fst (ndb_first_step (A j))).symm
  · funext j
    exact (congrArg Prod.snd (ndb_first_step (A j))).symm

end Transformer.DASH
