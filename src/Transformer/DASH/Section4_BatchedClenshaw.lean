/-
# DASH — optimized Clenshaw with batched products

arXiv:2602.02016v2, §4 and Appendix A. Coefficients may differ between
batch entries, but all entries share the degree. Each multiplication
is an actual call to the counted `bmm` primitive.
-/

import Transformer.DASH.Section4_BatchedCounts
import Transformer.DASH.SectionA_OptimizedClenshaw

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- The optimized backward loop, with one coefficient function per degree
and one batched product per remaining recurrence step. The highest two
states require only scalar operations.
Source: arXiv:2602.02016v2, §4 and Appendix A, optimized Clenshaw loop. -/
def countedBatchClenshawState (S : ι → Matrix (Fin n) (Fin n) ℝ) :
    List (ι → ℝ) →
      StateM ℕ ((ι → Matrix (Fin n) (Fin n) ℝ) × (ι → Matrix (Fin n) (Fin n) ℝ))
  | [] => pure (fun _ => 0, fun _ => 0)
  | [c] => pure (fun j => c j • 1, fun _ => 0)
  | [c₁, c₂] => pure (fun j => (2 * c₂ j) • S j + c₁ j • 1, fun j => c₂ j • 1)
  | c₁ :: c₂ :: c₃ :: cs => do
      let b ← countedBatchClenshawState S (c₂ :: c₃ :: cs)
      let SB ← countedBatchMatmul S b.1
      pure (fun j => 2 • SB j - b.2 j + c₁ j • 1, b.1)

/-- Full optimized batched evaluation, including the final `bmm` and the
degree-zero/one boundary cases. Construction of `S` is excluded from the
product count, as in the source.
Source: arXiv:2602.02016v2, §4 and Appendix A, optimized Clenshaw output. -/
def countedBatchClenshaw (S : ι → Matrix (Fin n) (Fin n) ℝ) :
    List (ι → ℝ) → StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ)
  | [] => pure (fun _ => 0)
  | [c] => pure (fun j => c j • 1)
  | [c₀, c₁] => pure (fun j => c₁ j • S j + c₀ j • 1)
  | c₀ :: c₁ :: c₂ :: cs => do
      let b ← countedBatchClenshawState S (c₁ :: c₂ :: cs)
      let SB ← countedBatchMatmul S b.1
      pure (fun j => SB j - b.2 j + c₀ j • 1)

/-- Both backward state buffers equal the single-matrix optimized loop
at every batch index. The entire loop uses `max(length-2,0)` batched calls.
Source: arXiv:2602.02016v2, §4 and Appendix A, unchanged stacked Clenshaw. -/
theorem countedBatchClenshawState_correct (S : ι → Matrix (Fin n) (Fin n) ℝ)
    (cs : List (ι → ℝ)) (count : ℕ) :
    (countedBatchClenshawState S cs).run count =
      ((fun j => (optimizedClenshawState (S j) (cs.map (fun c => c j))).1.1,
        fun j => (optimizedClenshawState (S j) (cs.map (fun c => c j))).1.2),
        count + (cs.length - 2)) := by
  induction cs generalizing count with
  | nil => rfl
  | cons c cs ih =>
    cases cs with
    | nil => rfl
    | cons c' cs =>
      cases cs with
      | nil => rfl
      | cons c'' cs =>
        change ((countedBatchClenshawState S (c' :: c'' :: cs) >>= fun b => do
          let SB ← countedBatchMatmul S b.1
          pure (fun j => 2 • SB j - b.2 j + c j • 1, b.1)).run count) = _
        rw [StateT.run_bind, ih]
        change ((fun j => 2 • (S j *
            (optimizedClenshawState (S j) ((c' :: c'' :: cs).map (fun a => a j))).1.1) -
              (optimizedClenshawState (S j) ((c' :: c'' :: cs).map (fun a => a j))).1.2 +
                c j • 1,
            fun j => (optimizedClenshawState (S j)
              ((c' :: c'' :: cs).map (fun a => a j))).1.1),
            count + ((c' :: c'' :: cs).length - 2) + 1) = _
        simp only [List.map_cons, optimizedClenshawState, List.length_cons]
        congr 1

/-- Batched evaluation returns the single-matrix polynomial on every
component, using `d-1` batched products for degree `d≥2`, irrespective of
batch size. Degrees zero and one require none.
Source: arXiv:2602.02016v2, §4 and Appendix A, `d-1` products. -/
theorem countedBatchClenshaw_correct (S : ι → Matrix (Fin n) (Fin n) ℝ)
    (cs : List (ι → ℝ)) (count : ℕ) :
    (countedBatchClenshaw S cs).run count =
      (fun j => (optimizedClenshaw (S j) (cs.map (fun c => c j))).1,
        count + (cs.length - 2)) := by
  cases cs with
  | nil => rfl
  | cons c cs =>
    cases cs with
    | nil => rfl
    | cons c' cs =>
      cases cs with
      | nil => rfl
      | cons c'' cs =>
        change ((countedBatchClenshawState S (c' :: c'' :: cs) >>= fun b => do
          let SB ← countedBatchMatmul S b.1
          pure (fun j => SB j - b.2 j + c j • 1)).run count) = _
        rw [StateT.run_bind, countedBatchClenshawState_correct]
        change (fun j => S j *
            (optimizedClenshawState (S j) ((c' :: c'' :: cs).map (fun a => a j))).1.1 -
              (optimizedClenshawState (S j) ((c' :: c'' :: cs).map (fun a => a j))).1.2 +
                c j • 1,
            count + ((c' :: c'' :: cs).length - 2) + 1) = _
        simp only [List.map_cons, optimizedClenshaw, List.length_cons]
        congr 1

end Transformer.DASH
