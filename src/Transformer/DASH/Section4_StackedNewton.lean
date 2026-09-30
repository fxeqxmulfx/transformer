/-
# DASH — exact bucket stacking for batched Newton solves

arXiv:2602.02016v2, §4. The source combines left and right full-block
preconditioners and compatible residual preconditioners into one bucket.
Running the actual batched recurrence on it preserves every output buffer.
-/

import Transformer.DASH.Section4_BatchedNewton

noncomputable section

namespace Transformer.DASH

variable {α β γ : Type*} {n : ℕ}

/-- One batched CN call on the combined bucket has exactly the same
output states as separate batched calls, with every input's own scale.
Source: arXiv:2602.02016v2, §4, `stack(L_full,R_full,R_rest)`. -/
theorem batchCnIterate_stackThree (p : ℕ)
    (A : α → Matrix (Fin n) (Fin n) ℝ) (B : β → Matrix (Fin n) (Fin n) ℝ)
    (C : γ → Matrix (Fin n) (Fin n) ℝ) (cA : α → ℝ) (cB : β → ℝ) (cC : γ → ℝ)
    (k : ℕ) :
    batchCnIterate p (stackThree A B C) (stackThree cA cB cC) k =
      (stackThree (batchCnIterate p A cA k).1 (batchCnIterate p B cB k).1
        (batchCnIterate p C cC k).1,
       stackThree (batchCnIterate p A cA k).2 (batchCnIterate p B cB k).2
        (batchCnIterate p C cC k).2) := by
  simp_rw [batchCnIterate_eq]
  apply Prod.ext
  · funext j
    rcases j with j | j
    · rfl
    · rcases j with j | j <;> rfl
  · funext j
    rcases j with j | j
    · rfl
    · rcases j with j | j <;> rfl

/-- One batched NDB recurrence on the source's combined bucket preserves
both the square-root and inverse-square-root buffers of each component.
Source: arXiv:2602.02016v2, §4, a single inverse-root call for stacked blocks. -/
theorem batchNdbIterate_stackThree
    (A : α → Matrix (Fin n) (Fin n) ℝ) (B : β → Matrix (Fin n) (Fin n) ℝ)
    (C : γ → Matrix (Fin n) (Fin n) ℝ) (k : ℕ) :
    batchNdbIterate (stackThree A B C) k =
      (stackThree (batchNdbIterate A k).1 (batchNdbIterate B k).1 (batchNdbIterate C k).1,
       stackThree (batchNdbIterate A k).2 (batchNdbIterate B k).2 (batchNdbIterate C k).2) := by
  simp_rw [batchNdbIterate_eq]
  apply Prod.ext
  · funext j
    rcases j with j | j
    · rfl
    · rcases j with j | j <;> rfl
  · funext j
    rcases j with j | j
    · rfl
    · rcases j with j | j <;> rfl

end Transformer.DASH
