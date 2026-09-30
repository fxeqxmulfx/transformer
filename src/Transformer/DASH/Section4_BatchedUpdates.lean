/-
# DASH — batched preconditioning, grafting and parameter updates

arXiv:2602.02016v2, §2 and §4. The two preconditioning products
are batched; grafting uses each block's own Frobenius norm.
The exact regularized direction uses the updated EMA history.
-/

import Transformer.DASH.Section4_BatchedCounts
import Transformer.DASH.Section4_BatchedHistory
import Transformer.DASH.Section2_GraftingDomain

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {m n : ℕ}

/-- Precondition the entire gradient bucket with two counted batched
products, retaining the left/right multiplication order.
Source: arXiv:2602.02016v2, §2.3 and §4, `U=L^(-1/4)GR^(-1/4)`. -/
def countedBatchPrecondition (P : ι → Matrix (Fin m) (Fin m) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) (Q : ι → Matrix (Fin n) (Fin n) ℝ) :
    StateM ℕ (ι → Matrix (Fin m) (Fin n) ℝ) := do
  let PG ← countedBatchMatmul P G
  countedBatchMatmul PG Q

/-- The two batched calls preserve every individual preconditioned
gradient. Source: arXiv:2602.02016v2, §4, blockwise Shampoo products. -/
theorem countedBatchPrecondition_correct (P : ι → Matrix (Fin m) (Fin m) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) (Q : ι → Matrix (Fin n) (Fin n) ℝ)
    (count : ℕ) :
    (countedBatchPrecondition P G Q).run count =
      (fun j => preconditionedGradient (P j) (G j) (Q j), count + 2) := by
  rfl

/-- Blockwise Adam grafting scales each block by its own pair of norms.
Source: arXiv:2602.02016v2, §2.3, blockwise grafting, and §4, stacked updates. -/
def batchGraft (U P : ι → Matrix (Fin m) (Fin n) ℝ) :
    ι → Matrix (Fin m) (Fin n) ℝ :=
  fun j => (frobeniusNorm (P j) / frobeniusNorm (U j)) • U j

/-- The parameter update on a stacked bucket, after per-block grafting.
Source: arXiv:2602.02016v2, §2.3 and §4, `θ_(t+1)=θ_t-η_t s_t U_t`. -/
def batchGraftedStep (η : ℝ) (θ U P : ι → Matrix (Fin m) (Fin n) ℝ) :
    ι → Matrix (Fin m) (Fin n) ℝ := fun j => θ j - η • batchGraft U P j

/-- Every stacked parameter update agrees with its original blockwise
grafted update. Source: arXiv:2602.02016v2, §2.3 and §4. -/
theorem batchGraftedStep_eq (η : ℝ) (θ U P : ι → Matrix (Fin m) (Fin n) ℝ) :
    batchGraftedStep η θ U P = fun j => graftedStep η (θ j) (U j) (P j) := by
  rfl

/-- Exact batched Shampoo direction after incorporating `G t`, the
manuscript's gradient at step `t+1`. The root buffers therefore use history
state `t+1`, not the previous state `t`.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §4, stacked preconditioning. -/
def batchShampooDirection (β ε : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hε : 0 < ε) (t : ℕ) :
    ι → Matrix (Fin m) (Fin n) ℝ :=
  batchMul
    (batchMul
      (fun j => evdInverseRoot (batchLeftHistory β G (t + 1) j + ε • 1)
        (regularized_posDef _ ε (batchHistory_posSemidef β G hβ hβ' (t + 1) j).1 hε) 4)
      (G t))
    (fun j => evdInverseRoot (batchRightHistory β G (t + 1) j + ε • 1)
      (regularized_posDef _ ε (batchHistory_posSemidef β G hβ hβ' (t + 1) j).2 hε) 4)

/-- The complete batched direction agrees with exact Shampoo using each
block's own temporal history. Source: arXiv:2602.02016v2, §2 and §4. -/
theorem batchShampooDirection_eq (β ε : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hε : 0 < ε) (t : ℕ) :
    batchShampooDirection β ε G hβ hβ' hε t = fun j =>
      preconditionedGradient
        (evdInverseRoot (leftHistory β (fun u => G u j) (t + 1) + ε • 1)
          (regularized_posDef _ ε
            (history_posSemidef β (fun u => G u j) hβ hβ' (t + 1)).1 hε) 4)
        (G t j)
        (evdInverseRoot (rightHistory β (fun u => G u j) (t + 1) + ε • 1)
          (regularized_posDef _ ε
            (history_posSemidef β (fun u => G u j) hβ hβ' (t + 1)).2 hε) 4) := by
  have hroot {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ)
      (hA : A.PosDef) (hB : B.PosDef) (h : A = B) :
      evdInverseRoot A hA 4 = evdInverseRoot B hB 4 := by
    cases h
    rfl
  funext j
  apply congrArg₂ (fun P Q => preconditionedGradient P (G t j) Q)
  · apply hroot
    exact congrArg (fun A => A + ε • 1) (congrFun (batchHistory_eq β G (t + 1)).1 j)
  · apply hroot
    exact congrArg (fun A => A + ε • 1) (congrFun (batchHistory_eq β G (t + 1)).2 j)

/-- EMA and regularization parameters for a nonempty batch exist,
arXiv:2602.02016v2, §2 and §4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

/-- At every nonzero-gradient block, grafting the actual regularized
batched history direction matches the reference norm. Positive regularization
and the EMA invariant supply the nonzero denominator.
Source: arXiv:2602.02016v2, §2.3 and §4, blockwise grafting. -/
theorem batchShampooGraft_norm (β ε : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (P : ι → Matrix (Fin m) (Fin n) ℝ) (hβ : 0 ≤ β) (hβ' : β ≤ 1)
    (hε : 0 < ε) (t : ℕ) (j : ι) (hG : G t j ≠ 0) :
    frobeniusNorm (batchGraft (batchShampooDirection β ε G hβ hβ' hε t) P j) =
      frobeniusNorm (P j) := by
  rw [batchShampooDirection_eq]
  exact regularized_graft_norm _ _ (G t j) (P j) ε
    (history_posSemidef β (fun u => G u j) hβ hβ' (t + 1)).1
    (history_posSemidef β (fun u => G u j) hβ hβ' (t + 1)).2 hε hG

/-- Nonzero gradients and valid parameters coexist in a nonempty batch,
arXiv:2602.02016v2, §2.3 and §4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (fun (_ : ℕ) (_ : Fin 2) => (1 : Matrix (Fin 1) (Fin 1) ℝ)) 0 0 ≠ 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro h
  have := congrFun (congrFun h 0) 0
  norm_num at this

/-- Reassembling the updated bucket is exactly the model parameter update
with the reassembled grafted direction, with no lost or duplicated entries.
Source: arXiv:2602.02016v2, §4, conversion back to the layer layout. -/
theorem unblock_batchGraftedStep {r c B : ℕ}
    (θ : Matrix (Fin (r * B)) (Fin (c * B)) ℝ)
    (U P : (Fin r × Fin c) → Matrix (Fin B) (Fin B) ℝ) (η : ℝ) :
    unblockGradient (batchGraftedStep η (blockGradient θ) U P) =
      θ - η • unblockGradient (batchGraft U P) :=
  unblockGradient_update θ (batchGraft U P) η

end Transformer.DASH
