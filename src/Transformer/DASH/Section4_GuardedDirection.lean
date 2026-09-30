/-
# DASH — the numerical inverse-root pipeline inside batched Shampoo

arXiv:2602.02016v2, §2–4. Both finite inverse-fourth-root buffers
feed the two actual batched preconditioning products. For temporal
Shampoo, regularization and the EMA invariant prove their input domain;
callers supply neither eigenframes nor a PI reliability assumption.
-/

import Transformer.DASH.Section4_GuardedNdb
import Transformer.DASH.Section4_BatchedUpdates
import Transformer.DASH.Section2_GraftingContinuity

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {m n : ℕ}

/-- Compute both automatically scaled finite inverse fourth roots and
apply them to the gradient using two counted batched products.
Source: arXiv:2602.02016v2, §2.2, §3.3–3.4 and §4, corrected numerical update. -/
def countedBatchGuardedDirection (L : ι → Matrix (Fin m) (Fin m) ℝ)
    (G : ι → Matrix (Fin m) (Fin n) ℝ) (R : ι → Matrix (Fin n) (Fin n) ℝ)
    (rL rR : ι → ℝ) (k : ℕ) : StateM ℕ (ι → Matrix (Fin m) (Fin n) ℝ) := do
  let P ← countedBatchGuardedInverseFourth L rL k
  let Q ← countedBatchGuardedInverseFourth R rR k
  countedBatchPrecondition P G Q

/-- The full pipeline uses the actual finite numerical buffers for each
block, with `4*max(3k-2,0)+2` batched products. PI, certification and scalar
operations are outside this root/preconditioning product count.
Source: arXiv:2602.02016v2, §2.2, §3.3 and §4, counted numerical Shampoo. -/
theorem countedBatchGuardedDirection_correct
    (L : ι → Matrix (Fin m) (Fin m) ℝ) (G : ι → Matrix (Fin m) (Fin n) ℝ)
    (R : ι → Matrix (Fin n) (Fin n) ℝ) (rL rR : ι → ℝ) (k count : ℕ) :
    (countedBatchGuardedDirection L G R rL rR k).run count =
      (fun j => preconditionedGradient (guardedInverseFourth (L j) (rL j) k) (G j)
        (guardedInverseFourth (R j) (rR j) k), count + 4 * (3 * k - 2) + 2) := by
  unfold countedBatchGuardedDirection
  rw [StateT.run_bind, countedBatchGuardedInverseFourth_correct]
  change ((countedBatchGuardedInverseFourth R rR k >>= fun Q =>
    countedBatchPrecondition (fun j => guardedInverseFourth (L j) (rL j) k) G Q).run
      (count + 2 * (3 * k - 2))) = _
  rw [StateT.run_bind, countedBatchGuardedInverseFourth_correct]
  change (countedBatchPrecondition (fun j => guardedInverseFourth (L j) (rL j) k)
    G (fun j => guardedInverseFourth (R j) (rR j) k)).run
      (count + 2 * (3 * k - 2) + 2 * (3 * k - 2)) = _
  rw [countedBatchPrecondition_correct]
  congr 1
  omega

/-- The numerical direction converges to the exact regularized Shampoo
direction for arbitrary candidate estimates. Both finite root calls and
the rectangular preconditioning products are included in this limit.
Source: arXiv:2602.02016v2, §2.2, §3.3–3.4 and §4. -/
theorem countedBatchGuardedDirection_convergence
    (L : ι → Matrix (Fin m) (Fin m) ℝ) (G : ι → Matrix (Fin m) (Fin n) ℝ)
    (R : ι → Matrix (Fin n) (Fin n) ℝ) (rL rR : ι → ℝ)
    (hL : ∀ j, (L j).PosDef) (hR : ∀ j, (R j).PosDef) :
    Tendsto (fun k : ℕ => ((countedBatchGuardedDirection L G R rL rR (k + 1)).run 0).1)
      atTop (𝓝 (fun j => preconditionedGradient
        (evdInverseRoot (L j) (hL j) 4) (G j) (evdInverseRoot (R j) (hR j) 4))) := by
  apply tendsto_pi_nhds.mpr
  intro j
  simpa only [countedBatchGuardedDirection_correct] using
    preconditionedGradient_tendsto _ _ _ _ (G j)
      (guardedInverseFourth_posDef (L j) (rL j) (hL j))
      (guardedInverseFourth_posDef (R j) (rR j) (hR j))

/-- A nonempty batch meets the numerical preconditioning input domain,
arXiv:2602.02016v2, §2.2, §3.3–3.4 and §4. -/
example : (∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef) ∧
    (∀ j : Fin 2, ((fun _ : Fin 2 => (1 : Matrix (Fin 1) (Fin 1) ℝ)) j).PosDef) := by
  constructor <;> intro j <;>
    simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

/-- The actual temporal multi-PI/NDB direction, after the current gradient
has updated both histories. It requires no proof arguments at run time.
Source: arXiv:2602.02016v2, §2, Algorithm 1, §3.3–3.5 and §4. -/
def countedBatchPiShampooDirection {κ : Type} [Fintype κ] [Nonempty κ]
    (β ε : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (startsL : ι → κ → Fin m → ℝ) (startsR : ι → κ → Fin n → ℝ)
    (t piSteps rootSteps : ℕ) : StateM ℕ (ι → Matrix (Fin m) (Fin n) ℝ) :=
  let L := fun j => batchLeftHistory β G (t + 1) j + ε • 1
  let R := fun j => batchRightHistory β G (t + 1) j + ε • 1
  countedBatchGuardedDirection L (G t) R
    (fun j => pooledRayleigh (L j) (startsL j) piSteps)
    (fun j => pooledRayleigh (R j) (startsR j) piSteps) rootSteps

/-- For any gradient history, valid EMA parameters and positive
regularization suffice for the complete multi-PI/NDB direction to converge
to exact Shampoo. The domain is derived from the actual histories.
Source: arXiv:2602.02016v2, §2, Algorithm 1, §3.3–3.5 and §4. -/
theorem countedBatchPiShampooDirection_convergence {κ : Type} [Fintype κ] [Nonempty κ]
    (β ε : ℝ) (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (startsL : ι → κ → Fin m → ℝ) (startsR : ι → κ → Fin n → ℝ)
    (t piSteps : ℕ) (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hε : 0 < ε) :
    Tendsto (fun k : ℕ =>
      ((countedBatchPiShampooDirection β ε G startsL startsR t piSteps (k + 1)).run 0).1)
      atTop (𝓝 (batchShampooDirection β ε G hβ hβ' hε t)) := by
  exact countedBatchGuardedDirection_convergence _ (G t) _ _ _
    (fun j => regularized_posDef _ ε
      (batchHistory_posSemidef β G hβ hβ' (t + 1) j).1 hε)
    (fun j => regularized_posDef _ ε
      (batchHistory_posSemidef β G hβ hβ' (t + 1) j).2 hε)

/-- Complete temporal Shampoo has admissible parameters,
arXiv:2602.02016v2, §2–4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.DASH
