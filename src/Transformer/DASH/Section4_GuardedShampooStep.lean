/-
# DASH — corrected numerical roots through the complete grafted update

arXiv:2602.02016v2, §2–4. The entire finite multi-PI/NDB pipeline
feeds the actual parameter update. Positive regularization and the
EMA invariant derive the root domain. Nonzero gradients derive the
grafting denominator; zero gradients give exactly zero directions.
-/

import Transformer.DASH.Section4_GuardedDirection

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {m n : ℕ}

/-- Complete numerical temporal Shampoo step with per-block grafting.
The reference direction `P` is supplied by the chosen grafting optimizer.
Source: arXiv:2602.02016v2, §2.3, §3.3–3.5 and §4, corrected parameter update. -/
def countedBatchPiShampooStep {κ : Type} [Fintype κ] [Nonempty κ]
    (β ε η : ℝ) (θ P : ι → Matrix (Fin m) (Fin n) ℝ)
    (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (startsL : ι → κ → Fin m → ℝ) (startsR : ι → κ → Fin n → ℝ)
    (t piSteps rootSteps : ℕ) : StateM ℕ (ι → Matrix (Fin m) (Fin n) ℝ) := do
  let U ← countedBatchPiShampooDirection β ε G startsL startsR t piSteps rootSteps
  pure (batchGraftedStep η θ U P)

/-- The complete step uses the actual finite numerical direction and
preserves the `4*max(3k-2,0)+2` root/preconditioning product count.
Grafting adds no matrix products; PI and certification are outside this count.
Source: arXiv:2602.02016v2, §2.3, §3.3 and §4, complete batched update. -/
theorem countedBatchPiShampooStep_correct {κ : Type} [Fintype κ] [Nonempty κ]
    (β ε η : ℝ) (θ P : ι → Matrix (Fin m) (Fin n) ℝ)
    (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (startsL : ι → κ → Fin m → ℝ) (startsR : ι → κ → Fin n → ℝ)
    (t piSteps rootSteps count : ℕ) :
    (countedBatchPiShampooStep β ε η θ P G startsL startsR t piSteps rootSteps).run count =
      (batchGraftedStep η θ
        ((countedBatchPiShampooDirection β ε G startsL startsR t piSteps rootSteps).run count).1
        P, count + 4 * (3 * rootSteps - 2) + 2) := by
  unfold countedBatchPiShampooStep
  rw [StateT.run_bind]
  apply Prod.ext
  · rfl
  · simp only [countedBatchPiShampooDirection, countedBatchGuardedDirection_correct]
    rfl

/-- The full finite multi-PI/NDB/grafting parameter update converges to
the exact Shampoo update at every fixed history step. Only valid EMA
parameters and positive regularization are required. No eigenframe, PI
accuracy, root-domain or nonzero-gradient assumption is supplied by callers.
At zero-gradient blocks both numerical and exact directions vanish.
Source: arXiv:2602.02016v2, §2.3, §3.3–3.5 and §4, corrected complete pipeline. -/
theorem countedBatchPiShampooStep_convergence {κ : Type} [Fintype κ] [Nonempty κ]
    (β ε η : ℝ) (θ P : ι → Matrix (Fin m) (Fin n) ℝ)
    (G : ℕ → ι → Matrix (Fin m) (Fin n) ℝ)
    (startsL : ι → κ → Fin m → ℝ) (startsR : ι → κ → Fin n → ℝ)
    (t piSteps : ℕ) (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hε : 0 < ε) :
    Tendsto (fun k : ℕ =>
      ((countedBatchPiShampooStep β ε η θ P G startsL startsR t piSteps (k + 1)).run 0).1)
      atTop (𝓝 (batchGraftedStep η θ (batchShampooDirection β ε G hβ hβ' hε t) P)) := by
  apply tendsto_pi_nhds.mpr
  intro j
  by_cases hG : G t j = 0
  · simpa only [countedBatchPiShampooStep_correct, countedBatchPiShampooDirection,
      countedBatchGuardedDirection_correct, batchGraftedStep, batchGraft,
      batchShampooDirection, batchMul, preconditionedGradient, hG, Matrix.mul_zero,
      Matrix.zero_mul, smul_zero, sub_zero] using (tendsto_const_nhds (x := θ j))
  · have hU := tendsto_pi_nhds.mp
      (countedBatchPiShampooDirection_convergence β ε G startsL startsR t piSteps
        hβ hβ' hε) j
    let L := batchLeftHistory β G (t + 1) j + ε • 1
    let R := batchRightHistory β G (t + 1) j + ε • 1
    have hL : L.PosDef := regularized_posDef _ ε
      (batchHistory_posSemidef β G hβ hβ' (t + 1) j).1 hε
    have hR : R.PosDef := regularized_posDef _ ε
      (batchHistory_posSemidef β G hβ hβ' (t + 1) j).2 hε
    obtain ⟨L', _, hL'⟩ := evdInverseRoot_invertible L hL 4
    obtain ⟨R', hR', _⟩ := evdInverseRoot_invertible R hR 4
    have hV : batchShampooDirection β ε G hβ hβ' hε t j ≠ 0 := by
      change preconditionedGradient (evdInverseRoot L hL 4) (G t j)
        (evdInverseRoot R hR 4) ≠ 0
      exact fun h => hG ((preconditionedGradient_eq_zero_iff _ L' _ R' _ hL' hR').mp h)
    have h := graftedStep_tendsto _ _ (θ j) (P j) η hU hV
    simpa only [countedBatchPiShampooStep_correct, batchGraftedStep, batchGraft,
      graftedStep, graft] using h

/-- A complete nonempty temporal update has admissible parameters,
arXiv:2602.02016v2, §2–4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.DASH
